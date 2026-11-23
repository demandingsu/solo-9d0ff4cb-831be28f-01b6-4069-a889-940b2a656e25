"""核心计算单测：窗口语义、配对、10 分钟连续运行、剔除原因、受影响轮次。"""

import pytest

from coldstore.analysis import (
    MIN_RUN_MINUTES,
    REASON_MISSING_A,
    REASON_MISSING_B,
    REASON_RUN_A,
    REASON_RUN_B,
    REASON_STRATEGY_B,
    REASON_UNKNOWN_A,
    RoundPlan,
    SwitchRecord,
    ValidationError,
    compute_round,
    impacted_round_ids,
    parse_ts,
)


def plan(a_start, a_end, b_start, b_end, rid="R1"):
    return RoundPlan(rid, a_start, a_end, b_start, b_end)


def readings_for(pairs, base=0.0):
    """给 A/B 侧全部分钟生成读数：偏移越大读数越高，便于核对差值。"""
    out = {}
    for minute, value in pairs:
        out[minute] = value
    return out


class TestTimeParsing:
    def test_accepts_z_and_offset(self):
        assert parse_ts("2026-03-01T00:05Z") == parse_ts("2026-03-01 00:05")
        assert parse_ts("2026-03-01T00:05+00:00") == parse_ts("2026-03-01 00:05")

    @pytest.mark.parametrize(
        "value", ["2026-03-01 00:05:30", "2026-03-01T00:05+08:00", "", "not-a-time"]
    )
    def test_rejects_non_minute_or_non_utc(self, value):
        with pytest.raises(ValidationError):
            parse_ts(value)

    def test_window_must_be_equal_and_non_overlapping(self):
        with pytest.raises(ValidationError, match="不等长"):
            plan("2026-03-01 00:00", "2026-03-01 01:00", "2026-03-01 02:00", "2026-03-01 02:30").validate()
        with pytest.raises(ValidationError, match="重叠"):
            plan("2026-03-01 00:00", "2026-03-01 01:00", "2026-03-01 00:59", "2026-03-01 01:59").validate()

    def test_adjacent_windows_allowed(self):
        plan("2026-03-01 00:00", "2026-03-01 01:00", "2026-03-01 01:00", "2026-03-01 02:00").validate()


class TestWindowSemantics:
    def test_window_includes_start_excludes_end(self):
        p = plan("2026-03-01 00:00", "2026-03-01 00:03", "2026-03-01 01:00", "2026-03-01 01:03")
        switches = [
            SwitchRecord("2026-02-28 23:00", "A"),
            SwitchRecord("2026-03-01 00:50", "B"),
        ]
        readings = {
            "2026-03-01 00:00": 1.0,
            "2026-03-01 00:01": 1.1,
            "2026-03-01 00:02": 1.2,
            "2026-03-01 00:03": 99.0,  # 终点分钟：不属于 A 窗
            "2026-03-01 01:00": 1.5,
            "2026-03-01 01:01": 1.6,
            "2026-03-01 01:02": 1.7,
            "2026-03-01 01:03": 99.0,  # 终点分钟：不属于 B 窗
        }
        result = compute_round(p, switches, readings)
        assert result.valid_pairs == 3
        assert [pair.a_minute for pair in result.pairs] == [
            "2026-03-01 00:00",
            "2026-03-01 00:01",
            "2026-03-01 00:02",
        ]
        assert all(pair.b_minute != "2026-03-01 01:03" for pair in result.pairs)


class TestPairingAndRunRequirement:
    def test_pairs_by_offset_b_minus_a(self):
        p = plan("2026-03-01 00:00", "2026-03-01 00:02", "2026-03-01 01:00", "2026-03-01 01:02")
        switches = [
            SwitchRecord("2026-02-28 23:00", "A"),
            SwitchRecord("2026-03-01 00:50", "B"),
        ]
        readings = {
            "2026-03-01 00:00": 1.0,
            "2026-03-01 00:01": 1.5,
            "2026-03-01 01:00": 1.25,
            "2026-03-01 01:01": 1.75,
        }
        result = compute_round(p, switches, readings)
        assert result.valid_pairs == 2
        assert result.mean_diff_kwh == pytest.approx(0.25)
        assert [p_.diff_kwh for p_ in result.pairs] == [0.25, 0.25]

    def test_exactly_ten_minutes_before_window_is_included(self):
        p = plan("2026-03-01 00:00", "2026-03-01 00:02", "2026-03-01 01:00", "2026-03-01 01:02")
        switches = [
            SwitchRecord("2026-02-28 23:50", "A"),  # 恰好 10 分钟
            SwitchRecord("2026-03-01 00:50", "B"),   # 恰好 10 分钟
        ]
        readings = {
            "2026-03-01 00:00": 1.0,
            "2026-03-01 00:01": 1.0,
            "2026-03-01 01:00": 2.0,
            "2026-03-01 01:01": 2.0,
        }
        result = compute_round(p, switches, readings)
        assert result.valid_pairs == 2
        assert result.mean_diff_kwh == pytest.approx(1.0)

    def test_nine_minutes_is_excluded(self):
        p = plan("2026-03-01 00:00", "2026-03-01 00:02", "2026-03-01 01:00", "2026-03-01 01:02")
        switches = [
            SwitchRecord("2026-02-28 23:51", "A"),  # 只有 9 分钟
            SwitchRecord("2026-03-01 00:50", "B"),
        ]
        readings = {
            "2026-03-01 00:00": 1.0,
            "2026-03-01 00:01": 1.0,
            "2026-03-01 01:00": 2.0,
            "2026-03-01 01:01": 2.0,
        }
        result = compute_round(p, switches, readings)
        # 00:00 时只运行了 9 分钟；00:01 时正好 10 分钟，故仅偏移 0 被剔除
        assert result.valid_pairs == 1
        assert result.pairs[0].reasons == [REASON_RUN_A]
        assert result.pairs[1].included is True

    def test_run_requirement_advances_minute_by_minute(self):
        """窗首 3 分钟未满 10 分钟，之后合格；后面的分钟照常纳入。"""
        p = plan("2026-03-01 00:00", "2026-03-01 00:05", "2026-03-01 01:00", "2026-03-01 01:05")
        switches = [
            SwitchRecord("2026-02-28 23:53", "A"),  # 00:00 时已运行 7 分钟
            SwitchRecord("2026-03-01 00:50", "B"),
        ]
        readings = {
            f"2026-03-01 00:0{i}": 1.0 for i in range(5)
        } | {
            f"2026-03-01 01:00": 2.0,
            "2026-03-01 01:01": 2.0,
            "2026-03-01 01:02": 2.0,
            "2026-03-01 01:03": 2.0,
            "2026-03-01 01:04": 2.0,
        }
        result = compute_round(p, switches, readings)
        included = [pair.offset for pair in result.pairs if pair.included]
        assert included == [3, 4]
        assert [p_.reasons for p_ in result.pairs[:3]] == [[REASON_RUN_A]] * 3

    def test_missing_reading_reported_per_side(self):
        p = plan("2026-03-01 00:00", "2026-03-01 00:03", "2026-03-01 01:00", "2026-03-01 01:03")
        switches = [
            SwitchRecord("2026-02-28 23:00", "A"),
            SwitchRecord("2026-03-01 00:50", "B"),
        ]
        readings = {
            # 偏移 0：缺 A；偏移 1：缺 B；偏移 2：两侧齐全
            "2026-03-01 00:01": 1.0,
            "2026-03-01 00:02": 1.0,
            "2026-03-01 01:00": 2.0,
            "2026-03-01 01:02": 2.0,
        }
        result = compute_round(p, switches, readings)
        assert result.valid_pairs == 1
        reasons = {pair.offset: pair.reasons for pair in result.pairs}
        assert reasons[0] == [REASON_MISSING_A]
        assert reasons[1] == [REASON_MISSING_B]
        assert reasons[2] == []
        assert result.pairs[0].included is False and result.pairs[1].included is False

    def test_both_missing_reports_both(self):
        p = plan("2026-03-01 00:00", "2026-03-01 00:01", "2026-03-01 01:00", "2026-03-01 01:01")
        switches = [
            SwitchRecord("2026-02-28 23:00", "A"),
            SwitchRecord("2026-03-01 00:50", "B"),
        ]
        result = compute_round(p, switches, {})
        assert result.pairs[0].reasons == [REASON_MISSING_A, REASON_MISSING_B]
        assert result.comparable is False

    def test_wrong_strategy_in_b_window(self):
        p = plan("2026-03-01 00:00", "2026-03-01 00:02", "2026-03-01 01:00", "2026-03-01 01:02")
        switches = [SwitchRecord("2026-02-28 23:00", "A")]  # B 窗内仍是 A
        readings = {"2026-03-01 00:00": 1.0, "2026-03-01 00:01": 1.0, "2026-03-01 01:00": 1.0, "2026-03-01 01:01": 1.0}
        result = compute_round(p, switches, readings)
        assert result.valid_pairs == 0
        assert all(REASON_STRATEGY_B in pair.reasons for pair in result.pairs)

    def test_no_switch_before_window_is_unknown(self):
        p = plan("2026-03-01 00:00", "2026-03-01 00:02", "2026-03-01 01:00", "2026-03-01 01:02")
        switches = [SwitchRecord("2026-03-01 00:30", "B")]
        readings = {"2026-03-01 00:00": 1.0, "2026-03-01 00:01": 1.0, "2026-03-01 01:00": 1.0, "2026-03-01 01:01": 1.0}
        result = compute_round(p, switches, readings)
        assert result.valid_pairs == 0
        assert REASON_UNKNOWN_A in result.pairs[0].reasons

    def test_round_scoped_switches_do_not_leak(self):
        p1 = plan("2026-03-01 00:00", "2026-03-01 00:02", "2026-03-01 01:00", "2026-03-01 01:02", "R1")
        p2 = plan("2026-03-01 00:00", "2026-03-01 00:02", "2026-03-01 01:00", "2026-03-01 01:02", "R2")
        switches = [
            SwitchRecord("2026-02-28 23:00", "A"),
            SwitchRecord("2026-03-01 00:50", "B", "R1"),
        ]
        readings = {"2026-03-01 00:00": 1.0, "2026-03-01 00:01": 1.0, "2026-03-01 01:00": 2.0, "2026-03-01 01:01": 2.0}
        assert compute_round(p1, switches, readings).valid_pairs == 2
        assert compute_round(p2, switches, readings).valid_pairs == 0

    def test_mean_uses_only_included_pairs(self):
        p = plan("2026-03-01 00:00", "2026-03-01 00:03", "2026-03-01 01:00", "2026-03-01 01:03")
        switches = [
            SwitchRecord("2026-02-28 23:00", "A"),
            SwitchRecord("2026-03-01 00:50", "B"),
        ]
        readings = {
            "2026-03-01 00:00": 1.0,
            "2026-03-01 00:01": 1.0,
            "2026-03-01 00:02": 1.0,
            "2026-03-01 01:00": 3.0,
            "2026-03-01 01:01": 2.0,
            # 01:02 缺失
        }
        result = compute_round(p, switches, readings)
        assert result.valid_pairs == 2
        assert result.mean_diff_kwh == pytest.approx(1.5)  # (2.0 + 1.0)/2，不含缺失对


class TestImpactedRounds:
    def test_switch_outside_window_changes_ten_minute_status(self):
        """跨窗更正：12:50 在 B 窗 [13:00,13:30) 之外，但改成 12:51 后窗首不再满 10 分钟。"""
        p = plan("2026-03-01 12:00", "2026-03-01 12:30", "2026-03-01 13:00", "2026-03-01 13:30", "R4")
        before = [
            SwitchRecord("2026-03-01 11:50", "A", "R4", 1),
            SwitchRecord("2026-03-01 12:50", "B", "R4", 2),
        ]
        after = [
            SwitchRecord("2026-03-01 11:50", "A", "R4", 1),
            SwitchRecord("2026-03-01 12:51", "B", "R4", 2),
        ]
        assert impacted_round_ids(before, after, [p]) == {"R4"}

    def test_switch_only_shifts_outside_window_does_not_matter(self):
        """12:50 → 12:45 仍满 10 分钟，窗内状态不变 → 不重算。"""
        p = plan("2026-03-01 12:00", "2026-03-01 12:30", "2026-03-01 13:00", "2026-03-01 13:30", "R4")
        before = [
            SwitchRecord("2026-03-01 11:50", "A", "R4", 1),
            SwitchRecord("2026-03-01 12:50", "B", "R4", 2),
        ]
        after = [
            SwitchRecord("2026-03-01 11:50", "A", "R4", 1),
            SwitchRecord("2026-03-01 12:45", "B", "R4", 2),
        ]
        assert impacted_round_ids(before, after, [p]) == set()

    def test_only_affected_round_is_returned(self):
        p1 = plan("2026-03-01 00:00", "2026-03-01 00:10", "2026-03-01 01:00", "2026-03-01 01:10", "R1")
        p2 = plan("2026-03-01 05:00", "2026-03-01 05:10", "2026-03-01 06:00", "2026-03-01 06:10", "R2")
        before = [
            SwitchRecord("2026-02-28 23:00", "A", None, 1),
            SwitchRecord("2026-03-01 00:50", "B", "R1", 2),
            SwitchRecord("2026-03-01 05:50", "B", "R2", 3),
        ]
        after = [
            SwitchRecord("2026-02-28 23:00", "A", None, 1),
            SwitchRecord("2026-03-01 00:55", "B", "R1", 2),  # R1 窗内状态变化：仍满 10 分钟?
            SwitchRecord("2026-03-01 05:50", "B", "R2", 3),
        ]
        # 00:55 距 01:00 仅 5 分钟 → R1 前 5 分钟短运行；R2 不受影响
        assert impacted_round_ids(before, after, [p1, p2]) == {"R1"}

    def test_scope_change_affects_rounds(self):
        p1 = plan("2026-03-01 00:00", "2026-03-01 00:10", "2026-03-01 01:00", "2026-03-01 01:10", "R1")
        p2 = plan("2026-03-01 00:00", "2026-03-01 00:10", "2026-03-01 01:00", "2026-03-01 01:10", "R2")
        base = [SwitchRecord("2026-02-28 23:00", "A", None, 1)]
        scoped = base + [SwitchRecord("2026-03-01 00:50", "B", "R1", 2)]
        changed = base + [SwitchRecord("2026-03-01 00:50", "B", None, 2)]
        assert impacted_round_ids(scoped, changed, [p1, p2]) == {"R2"}
