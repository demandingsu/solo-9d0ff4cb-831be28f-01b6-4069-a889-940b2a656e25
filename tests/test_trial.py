"""峰段转移试算：输入校验、上下界算法、缺测/witness、峰段可不同。

上下界用小规模穷举（角点 0 / cap，以及内部点）交叉验证，避免只测自洽性。
"""

from __future__ import annotations

import itertools

import pytest

from coldstore.analysis import (
    PairResult,
    RoundPlan,
    RoundResult,
    SwitchRecord,
    compute_round,
)
from coldstore.trial import (
    EPS_KW,
    VERDICT_CERTAIN_DOWN,
    VERDICT_CERTAIN_UP,
    VERDICT_FLAT,
    VERDICT_UNCERTAIN,
    run_trial,
)

# ---------------------------------------------------------------------------
# 构造工具
# ---------------------------------------------------------------------------


def make_plan(length=30, rid="R1"):
    return RoundPlan(
        rid,
        "2026-03-01 00:00",
        f"2026-03-01 00:{length:02d}",
        "2026-03-01 01:00",
        f"2026-03-01 01:{length:02d}",
    )


SWITCHES_OK = [
    SwitchRecord("2026-02-28 23:00", "A"),
    SwitchRecord("2026-03-01 00:50", "B"),
]


def flat_readings(plan: RoundPlan, a_kwh=1.0, b_kwh=0.9, extra=()):
    """A/B 窗匀速读数；extra 给出 A 窗之外的计费分钟读数。"""
    length = plan.length_minutes
    out = {}
    for i in range(length):
        out[f"2026-03-01 00:{i:02d}"] = a_kwh
        out[f"2026-03-01 01:{i:02d}"] = b_kwh
    for stamp, value in extra:
        out[stamp] = value
    return out


def peak_kw(energy: list[float]) -> float:
    return 4.0 * max(
        sum(energy[k : k + 15]) for k in range(len(energy) - 14)
    )


def brute_force_bounds(plan, round_result, readings, start, end, cap, samples):
    """穷举缺测分钟在 samples 中的所有组合，返回 (min D, max D)（kW）。"""
    from coldstore.analysis import minutes_between

    n = minutes_between(start, end)
    base = [readings.get(f"2026-03-01 00:{i:02d}") for i in range(n)]
    delta = [0.0] * n
    for p in round_result.pairs:
        if p.included:
            idx = int(p.a_minute[-2:])
            delta[idx] = p.b_kwh - p.a_kwh
    missing_idx = [i for i, v in enumerate(base) if v is None]
    diffs = []
    for combo in itertools.product(samples, repeat=len(missing_idx)):
        before = [v if v is not None else 0.0 for v in base]
        for i, x in zip(missing_idx, combo):
            before[i] = x
        after = [before[i] + delta[i] for i in range(n)]
        diffs.append(peak_kw(after) - peak_kw(before))
    return min(diffs), max(diffs)


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


class TestValidation:
    def test_billing_must_be_utc_whole_minute(self):
        plan = make_plan()
        rr = compute_round(plan, SWITCHES_OK, flat_readings(plan))
        with pytest.raises(Exception):
            run_trial(plan, rr, {}, "2026-03-01 00:00:30", "2026-03-01 00:30", 1.0)
        with pytest.raises(Exception):
            run_trial(plan, rr, {}, "2026-03-01 00:00+08:00", "2026-03-01 00:30", 1.0)

    def test_billing_duration_bounds(self):
        plan = make_plan()
        rr = compute_round(plan, SWITCHES_OK, flat_readings(plan))
        # 14 分钟：低于下限
        with pytest.raises(Exception, match="至少 15 分钟"):
            run_trial(
                plan, rr, flat_readings(plan),
                "2026-03-01 00:00", "2026-03-01 00:14", 1.0,
            )
        # 24 小时零 1 分钟：超过上限
        with pytest.raises(Exception, match="24 小时"):
            run_trial(
                plan, rr, flat_readings(plan),
                "2026-03-01 00:00", "2026-03-02 00:01", 1.0,
            )
        # 恰好 24 小时：允许
        ok = run_trial(
            plan, rr, flat_readings(plan),
            "2026-03-01 00:00", "2026-03-02 00:00", 1.0,
        )
        assert ok.duration_minutes == 1440

    def test_billing_must_cover_whole_a_window(self):
        plan = make_plan(30)
        rr = compute_round(plan, SWITCHES_OK, flat_readings(plan))
        with pytest.raises(Exception, match="完整覆盖"):
            run_trial(
                plan, rr, flat_readings(plan),
                "2026-03-01 00:01", "2026-03-01 00:31", 1.0,
            )
        with pytest.raises(Exception, match="完整覆盖"):
            run_trial(
                plan, rr, flat_readings(plan),
                "2026-03-01 00:00", "2026-03-01 00:29", 1.0,
            )

    def test_cap_must_be_non_negative_finite(self):
        from coldstore.analysis import ValidationError

        plan = make_plan()
        rr = compute_round(plan, SWITCHES_OK, flat_readings(plan))
        with pytest.raises(ValidationError):
            run_trial(plan, rr, flat_readings(plan),
                      "2026-03-01 00:00", "2026-03-01 00:30", -0.1)
        with pytest.raises(ValidationError):
            run_trial(plan, rr, flat_readings(plan),
                      "2026-03-01 00:00", "2026-03-01 00:30", "not-a-number")
        with pytest.raises(ValidationError):
            run_trial(plan, rr, flat_readings(plan),
                      "2026-03-01 00:00", "2026-03-01 00:30", float("inf"))

    def test_no_valid_pair_prompts(self):
        # B 窗内实际仍是 A → 全部配对无效
        plan = make_plan(20)
        rr = compute_round(plan, [SwitchRecord("2026-02-28 23:00", "A")],
                           flat_readings(plan))
        assert rr.valid_pairs == 0
        with pytest.raises(Exception, match="没有可替换的有效分钟对"):
            run_trial(plan, rr, flat_readings(plan),
                      "2026-03-01 00:00", "2026-03-01 00:20", 1.0)


# ---------------------------------------------------------------------------
# 无缺测：峰段转移、单位与结论
# ---------------------------------------------------------------------------


class TestDeterministicTrials:
    def test_uniform_shift_is_certain_down(self):
        plan = make_plan(30)
        rr = compute_round(plan, SWITCHES_OK, flat_readings(plan, 1.0, 0.9))
        res = run_trial(
            plan, rr, flat_readings(plan, 1.0, 0.9),
            "2026-03-01 00:00", "2026-03-01 00:30", 1.0,
        )
        # 每个 15 分钟窗电量降 15*0.1=1.5 kWh → 平均功率降 6 kW
        assert res.n_replaced == 30
        assert res.n_missing == 0
        assert res.diff_min_kw == pytest.approx(-6.0)
        assert res.diff_max_kw == pytest.approx(-6.0)
        assert res.verdict == VERDICT_CERTAIN_DOWN
        assert res.certain_reduction is True
        assert res.witness is None
        # 峰值功率 = 15 kWh / 0.25h = 60 kW
        assert res.peak_before_at_zero.mean_power_kw == pytest.approx(60.0)
        assert res.peak_after_at_zero.mean_power_kw == pytest.approx(54.0)

    def test_uniform_increase_is_certain_up(self):
        plan = make_plan(20)
        rr = compute_round(plan, SWITCHES_OK, flat_readings(plan, 1.0, 1.2))
        res = run_trial(
            plan, rr, flat_readings(plan, 1.0, 1.2),
            "2026-03-01 00:00", "2026-03-01 00:20", 1.0,
        )
        assert res.verdict == VERDICT_CERTAIN_UP
        assert res.diff_min_kw == pytest.approx(12.0)
        assert res.diff_max_kw == pytest.approx(12.0)

    def test_identical_readings_are_flat(self):
        plan = make_plan(20)
        rr = compute_round(plan, SWITCHES_OK, flat_readings(plan, 1.0, 1.0))
        res = run_trial(
            plan, rr, flat_readings(plan, 1.0, 1.0),
            "2026-03-01 00:00", "2026-03-01 00:20", 1.0,
        )
        assert res.verdict == VERDICT_FLAT
        assert abs(res.diff_min_kw) <= EPS_KW and abs(res.diff_max_kw) <= EPS_KW

    def test_peak_window_can_differ_between_scenarios(self):
        """替换后峰段从窗 0 转移到另一个窗：两种情形峰段可以不同。"""
        plan = make_plan(20)
        readings = flat_readings(plan, 1.0, 1.0)
        # 替换前峰值在窗 0（分钟 0..14 抬高）；替换后 B 只在分钟 15..19 抬高，
        # 使替换后峰段转移到窗 5（分钟 5..19）。
        for i in range(15):
            readings[f"2026-03-01 00:{i:02d}"] = 2.0
        for i in range(15, 20):
            readings[f"2026-03-01 01:{i:02d}"] = 3.0
        rr = compute_round(plan, SWITCHES_OK, readings)
        res = run_trial(
            plan, rr, readings,
            "2026-03-01 00:00", "2026-03-01 00:20", 1.0,
        )
        assert res.peak_before_at_zero.start == "2026-03-01 00:00"
        assert res.peak_after_at_zero.start == "2026-03-01 00:05"
        assert res.peak_after_at_zero.end == "2026-03-01 00:20"


# ---------------------------------------------------------------------------
# 缺测：上下界与 witness
# ---------------------------------------------------------------------------


class TestMissingMinutes:
    def _nonuniform_case(self, length=30, billing=60):
        plan = make_plan(length)
        readings = {}
        for i in range(billing):
            readings[f"2026-03-01 00:{i:02d}"] = 1.0 + 0.05 * ((i * 3) % 7)
        for i in range(length):
            j = i
            readings[f"2026-03-01 01:{i:02d}"] = (
                1.0 + 0.05 * ((j * 3) % 7)
                + (0.2 if i % 3 == 0 else -0.15)
            )
        return plan, readings

    def test_bounds_match_exhaustive_vertex_enumeration(self):
        plan, readings = self._nonuniform_case()
        missing = ["2026-03-01 00:02", "2026-03-01 00:30", "2026-03-01 00:55"]
        for m in missing:
            del readings[m]
        rr = compute_round(plan, SWITCHES_OK, readings)
        cap = 1.25
        res = run_trial(
            plan, rr, readings,
            "2026-03-01 00:00", "2026-03-01 01:00", cap,
        )
        assert res.n_missing == 3
        lo, hi = brute_force_bounds(
            plan, rr, readings,
            "2026-03-01 00:00", "2026-03-01 01:00", cap, [0.0, cap],
        )
        assert res.diff_min_kw == pytest.approx(lo, abs=1e-9)
        assert res.diff_max_kw == pytest.approx(hi, abs=1e-9)

    def test_interior_values_stay_inside_bounds(self):
        plan, readings = self._nonuniform_case(length=20, billing=40)
        missing = ["2026-03-01 00:07", "2026-03-01 00:33"]
        for m in missing:
            del readings[m]
        rr = compute_round(plan, SWITCHES_OK, readings)
        cap = 1.5
        res = run_trial(
            plan, rr, readings,
            "2026-03-01 00:00", "2026-03-01 00:40", cap,
        )
        _lo, hi = brute_force_bounds(
            plan, rr, readings,
            "2026-03-01 00:00", "2026-03-01 00:40", cap,
            [0.0, cap / 3, cap / 2, cap],
        )
        # 穷举内部点得到的最大值不可能超过算法给出的全局上界
        assert hi <= res.diff_max_kw + 1e-9

    def test_missing_value_shared_between_scenarios(self):
        """缺测分钟同时出现在替换前后的能量序列里（同一变量），不能替换成 B。"""
        plan = make_plan(15)
        readings = flat_readings(plan)
        del readings["2026-03-01 00:07"]  # A 窗缺测 → 该对无效，不产生替换
        rr = compute_round(plan, SWITCHES_OK, readings)
        replaced_minutes = {p.a_minute for p in rr.pairs if p.included}
        assert "2026-03-01 00:07" not in replaced_minutes
        res = run_trial(
            plan, rr, readings,
            "2026-03-01 00:00", "2026-03-01 00:15", 2.0,
        )
        assert res.missing_minutes == ["2026-03-01 00:07"]
        assert res.n_replaced == 14
        # 缺测取 0：窗能量 14 kWh；取上限 2：窗能量 14+2=16 kWh
        assert res.peak_before_at_zero.energy_kwh == pytest.approx(14.0)
        assert res.peak_before_at_cap.energy_kwh == pytest.approx(16.0)

    def test_witness_present_only_when_upper_bound_positive(self):
        plan, readings = self._nonuniform_case()
        missing = ["2026-03-01 00:02", "2026-03-01 00:30", "2026-03-01 00:55"]
        for m in missing:
            del readings[m]
        rr = compute_round(plan, SWITCHES_OK, readings)
        res = run_trial(
            plan, rr, readings,
            "2026-03-01 00:00", "2026-03-01 01:00", 1.25,
        )
        if res.diff_max_kw > EPS_KW:
            assert res.witness is not None
            assert res.witness.demand_diff_kw == pytest.approx(res.diff_max_kw)
            # witness 赋值必须落在 [0, cap]，且能复现上界
            n = 60
            base = [readings.get(f"2026-03-01 00:{i:02d}") for i in range(n)]
            delta = [0.0] * n
            for p in rr.pairs:
                if p.included:
                    delta[int(p.a_minute[-2:])] = p.b_kwh - p.a_kwh
            values = {a.minute: a.kwh for a in res.witness.assignments}
            before = [v if v is not None else 0.0 for v in base]
            for stamp, x in values.items():
                before[int(stamp[-2:])] = x
            after = [before[i] + delta[i] for i in range(n)]
            assert peak_kw(after) - peak_kw(before) == pytest.approx(
                res.diff_max_kw, abs=1e-9
            )
            for a in res.witness.assignments:
                assert a.kwh in (0.0, pytest.approx(1.25))
        else:
            assert res.witness is None

    def test_zero_cap_collapses_uncertainty(self):
        plan = make_plan(15)
        readings = flat_readings(plan, 1.0, 0.95)
        del readings["2026-03-01 00:03"]
        rr = compute_round(plan, SWITCHES_OK, readings)
        res = run_trial(
            plan, rr, readings,
            "2026-03-01 00:00", "2026-03-01 00:15", 0.0,
        )
        assert res.diff_min_kw == pytest.approx(res.diff_max_kw)
        assert res.n_missing == 1

    def test_uncertain_verdict_when_sign_can_change(self):
        # 冒烟场景：B 全面低 0.1，但一个缺测分钟可让替换后峰段不占它 → 差从 -4 到 0
        plan = make_plan(30)
        readings = flat_readings(plan, 1.0, 0.9)
        for i in range(30, 45):
            readings[f"2026-03-01 00:{i:02d}"] = 1.0
        del readings["2026-03-01 00:10"]
        rr = compute_round(plan, SWITCHES_OK, readings)
        res = run_trial(
            plan, rr, readings,
            "2026-03-01 00:00", "2026-03-01 00:45", 2.0,
        )
        assert res.verdict == VERDICT_UNCERTAIN
        assert res.diff_min_kw == pytest.approx(-4.0)
        assert abs(res.diff_max_kw) <= 1e-9
        assert res.certain_reduction is False


# ---------------------------------------------------------------------------
# 边界与数据结构
# ---------------------------------------------------------------------------


class TestStructureAndEdges:
    def test_minimum_15_minute_billing_window(self):
        plan = make_plan(15)
        rr = compute_round(plan, SWITCHES_OK, flat_readings(plan))
        res = run_trial(
            plan, rr, flat_readings(plan),
            "2026-03-01 00:00", "2026-03-01 00:15", 1.0,
        )
        # 只有一个 15 分钟窗
        assert res.peak_before_at_zero.start == "2026-03-01 00:00"
        assert res.peak_before_at_zero.end == "2026-03-01 00:15"

    def test_billing_extends_beyond_a_window(self):
        plan = make_plan(20)
        extra = [(f"2026-03-01 00:{i:02d}", 0.5) for i in range(20, 35)]
        readings = flat_readings(plan, extra=extra)
        rr = compute_round(plan, SWITCHES_OK, readings)
        res = run_trial(
            plan, rr, readings,
            "2026-03-01 00:00", "2026-03-01 00:35", 1.0,
        )
        assert res.duration_minutes == 35
        assert res.n_replaced == 20  # 只有 A 窗内的有效对被替换

    def test_as_dict_is_json_friendly(self):
        plan = make_plan(15)
        rr = compute_round(plan, SWITCHES_OK, flat_readings(plan))
        res = run_trial(
            plan, rr, flat_readings(plan),
            "2026-03-01 00:00", "2026-03-01 00:15", 1.0,
        )
        import json

        payload = json.dumps(res.as_dict(), ensure_ascii=False)
        assert json.loads(payload)["round_id"] == "R1"

    def test_replacement_uses_b_minus_a_delta(self):
        plan = make_plan(15)
        rr = compute_round(plan, SWITCHES_OK, flat_readings(plan, 1.0, 0.8))
        res = run_trial(
            plan, rr, flat_readings(plan, 1.0, 0.8),
            "2026-03-01 00:00", "2026-03-01 00:15", 1.0,
        )
        assert all(r.delta_kwh == pytest.approx(-0.2) for r in res.replacements)
        assert {r.minute for r in res.replacements} == {
            f"2026-03-01 00:{i:02d}" for i in range(15)
        }
