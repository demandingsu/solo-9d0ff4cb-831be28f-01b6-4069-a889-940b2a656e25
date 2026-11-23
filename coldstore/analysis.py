"""冷库 A/B 除霜策略耗电对比 —— 核心计算（无 Flask 依赖，便于单测）。

时间约定
--------
* 所有时间均为 UTC 整分钟；数据库中以 ``YYYY-MM-DD HH:MM`` 字符串保存，可直接比较。
* 时间窗左闭右开：[起点, 终点)，即含起点、到终点即止。

筛选规则
--------
1. 每轮有两个等长、互不重叠的 A/B 时间窗，按窗内相同的分钟偏移配对。
2. 同一偏移上 A、B 两侧都必须有用电读数；每侧在该分钟的实际策略必须等于计划策略，
   且该策略从区间起点到该分钟已连续运行满 ``MIN_RUN_MINUTES`` 分钟，这一对才有效。
3. 任何一侧不满足，整对剔除并记录原因；剔除的对不参与平均值。
4. 一轮没有任何有效对时，该轮标记为“不可比较”。

“连续运行”从实际策略区间的起点算起，区间起点可以早于时间窗。因此窗外的切换时刻
也可能改变窗内某些分钟是否已运行满 10 分钟——受影响轮次按“窗内逐分钟状态是否
变化”判定，而不是“切换时刻是否落在窗内”。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Mapping, Sequence

MIN_RUN_MINUTES = 10
TS_FORMAT = "%Y-%m-%d %H:%M"
STRATEGIES = ("A", "B")

# 逐分钟状态
STATE_OK = "ok"                    # 策略正确且已运行满 10 分钟
STATE_UNKNOWN = "unknown"          # 该分钟之前没有任何可判定策略的记录
STATE_MISMATCH = "mismatch"        # 实际策略与计划不符
STATE_SHORT_RUN = "short_run"      # 策略正确，但开始运行不足 10 分钟

REASON_MISSING_A = "missing_reading_a"
REASON_MISSING_B = "missing_reading_b"
REASON_STRATEGY_A = "strategy_mismatch_a"
REASON_STRATEGY_B = "strategy_mismatch_b"
REASON_UNKNOWN_A = "unknown_strategy_a"
REASON_UNKNOWN_B = "unknown_strategy_b"
REASON_RUN_A = "not_run_10min_a"
REASON_RUN_B = "not_run_10min_b"

REASON_LABELS = {
    REASON_MISSING_A: "A 侧该分钟无用电读数",
    REASON_MISSING_B: "B 侧该分钟无用电读数",
    REASON_STRATEGY_A: "A 窗该分钟实际策略不是 A",
    REASON_STRATEGY_B: "B 窗该分钟实际策略不是 B",
    REASON_UNKNOWN_A: "A 窗该分钟之前没有切换记录，无法确认实际策略",
    REASON_UNKNOWN_B: "B 窗该分钟之前没有切换记录，无法确认实际策略",
    REASON_RUN_A: f"A 侧到该分钟未按计划策略连续运行满 {MIN_RUN_MINUTES} 分钟",
    REASON_RUN_B: f"B 侧到该分钟未按计划策略连续运行满 {MIN_RUN_MINUTES} 分钟",
}


class ValidationError(ValueError):
    """输入数据不符合约束。"""


def parse_ts(value: str) -> datetime:
    """解析 UTC 整分钟时间戳。

    容忍 ``Z``、``+00:00``、``T`` 分隔符和带秒的写法；拒绝非 UTC 偏移与非整分钟。
    """
    text = (value or "").strip()
    if not text:
        raise ValidationError("时间为空")
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValidationError(f"无法解析时间 {value!r}（应为 UTC 整分钟）") from exc
    if dt.tzinfo is not None:
        offset = dt.utcoffset()
        if offset != timedelta(0):
            raise ValidationError(f"时间 {value!r} 不是 UTC（偏移 {offset}）")
        dt = dt.replace(tzinfo=None)
    if dt.second or dt.microsecond:
        raise ValidationError(f"时间 {value!r} 不是整分钟")
    return dt


def fmt_ts(dt: datetime) -> str:
    return dt.strftime(TS_FORMAT)


def parse_strategy(value: str) -> str:
    text = (value or "").strip().upper()
    if text not in STRATEGIES:
        raise ValidationError(f"策略必须是 A 或 B，收到 {value!r}")
    return text


def minutes_between(start: str, end: str) -> int:
    delta = datetime.strptime(end, TS_FORMAT) - datetime.strptime(start, TS_FORMAT)
    seconds = delta.total_seconds()
    if seconds % 60:
        raise ValidationError("窗口边界不是整分钟")
    minutes = int(seconds // 60)
    if minutes <= 0:
        raise ValidationError("窗口终点必须晚于起点")
    return minutes


@dataclass(frozen=True)
class RoundPlan:
    round_id: str
    a_start: str
    a_end: str
    b_start: str
    b_end: str

    @property
    def length_minutes(self) -> int:
        return minutes_between(self.a_start, self.a_end)

    def validate(self) -> None:
        if not self.round_id or not self.round_id.strip():
            raise ValidationError("轮次编号不能为空")
        a_len = self.length_minutes
        b_len = minutes_between(self.b_start, self.b_end)
        if a_len != b_len:
            raise ValidationError(
                f"轮次 {self.round_id}: A 窗 {a_len} 分钟与 B 窗 {b_len} 分钟不等长"
            )
        if self.a_start < self.b_end and self.b_start < self.a_end:
            raise ValidationError(
                f"轮次 {self.round_id}: A 窗 [{self.a_start}, {self.a_end}) 与 "
                f"B 窗 [{self.b_start}, {self.b_end}) 重叠"
            )

    def windows(self) -> list[tuple[str, str, str]]:
        return [("A", self.a_start, self.a_end), ("B", self.b_start, self.b_end)]


@dataclass(frozen=True)
class SwitchRecord:
    """实际切换记录：从 ``at`` 这一分钟起，实际策略为 ``strategy``。

    ``round_id`` 为空表示对全部轮次生效；否则只作用于该轮。
    """

    at: str
    strategy: str
    round_id: str | None = None
    switch_id: int | None = None

    def applies_to(self, round_id: str) -> bool:
        return self.round_id is None or self.round_id == round_id


@dataclass
class PairResult:
    offset: int
    a_minute: str
    b_minute: str
    a_kwh: float | None = None
    b_kwh: float | None = None
    included: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def diff_kwh(self) -> float | None:
        """B − A（正值表示 B 更耗电）。"""
        if self.a_kwh is None or self.b_kwh is None:
            return None
        return round(self.b_kwh - self.a_kwh, 6)

    @property
    def reason_texts(self) -> list[str]:
        return [REASON_LABELS[r] for r in self.reasons]

    def as_dict(self) -> dict:
        return {
            "offset": self.offset,
            "a_minute": self.a_minute,
            "b_minute": self.b_minute,
            "a_kwh": self.a_kwh,
            "b_kwh": self.b_kwh,
            "diff_kwh": self.diff_kwh,
            "included": self.included,
            "reasons": list(self.reasons),
            "reason_texts": self.reason_texts,
        }


@dataclass
class RoundResult:
    round_id: str
    plan: RoundPlan
    pairs: list[PairResult]
    valid_pairs: int
    mean_diff_kwh: float | None
    comparable: bool

    def as_dict(self) -> dict:
        return {
            "round_id": self.round_id,
            "plan": {
                "a_start": self.plan.a_start,
                "a_end": self.plan.a_end,
                "b_start": self.plan.b_start,
                "b_end": self.plan.b_end,
                "length_minutes": self.plan.length_minutes,
            },
            "valid_pairs": self.valid_pairs,
            "mean_diff_kwh": self.mean_diff_kwh,
            "comparable": self.comparable,
            "pairs": [p.as_dict() for p in self.pairs],
        }


# ---------------------------------------------------------------------------
# 实际策略区间
# ---------------------------------------------------------------------------


def build_intervals(
    switches: Sequence[SwitchRecord], span_start: str, span_end: str
) -> list[tuple[str, str, str]]:
    """构造覆盖 ``[span_start, span_end)`` 的 ``(strategy, start, end)`` 区间。

    * ``start`` 是实际区间起点（可能早于 ``span_start``），用于计算连续运行时长。
    * 同一时刻的多条记录以后出现者为准（按 ``switch_id`` 排序）。
    * 相邻且策略相同的区间合并，避免重复记录打断“连续运行”。
    * ``span_start`` 之前没有可确定策略时，用占位策略 ``"?"`` 表示未知。
    """
    ordered = sorted(switches, key=lambda s: (s.at, s.switch_id if s.switch_id is not None else -1))
    dedup: list[tuple[str, str]] = []
    for rec in ordered:
        if dedup and dedup[-1][0] == rec.at:
            dedup[-1] = (rec.at, rec.strategy)
        else:
            dedup.append((rec.at, rec.strategy))

    prior: tuple[str, str] | None = None
    inside: list[tuple[str, str]] = []
    for at, strategy in dedup:
        if at <= span_start:
            prior = (at, strategy)
        elif at < span_end:
            inside.append((at, strategy))

    intervals: list[tuple[str, str, str]] = []

    def append(strategy: str, start: str, end: str) -> None:
        if end <= start:
            return
        if intervals and intervals[-1][0] == strategy:
            prev_strategy, prev_start, _ = intervals[-1]
            intervals[-1] = (prev_strategy, prev_start, end)
        else:
            intervals.append((strategy, start, end))

    if prior is not None:
        append(prior[1], prior[0], inside[0][0] if inside else span_end)
    elif inside:
        append("?", span_start, inside[0][0])

    for idx, (at, strategy) in enumerate(inside):
        end = inside[idx + 1][0] if idx + 1 < len(inside) else span_end
        append(strategy, at, end)

    return intervals


def state_at(
    intervals: Sequence[tuple[str, str, str]], minute: str
) -> tuple[str, str] | None:
    """返回覆盖该分钟的 ``(strategy, interval_start)``；未知返回 ``None``。"""
    for strategy, start, end in intervals:
        if start <= minute < end:
            if strategy == "?":
                return None
            return strategy, start
    return None


def assess_minute(
    intervals: Sequence[tuple[str, str, str]], planned: str, minute: str
) -> str:
    """返回该分钟相对计划策略的状态码。"""
    state = state_at(intervals, minute)
    if state is None:
        return STATE_UNKNOWN
    strategy, run_start = state
    if strategy != planned:
        return STATE_MISMATCH
    elapsed = datetime.strptime(minute, TS_FORMAT) - datetime.strptime(
        run_start, TS_FORMAT
    )
    return STATE_OK if elapsed >= timedelta(minutes=MIN_RUN_MINUTES) else STATE_SHORT_RUN


def window_states(
    switches: Sequence[SwitchRecord],
    planned: str,
    window_start: str,
    window_end: str,
) -> list[tuple[str, str]]:
    """窗内逐分钟 ``(minute, state)``，用于判定更正影响范围。"""
    intervals = build_intervals(switches, window_start, window_end)
    length = minutes_between(window_start, window_end)
    start_dt = datetime.strptime(window_start, TS_FORMAT)
    return [
        (
            fmt_ts(start_dt + timedelta(minutes=offset)),
            assess_minute(intervals, planned, fmt_ts(start_dt + timedelta(minutes=offset))),
        )
        for offset in range(length)
    ]


def round_signature(
    plan: RoundPlan, switches: Sequence[SwitchRecord]
) -> dict[str, list[tuple[str, str]]]:
    """只包含作用于该轮的切换记录。"""
    scoped = [s for s in switches if s.applies_to(plan.round_id)]
    return {
        side: window_states(scoped, side, ws, we)
        for side, ws, we in plan.windows()
    }


# ---------------------------------------------------------------------------
# 单轮计算
# ---------------------------------------------------------------------------


def compute_round(
    plan: RoundPlan,
    switches: Sequence[SwitchRecord],
    readings: Mapping[str, float],
) -> RoundResult:
    plan.validate()
    scoped = [s for s in switches if s.applies_to(plan.round_id)]
    length = plan.length_minutes
    a0 = datetime.strptime(plan.a_start, TS_FORMAT)
    b0 = datetime.strptime(plan.b_start, TS_FORMAT)
    a_iv = build_intervals(scoped, plan.a_start, plan.a_end)
    b_iv = build_intervals(scoped, plan.b_start, plan.b_end)

    pairs: list[PairResult] = []
    for offset in range(length):
        a_minute = fmt_ts(a0 + timedelta(minutes=offset))
        b_minute = fmt_ts(b0 + timedelta(minutes=offset))
        pair = PairResult(offset=offset, a_minute=a_minute, b_minute=b_minute)
        pair.a_kwh = readings.get(a_minute)
        pair.b_kwh = readings.get(b_minute)

        if pair.a_kwh is None:
            pair.reasons.append(REASON_MISSING_A)
        if pair.b_kwh is None:
            pair.reasons.append(REASON_MISSING_B)

        a_state = assess_minute(a_iv, "A", a_minute)
        b_state = assess_minute(b_iv, "B", b_minute)
        if a_state == STATE_UNKNOWN:
            pair.reasons.append(REASON_UNKNOWN_A)
        elif a_state == STATE_MISMATCH:
            pair.reasons.append(REASON_STRATEGY_A)
        elif a_state == STATE_SHORT_RUN:
            pair.reasons.append(REASON_RUN_A)

        if b_state == STATE_UNKNOWN:
            pair.reasons.append(REASON_UNKNOWN_B)
        elif b_state == STATE_MISMATCH:
            pair.reasons.append(REASON_STRATEGY_B)
        elif b_state == STATE_SHORT_RUN:
            pair.reasons.append(REASON_RUN_B)

        pair.included = not pair.reasons
        pairs.append(pair)

    included = [p for p in pairs if p.included]
    valid = len(included)
    mean_diff = round(sum(p.diff_kwh for p in included) / valid, 6) if valid else None
    return RoundResult(
        round_id=plan.round_id,
        plan=plan,
        pairs=pairs,
        valid_pairs=valid,
        mean_diff_kwh=mean_diff,
        comparable=valid > 0,
    )


def impacted_round_ids(
    before: Sequence[SwitchRecord],
    after: Sequence[SwitchRecord],
    plans: Sequence[RoundPlan],
) -> set[str]:
    """比较更正前后窗内逐分钟状态，返回必须重算的轮次。

    切换记录落在窗外也可能影响窗内状态（更早的切换时刻会改变窗首若干分钟是否
    已连续运行满 10 分钟），因此不能只按“切换时刻是否落在窗内”判断。
    """
    impacted: set[str] = set()
    for plan in plans:
        if round_signature(plan, before) != round_signature(plan, after):
            impacted.add(plan.round_id)
    return impacted
