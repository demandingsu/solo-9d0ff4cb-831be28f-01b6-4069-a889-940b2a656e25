"""峰段转移试算 —— 纯计算模块（只读，不依赖 Flask / SQLite）。

业务问题
--------
选一轮，给定一个**完整覆盖该轮 A 窗**的计费时段 ``[start, end)``：把计费时段内
所有**有效分钟对**对应的 A 分钟电量替换成 B 分钟电量（B − A 增量记为 Δ），其余
分钟保持原样；分别求替换前后计费时段内最高的连续 15 分钟平均功率（需量），以及
需量差 ``D = 替换后峰值 − 替换前峰值`` 随缺测分钟取值能达到的上下界。

缺测分钟
--------
计费时段内没有读数的分钟为缺测分钟，其电量 ``x_m`` 可在 ``[0, cap]`` 之间取值，
**替换前后共用同一个值**（被替换的 A 分钟两侧都有读数，不可能同时是缺测分钟）。

记号与结论
----------
设计费时段共 n 分钟，15 分钟窗共 ``W = n − 14`` 个（n ≤ 1440）：

* ``S_j(x)``  ：第 j 个 15 分钟窗**替换前**电量合计；
* ``Δ_j``     ：该窗替换增量之和（有效对 B − A），与缺测取值无关；
* ``S'_j(x) = S_j(x) + Δ_j``：替换后合计；
* 峰值平均功率 ``P(x) = 4·max_j S_j(x)``、``P'(x) = 4·max_j S'_j(x)``（kW）；
* 需量差 ``D(x) = P'(x) − P(x)``。

对每个窗 w 定义角点赋值 ``x^(w)``：落在第 w 个 15 分钟窗内的缺测分钟取上限
``cap``，其余缺测分钟取 0。展开 max 后每个仿射项里 ``x_m`` 的系数只可能是
``−1 / 0 / +1``，且符号完全由“m 是否在窗 w 内”决定，因此同一赋值可以同时
优化所有仿射项，于是：

    max_x D(x) = 4·max_w ( S'_w(x^(w)) − max_k S_k(x^(w)) )
    min_x D(x) = 4·min_w ( max_k S'_k(x^(w)) − S_w(x^(w)) )

即全局上下界只需评估 W 个角点赋值；配合前缀和，每个窗的求和为 O(1)，
整体复杂度 O(W²)，24 小时（W=1426）约两百万次简单运算。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Mapping, Sequence

from .analysis import (
    TS_FORMAT,
    RoundPlan,
    RoundResult,
    ValidationError,
    fmt_ts,
    minutes_between,
    parse_ts,
)

PEAK_WINDOW_MINUTES = 15
MIN_BILLING_MINUTES = 15
MAX_BILLING_MINUTES = 24 * 60
# 判定“确定降峰/抬升”的容差（kW），避免浮点噪声把 0 判成正负。
EPS_KW = 1e-9

VERDICT_CERTAIN_DOWN = "certain_down"
VERDICT_CERTAIN_UP = "certain_up"
VERDICT_FLAT = "flat"
VERDICT_UNCERTAIN = "uncertain"

VERDICT_LABELS = {
    VERDICT_CERTAIN_DOWN: "确定降峰：缺测分钟取任何允许值，需量都下降",
    VERDICT_CERTAIN_UP: "确定抬升：缺测分钟取任何允许值，需量都上升",
    VERDICT_FLAT: "持平：缺测分钟取任何允许值，需量差都约为 0",
    VERDICT_UNCERTAIN: "无法确定：降峰与否取决于缺测分钟的实际电量",
}


@dataclass(frozen=True)
class PeakWindow:
    """一个连续 15 分钟窗的峰值信息。"""

    start: str
    end: str
    energy_kwh: float
    mean_power_kw: float

    def as_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "energy_kwh": round(self.energy_kwh, 6),
            "mean_power_kw": round(self.mean_power_kw, 6),
        }


@dataclass(frozen=True)
class Replacement:
    minute: str
    a_kwh: float
    b_kwh: float
    delta_kwh: float

    def as_dict(self) -> dict:
        return {
            "minute": self.minute,
            "a_kwh": round(self.a_kwh, 6),
            "b_kwh": round(self.b_kwh, 6),
            "delta_kwh": round(self.delta_kwh, 6),
        }


@dataclass(frozen=True)
class MissingAssignment:
    minute: str
    kwh: float
    at_cap: bool

    def as_dict(self) -> dict:
        return {
            "minute": self.minute,
            "kwh": round(self.kwh, 6),
            "at_cap": self.at_cap,
        }


@dataclass
class Witness:
    """上界 > 0 时，一组确实会抬高需量的缺测取值及其前后峰段。"""

    demand_diff_kw: float
    peak_before: PeakWindow
    peak_after: PeakWindow
    assignments: list[MissingAssignment] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "demand_diff_kw": round(self.demand_diff_kw, 6),
            "peak_before": self.peak_before.as_dict(),
            "peak_after": self.peak_after.as_dict(),
            "assignments": [a.as_dict() for a in self.assignments],
        }


@dataclass
class TrialResult:
    round_id: str
    billing_start: str
    billing_end: str
    duration_minutes: int
    missing_cap_kwh: float
    n_minutes: int
    n_missing: int
    n_replaced: int
    missing_minutes: list[str]
    replacements: list[Replacement]
    diff_min_kw: float
    diff_max_kw: float
    verdict: str
    certain_reduction: bool
    # 两个参照情形：缺测全部取 0、缺测全部取上限。
    peak_before_at_zero: PeakWindow
    peak_after_at_zero: PeakWindow
    peak_before_at_cap: PeakWindow
    peak_after_at_cap: PeakWindow
    witness: Witness | None

    def as_dict(self) -> dict:
        return {
            "round_id": self.round_id,
            "billing_start": self.billing_start,
            "billing_end": self.billing_end,
            "duration_minutes": self.duration_minutes,
            "missing_cap_kwh": self.missing_cap_kwh,
            "n_minutes": self.n_minutes,
            "n_missing": self.n_missing,
            "n_replaced": self.n_replaced,
            "missing_minutes": list(self.missing_minutes),
            "replacements": [r.as_dict() for r in self.replacements],
            "diff_min_kw": round(self.diff_min_kw, 6),
            "diff_max_kw": round(self.diff_max_kw, 6),
            "verdict": self.verdict,
            "verdict_text": VERDICT_LABELS[self.verdict],
            "certain_reduction": self.certain_reduction,
            "peak_before_at_zero": self.peak_before_at_zero.as_dict(),
            "peak_after_at_zero": self.peak_after_at_zero.as_dict(),
            "peak_before_at_cap": self.peak_before_at_cap.as_dict(),
            "peak_after_at_cap": self.peak_after_at_cap.as_dict(),
            "witness": self.witness.as_dict() if self.witness else None,
        }


def parse_billing_window(start: str, end: str) -> tuple[str, str, int]:
    """校验计费时段：UTC 整分钟、左闭右开、15 分钟至 24 小时。"""
    s = fmt_ts(parse_ts(start))
    e = fmt_ts(parse_ts(end))
    n = minutes_between(s, e)
    if n < MIN_BILLING_MINUTES:
        raise ValidationError(
            f"计费时段至少 {MIN_BILLING_MINUTES} 分钟，当前只有 {n} 分钟"
        )
    if n > MAX_BILLING_MINUTES:
        raise ValidationError(
            f"计费时段最长 {MAX_BILLING_MINUTES} 分钟（24 小时），当前为 {n} 分钟"
        )
    return s, e, n


def parse_cap(value: str | float) -> float:
    try:
        cap = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"缺测每分钟上限必须是数字，收到 {value!r}") from exc
    if not math.isfinite(cap) or cap < 0:
        raise ValidationError("缺测每分钟上限必须是非负有限数字")
    return cap


def _argmax_first(values: Sequence[float]) -> int:
    """返回首个最大值所在下标，保证峰段结果稳定可复现。"""
    return max(range(len(values)), key=lambda i: values[i])


def run_trial(
    plan: RoundPlan,
    round_result: RoundResult,
    readings: Mapping[str, float],
    billing_start: str,
    billing_end: str,
    missing_cap_kwh: float,
) -> TrialResult:
    """执行一次峰段转移试算。全程只读，不修改任何输入数据。"""
    start, end, n = parse_billing_window(billing_start, billing_end)
    if not (start <= plan.a_start and plan.a_end <= end):
        raise ValidationError(
            f"计费时段 [{start}, {end}) 必须完整覆盖该轮 A 窗 "
            f"[{plan.a_start}, {plan.a_end})"
        )
    try:
        cap = float(missing_cap_kwh)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"缺测每分钟上限必须是数字，收到 {missing_cap_kwh!r}"
        ) from exc
    if not math.isfinite(cap) or cap < 0:
        raise ValidationError("缺测每分钟上限必须是非负有限数字")

    start_dt = datetime.strptime(start, TS_FORMAT)
    minutes = [fmt_ts(start_dt + timedelta(minutes=i)) for i in range(n)]
    index_of = {m: i for i, m in enumerate(minutes)}

    # 替换前每分钟电量：缺测为 None；替换增量 delta（缺测分钟恒为 0）。
    base: list[float | None] = [readings.get(m) for m in minutes]
    delta = [0.0] * n
    replacements: list[Replacement] = []
    for pair in round_result.pairs:
        if not pair.included:
            continue
        i = index_of.get(pair.a_minute)
        if i is None or pair.a_kwh is None or pair.b_kwh is None:
            continue
        d = round(pair.b_kwh - pair.a_kwh, 6)
        delta[i] = d
        replacements.append(
            Replacement(
                minute=pair.a_minute,
                a_kwh=pair.a_kwh,
                b_kwh=pair.b_kwh,
                delta_kwh=d,
            )
        )
    if not replacements:
        raise ValidationError(
            f"轮次 {plan.round_id} 在计费时段内没有可替换的有效分钟对，无法试算"
        )

    missing_idx = [i for i, v in enumerate(base) if v is None]
    missing_set = set(missing_idx)

    # 前缀和：基础电量、缺测计数、替换增量。
    base_prefix = [0.0] * (n + 1)
    miss_prefix = [0] * (n + 1)
    delta_prefix = [0.0] * (n + 1)
    for i in range(n):
        base_prefix[i + 1] = base_prefix[i] + (base[i] or 0.0)
        miss_prefix[i + 1] = miss_prefix[i] + (1 if i in missing_set else 0)
        delta_prefix[i + 1] = delta_prefix[i] + delta[i]

    w_count = n - PEAK_WINDOW_MINUTES + 1
    base_sum = [
        base_prefix[k + PEAK_WINDOW_MINUTES] - base_prefix[k] for k in range(w_count)
    ]
    delta_sum = [
        delta_prefix[k + PEAK_WINDOW_MINUTES] - delta_prefix[k]
        for k in range(w_count)
    ]

    def cap_count_in_intersection(k: int, w: int) -> int:
        lo = max(k, w)
        hi = min(k + PEAK_WINDOW_MINUTES, w + PEAK_WINDOW_MINUTES)
        return miss_prefix[hi] - miss_prefix[lo] if hi > lo else 0

    def window_values(w: int) -> tuple[list[float], list[float]]:
        """角点赋值 x^(w) 下每个 15 分钟窗的替换前/替换后电量合计。"""
        s_values = [
            base_sum[k] + cap * cap_count_in_intersection(k, w)
            for k in range(w_count)
        ]
        sp_values = [s_values[k] + delta_sum[k] for k in range(w_count)]
        return s_values, sp_values

    def make_peak(values: Sequence[float], k: int) -> PeakWindow:
        w_start = fmt_ts(start_dt + timedelta(minutes=k))
        w_end = fmt_ts(
            start_dt + timedelta(minutes=k + PEAK_WINDOW_MINUTES)
        )
        return PeakWindow(
            start=w_start,
            end=w_end,
            energy_kwh=values[k],
            mean_power_kw=values[k] * 4.0,
        )

    # 参照情形 1：缺测全部取 0。
    s_zero = list(base_sum)
    sp_zero = [s_zero[k] + delta_sum[k] for k in range(w_count)]
    peak_before_zero = make_peak(s_zero, _argmax_first(s_zero))
    peak_after_zero = make_peak(sp_zero, _argmax_first(sp_zero))

    # 参照情形 2：缺测全部取上限。
    s_cap = [
        base_sum[k]
        + cap * (
            miss_prefix[k + PEAK_WINDOW_MINUTES] - miss_prefix[k]
        )
        for k in range(w_count)
    ]
    sp_cap = [s_cap[k] + delta_sum[k] for k in range(w_count)]
    peak_before_cap = make_peak(s_cap, _argmax_first(s_cap))
    peak_after_cap = make_peak(sp_cap, _argmax_first(sp_cap))

    # 上下界：评估 W 个角点赋值 x^(w)。
    max_f = -math.inf
    max_f_w = 0
    min_g = math.inf
    for w in range(w_count):
        s_values, sp_values = window_values(w)
        f_w = sp_values[w] - max(s_values)
        if f_w > max_f:
            max_f = f_w
            max_f_w = w
        g_w = max(sp_values) - s_values[w]
        if g_w < min_g:
            min_g = g_w

    diff_max_kw = 4.0 * max_f
    diff_min_kw = 4.0 * min_g

    if diff_max_kw <= -EPS_KW:
        verdict = VERDICT_CERTAIN_DOWN
    elif diff_min_kw >= EPS_KW:
        verdict = VERDICT_CERTAIN_UP
    elif diff_min_kw > -EPS_KW and diff_max_kw < EPS_KW:
        verdict = VERDICT_FLAT
    else:
        verdict = VERDICT_UNCERTAIN

    # 上界 > 0：给出一组会抬高需量的缺测取值与前后峰段，方便补查。
    witness: Witness | None = None
    if diff_max_kw > EPS_KW:
        s_values, sp_values = window_values(max_f_w)
        before_k = _argmax_first(s_values)
        after_k = _argmax_first(sp_values)
        peak_before = make_peak(s_values, before_k)
        peak_after = make_peak(sp_values, after_k)
        assignments = [
            MissingAssignment(
                minute=minutes[i],
                kwh=cap if max_f_w <= i < max_f_w + PEAK_WINDOW_MINUTES else 0.0,
                at_cap=max_f_w <= i < max_f_w + PEAK_WINDOW_MINUTES,
            )
            for i in missing_idx
        ]
        witness = Witness(
            demand_diff_kw=diff_max_kw,
            peak_before=peak_before,
            peak_after=peak_after,
            assignments=assignments,
        )

    return TrialResult(
        round_id=plan.round_id,
        billing_start=start,
        billing_end=end,
        duration_minutes=n,
        missing_cap_kwh=cap,
        n_minutes=n,
        n_missing=len(missing_idx),
        n_replaced=len(replacements),
        missing_minutes=[minutes[i] for i in missing_idx],
        replacements=sorted(replacements, key=lambda r: r.minute),
        diff_min_kw=diff_min_kw,
        diff_max_kw=diff_max_kw,
        verdict=verdict,
        certain_reduction=verdict == VERDICT_CERTAIN_DOWN,
        peak_before_at_zero=peak_before_zero,
        peak_after_at_zero=peak_after_zero,
        peak_before_at_cap=peak_before_cap,
        peak_after_at_cap=peak_after_cap,
        witness=witness,
    )
