"""SQLite 存储层：导入、查询、更正切换记录、按受影响轮次增量重算。

事务约定
--------
* 导入三份 CSV 在单一事务中完成：任一文件有致命错误则整体回滚，不留半成品。
* ``results`` / ``result_pairs`` 是缓存表。更正切换时刻时，只删除并重写受影响轮次
  的缓存，其余轮次原样保留。
* 切换记录、配对明细、汇总在同一事务提交；失败整体回滚，原记录与结果仍可查询。
* ``audit_log`` 记录每次导入/更正/重算，便于追溯。
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Sequence

from .analysis import (
    MIN_RUN_MINUTES,
    PairResult,
    RoundPlan,
    SwitchRecord,
    ValidationError,
    compute_round,
    fmt_ts,
    impacted_round_ids,
    minutes_between,
    parse_strategy,
    parse_ts,
)
from .trial import TrialResult, parse_billing_window, parse_cap, run_trial

SCHEMA = """
CREATE TABLE IF NOT EXISTS rounds (
    round_id TEXT PRIMARY KEY,
    a_start  TEXT NOT NULL,
    a_end    TEXT NOT NULL,
    b_start  TEXT NOT NULL,
    b_end    TEXT NOT NULL,
    imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS readings (
    minute TEXT PRIMARY KEY,
    kwh REAL NOT NULL,
    imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS switches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    strategy TEXT NOT NULL CHECK (strategy IN ('A','B')),
    round_id TEXT,
    source TEXT NOT NULL DEFAULT 'import',
    updated_at TEXT NOT NULL,
    FOREIGN KEY (round_id) REFERENCES rounds(round_id)
);
CREATE INDEX IF NOT EXISTS idx_switches_at ON switches(at);

CREATE TABLE IF NOT EXISTS results (
    round_id TEXT PRIMARY KEY,
    comparable INTEGER NOT NULL,
    valid_pairs INTEGER NOT NULL,
    mean_diff_kwh REAL,
    computed_at TEXT NOT NULL,
    FOREIGN KEY (round_id) REFERENCES rounds(round_id)
);

CREATE TABLE IF NOT EXISTS result_pairs (
    round_id TEXT NOT NULL,
    offset INTEGER NOT NULL,
    a_minute TEXT NOT NULL,
    b_minute TEXT NOT NULL,
    a_kwh REAL,
    b_kwh REAL,
    diff_kwh REAL,
    included INTEGER NOT NULL,
    reasons_json TEXT NOT NULL,
    PRIMARY KEY (round_id, offset),
    FOREIGN KEY (round_id) REFERENCES rounds(round_id)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT NOT NULL,
    payload_json TEXT
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, isolation_level=None, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    return conn


# ---------------------------------------------------------------------------
# CSV 解析
# ---------------------------------------------------------------------------


def _read_csv(
    text: str | bytes,
    required: Sequence[str],
    label: str,
    optional: Sequence[str] = (),
) -> list[dict[str, str]]:
    if isinstance(text, bytes):
        text = text.decode("utf-8-sig", errors="replace")
    text = text.lstrip("﻿")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValidationError(f"{label}: 空文件或缺少表头")
    field_map = {name.strip().lower(): name for name in reader.fieldnames if name}
    missing = [col for col in required if col not in field_map]
    if missing:
        raise ValidationError(
            f"{label}: 缺少列 {', '.join(missing)}"
            f"（实际表头: {', '.join(n for n in reader.fieldnames if n)}）"
        )
    columns = list(required) + [c for c in optional if c in field_map]
    rows: list[dict[str, str]] = []
    for raw in reader:
        row = {}
        for col in columns:
            value = raw.get(field_map[col])
            row[col] = value.strip() if isinstance(value, str) else ""
        rows.append(row)
    return rows


def parse_rounds_csv(text: str | bytes) -> list[RoundPlan]:
    rows = _read_csv(
        text, ["round_id", "a_start", "a_end", "b_start", "b_end"], "轮次计划"
    )
    plans: dict[str, RoundPlan] = {}
    for line_no, row in enumerate(rows, start=2):
        if not any(row.values()):
            continue
        rid = row["round_id"]
        try:
            plan = RoundPlan(
                round_id=rid,
                a_start=fmt_ts(parse_ts(row["a_start"])),
                a_end=fmt_ts(parse_ts(row["a_end"])),
                b_start=fmt_ts(parse_ts(row["b_start"])),
                b_end=fmt_ts(parse_ts(row["b_end"])),
            )
            plan.validate()
        except ValidationError as exc:
            raise ValidationError(f"轮次计划第 {line_no} 行: {exc}") from exc
        if rid in plans:
            raise ValidationError(f"轮次计划第 {line_no} 行: 轮次编号 {rid!r} 重复")
        plans[rid] = plan
    if not plans:
        raise ValidationError("轮次计划没有任何数据行")
    return list(plans.values())


def parse_switches_csv(
    text: str | bytes,
) -> list[tuple[str, str, str | None]]:
    # round_id 列可选：没有该列时全部记录对全部轮次生效
    rows = _read_csv(
        text, ["switched_at", "strategy"], "切换记录", optional=["round_id"]
    )
    seen: dict[tuple[str, str | None], int] = {}
    out: list[tuple[str, str, str | None]] = []
    for line_no, row in enumerate(rows, start=2):
        if not any(row.values()):
            continue
        try:
            at = fmt_ts(parse_ts(row["switched_at"]))
            strategy = parse_strategy(row["strategy"])
        except ValidationError as exc:
            raise ValidationError(f"切换记录第 {line_no} 行: {exc}") from exc
        rid = (row.get("round_id") or "").strip() or None
        if (at, rid) in seen:
            raise ValidationError(
                f"切换记录第 {line_no} 行: 与第 {seen[(at, rid)]} 行重复"
                "（同一时刻/轮次只能有一条），请合并"
            )
        seen[(at, rid)] = line_no
        out.append((at, strategy, rid))
    return out


def parse_readings_csv(text: str | bytes) -> list[tuple[str, float]]:
    rows = _read_csv(text, ["minute", "kwh"], "用电量")
    seen: dict[str, int] = {}
    out: list[tuple[str, float]] = []
    for line_no, row in enumerate(rows, start=2):
        if not any(row.values()):
            continue
        try:
            minute = fmt_ts(parse_ts(row["minute"]))
        except ValidationError as exc:
            raise ValidationError(f"用电量第 {line_no} 行: {exc}") from exc
        try:
            kwh = float(row["kwh"])
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"用电量第 {line_no} 行: 读数 {row['kwh']!r} 不是数字"
            ) from exc
        if kwh < 0:
            raise ValidationError(f"用电量第 {line_no} 行: 读数不能为负")
        if minute in seen:
            raise ValidationError(
                f"用电量第 {line_no} 行: 分钟 {minute} 与第 {seen[minute]} 行重复"
            )
        seen[minute] = line_no
        out.append((minute, kwh))
    return out


# ---------------------------------------------------------------------------
# 导入
# ---------------------------------------------------------------------------


def _log(
    conn: sqlite3.Connection, action: str, detail: str, payload=None
) -> None:
    conn.execute(
        "INSERT INTO audit_log (at, action, detail, payload_json) VALUES (?,?,?,?)",
        (
            utc_now(),
            action,
            detail,
            json.dumps(payload, ensure_ascii=False) if payload is not None else None,
        ),
    )


def import_all(
    conn: sqlite3.Connection,
    rounds_text: str | bytes,
    switches_text: str | bytes,
    readings_text: str | bytes,
) -> dict:
    """在同一事务里导入三份 CSV 并全量重算；任一步失败整体回滚。"""
    plans = parse_rounds_csv(rounds_text)
    switches = parse_switches_csv(switches_text)
    readings = parse_readings_csv(readings_text)

    known_rounds = {p.round_id for p in plans}
    for at, _strategy, rid in switches:
        if rid is not None and rid not in known_rounds:
            raise ValidationError(
                f"切换记录 {at} 指向未知轮次 {rid!r}（该轮不在轮次计划中）"
            )

    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DELETE FROM result_pairs")
        conn.execute("DELETE FROM results")
        conn.execute("DELETE FROM switches")
        conn.execute("DELETE FROM readings")
        conn.execute("DELETE FROM rounds")

        now = utc_now()
        conn.executemany(
            "INSERT INTO rounds (round_id, a_start, a_end, b_start, b_end, imported_at)"
            " VALUES (?,?,?,?,?,?)",
            [
                (p.round_id, p.a_start, p.a_end, p.b_start, p.b_end, now)
                for p in plans
            ],
        )
        conn.executemany(
            "INSERT INTO switches (at, strategy, round_id, source, updated_at)"
            " VALUES (?,?,?,'import',?)",
            [(at, strategy, rid, now) for at, strategy, rid in switches],
        )
        conn.executemany(
            "INSERT INTO readings (minute, kwh, imported_at) VALUES (?,?,?)",
            [(minute, kwh, now) for minute, kwh in readings],
        )
        recomputed = recompute_rounds(conn, [p.round_id for p in plans])
        _log(
            conn,
            "import",
            f"导入 {len(plans)} 轮、{len(switches)} 条切换、{len(readings)} 条读数，"
            f"重算 {len(recomputed)} 轮",
            {"round_ids": [p.round_id for p in plans]},
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return {
        "rounds": len(plans),
        "switches": len(switches),
        "readings": len(readings),
        "recomputed": recomputed,
    }


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------


def load_plans(conn: sqlite3.Connection) -> list[RoundPlan]:
    return [
        RoundPlan(r["round_id"], r["a_start"], r["a_end"], r["b_start"], r["b_end"])
        for r in conn.execute("SELECT * FROM rounds ORDER BY a_start, round_id")
    ]


def load_switches(conn: sqlite3.Connection) -> list[SwitchRecord]:
    return [
        SwitchRecord(
            at=r["at"],
            strategy=r["strategy"],
            round_id=r["round_id"],
            switch_id=r["id"],
        )
        for r in conn.execute("SELECT id, at, strategy, round_id FROM switches ORDER BY at, id")
    ]


def load_readings(conn: sqlite3.Connection) -> dict[str, float]:
    return {
        r["minute"]: r["kwh"]
        for r in conn.execute("SELECT minute, kwh FROM readings")
    }


# ---------------------------------------------------------------------------
# 重算
# ---------------------------------------------------------------------------


def recompute_rounds(
    conn: sqlite3.Connection, round_ids: Iterable[str]
) -> list[str]:
    """只重算给定轮次并重写其缓存；返回实际写入的轮次（未知轮次忽略）。"""
    ids = list(dict.fromkeys(round_ids))
    if not ids:
        return []
    plans = {p.round_id: p for p in load_plans(conn)}
    switches = load_switches(conn)
    readings = load_readings(conn)
    written: list[str] = []
    for rid in ids:
        plan = plans.get(rid)
        if plan is None:
            continue
        result = compute_round(plan, switches, readings)
        conn.execute(
            "INSERT INTO results (round_id, comparable, valid_pairs, mean_diff_kwh, computed_at)"
            " VALUES (?,?,?,?,?)"
            " ON CONFLICT(round_id) DO UPDATE SET comparable=excluded.comparable,"
            " valid_pairs=excluded.valid_pairs, mean_diff_kwh=excluded.mean_diff_kwh,"
            " computed_at=excluded.computed_at",
            (
                rid,
                int(result.comparable),
                result.valid_pairs,
                result.mean_diff_kwh,
                utc_now(),
            ),
        )
        conn.execute("DELETE FROM result_pairs WHERE round_id = ?", (rid,))
        conn.executemany(
            "INSERT INTO result_pairs (round_id, offset, a_minute, b_minute, a_kwh,"
            " b_kwh, diff_kwh, included, reasons_json) VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (
                    rid,
                    p.offset,
                    p.a_minute,
                    p.b_minute,
                    p.a_kwh,
                    p.b_kwh,
                    p.diff_kwh,
                    int(p.included),
                    json.dumps(p.reasons, ensure_ascii=False),
                )
                for p in result.pairs
            ],
        )
        written.append(rid)
    return written


def correct_switch(
    conn: sqlite3.Connection,
    switch_id: int,
    new_at: str,
    new_strategy: str | None = None,
    new_round_id: str | None = None,
    update_round_scope: bool = False,
) -> dict:
    """更正一条切换记录（时刻、可选策略/作用轮次），只重算受影响的轮次。

    影响判定按“窗内逐分钟状态是否变化”，因此窗外切换也可能触发重算；窗内切换若
    不改变窗内任何分钟状态，则不重算。切换记录、配对明细、汇总在同一事务提交；
    失败整体回滚，原记录与结果仍可查。
    """
    row = conn.execute("SELECT * FROM switches WHERE id = ?", (switch_id,)).fetchone()
    if row is None:
        raise ValidationError(f"切换记录 #{switch_id} 不存在")

    at = fmt_ts(parse_ts(new_at))
    strategy = parse_strategy(new_strategy) if new_strategy else row["strategy"]
    if update_round_scope:
        scope = (new_round_id or "").strip() or None
    else:
        scope = row["round_id"]
    if scope is not None:
        exists = conn.execute(
            "SELECT 1 FROM rounds WHERE round_id = ?", (scope,)
        ).fetchone()
        if not exists:
            raise ValidationError(f"作用轮次 {scope!r} 不在轮次计划中")
    duplicate = conn.execute(
        "SELECT id FROM switches WHERE at = ? AND round_id IS ? AND id != ?",
        (at, scope, switch_id),
    ).fetchone()
    if duplicate is not None:
        raise ValidationError(
            f"更正后与切换记录 #{duplicate['id']} 重复（同一时刻、同一作用轮次），"
            "请先合并或删除另一条"
        )

    if (at, strategy, scope) == (row["at"], row["strategy"], row["round_id"]):
        return {
            "changed": False,
            "recomputed": [],
            "switch_id": switch_id,
            "message": "记录未变化，未重算任何轮次",
        }

    plans = load_plans(conn)
    before = load_switches(conn)
    after = [
        SwitchRecord(at=at, strategy=strategy, round_id=scope, switch_id=s.switch_id)
        if s.switch_id == switch_id
        else s
        for s in before
    ]
    impacted = impacted_round_ids(before, after, plans)

    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "UPDATE switches SET at = ?, strategy = ?, round_id = ?, source = 'correction',"
            " updated_at = ? WHERE id = ?",
            (at, strategy, scope, utc_now(), switch_id),
        )
        recomputed = recompute_rounds(conn, sorted(impacted))
        _log(
            conn,
            "correct_switch",
            f"更正切换 #{switch_id}: {row['at']} {row['strategy']} → {at} {strategy}；"
            f"重算 {len(recomputed)} 轮"
            + (f"（{', '.join(recomputed)}）" if recomputed else ""),
            {
                "switch_id": switch_id,
                "before": {"at": row["at"], "strategy": row["strategy"]},
                "after": {"at": at, "strategy": strategy},
                "recomputed": recomputed,
            },
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return {
        "changed": True,
        "recomputed": recomputed,
        "switch_id": switch_id,
        "message": (
            f"已更正切换 #{switch_id}，重算 {len(recomputed)} 轮"
            + (f"：{', '.join(recomputed)}" if recomputed else "（其余轮次状态未变）")
        ),
    }


# ---------------------------------------------------------------------------
# 查询
# ---------------------------------------------------------------------------


def round_rows(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT r.round_id, r.a_start, r.a_end, r.b_start, r.b_end,
               res.valid_pairs, res.mean_diff_kwh, res.comparable, res.computed_at
          FROM rounds r
          LEFT JOIN results res ON res.round_id = r.round_id
         ORDER BY r.a_start, r.round_id
        """
    ).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        d["length_minutes"] = minutes_between(d["a_start"], d["a_end"])
        d["comparable"] = bool(d["comparable"])
        out.append(d)
    return out


def pair_rows(conn: sqlite3.Connection, round_id: str) -> list[dict]:
    from .analysis import REASON_LABELS

    rows = conn.execute(
        "SELECT * FROM result_pairs WHERE round_id = ? ORDER BY offset", (round_id,)
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        reasons = json.loads(d.pop("reasons_json"))
        d["reasons"] = reasons
        d["reason_texts"] = [REASON_LABELS.get(x, x) for x in reasons]
        d["included"] = bool(d["included"])
        out.append(d)
    return out


def switch_rows(conn: sqlite3.Connection) -> list[dict]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT id, at, strategy, round_id, source, updated_at FROM switches ORDER BY at, id"
        )
    ]


def audit_rows(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        )
    ]


def summary(conn: sqlite3.Connection) -> dict:
    from .analysis import REASON_LABELS

    rounds = round_rows(conn)
    comparable = [
        r for r in rounds if r["comparable"] and r["mean_diff_kwh"] is not None
    ]
    overall = (
        round(sum(r["mean_diff_kwh"] for r in comparable) / len(comparable), 6)
        if comparable
        else None
    )
    return {
        "rounds": rounds,
        "n_rounds": len(rounds),
        "n_comparable": len(comparable),
        "n_incomparable": len(rounds) - len(comparable),
        "overall_mean_diff_kwh": overall,
        "min_run_minutes": MIN_RUN_MINUTES,
        "reason_labels": REASON_LABELS,
    }


# ---------------------------------------------------------------------------
# 峰段转移试算（只读：不写回任何数据表，也不写 audit_log）
# ---------------------------------------------------------------------------


def trial_from_store(
    conn: sqlite3.Connection,
    round_id: str,
    billing_start: str,
    billing_end: str,
    missing_cap_kwh: str | float,
) -> TrialResult:
    """基于当前已提交状态的配对缓存与读数做一次试算。

    配对直接取自 ``result_pairs``（即最近一次导入/更正提交后的结果），读数取自
    ``readings``；切换更正重算后再试算自然使用更新后的配对。整个过程只读。
    """
    row = conn.execute(
        "SELECT * FROM rounds WHERE round_id = ?", (round_id,)
    ).fetchone()
    if row is None:
        raise ValidationError(f"轮次 {round_id!r} 不存在")
    plan = RoundPlan(
        row["round_id"], row["a_start"], row["a_end"], row["b_start"], row["b_end"]
    )
    start, end, _n = parse_billing_window(billing_start, billing_end)
    cap = parse_cap(missing_cap_kwh)

    pair_rows = conn.execute(
        "SELECT * FROM result_pairs WHERE round_id = ? ORDER BY offset", (round_id,)
    ).fetchall()
    if not pair_rows:
        raise ValidationError(
            f"轮次 {round_id} 还没有配对结果，请先导入数据或等待重算完成"
        )
    pairs = [
        PairResult(
            offset=r["offset"],
            a_minute=r["a_minute"],
            b_minute=r["b_minute"],
            a_kwh=r["a_kwh"],
            b_kwh=r["b_kwh"],
            included=bool(r["included"]),
            reasons=json.loads(r["reasons_json"]),
        )
        for r in pair_rows
    ]
    valid = sum(1 for p in pairs if p.included)
    result_row = conn.execute(
        "SELECT * FROM results WHERE round_id = ?", (round_id,)
    ).fetchone()
    from .analysis import RoundResult

    round_result = RoundResult(
        round_id=round_id,
        plan=plan,
        pairs=pairs,
        valid_pairs=valid,
        mean_diff_kwh=result_row["mean_diff_kwh"] if result_row else None,
        comparable=bool(result_row["comparable"]) if result_row else valid > 0,
    )
    readings = load_readings(conn)
    return run_trial(plan, round_result, readings, start, end, cap)
