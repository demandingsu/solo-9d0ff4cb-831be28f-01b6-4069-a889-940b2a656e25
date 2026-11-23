"""回归测试：更正后的边界与冲突处理。"""

import pytest

from coldstore import store
from coldstore.analysis import ValidationError

from conftest import import_files


class TestCorrectionEdges:
    def test_correction_onto_existing_switch_rejected(self, conn, tmp_path):
        import_files(conn, tmp_path)
        target = next(s for s in store.switch_rows(conn) if s["at"] == "2026-03-01 01:15")
        # 04:00 A(R2) 已存在；把 01:15 改到 04:00 并挂到 R2 会与它冲突
        with pytest.raises(ValidationError, match="重复"):
            store.correct_switch(
                conn,
                target["id"],
                "2026-03-01 04:00",
                "B",
                new_round_id="R2",
                update_round_scope=True,
            )
        after = next(s for s in store.switch_rows(conn) if s["id"] == target["id"])
        assert after["at"] == "2026-03-01 01:15"

    def test_noop_correction_does_not_touch_results(self, conn, tmp_path):
        import_files(conn, tmp_path)
        target = next(s for s in store.switch_rows(conn) if s["at"] == "2026-03-01 01:15")
        before = store.round_rows(conn)
        result = store.correct_switch(conn, target["id"], target["at"], "B")
        assert result["changed"] is False
        assert store.round_rows(conn) == before
        assert all(row["action"] != "correct_switch" for row in store.audit_rows(conn))

    def test_scope_change_makes_round_incomparable(self, conn, tmp_path):
        import_files(conn, tmp_path)
        # 01:15 的 B 切换原只作用于 R1；改挂到 R2 后 R1 的 B 窗回落到基线 A → 不可比较
        target = next(s for s in store.switch_rows(conn) if s["at"] == "2026-03-01 01:15")
        result = store.correct_switch(
            conn,
            target["id"],
            "2026-03-01 01:15",
            "B",
            new_round_id="R2",
            update_round_scope=True,
        )
        assert result["changed"] is True
        assert result["recomputed"] == ["R1"]
        r1 = next(r for r in store.round_rows(conn) if r["round_id"] == "R1")
        assert r1["comparable"] is False and r1["valid_pairs"] == 0

    def test_early_outside_window_switch_does_not_matter(self, conn, tmp_path):
        import_files(conn, tmp_path)
        target = next(s for s in store.switch_rows(conn) if s["at"] == "2026-02-28 23:30")
        result = store.correct_switch(conn, target["id"], "2026-02-28 23:20")
        assert result["changed"] is True
        assert result["recomputed"] == []

    def test_unknown_switch_id(self, conn, tmp_path):
        import_files(conn, tmp_path)
        with pytest.raises(ValidationError, match="不存在"):
            store.correct_switch(conn, 99999, "2026-03-01 00:00")
