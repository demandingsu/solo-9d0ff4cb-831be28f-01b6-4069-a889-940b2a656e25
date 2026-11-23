"""存储层与 Flask 端到端测试：导入事务、增量重算、失败回滚、跨窗更正。"""

import io
import json

import pytest

from coldstore import store
from coldstore.analysis import ValidationError

from conftest import import_files, write_files


def _upload_data(tmp_path):
    rounds, switches, readings = write_files(tmp_path)
    return {
        "rounds": (io.BytesIO(rounds.read_bytes()), "rounds.csv"),
        "switches": (io.BytesIO(switches.read_bytes()), "switches.csv"),
        "readings": (io.BytesIO(readings.read_bytes()), "readings.csv"),
    }


def _switch_at(client, at):
    conn = store.connect(client.application.config["DB_PATH"])
    try:
        return next(s for s in store.switch_rows(conn) if s["at"] == at)
    finally:
        conn.close()


class TestImport:
    def test_successful_import_computes_rounds(self, conn, tmp_path):
        result = import_files(conn, tmp_path)
        assert result["rounds"] == 2
        rows = {r["round_id"]: r for r in store.round_rows(conn)}
        assert rows["R1"]["comparable"] is True
        assert rows["R1"]["valid_pairs"] == 60
        assert rows["R1"]["mean_diff_kwh"] == pytest.approx(0.05, abs=0.01)
        # R2：A 窗首 10 分钟短运行、B 窗首 5 分钟短运行 → 有效对 45
        assert rows["R2"]["valid_pairs"] == 45
        assert len(store.pair_rows(conn, "R2")) == 60

    def test_bad_readings_roll_back_everything(self, conn, tmp_path):
        rounds, switches, readings = write_files(tmp_path)
        bad = readings.read_text(encoding="utf-8").replace(
            "2026-03-01 00:05", "2026-03-01 00:05:30"
        )
        with pytest.raises(ValidationError, match="整分钟"):
            store.import_all(
                conn,
                rounds.read_text(encoding="utf-8"),
                switches.read_text(encoding="utf-8"),
                bad,
            )
        assert store.round_rows(conn) == []
        assert store.switch_rows(conn) == []

    def test_duplicate_reading_rejected(self, conn, tmp_path):
        rounds, switches, readings = write_files(tmp_path)
        text = readings.read_text(encoding="utf-8") + "2026-03-01 00:05,1.23\n"
        with pytest.raises(ValidationError, match="重复"):
            store.import_all(
                conn,
                rounds.read_text(encoding="utf-8"),
                switches.read_text(encoding="utf-8"),
                text,
            )

    def test_duplicate_round_rejected(self, conn, tmp_path):
        rounds, switches, readings = write_files(tmp_path)
        text = rounds.read_text(encoding="utf-8") + (
            "R1,2026-03-01 08:00,2026-03-01 09:00,2026-03-01 09:30,2026-03-01 10:30\n"
        )
        with pytest.raises(ValidationError, match="重复"):
            store.import_all(
                conn,
                text,
                switches.read_text(encoding="utf-8"),
                readings.read_text(encoding="utf-8"),
            )

    def test_unknown_round_scope_rejected(self, conn, tmp_path):
        rounds, switches, readings = write_files(tmp_path)
        text = switches.read_text(encoding="utf-8") + "2026-03-01 02:00,A,R9\n"
        with pytest.raises(ValidationError, match="未知轮次"):
            store.import_all(
                conn,
                rounds.read_text(encoding="utf-8"),
                text,
                readings.read_text(encoding="utf-8"),
            )

    def test_bad_csv_keeps_previous_data(self, conn, tmp_path):
        import_files(conn, tmp_path)
        before = store.round_rows(conn)
        with pytest.raises(ValidationError):
            store.import_all(conn, "wrong,header\n1,2\n", "a,b\n", "c,d\n")
        assert store.round_rows(conn) == before
        assert len(store.switch_rows(conn)) == 5


class TestCorrection:
    def test_cross_window_correction_recomputes_only_affected_round(self, conn, tmp_path):
        import_files(conn, tmp_path)
        switches = store.switch_rows(conn)
        target = next(s for s in switches if s["at"] == "2026-03-01 01:15")
        before_r2 = next(r for r in store.round_rows(conn) if r["round_id"] == "R2")

        # 01:15 在 B 窗 [01:30, 02:30) 之外；改成 01:25 后窗首不再满 10 分钟
        result = store.correct_switch(conn, target["id"], "2026-03-01 01:25")
        assert result["recomputed"] == ["R1"]
        after_r1 = next(r for r in store.round_rows(conn) if r["round_id"] == "R1")
        after_r2 = next(r for r in store.round_rows(conn) if r["round_id"] == "R2")
        assert after_r1["valid_pairs"] == 55
        assert after_r2 == before_r2  # R2 未重算，结果原样保留

    def test_switch_outside_window_that_keeps_run_status_does_not_recompute(self, conn, tmp_path):
        import_files(conn, tmp_path)
        switches = store.switch_rows(conn)
        target = next(s for s in switches if s["at"] == "2026-03-01 01:15")
        result = store.correct_switch(conn, target["id"], "2026-03-01 01:00")
        assert result["changed"] is True
        assert result["recomputed"] == []

    def test_strategy_correction_changes_comparability(self, conn, tmp_path):
        import_files(conn, tmp_path)
        switches = store.switch_rows(conn)
        target = next(s for s in switches if s["at"] == "2026-03-01 05:35")
        result = store.correct_switch(
            conn, target["id"], "2026-03-01 05:35", new_strategy="A"
        )
        assert result["recomputed"] == ["R2"]
        r2 = next(r for r in store.round_rows(conn) if r["round_id"] == "R2")
        assert r2["comparable"] is False
        assert r2["valid_pairs"] == 0

    def test_correction_failure_keeps_original_record_and_results(self, conn, tmp_path):
        import_files(conn, tmp_path)
        switches = store.switch_rows(conn)
        target = next(s for s in switches if s["at"] == "2026-03-01 01:15")
        before_switch = dict(target)
        before_rows = store.round_rows(conn)
        with pytest.raises(ValidationError):
            store.correct_switch(conn, target["id"], "not-a-time")
        after = next(s for s in store.switch_rows(conn) if s["id"] == target["id"])
        assert after["at"] == before_switch["at"]
        assert store.round_rows(conn) == before_rows

    def test_correction_rolls_back_on_recompute_failure(self, conn, tmp_path, monkeypatch):
        import_files(conn, tmp_path)
        switches = store.switch_rows(conn)
        target = next(s for s in switches if s["at"] == "2026-03-01 01:15")
        before_switch = dict(target)
        before_rows = store.round_rows(conn)
        before_pairs = store.pair_rows(conn, "R1")

        def boom(*_args, **_kwargs):
            raise RuntimeError("recompute blew up")

        monkeypatch.setattr(store, "recompute_rounds", boom)
        with pytest.raises(RuntimeError):
            store.correct_switch(conn, target["id"], "2026-03-01 01:25")

        after = next(s for s in store.switch_rows(conn) if s["id"] == target["id"])
        assert after["at"] == before_switch["at"]
        assert store.round_rows(conn) == before_rows
        assert store.pair_rows(conn, "R1") == before_pairs

    def test_audit_log_records_correction(self, conn, tmp_path):
        import_files(conn, tmp_path)
        target = next(s for s in store.switch_rows(conn) if s["at"] == "2026-03-01 01:15")
        store.correct_switch(conn, target["id"], "2026-03-01 01:25")
        log = store.audit_rows(conn)
        assert log[0]["action"] == "correct_switch"
        assert "R1" in log[0]["detail"]
        assert json.loads(log[0]["payload_json"])["recomputed"] == ["R1"]


class TestWeb:
    def test_index_empty(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "还没有数据" in resp.get_data(as_text=True)

    def test_import_via_web_and_view(self, client, tmp_path):
        resp = client.post("/import", data=_upload_data(tmp_path), follow_redirects=True)
        body = resp.get_data(as_text=True)
        assert resp.status_code == 200
        assert "导入成功" in body
        assert "可比较" in body
        assert "B−A 平均" in body
        assert "配对明细" in body

    def test_import_missing_file_shows_error(self, client, tmp_path):
        rounds, _switches, _readings = write_files(tmp_path)
        data = {"rounds": (io.BytesIO(rounds.read_bytes()), "rounds.csv")}
        resp = client.post(
            "/import", data=data, content_type="multipart/form-data", follow_redirects=True
        )
        assert "请选择全部三份 CSV" in resp.get_data(as_text=True)

    def test_correct_via_web(self, client, tmp_path):
        client.post("/import", data=_upload_data(tmp_path), content_type="multipart/form-data")
        summary = client.get("/api/summary").get_json()
        r1 = next(r for r in summary["rounds"] if r["round_id"] == "R1")
        assert r1["valid_pairs"] == 60

        target = _switch_at(client, "2026-03-01 01:15")
        resp = client.post(
            f"/switches/{target['id']}/correct",
            data={"new_at": "2026-03-01 01:25", "new_strategy": "B"},
            follow_redirects=True,
        )
        assert "重算 1 轮" in resp.get_data(as_text=True)
        summary = client.get("/api/summary").get_json()
        r1 = next(r for r in summary["rounds"] if r["round_id"] == "R1")
        assert r1["valid_pairs"] == 55

    def test_api_pairs_404_for_unknown_round(self, client):
        assert client.get("/api/rounds/NOPE/pairs").status_code == 404

    def test_api_pairs_include_reasons(self, client, tmp_path):
        client.post("/import", data=_upload_data(tmp_path), content_type="multipart/form-data")
        payload = client.get("/api/rounds/R2/pairs").get_json()
        excluded = [p for p in payload["pairs"] if not p["included"]]
        assert len(excluded) == 15
        assert all(p["reasons"] for p in excluded)
        assert all(p["reason_texts"] for p in excluded)

    def test_invalid_correction_via_web_keeps_data(self, client, tmp_path):
        client.post("/import", data=_upload_data(tmp_path), content_type="multipart/form-data")
        target = _switch_at(client, "2026-03-01 01:15")
        before = client.get("/api/summary").get_json()
        resp = client.post(
            f"/switches/{target['id']}/correct",
            data={"new_at": "bad-input"},
            follow_redirects=True,
        )
        assert "更正失败" in resp.get_data(as_text=True)
        assert client.get("/api/summary").get_json()["rounds"] == before["rounds"]

    def test_healthz(self, client):
        assert client.get("/healthz").get_json() == {"ok": True}
