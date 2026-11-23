"""峰段转移试算的存储层与网页端到端测试。

覆盖：只读不写回、无有效对提示、切换更正后按更新配对试算、表单页与 JSON API。
"""

import io

import pytest

from coldstore import store
from coldstore.analysis import ValidationError

from conftest import import_files, write_files


def _upload(client, tmp_path):
    rounds, switches, readings = write_files(tmp_path)
    return client.post(
        "/import",
        data={
            "rounds": (io.BytesIO(rounds.read_bytes()), "rounds.csv"),
            "switches": (io.BytesIO(switches.read_bytes()), "switches.csv"),
            "readings": (io.BytesIO(readings.read_bytes()), "readings.csv"),
        },
        content_type="multipart/form-data",
    )


# conftest 样例：R1 的 A 窗 [00:00, 01:00)，B 窗 [01:30, 02:30)，60 个有效对
R1_A_START, R1_A_END = "2026-03-01 00:00", "2026-03-01 01:00"


class TestStoreTrial:
    def test_trial_uses_committed_pairs_and_is_read_only(self, conn, tmp_path):
        import_files(conn, tmp_path)
        readings_before = store.load_readings(conn)
        audit_before = store.audit_rows(conn)
        rows_before = store.round_rows(conn)

        result = store.trial_from_store(
            conn, "R1", R1_A_START, R1_A_END, 2.0
        )
        assert result.n_replaced == 60
        assert result.n_missing == 0

        # 试算不得写回任何业务数据或日志
        assert store.load_readings(conn) == readings_before
        assert store.audit_rows(conn) == audit_before
        assert store.round_rows(conn) == rows_before

    def test_unknown_round_rejected(self, conn, tmp_path):
        import_files(conn, tmp_path)
        with pytest.raises(ValidationError, match="不存在"):
            store.trial_from_store(conn, "NOPE", R1_A_START, R1_A_END, 1.0)

    def test_billing_not_covering_a_window_rejected(self, conn, tmp_path):
        import_files(conn, tmp_path)
        with pytest.raises(ValidationError, match="完整覆盖"):
            store.trial_from_store(
                conn, "R1", "2026-03-01 00:01", "2026-03-01 01:01", 1.0
            )

    def test_trial_after_correction_uses_updated_pairs(self, conn, tmp_path):
        import_files(conn, tmp_path)
        first = store.trial_from_store(conn, "R1", R1_A_START, R1_A_END, 2.0)
        assert first.n_replaced == 60

        # 把 B 切入时刻从 01:15 改到 01:25 → R1 前 5 对失效（既有测试口径）
        target = next(
            s for s in store.switch_rows(conn) if s["at"] == "2026-03-01 01:15"
        )
        outcome = store.correct_switch(
            conn, target["id"], "2026-03-01 01:25"
        )
        assert outcome["recomputed"] == ["R1"]

        second = store.trial_from_store(conn, "R1", R1_A_START, R1_A_END, 2.0)
        assert second.n_replaced == 55
        # 试算结果反映新配对：被剔除的前 5 个 A 分钟不再出现在替换明细里
        replaced = {r.minute for r in second.replacements}
        for minute in (
            "2026-03-01 00:00",
            "2026-03-01 00:01",
            "2026-03-01 00:02",
            "2026-03-01 00:03",
            "2026-03-01 00:04",
        ):
            assert minute not in replaced

    def test_round_without_valid_pair_prompts(self, conn, tmp_path):
        # R2 的 B 窗切入改挂 A 策略会让配对失效；这里直接构造不可比较场景：
        # conftest 数据里 R2 仍有 45 个有效对，所以改用更正制造 0 有效对。
        import_files(conn, tmp_path)
        target = next(
            s for s in store.switch_rows(conn) if s["at"] == "2026-03-01 05:35"
        )
        store.correct_switch(
            conn, target["id"], "2026-03-01 05:35", new_strategy="A"
        )
        with pytest.raises(ValidationError, match="没有可替换的有效分钟对"):
            store.trial_from_store(
                conn, "R2",
                "2026-03-01 04:00", "2026-03-01 05:00", 1.0,
            )


class TestTrialWeb:
    def test_trial_page_renders_form(self, client, tmp_path):
        _upload(client, tmp_path)
        resp = client.get("/trial")
        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        assert "峰段转移试算" in body
        assert "billing_start" in body and "missing_cap_kwh" in body

    def test_trial_form_submission_renders_result(self, client, tmp_path):
        _upload(client, tmp_path)
        resp = client.post(
            "/trial",
            data={
                "round_id": "R1",
                "billing_start": R1_A_START,
                "billing_end": R1_A_END,
                "missing_cap_kwh": "2.0",
            },
        )
        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        assert "试算结果" in body
        assert "需量差" in body

    def test_trial_form_error_keeps_input(self, client, tmp_path):
        _upload(client, tmp_path)
        resp = client.post(
            "/trial",
            data={
                "round_id": "R1",
                "billing_start": "2026-03-01 00:01",
                "billing_end": R1_A_END,
                "missing_cap_kwh": "2.0",
            },
        )
        assert resp.status_code == 400
        body = resp.get_data(as_text=True)
        assert "完整覆盖" in body
        assert 'value="2026-03-01 00:01"' in body  # 表单回填

    def test_trial_json_api_success(self, client, tmp_path):
        _upload(client, tmp_path)
        resp = client.post(
            "/api/trials",
            json={
                "round_id": "R1",
                "billing_start": R1_A_START,
                "billing_end": R1_A_END,
                "missing_cap_kwh": 2.0,
            },
        )
        assert resp.status_code == 200
        payload = resp.get_json()
        assert payload["round_id"] == "R1"
        assert payload["n_replaced"] == 60
        assert payload["verdict"] in (
            "certain_down", "certain_up", "flat", "uncertain"
        )
        assert payload["witness"] is None or "assignments" in payload["witness"]

    def test_trial_json_api_validation_error(self, client, tmp_path):
        _upload(client, tmp_path)
        resp = client.post(
            "/api/trials",
            json={
                "round_id": "R1",
                "billing_start": R1_A_START,
                "billing_end": "2026-03-01 00:14",
                "missing_cap_kwh": 1.0,
            },
        )
        assert resp.status_code == 400
        assert "至少 15 分钟" in resp.get_json()["error"]

    def test_trial_json_api_bad_body(self, client):
        resp = client.post("/api/trials", data="not-json")
        assert resp.status_code == 400

    def test_index_links_to_trial(self, client, tmp_path):
        _upload(client, tmp_path)
        body = client.get("/").get_data(as_text=True)
        assert "/trial" in body
