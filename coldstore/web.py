"""Flask 分析网页：导入三份 CSV、查看配对明细与汇总、更正切换时刻并增量重算。"""

from __future__ import annotations

import os
import sqlite3

from flask import (
    Flask,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

from . import store
from .analysis import ValidationError

DEFAULT_DB = os.environ.get(
    "COLDSTORE_DB", os.path.join(os.getcwd(), "data", "coldstore.sqlite3")
)


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def create_app(
    db_path: str | None = None,
    testing: bool = False,
    secret_key: str | None = None,
) -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=secret_key
        or os.environ.get("COLDSTORE_SECRET", "dev-only-secret"),
        DB_PATH=db_path or DEFAULT_DB,
        TESTING=testing,
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    )

    def get_db() -> sqlite3.Connection:
        if "db" not in g:
            _ensure_parent(app.config["DB_PATH"])
            g.db = store.connect(app.config["DB_PATH"])
        return g.db

    @app.teardown_appcontext
    def close_db(exc):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    @app.get("/")
    def index():
        db = get_db()
        data = store.summary(db)
        pairs_by_round = {
            r["round_id"]: store.pair_rows(db, r["round_id"]) for r in data["rounds"]
        }
        return render_template(
            "index.html",
            data=data,
            pairs_by_round=pairs_by_round,
            switches=store.switch_rows(db),
            audit=store.audit_rows(db),
        )

    @app.post("/import")
    def import_csv():
        db = get_db()
        files = {
            "rounds": request.files.get("rounds"),
            "switches": request.files.get("switches"),
            "readings": request.files.get("readings"),
        }
        missing = [name for name, f in files.items() if f is None or not f.filename]
        if missing:
            flash(f"请选择全部三份 CSV，缺少: {', '.join(missing)}", "error")
            return redirect(url_for("index"))
        try:
            payload = {name: files[name].read() for name in files}
            result = store.import_all(
                db, payload["rounds"], payload["switches"], payload["readings"]
            )
        except ValidationError as exc:
            flash(f"导入失败（未写入任何数据）：{exc}", "error")
            return redirect(url_for("index"))
        except sqlite3.DatabaseError as exc:
            flash(f"导入失败，数据库已回滚：{exc}", "error")
            return redirect(url_for("index"))
        flash(
            f"导入成功：{result['rounds']} 轮、{result['switches']} 条切换、"
            f"{result['readings']} 条分钟读数；已重算 {len(result['recomputed'])} 轮。",
            "success",
        )
        return redirect(url_for("index"))

    @app.post("/switches/<int:switch_id>/correct")
    def correct(switch_id: int):
        db = get_db()
        new_at = (request.form.get("new_at") or "").strip()
        new_strategy = (request.form.get("new_strategy") or "").strip() or None
        new_scope = (request.form.get("new_round_id") or "").strip() or None
        allow_scope = "new_round_id" in request.form
        try:
            result = store.correct_switch(
                db,
                switch_id,
                new_at,
                new_strategy,
                new_round_id=new_scope,
                update_round_scope=allow_scope,
            )
        except ValidationError as exc:
            flash(f"更正失败（记录与结果未改动）：{exc}", "error")
            return redirect(url_for("index"))
        except sqlite3.DatabaseError as exc:
            flash(f"更正失败，事务已回滚：{exc}", "error")
            return redirect(url_for("index"))
        flash(result["message"], "success" if result["changed"] else "info")
        return redirect(url_for("index"))

    @app.get("/api/summary")
    def api_summary():
        return jsonify(store.summary(get_db()))

    def _run_trial_payload(payload) -> tuple[dict | None, str | None]:
        round_id = str(payload.get("round_id") or "").strip()
        start = str(payload.get("billing_start") or "").strip()
        end = str(payload.get("billing_end") or "").strip()
        cap_raw = payload.get("missing_cap_kwh")
        cap = str(cap_raw).strip() if cap_raw is not None else ""
        if not round_id:
            return None, "请选择要试算的轮次"
        try:
            result = store.trial_from_store(get_db(), round_id, start, end, cap)
        except ValidationError as exc:
            return None, str(exc)
        return result.as_dict(), None

    @app.get("/trial")
    def trial_page():
        db = get_db()
        data = store.summary(db)
        return render_template(
            "trial.html",
            data=data,
            form={},
            trial=None,
            error=None,
        )

    @app.post("/trial")
    def trial_run():
        db = get_db()
        data = store.summary(db)
        form = {
            "round_id": (request.form.get("round_id") or "").strip(),
            "billing_start": (request.form.get("billing_start") or "").strip(),
            "billing_end": (request.form.get("billing_end") or "").strip(),
            "missing_cap_kwh": (request.form.get("missing_cap_kwh") or "").strip(),
        }
        try:
            result = store.trial_from_store(
                db,
                form["round_id"],
                form["billing_start"],
                form["billing_end"],
                form["missing_cap_kwh"],
            )
        except ValidationError as exc:
            return render_template(
                "trial.html", data=data, form=form, trial=None, error=str(exc)
            ), 400
        return render_template(
            "trial.html", data=data, form=form, trial=result.as_dict(), error=None
        )

    @app.post("/api/trials")
    def api_trial():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "请求体必须是 JSON 对象"}), 400
        result, error = _run_trial_payload(payload)
        if error is not None:
            return jsonify({"error": error}), 400
        return jsonify(result)

    @app.get("/api/rounds/<round_id>/pairs")
    def api_pairs(round_id: str):
        db = get_db()
        plans = {p.round_id for p in store.load_plans(db)}
        if round_id not in plans:
            return jsonify({"error": f"轮次 {round_id} 不存在"}), 404
        return jsonify(
            {
                "round_id": round_id,
                "pairs": store.pair_rows(db, round_id),
            }
        )

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True})

    @app.errorhandler(413)
    def too_large(_exc):
        flash("上传文件过大（上限 16 MB）", "error")
        return redirect(url_for("index"))

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
