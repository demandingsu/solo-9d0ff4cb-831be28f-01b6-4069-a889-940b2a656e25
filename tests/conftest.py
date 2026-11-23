"""共享测试夹具与样例数据构造。"""

import io

import pytest

from coldstore import store
from coldstore.web import create_app

ROUNDS_CSV = """round_id,a_start,a_end,b_start,b_end
R1,2026-03-01 00:00,2026-03-01 01:00,2026-03-01 01:30,2026-03-01 02:30
R2,2026-03-01 04:00,2026-03-01 05:00,2026-03-01 05:30,2026-03-01 06:30
"""

SWITCHES_CSV = """switched_at,strategy,round_id
2026-02-28 23:30,A,
2026-03-01 01:15,B,R1
2026-03-01 03:50,B,R2
2026-03-01 04:00,A,R2
2026-03-01 05:35,B,R2
"""


def readings_csv(minutes=391, base=1.2, b_windows=()):
    """00:00 起连续 minutes 分钟的读数；B 窗加 0.05 kWh，带轻微周期波动。"""
    lines = ["minute,kwh"]
    for i in range(minutes):
        stamp = f"2026-03-01 {i // 60:02d}:{i % 60:02d}"
        extra = 0.05 if any(s <= i < e for s, e in b_windows) else 0.0
        lines.append(f"{stamp},{base + extra + 0.0001 * (i % 7):.6f}")
    return "\n".join(lines) + "\n"


def write_files(tmp_path):
    rounds = tmp_path / "rounds.csv"
    switches = tmp_path / "switches.csv"
    readings = tmp_path / "readings.csv"
    rounds.write_text(ROUNDS_CSV, encoding="utf-8")
    switches.write_text(SWITCHES_CSV, encoding="utf-8")
    readings.write_text(
        readings_csv(b_windows=[(90, 150), (330, 390)]), encoding="utf-8"
    )
    return rounds, switches, readings


def import_files(conn, tmp_path):
    rounds, switches, readings = write_files(tmp_path)
    return store.import_all(
        conn,
        rounds.read_text(encoding="utf-8"),
        switches.read_text(encoding="utf-8"),
        readings.read_text(encoding="utf-8"),
    )


def upload(client, tmp_path):
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


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "test.sqlite3")


@pytest.fixture()
def conn(db_path):
    connection = store.connect(db_path)
    yield connection
    connection.close()


@pytest.fixture()
def client(db_path):
    app = create_app(db_path, testing=True, secret_key="test-secret")
    with app.test_client() as c:
        yield c
