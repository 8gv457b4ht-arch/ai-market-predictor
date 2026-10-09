"""Long real-data observation of the CONTINUOUS components (persistent WebSocket collector + predictor),
for example inside a manually started GitHub Actions job (up to ~5.5 h). Works on a COPY of the state
database, so the published ledger is not touched. Results are printed as ::notice annotations and to the
job summary.

    python scripts/soak_test.py --state state --minutes 180

This verifies hours of continuous operation, not 24/7 operation: a real always-on server is still needed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state")
    ap.add_argument("--minutes", type=float, default=180)
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp())
    src = Path(a.state)
    if (src / "market.sqlite3").exists():
        shutil.copy(src / "market.sqlite3", work / "market.sqlite3")
    if (src / "models").exists():
        shutil.copytree(src / "models", work / "models")
    os.environ.update({"DATABASE_URL": f"sqlite:///{work / 'market.sqlite3'}", "MODEL_DIR": str(work / "models"),
                       "BACKUP_DIR": str(work / "backups"), "PRIMARY_EXCHANGE": os.getenv("PRIMARY_EXCHANGE", "binance"),
                       "STORE_RAW_TRADES": "0", "LOG_FORMAT": "text", "CONTROL_DIR": str(work / "control")})
    from backend.app.config import TIMEFRAME_MS, get_settings
    from backend.app.db import get_db, now_ms
    from backend.app.learning.predictor import Predictor
    from backend.app.market.collector import run_collector
    from backend.app.workers.runner import setup_logging
    setup_logging("soak")
    s, db = get_settings(), get_db()
    if s.primary_exchange == "auto":
        s.primary_exchange = db.get_state("primary_exchange") or "binance"
    pred = Predictor(db, s)
    samples: dict[str, list[int]] = {ex: [] for ex in s.exchanges}
    made: list[dict] = []
    t_end = time.time() + a.minutes * 60

    async def amain():
        stop = asyncio.Event()
        task = asyncio.create_task(run_collector(db, s, stop))
        while time.time() < t_end:
            await asyncio.sleep(10)
            now = now_ms()
            for ex in s.exchanges:
                rows = db.query("SELECT last_msg_ms FROM stream_status WHERE exchange=?", (ex,))
                live = bool(rows) and all(r["last_msg_ms"] and now - r["last_msg_ms"] <= s.stale_after_sec * 1000 for r in rows) \
                    and len(rows) == len(s.symbols)
                samples[ex].append(int(live))
            out = await asyncio.to_thread(pred.run_once)
            for k, v in out.items():
                if v.get("status") == "predicted":
                    made.append({"key": k, "delay": v["gate"]["decision_delay_sec"], "prediction": v["prediction"],
                                 "reasons": v["reasons"]})
        stop.set()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(amain())
    streams = {r["exchange"]: r for r in db.query(
        "SELECT exchange, SUM(reconnects) AS reconnects, SUM(gaps) AS gaps, SUM(duplicates) AS duplicates, "
        "SUM(invalid) AS invalid, SUM(stale_events) AS stale, MAX(ABS(clock_skew_ms)) AS skew FROM stream_status GROUP BY exchange")}
    report = {"minutes": a.minutes, "exchanges": {}, "predictions": len(made),
              "decision_delay_sec": {"median": statistics.median([m["delay"] for m in made]) if made else None,
                                     "max": max([m["delay"] for m in made]) if made else None},
              "by_prediction": {p: sum(1 for m in made if m["prediction"] == p) for p in {m["prediction"] for m in made}}}
    for ex in s.exchanges:
        sm = samples[ex]
        st = streams.get(ex, {})
        report["exchanges"][ex] = {"live_share": round(sum(sm) / len(sm), 4) if sm else 0.0, "samples": len(sm),
                                   **{k: st.get(k) for k in ("reconnects", "gaps", "duplicates", "invalid", "stale", "skew")}}
    # expected candle closes vs predictions made (any timeframe)
    expected = sum(int(a.minutes * 60_000 // TIMEFRAME_MS[tf]) for tf in s.predict_timeframes) * len(s.symbols)
    report["expected_closes_approx"] = expected
    text = json.dumps(report, indent=1, default=str)
    print(text)
    for ex, r in report["exchanges"].items():
        print(f"::notice title=soak {ex}::live {r['live_share']:.1%} of {r['samples']} samples, reconnects {r['reconnects']}, "
              f"gaps {r['gaps']}, stale {r['stale']}, max skew {r['skew']} ms")
    print(f"::notice title=soak predictions::{len(made)} predictions (≈{expected} closes expected), "
          f"delay median {report['decision_delay_sec']['median']} s, max {report['decision_delay_sec']['max']} s, {report['by_prediction']}")
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("## Continuous soak test (real data)\n\n```json\n" + text + "\n```\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
