"""Service entry points. Each docker-compose service runs one of these:

    python -m backend.app.workers.runner collector|news|predictor|learner|backup
"""
from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import threading
import time

from ..config import get_settings
from ..db import get_db, now_ms

log = logging.getLogger("runner")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)), "level": record.levelname,
               "logger": record.name, "msg": record.getMessage()}
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False)


def setup_logging(service: str) -> None:
    import os
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter() if os.getenv("LOG_FORMAT", "json") == "json" else
                   logging.Formatter(f"%(asctime)s {service} %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


def _stop_event() -> threading.Event:
    ev = threading.Event()

    def handler(signum, _frame):
        log.info("signal %s received, stopping", signum)
        ev.set()
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    return ev


def periodic(service: str, fn, interval, request_key: str | None = None, run_first: bool = True) -> None:
    """Run fn() every `interval` seconds (number or zero-arg callable); a request flag in
    system_state triggers an immediate run."""
    db = get_db()
    stop = _stop_event()
    get_interval = interval if callable(interval) else (lambda: interval)
    next_run = time.monotonic() if run_first else time.monotonic() + get_interval()
    last_result: dict = {}
    while not stop.is_set():
        forced = False
        if request_key:
            req = db.get_state(request_key)
            if req and not req.get("handled"):
                forced = True
        if forced or time.monotonic() >= next_run:
            t0 = time.monotonic()
            try:
                last_result = fn(forced) or {}
                status = "ok"
            except Exception as exc:  # noqa: BLE001 - a crash loop is worse than a logged error
                log.exception("%s cycle failed", service)
                last_result, status = {"error": f"{type(exc).__name__}: {exc}"}, "error"
            if forced:
                db.set_state(request_key, {**(db.get_state(request_key) or {}), "handled": True,
                                           "handled_ms": now_ms(), "status": status})
            db.heartbeat(service, {"status": status, "last_cycle_sec": round(time.monotonic() - t0, 2),
                                   "summary": json.loads(json.dumps(last_result, default=str))})
            next_run = time.monotonic() + get_interval()
        else:
            db.heartbeat(service, {"status": "idle", "next_run_in_sec": round(next_run - time.monotonic(), 1),
                                   "summary": json.loads(json.dumps(last_result, default=str))})
        stop.wait(min(5.0, max(0.5, next_run - time.monotonic())))
    log.info("%s stopped", service)


def main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] not in {"collector", "news", "predictor", "learner", "backup"}:
        print("usage: python -m backend.app.workers.runner collector|news|predictor|learner|backup", file=sys.stderr)
        return 2
    service = argv[0]
    setup_logging(service)
    s = get_settings()
    db = get_db()
    log.info("starting %s (db=%s, exchanges=%s, symbols=%s)", service, db.dialect, s.exchanges, s.symbols)

    if service == "collector":
        from ..market.collector import run_collector

        async def amain():
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, stop.set)
            await run_collector(db, s, stop)
        asyncio.run(amain())
        return 0
    if service == "news":
        from ..news.service import run_news_once
        periodic("news", lambda forced: run_news_once(db, s), s.news_interval_sec, "news_refresh_request")
    elif service == "predictor":
        from ..learning.predictor import Predictor
        p = Predictor(db, s)
        periodic("predictor", lambda forced: p.run_once(), s.predictor_poll_sec)
    elif service == "learner":
        from ..learning import registry
        from ..learning.engine import run_learning_once

        def learner_interval() -> float:
            # retry quickly until every market has a production model, then the normal cadence
            keys = [registry.model_key(sym, tf, s.horizon_bars) for sym in s.symbols for tf in s.predict_timeframes]
            missing = any(registry.get_production(db, k) is None for k in keys)
            return min(120.0, s.learning_interval_sec) if missing else s.learning_interval_sec
        periodic("learner", lambda forced: run_learning_once(db, s, force=forced), learner_interval,
                 "learning_request")
    elif service == "backup":
        from .backup import run_backup_once
        periodic("backup", lambda forced: run_backup_once(db, s), s.backup_interval_sec, "backup_request")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
