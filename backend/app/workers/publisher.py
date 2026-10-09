"""Continuous mode (VPS / Docker): publish the dashboard data to the same `data` branch the GitHub Pages site reads.

Runs next to the always-on services (collector with persistent WebSocket streams, predictor, learner, news,
backup). Every PUBLISH_INTERVAL_SEC it records source checks from the live stream status, exports the public
JSON with BACKEND_MODE=continuous, and replaces the `data` branch with one new commit through the GitHub Git
Data API (no git binary needed). Every PUBLISH_STATE_INTERVAL_SEC the commit also carries a gzipped database
snapshot and the production models, so the scheduled GitHub job could take over again.

Secrets: PUBLISH_TOKEN (fine-grained token, Contents: read & write, this repository only) stays in .env on the
server. When the continuous backend runs, set the repository variable BACKEND_MODE=external so the scheduled
GitHub job stops writing to the same branch.
"""
from __future__ import annotations

import base64
import gzip
import json
import logging
import os
import sqlite3
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..config import Settings
from ..db import Database, now_ms

log = logging.getLogger("publisher")
API = "https://api.github.com"


class GitHubPublisher:
    def __init__(self, repo: str, token: str, branch: str = "data", request=None):
        self.repo, self.token, self.branch = repo, token, branch
        self._request = request  # injectable for tests
        self.state_blobs: dict[str, str] = {}  # path -> blob sha of the last full-state upload

    def call(self, method: str, path: str, body: dict | None = None) -> dict:
        if self._request:
            return self._request(method, path, body)
        req = urllib.request.Request(f"{API}{path}", method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                                              "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "ai-market-predictor"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as exc:  # never include the token in errors
            raise RuntimeError(f"GitHub API {method} {path.split('?')[0]}: HTTP {exc.code}") from None

    def blob(self, data: bytes) -> str:
        return self.call("POST", f"/repos/{self.repo}/git/blobs",
                         {"content": base64.b64encode(data).decode(), "encoding": "base64"})["sha"]

    def publish(self, files: dict[str, bytes], keep_state: bool = True, message: str = "continuous backend") -> str:
        """One orphan commit containing `files` (+ the last uploaded state files) -> force-update the branch."""
        tree = [{"path": p, "mode": "100644", "type": "blob", "sha": self.blob(b)} for p, b in files.items()]
        if keep_state:
            tree += [{"path": p, "mode": "100644", "type": "blob", "sha": s} for p, s in self.state_blobs.items() if p not in files]
        t = self.call("POST", f"/repos/{self.repo}/git/trees", {"tree": tree})["sha"]
        c = self.call("POST", f"/repos/{self.repo}/git/commits", {"message": message, "tree": t, "parents": []})["sha"]
        try:
            self.call("PATCH", f"/repos/{self.repo}/git/refs/heads/{self.branch}", {"sha": c, "force": True})
        except RuntimeError:
            self.call("POST", f"/repos/{self.repo}/git/refs", {"ref": f"refs/heads/{self.branch}", "sha": c})
        return c


def record_stream_checks(db: Database, stale_sec: float = 30.0) -> int:
    """Continuous mode: one WebSocket check per exchange from the collector's live stream status."""
    now = now_ms()
    rows = []
    for r in db.query("SELECT exchange, MAX(last_msg_ms) AS last, MAX(ABS(COALESCE(clock_skew_ms,0))) AS skew, "
                      "MAX(last_error) AS err FROM stream_status GROUP BY exchange"):
        ok = bool(r["last"]) and now - int(r["last"]) <= stale_sec * 1000
        rows.append((now, r["exchange"], "ws", int(ok), None if ok else "stale" if r["last"] else "no_data",
                     None, None, None if ok else (r["err"] or "")[:300] or None, r["skew"]))
    db.executemany("INSERT INTO source_checks(ts_ms,exchange,channel,ok,kind,latency_ms,host,detail,skew_ms) "
                   "VALUES(?,?,?,?,?,?,?,?,?)", rows)
    return len(rows)


def record_rest_checks(db: Database, settings: Settings) -> dict:
    from ..exchanges import rest
    out = {}
    for ex in settings.exchanges:
        t0 = time.monotonic()
        try:
            rest.fetch_ticker(ex, settings.symbols[0])
            ok, kind, detail = True, None, None
        except Exception as exc:  # noqa: BLE001
            ok, kind, detail = False, getattr(exc, "kind", type(exc).__name__), str(exc)[:300]
        db.execute("INSERT INTO source_checks(ts_ms,exchange,channel,ok,kind,latency_ms,host,detail,skew_ms) "
                   "VALUES(?,?,?,?,?,?,?,?,?)", (now_ms(), ex, "rest", int(ok), kind, int((time.monotonic() - t0) * 1000),
                                                 rest._working_host.get(ex), detail, None))
        out[ex] = ok
    return out


def probes_from_live(db: Database, settings: Settings, rest_ok: dict) -> dict:
    """The same `exchange_probes` structure the scheduled cycle writes, built from live stream status."""
    now = now_ms()
    probes = {}
    for ex in settings.exchanges:
        syms = {}
        for r in db.query("SELECT * FROM stream_status WHERE exchange=?", (ex,)):
            fresh = bool(r["last_msg_ms"]) and now - int(r["last_msg_ms"]) <= settings.stale_after_sec * 1000
            syms[r["symbol"]] = {"trade_received": fresh, "book_received": fresh, "clock_skew_ms": r["clock_skew_ms"],
                                 "last_verified_age_sec": round((now - r["last_msg_ms"]) / 1000, 1) if r["last_msg_ms"] else None,
                                 "gaps": r["gaps"], "duplicates": r["duplicates"], "invalid": r["invalid"]}
        good = [s for s, v in syms.items() if v["trade_received"]]
        live = bool(syms) and len(good) == len(settings.symbols)
        primary = settings.primary_exchange
        candles = {}
        if ex == primary:
            for sym in settings.symbols:
                for tf in settings.predict_timeframes:
                    from ..config import TIMEFRAME_MS
                    last = db.scalar("SELECT MAX(open_ts) FROM candles WHERE exchange=? AND symbol=? AND timeframe=? AND closed=1",
                                     (ex, sym, tf))
                    candles[f"{sym} {tf}"] = {"ok": last is not None, "last_closed_ts": last,
                                              "last_close_age_sec": round((now - last - TIMEFRAME_MS[tf]) / 1000, 1) if last else None}
        probes[ex] = {"rest": {"ok": bool(rest_ok.get(ex)), "kind": None if rest_ok.get(ex) else "error", "candles": candles},
                      "ws": {"ok": live, "verified_live": live, "verified_symbols": good, "kind": None if live else "stale",
                             "error": None if live else "stream not live", "symbols": syms, "events": None}}
    return {"ts_ms": now, "primary": settings.primary_exchange, "probes": probes}


def snapshot_db(db: Database) -> bytes:
    """Consistent copy through the SQLite backup API, gzipped."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "snap.sqlite3"
        src = sqlite3.connect(db.path)
        dst = sqlite3.connect(tmp)
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        return gzip.compress(tmp.read_bytes(), compresslevel=6)


def run_publish_once(db: Database, settings: Settings, pub: GitHubPublisher | None, with_state: bool, rest_check: bool) -> dict:
    from ..cloud.export import export_public
    settings.backend_mode = "continuous"
    rest_ok = record_rest_checks(db, settings) if rest_check else (db.get_state("publisher_rest_ok") or {})
    if rest_check:
        db.set_state("publisher_rest_ok", rest_ok)
    record_stream_checks(db, settings.stale_after_sec)
    db.set_state("exchange_probes", probes_from_live(db, settings, rest_ok))
    lc = db.get_state("heartbeat:collector") or {}
    db.set_state("last_cycle", {"started_ms": lc.get("ts_ms") or now_ms(), "finished_ms": now_ms(), "ok": True, "errors": [],
                                "duration_sec": 0, "primary_exchange": settings.primary_exchange,
                                "db": {"ok": True, "check": "ok", "ts_ms": now_ms(),
                                       "predictions": db.scalar("SELECT COUNT(*) FROM predictions")}})
    export_public(db, settings, settings.public_dir)
    out = {"exported": True}
    if pub is None:
        return out
    files = {f"public/{p.name}": p.read_bytes() for p in Path(settings.public_dir).glob("*.json")}
    if with_state and db.dialect == "sqlite":
        state = {"state/market.sqlite3.gz": snapshot_db(db)}
        for sym in settings.symbols:
            for tf in settings.predict_timeframes:
                from ..learning import registry
                prod = registry.get_production(db, registry.model_key(sym, tf, settings.horizon_bars))
                p = prod and registry.resolve_path(prod["artifact_path"], settings.model_dir)
                if p and p.exists():
                    state[f"state/models/{p.name}"] = p.read_bytes()
        pub.state_blobs = {k: pub.blob(v) for k, v in state.items()}
    out["commit"] = pub.publish(files, message=f"continuous backend {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    return out


def run_publisher(db: Database, settings: Settings, stop) -> None:
    repo, token = os.getenv("PUBLISH_REPO", "").strip(), os.getenv("PUBLISH_TOKEN", "").strip()
    pub = GitHubPublisher(repo, token, os.getenv("PUBLISH_BRANCH", "data")) if repo and token else None
    if pub is None:
        log.warning("PUBLISH_REPO/PUBLISH_TOKEN not set: exporting locally only (%s)", settings.public_dir)
    every = float(os.getenv("PUBLISH_INTERVAL_SEC", "120"))
    state_every = float(os.getenv("PUBLISH_STATE_INTERVAL_SEC", "3600"))
    rest_every = 300.0
    last_state = last_rest = 0.0
    while not stop.is_set():
        t0 = time.monotonic()
        try:
            res = run_publish_once(db, settings, pub, with_state=t0 - last_state >= state_every or not last_state,
                                   rest_check=t0 - last_rest >= rest_every or not last_rest)
            if res.get("commit") and (t0 - last_state >= state_every or not last_state):
                last_state = t0
            if t0 - last_rest >= rest_every or not last_rest:
                last_rest = t0
            db.heartbeat("publisher", {"status": "ok", **res})
        except Exception as exc:  # noqa: BLE001 - keep publishing on the next round
            log.exception("publish failed")
            db.heartbeat("publisher", {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:300]})
        stop.wait(max(5.0, every - (time.monotonic() - t0)))
