"""API tested over real HTTP against a uvicorn server running in a thread:
monitoring endpoints, auth, rate limiting, validation, action endpoints, static dashboard."""
import json
import socket
import threading
import time
import urllib.error
import urllib.request

import uvicorn

from tests.helpers import fresh_settings


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Server:
    def __init__(self, app):
        self.port = _free_port()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("server did not start")

    def __exit__(self, *a):
        self.server.should_exit = True
        self.thread.join(5)

    def get(self, path, key=None, method="GET"):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", method=method,
                                     headers={"X-API-Key": key} if key else {})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                body = r.read()
                return r.status, dict(r.headers), body
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()


def _app(monkeypatch, tmp_path, **env):
    fresh_settings(monkeypatch, tmp_path, **env)
    from backend.app.api.main import create_app
    return create_app()


def _j(body):
    return json.loads(body)


def test_monitoring_and_dashboard(monkeypatch, tmp_path):
    with Server(_app(monkeypatch, tmp_path)) as srv:
        st, _, b = srv.get("/api/health")
        assert st == 200 and _j(b)["database"] is True
        st, _, b = srv.get("/api/ready")
        assert st == 503 and _j(b)["ready"] is False  # no data, no model yet: honestly not ready
        st, _, b = srv.get("/api/status")
        s = _j(b)
        assert st == 200
        for k in ("api", "database", "websocket", "market_data", "news", "model", "learning", "services",
                  "last_prediction", "last_backup"):
            assert k in s, k
        assert s["websocket"]["state"] == "DISCONNECTED" and s["market_data"]["state"] == "STALE"
        assert s["model"]["state"] == "NOT TRAINED" and s["auth"].startswith("DISABLED")
        st, h, b = srv.get("/")
        assert st == 200 and b"AI Market Predictor" in b
        assert "default-src 'self'" in h.get("content-security-policy", "")
        st, _, b = srv.get("/static/app.js")
        assert st == 200 and b"async function api" in b
        for path in ("/api/overview", "/api/candles", "/api/orderbook", "/api/orderflow", "/api/news",
                     "/api/predictions", "/api/model", "/api/learning"):
            st, _, b = srv.get(path + "?symbol=BTC/USDT&tf=15m")
            assert st == 200, (path, b[:200])
        st, _, b = srv.get("/api/backtest?symbol=BTC/USDT&tf=15m")
        assert st == 404 and "no production model" in _j(b)["error"]


def test_validation_errors(monkeypatch, tmp_path):
    with Server(_app(monkeypatch, tmp_path)) as srv:
        st, _, b = srv.get("/api/overview?symbol=DOGE/USDT")
        assert st == 400 and "symbol must be one of" in _j(b)["detail"]
        st, _, b = srv.get("/api/candles?symbol=BTC/USDT&tf=2h")
        assert st == 400
        st, _, b = srv.get("/api/candles?symbol=BTC/USDT&tf=15m&limit=abc")
        assert st == 400


def test_auth_and_actions(monkeypatch, tmp_path):
    with Server(_app(monkeypatch, tmp_path, API_KEY="s3cret-key")) as srv:
        assert srv.get("/api/health")[0] == 200            # liveness stays public for healthchecks
        assert srv.get("/api/config")[0] == 200            # tells the UI that a key is required
        assert _j(srv.get("/api/config")[2])["auth_required"] is True
        assert srv.get("/api/status")[0] == 401
        assert srv.get("/api/status", key="wrong")[0] == 401
        assert srv.get("/api/status", key="s3cret-key")[0] == 200
        assert srv.get("/api/learning/run", method="POST")[0] == 401
        st, _, b = srv.get("/api/learning/run", key="s3cret-key", method="POST")
        assert st == 202 and _j(b)["status"] == "queued"
        st, _, b = srv.get("/api/learning/run", key="s3cret-key", method="POST")
        assert _j(b)["status"] == "already_queued"
        assert srv.get("/api/news/refresh", key="s3cret-key", method="POST")[0] == 202
        assert srv.get("/api/backup/run", key="s3cret-key", method="POST")[0] == 202
        assert srv.get("/api/learning/run", key="s3cret-key")[0] == 405  # GET is not allowed for actions
        from backend.app.db import get_db
        assert get_db().get_state("learning_request")["handled"] is False


def test_rate_limit(monkeypatch, tmp_path):
    with Server(_app(monkeypatch, tmp_path, RATE_LIMIT_PER_MIN=5)) as srv:
        codes = [srv.get("/api/health")[0] for _ in range(8)]
        assert codes[:5] == [200] * 5 and 429 in codes[5:]
        st, h, _ = srv.get("/api/health")
        assert st == 429 and "retry-after" in {k.lower() for k in h}
