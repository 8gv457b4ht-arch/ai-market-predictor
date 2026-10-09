"""Deployment configuration checks: compose services, restart policies, healthchecks,
persistent volumes, non-root image, env documentation, no committed secrets, healthcheck script."""
import re
import subprocess
import sys
from pathlib import Path

import yaml

from tests.helpers import fresh_settings

ROOT = Path(__file__).resolve().parents[1]
SERVICES = {"api", "collector", "news", "predictor", "learner", "backup"}


def test_compose_services_restart_health_volumes():
    c = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    svcs = c["services"]
    assert SERVICES <= set(svcs)
    for name in SERVICES:
        s = svcs[name]
        assert s["restart"] == "unless-stopped", name
        assert "healthcheck" in s and s["healthcheck"]["test"][0] == "CMD", name
        vols = s["volumes"]
        for v in ("./data:/app/data", "./models:/app/models", "./backups:/app/backups"):
            assert v in vols, (name, v)
        assert s["logging"]["options"]["max-size"]
        assert s["env_file"] == ".env"
    for name in SERVICES - {"api"}:
        assert svcs[name]["command"][-1] == name
    assert svcs["caddy"]["profiles"] == ["https"]


def test_dockerfile_runs_as_non_root_and_has_test_stage():
    d = (ROOT / "Dockerfile").read_text()
    assert "USER app" in d and "AS test" in d and "python -m pytest" in d and "HEALTHCHECK" in d
    assert "--no-proxy-headers" in d


def test_env_example_documents_every_setting():
    src = (ROOT / "backend/app/config.py").read_text()
    names = set(re.findall(r'(?:_list|_float|_int|_bool|os\.getenv)\("([A-Z0-9_]+)"', src))
    env = (ROOT / ".env.example").read_text()
    documented = set(re.findall(r"^#?\s?([A-Z0-9_]+)=", env, re.M))
    missing = names - documented - {"MODEL_DIR", "BACKUP_DIR"}  # set by docker-compose
    assert not missing, missing


def test_no_secrets_committed():
    pat = re.compile(r"(sk-[A-Za-z0-9]{20,}|sk-ant-[A-Za-z0-9-]{20,}|AKIA[0-9A-Z]{16})")
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".js", ".html", ".yml", ".md", ".txt", ".sh", ".example"} and ".git" not in p.parts:
            assert not pat.search(p.read_text(errors="ignore")), p
    env = (ROOT / ".env.example").read_text()
    for key in ("API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        assert re.search(rf"^{key}=$", env, re.M), key
    assert ".env" in (ROOT / ".gitignore").read_text().split()


def test_no_trading_endpoints_or_private_api_usage():
    code = "\n".join(p.read_text() for p in (ROOT / "backend").rglob("*.py"))
    for forbidden in ("/api/v3/order", "/v5/order", "/api/v5/trade", "/sapi/v1/capital/withdraw", "/api/v5/asset/withdrawal", "/v5/asset/withdraw", "X-MBX-APIKEY", "X-BAPI-API-KEY", "OK-ACCESS-KEY"):
        assert forbidden not in code, forbidden


def test_service_healthcheck_script(monkeypatch, tmp_path):
    s, db = fresh_settings(monkeypatch, tmp_path)
    env = {"DATABASE_URL": s.database_url, "PATH": "/usr/bin:/bin"}
    run = lambda *a: subprocess.run([sys.executable, str(ROOT / "scripts/healthcheck.py"), *a], env=env,
                                    capture_output=True, text=True)
    assert run("service", "predictor", "60").returncode == 1  # no heartbeat yet
    db.heartbeat("predictor")
    assert run("service", "predictor", "60").returncode == 0
    assert run("api").returncode == 1  # nothing listening in the test
