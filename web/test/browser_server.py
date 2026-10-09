"""Browser test 3 (server mode): the Python backend cycle runs against SYNTHETIC exchanges (test only),
exports public JSON, and the published dashboard renders it. Checks the Python -> JSON -> JS contract."""
import asyncio
import functools
import http.server
import os
import sys
import tempfile
import threading
from pathlib import Path

from playwright.async_api import async_playwright


def _annotate(exc: BaseException) -> None:
    import traceback
    msg = "".join(traceback.format_exception(exc)).replace("%", "%25").replace("\r", "").replace("\n", "%0A")
    print(f"::error title={Path(__file__).name} failed::{msg[-3000:]}", flush=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
APP = ROOT / "web" / "dist" / "AI_Market_Predictor.html"
CHROME = os.getenv("CHROME_PATH") or ("/opt/pw-browsers/chromium-1194/chrome-linux/chrome" if os.path.exists("/opt/pw-browsers/chromium-1194/chrome-linux/chrome") else None)


def produce_backend_output(tmp: Path) -> None:
    from tests.helpers import make_env
    env = make_env(tmp, PRIMARY_EXCHANGE="auto", SYMBOLS="BTC/USDT,ETH/USDT", PREDICT_TIMEFRAMES="15m",
                   TIMEFRAMES="15m,1h,4h,1d", HISTORY_BARS=2400, CONTEXT_HISTORY_BARS=800, WS_SAMPLE_SEC=3,
                   WALK_FORWARD_FOLDS=3, MIN_TRAIN_ROWS=600, PUBLIC_DIR=tmp / "site" / "data")
    os.environ.update({k: str(v) for k, v in env.items()})
    from backend.app import config
    from backend.app.cloud.cycle import run_cycle
    from backend.app.db import get_db
    from backend.app.exchanges import rest
    import tests.test_cloud_cycle as fx
    rest.fetch_klines, rest.fetch_history, rest.fetch_ticker = fx.fake_klines, fx.fake_history, fx.fake_ticker
    config.reset_settings()
    s, db = config.get_settings(), get_db()
    run_cycle(db, s, connect=fx.fake_connect, news=False)
    run_cycle(db, s, connect=fx.fake_connect, news=False, learn=False)


async def main(tmp: Path):
    site = tmp / "site"
    (site / "index.html").write_text(APP.read_text())
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
    handler.log_message = lambda *a: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = srv.server_address[1]
    (site / "config.json").write_text(f'{{"state_url":"http://127.0.0.1:{port}/data/"}}')
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path=CHROME)
        errors = []
        for name, vp in (("desktop", {"width": 1440, "height": 1000}), ("mobile", {"width": 390, "height": 844})):
            ctx = await b.new_context(viewport=vp, device_scale_factor=2 if name == "mobile" else 1)
            page = await ctx.new_page()
            page.on("pageerror", lambda e: errors.append(str(e)))
            # exchanges unreachable from this browser: everything must come from the backend
            await page.route("**/*", lambda r: r.continue_() if r.request.url.startswith(f"http://127.0.0.1:{port}") else r.abort())
            await page.route_web_socket("wss://**", lambda ws: ws.close())
            await page.goto(f"http://127.0.0.1:{port}/")
            await page.wait_for_function("() => /backend/.test(document.querySelector('#modeNote').textContent)", timeout=30000)
            await page.wait_for_timeout(2500)
            status = await page.text_content("#statusGrid")
            if name == "desktop":
                print("mode:", await page.text_content("#modeNote"))
                print("status:", " | ".join(x.strip()[:70] for x in status.split("●")))
                print("price:", await page.text_content("#price"), "| signal:", await page.text_content("#signal"))
                print("reasons:", await page.text_content("#reasons"))
                print("why:", (await page.text_content("#whyNoTrade"))[:400])
                assert "24/7 backend" in await page.text_content("#modeNote")
                assert "RUNNING" in status and "VERIFIED" in status and "READY" in status
                assert "PARTIAL" in status and "BTC/USDT only" in status  # fixture streams BTC only: ETH must not count as verified
                assert (await page.text_content("#price")).startswith("$")
                assert "out-of-sample" in await page.text_content("#whyNoTrade")
                assert "region" in await page.text_content("#exTable")  # binance blocked at the backend
                ledger = await page.text_content("#ledger")
                assert "Candle" in ledger and ("NO TRADE" in ledger or "UP" in ledger or "DOWN" in ledger)
                assert not await page.is_visible("#welcome")
                for tab in ("volume", "cvd", "book", "ind", "price"):
                    await page.click(f"button[data-chart={tab}]")
                    await page.wait_for_timeout(250)
                    if tab != "cvd":
                        assert not await page.is_visible("#chartEmpty"), (tab, await page.text_content("#chartEmpty"))
                await page.select_option("#ledgerFilter", "notrade")
                await page.wait_for_timeout(300)
            else:
                print("mobile scrollWidth:", await page.evaluate("document.documentElement.scrollWidth"))
                assert await page.evaluate("document.documentElement.scrollWidth") <= 390
            await page.screenshot(path=str(APP.parent / f"server_{name}.png"), full_page=True)
            await ctx.close()
        assert not errors, errors
        print("SERVER-MODE TEST PASSED; page errors: none")
        await b.close()
    srv.shutdown()


if __name__ == "__main__":
    TMP = Path(tempfile.mkdtemp())
    produce_backend_output(TMP)  # synchronous backend cycle first, outside the browser's event loop
    try:
        asyncio.run(main(TMP))
    except BaseException as exc:
        _annotate(exc)
        raise
