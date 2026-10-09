"""Browser test 2: exchanges answered by an in-test mock (SYNTHETIC data, test only).
Exercises the full path in a real Chromium: REST history -> Web Worker training ->
WebSocket trades/books -> prediction -> ledger -> statuses -> buttons -> settings."""
import asyncio
import os
import json
import math
import random
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.async_api import async_playwright


def _annotate(exc: BaseException) -> None:
    import traceback
    msg = "".join(traceback.format_exception(exc)).replace("%", "%25").replace("\r", "").replace("\n", "%0A")
    print(f"::error title={Path(__file__).name} failed::{msg[-3000:]}", flush=True)

APP = Path(__file__).resolve().parents[1] / "dist" / "AI_Market_Predictor.html"
CHROME = os.getenv("CHROME_PATH") or ("/opt/pw-browsers/chromium-1194/chrome-linux/chrome" if os.path.exists("/opt/pw-browsers/chromium-1194/chrome-linux/chrome") else None)
TF = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
NOW = int(time.time() * 1000)


def make_series(tf, n, seed, last_price):
    rng = random.Random(seed)
    ms = TF[tf]
    cur = NOW - NOW % ms  # the currently forming bar
    trend, p, rows = 0.0, 1.0, []
    for i in range(n):
        trend = 0.97 * trend + rng.gauss(0, 0.0012)
        o = p
        p = p * math.exp(trend + rng.gauss(0, 0.002))
        rows.append([cur - (n - 1 - i) * ms, o, p, rng.random(), rng.random()])
    k = last_price / rows[-1][2]
    out = []
    for ts, o, c, a, b in rows:
        o, c = o * k, c * k
        hi, lo = max(o, c) * (1 + 0.001 * a), min(o, c) * (1 - 0.001 * b)
        v = 50 + 100 * a
        out.append([ts, f"{o:.2f}", f"{hi:.2f}", f"{lo:.2f}", f"{c:.2f}", f"{v:.4f}", ts + ms - 1, "0", 100, f"{v * (0.3 + 0.4 * b):.4f}", "0", "0"])
    return out


SERIES = {}
for si, (sym, px) in enumerate((("BTCUSDT", 62000.0), ("ETHUSDT", 2400.0))):
    for ti, tf in enumerate(TF):
        SERIES[(sym, tf)] = make_series(tf, 3200, 100 * si + ti, px)


async def binance_rest(route):
    u = urlparse(route.request.url)
    q = {k: v[0] for k, v in parse_qs(u.query).items()}
    if u.path.endswith("/klines"):
        rows = SERIES[(q["symbol"], q["interval"])]
        end = int(q.get("endTime", 10**15))
        rows = [r for r in rows if r[0] <= end][-int(q.get("limit", 500)):]
        return await route.fulfill(status=200, json=rows, headers={"access-control-allow-origin": "*"})
    if u.path.endswith("/ticker/24hr"):
        last = SERIES[(q["symbol"], "15m")][-1]
        return await route.fulfill(json={"lastPrice": last[4], "priceChangePercent": "1.25", "volume": "1000", "quoteVolume": "1"},
                                   headers={"access-control-allow-origin": "*"})
    await route.fulfill(status=404, body="{}")


NEWS = {"Data": [
    {"TITLE": "Bitcoin ETF inflows rise as price surges", "URL": "https://example.test/a", "PUBLISHED_ON": NOW // 1000 - 600,
     "BODY": "Spot bitcoin ETFs recorded inflows.", "SOURCE_DATA": {"NAME": "TestWire"}},
    {"TITLE": "Fed holds rates, signals rate cut later", "URL": "https://example.test/b", "PUBLISHED_ON": NOW // 1000 - 3600,
     "BODY": "FOMC kept policy unchanged.", "SOURCE_DATA": {"NAME": "TestWire"}}]}


def ws_handler(exchange):
    async def handler(ws):
        sym_px = {"BTC": float(SERIES[("BTCUSDT", "15m")][-1][4]), "ETH": float(SERIES[("ETHUSDT", "15m")][-1][4])}

        async def pump():
            tid = 1000
            for _ in range(400):
                for base, px in sym_px.items():
                    tid += 1
                    t = int(time.time() * 1000)
                    if exchange == "binance":
                        s = f"{base}USDT"
                        ws.send(json.dumps({"stream": f"{s.lower()}@trade", "data": {"e": "trade", "s": s, "t": tid, "p": f"{px:.2f}", "q": "0.01", "T": t, "m": tid % 2 == 0}}))
                        ws.send(json.dumps({"stream": f"{s.lower()}@depth20@100ms", "data": {"lastUpdateId": tid,
                            "bids": [[f"{px - 0.5 - i:.2f}", "1.5"] for i in range(20)], "asks": [[f"{px + 0.5 + i:.2f}", "1.0"] for i in range(20)]}}))
                    elif exchange == "bybit":
                        s = f"{base}USDT"
                        ws.send(json.dumps({"topic": f"publicTrade.{s}", "ts": t, "data": [{"T": t, "s": s, "S": "Buy", "v": "0.02", "p": f"{px:.2f}", "i": str(tid)}]}))
                        ws.send(json.dumps({"topic": f"orderbook.50.{s}", "type": "snapshot", "ts": t, "data": {"s": s, "b": [[f"{px - 0.4:.2f}", "2"]], "a": [[f"{px + 0.4:.2f}", "2"]], "u": tid, "seq": tid}}))
                    else:
                        s = f"{base}-USDT"
                        ws.send(json.dumps({"arg": {"channel": "trades", "instId": s}, "data": [{"instId": s, "tradeId": str(tid), "px": f"{px:.2f}", "sz": "0.03", "side": "sell", "ts": str(t)}]}))
                        ws.send(json.dumps({"arg": {"channel": "books5", "instId": s}, "data": [{"bids": [[f"{px - 0.6:.2f}", "1", "0", "1"]], "asks": [[f"{px + 0.6:.2f}", "1", "0", "1"]], "ts": str(t), "seqId": tid}]}))
                await asyncio.sleep(0.25)
        ws.on_message(lambda m: None)
        asyncio.ensure_future(pump())
    return handler


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path=CHROME)
        ctx = await b.new_context(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
        page = await ctx.new_page()
        errors, console = [], []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: console.append(m.text) if m.type == "error" else None)
        await page.route("**/*", lambda r: r.abort() if not r.request.url.startswith("file:") else r.continue_())
        for host in ("https://data-api.binance.vision/**", "https://api.binance.com/**"):
            await page.route(host, binance_rest)
        await page.route("https://data-api.coindesk.com/**", lambda r: r.fulfill(json=NEWS, headers={"access-control-allow-origin": "*"}))
        await page.route_web_socket("wss://stream.binance.com:9443/**", ws_handler("binance"))
        await page.route_web_socket("wss://stream.bybit.com/**", ws_handler("bybit"))
        await page.route_web_socket("wss://ws.okx.com:8443/**", ws_handler("okx"))

        await page.goto(APP.as_uri())
        await page.click("#startBtn")
        t0 = time.time()
        await page.wait_for_function("() => document.querySelector('#ledger') && document.querySelectorAll('#ledger tr').length > 1", timeout=600_000)
        print(f"first model trained and first prediction recorded after {time.time() - t0:.0f}s")
        await page.wait_for_timeout(1500)
        price = await page.text_content("#price")
        status = await page.text_content("#statusGrid")
        print("price:", price, "| signal:", await page.text_content("#signal"))
        print("status:", " | ".join(x.strip()[:60] for x in status.split("●")))
        assert price.replace("\u202f", "").replace("\xa0", "").replace(" ", "").startswith("$62"), price  # ru-RU: "$62 000,00"
        assert "LIVE" in status and "OK" in status
        for ex in ("binance", "bybit", "okx"):
            assert ex in await page.text_content("#exTable")
        assert "LIVE" in await page.text_content("#exTable")
        state = await page.evaluate("() => { const S = window.AMPApp.state; const r=[...S.ledger.values()][0]; return {n:S.ledger.size, keys:Object.keys(r), feats:Object.keys(r.features).length, reg:S.registry.versions.map(v=>[v.key,v.status,v.metrics.log_loss,v.metrics.baseline_prior.log_loss]), news:S.news.length} }")
        print("ledger records:", state["n"], "| features per record:", state["feats"], "| news events:", state["news"])
        print("registry:", state["reg"])
        for k in ("prediction", "candle_ts", "symbol", "timeframe", "price", "p_up", "p_down", "confidence", "model_version", "features", "regime", "quality_score", "gate"):
            assert k in state["keys"], k
        assert state["news"] >= 2 and "TestWire" in await page.text_content("#newsList")
        for tab in ("volume", "cvd", "book", "ind", "price"):
            await page.click(f"button[data-chart={tab}]")
            await page.wait_for_timeout(300)
            empty = await page.is_visible("#chartEmpty")
            assert not empty, (tab, await page.text_content("#chartEmpty"))
        await page.click("#learnBtn")
        await page.wait_for_timeout(8000)
        mk = await page.text_content("#modelKv")
        print("model panel:", mk[:300].replace("\n", " "))
        assert "Последняя проверка претендента" in mk
        await page.click("text=Бэктест с издержками")
        await page.click("#btForm button")
        await page.wait_for_timeout(500)
        print("backtest:", (await page.text_content("#btResult"))[:120])
        async with page.expect_download() as d:
            await page.click("#exportBtn")
        dl = await d.value
        print("export:", dl.suggested_filename)
        await page.click("#settingsBtn")
        await page.fill("#setThreshold", "0.6")
        await page.click("#settingsForm button[type=submit]")
        await page.wait_for_timeout(500)
        print("toast:", await page.text_content("#toast"))
        await page.screenshot(path=str(APP.parent / "mocked_desktop.png"), full_page=True)
        m = await b.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True)
        mp = await m.new_page()
        await mp.route("**/*", lambda r: r.abort() if not r.request.url.startswith("file:") else r.continue_())
        await mp.goto(APP.as_uri())
        await mp.wait_for_timeout(1500)
        print("mobile scrollWidth:", await mp.evaluate("document.documentElement.scrollWidth"))
        await mp.screenshot(path=str(APP.parent / "mobile_first_run.png"))
        assert not errors, errors
        print("MOCKED TEST PASSED; page errors:", errors or "none", "| console errors:", console[:3] or "none")
        await b.close()

try:
    asyncio.run(main())
except BaseException as exc:
    _annotate(exc)
    raise
