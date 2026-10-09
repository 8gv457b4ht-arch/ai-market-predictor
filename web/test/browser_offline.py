"""Browser test 1: no exchange reachable (the real situation in the build sandbox).
The app must show DATA SOURCE UNAVAILABLE / OFFLINE and never invent prices or predictions."""
import asyncio
import os
import sys
from pathlib import Path

from playwright.async_api import async_playwright


def _annotate(exc: BaseException) -> None:
    import traceback
    msg = "".join(traceback.format_exception(exc)).replace("%", "%25").replace("\r", "").replace("\n", "%0A")
    print(f"::error title={Path(__file__).name} failed::{msg[-3000:]}", flush=True)

APP = Path(__file__).resolve().parents[1] / "dist" / "AI_Market_Predictor.html"
CHROME = os.getenv("CHROME_PATH") or ("/opt/pw-browsers/chromium-1194/chrome-linux/chrome" if os.path.exists("/opt/pw-browsers/chromium-1194/chrome-linux/chrome") else None)


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path=CHROME)
        ctx = await b.new_context(viewport={"width": 1280, "height": 900})
        page = await ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        # every network request fails, like a machine without access to the exchanges
        await page.route("**/*", lambda r: r.abort() if not r.request.url.startswith("file:") else r.continue_())
        await page.route_web_socket("**", lambda ws: ws.close())
        await page.goto(APP.as_uri())
        await page.wait_for_selector("#welcome", state="visible", timeout=15000)  # first run shows the Start screen
        await page.click("#startBtn")
        await page.wait_for_function("() => document.querySelector('#price').textContent !== '—'", timeout=30000)
        await page.wait_for_timeout(9000)
        price = await page.text_content("#price")
        status = await page.text_content("#statusGrid")
        ex = await page.text_content("#exTable")
        print("price:", price)
        print("status:", " | ".join(status.split("●")))
        assert price == "DATA SOURCE UNAVAILABLE", price
        assert "Market data● DATA SOURCE UNAVAILABLE" in status.replace(" ", " ") and "MODEL NOT READY" in status
        assert "LIVE" not in status.replace("DELIVER", "")
        assert "$" not in price
        assert await page.locator("#ledger td").first.text_content() != "" and "No predictions recorded yet" in await page.text_content("#ledger")
        assert "DATA SOURCE UNAVAILABLE" in await page.text_content("#newsList")
        await page.click("#probeBtn")
        await page.wait_for_function("() => !document.querySelector('#probeBtn').disabled", timeout=60000)
        probe = await page.text_content("#probeTable")
        print("connection test:", probe[:300])
        assert probe.count("network/DNS/firewall") >= 3 and "verified" not in probe
        await page.screenshot(path=str(APP.parent / "offline.png"))
        assert not errors, errors
        print("OFFLINE TEST PASSED; page errors:", errors or "none")
        await b.close()

try:
    asyncio.run(main())
except BaseException as exc:
    _annotate(exc)
    raise
