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
        print("status:", status[:400])
        # Russian is the default language on the first visit
        assert await page.evaluate("document.documentElement.lang") == "ru"
        assert await page.text_content("#sysTitle") == "Состояние системы"
        assert price.startswith("DATA SOURCE UNAVAILABLE"), price
        assert "Исторические данные" in status and "OFFLINE" in status and "MODEL NOT READY" in status
        assert "LIVE" not in status and "DELAYED" not in status  # nothing real-time arrives here
        assert "$" not in price
        assert "Прогнозов пока нет" in await page.text_content("#ledger")
        assert "DATA SOURCE UNAVAILABLE" in await page.text_content("#newsList")
        await page.click("#probeBtn")
        await page.wait_for_function("() => !document.querySelector('#probeBtn').disabled", timeout=60000)
        probe = await page.text_content("#probeTable")
        print("connection test:", probe[:300])
        assert probe.count("сеть/DNS/файрвол") >= 3 and "подтверждён" not in probe
        # language switch: no reload, same state, every language complete
        await page.evaluate("window.__marker = 42")
        for lang, title, ledger_word in (("uk", "Стан системи", "Прогнозів поки немає"), ("en", "System status", "No predictions recorded yet"),
                                          ("ru", "Состояние системы", "Прогнозов пока нет")):
            await page.select_option("#lang", lang)
            await page.wait_for_timeout(400)
            assert await page.evaluate("window.__marker") == 42, "page reloaded on language switch"
            assert await page.evaluate("document.documentElement.lang") == lang
            assert await page.text_content("#sysTitle") == title, lang
            assert ledger_word in await page.text_content("#ledger"), lang
            assert "network" in (await page.text_content("#probeTable")) or lang != "en"
        # the choice survives a reload
        await page.select_option("#lang", "uk")
        await page.reload()
        await page.wait_for_timeout(1500)
        assert await page.evaluate("document.documentElement.lang") == "uk" and await page.text_content("#sysTitle") == "Стан системи"
        await page.select_option("#lang", "ru")
        # phone width: no horizontal page scroll in any language
        m = await b.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=3, is_mobile=True, has_touch=True)
        mp = await m.new_page()
        mp.on("pageerror", lambda e: errors.append(str(e)))
        await mp.route("**/*", lambda r: r.abort() if not r.request.url.startswith("file:") else r.continue_())
        await mp.route_web_socket("**", lambda ws: ws.close())
        await mp.goto(APP.as_uri())
        await mp.wait_for_selector("#welcome", state="visible", timeout=15000)
        await mp.click("#startBtn")
        await mp.wait_for_timeout(4000)
        for lang in ("ru", "uk", "en"):
            await mp.select_option("#lang", lang)
            await mp.wait_for_timeout(300)
            sw = await mp.evaluate("document.documentElement.scrollWidth")
            small = await mp.evaluate("""() => [...document.querySelectorAll('button, select')].filter(e => e.offsetParent && e.getBoundingClientRect().height < 40).map(e => e.id || e.textContent).slice(0, 5)""")
            print("mobile", lang, "scrollWidth:", sw, "small tap targets:", small)
            assert sw <= 390, (lang, sw)
            assert not small, (lang, small)
            await mp.screenshot(path=str(APP.parent / f"offline_mobile_{lang}.png"), full_page=True)
        await m.close()
        await page.screenshot(path=str(APP.parent / "offline.png"))
        assert not errors, errors
        print("OFFLINE TEST PASSED; page errors:", errors or "none")
        await b.close()

try:
    asyncio.run(main())
except BaseException as exc:
    _annotate(exc)
    raise
