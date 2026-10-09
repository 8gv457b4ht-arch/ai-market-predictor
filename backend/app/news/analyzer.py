"""News -> structured market events.

Two analyzers produce the same schema:
  * LLMAnalyzer  - Anthropic Messages API or any OpenAI-compatible chat API,
                   enabled when an API key is configured.
  * RuleAnalyzer - transparent keyword/lexicon model, always available, used
                   when no key is set or the LLM call fails. Its confidence is
                   deliberately capped low.
Events are context only: they never create a trading signal by themselves.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request

log = logging.getLogger("news")

CATEGORIES = {
    "crypto": ["bitcoin", "btc", "ethereum", "ether", "crypto", "cryptocurrency", "stablecoin", "solana", "altcoin",
               "blockchain", "defi", "token", "xrp", "memecoin", "halving", "on-chain"],
    "regulation": ["sec", "regulation", "regulator", "regulatory", "lawsuit", "ban", "approval", "approves", "cftc",
                   "mica", "compliance", "enforcement", "court", "legislation", "bill", "license"],
    "macroeconomics": ["gdp", "recession", "unemployment", "payrolls", "nonfarm", "jobs report", "retail sales", "pmi",
                       "economy", "economic growth", "consumer confidence", "jobless claims"],
    "central_banks": ["fed", "federal reserve", "fomc", "ecb", "boj", "bank of japan", "bank of england", "powell",
                      "lagarde", "central bank", "pboc", "monetary policy"],
    "inflation": ["inflation", "cpi", "ppi", "pce", "consumer prices", "price index"],
    "interest_rates": ["rate hike", "rate cut", "interest rate", "interest rates", "basis points", "rate decision",
                       "hikes rates", "cuts rates", "holds rates"],
    "geopolitics": ["war", "sanctions", "missile", "conflict", "tariff", "tariffs", "invasion", "ceasefire",
                    "military", "geopolitical", "election", "strike on"],
    "etf": ["etf", "etfs", "spot etf", "inflows", "outflows", "blackrock", "grayscale", "fidelity"],
    "exchange_incidents": ["outage", "halts", "halted", "suspends withdrawals", "suspended withdrawals", "downtime",
                           "delist", "delisting", "insolvency", "bankruptcy"],
    "hacks": ["hack", "hacked", "exploit", "exploited", "stolen", "breach", "drained", "attacker", "phishing"],
    "liquidations": ["liquidation", "liquidations", "liquidated", "margin call", "short squeeze", "long squeeze"],
    "major_companies": ["microstrategy", "strategy inc", "tesla", "nvidia", "apple", "coinbase", "earnings",
                        "quarterly results", "revenue", "guidance"],
    "commodities": ["oil", "crude", "brent", "gold", "silver", "opec", "natural gas", "copper", "commodities"],
    "equities": ["stocks", "s&p 500", "nasdaq", "dow jones", "equities", "wall street", "stock market"],
    "bonds": ["bond", "bonds", "treasury", "treasuries", "yields", "gilts", "bunds"],
    "currencies": ["dollar", "dxy", "euro", "yen", "forex", "currency", "currencies", "yuan"],
}
POSITIVE = ["surge", "surges", "rally", "rallies", "soar", "soars", "jumps", "approve", "approves", "approved",
            "approval", "inflows", "beats", "record high", "all-time high", "rate cut", "cuts rates", "cools",
            "eases", "gains", "bullish", "adopts", "adoption", "partnership", "upgrade", "rebound", "recovers",
            "ceasefire", "stimulus", "dovish", "buys"]
NEGATIVE = ["plunge", "plunges", "crash", "crashes", "tumbles", "slumps", "hack", "hacked", "exploit", "ban",
            "bans", "lawsuit", "sues", "outflows", "liquidation", "liquidations", "sanctions", "war", "rate hike",
            "hikes rates", "hotter", "misses", "bearish", "downgrade", "halt", "halts", "sell-off", "selloff",
            "fraud", "charges", "charged", "hawkish", "drained", "stolen", "recession", "default", "invasion",
            "sells", "dumps", "insolvency", "bankruptcy"]
ASSETS = {
    "BTC": ["bitcoin", "btc"], "ETH": ["ethereum", "ether", "eth"], "SOL": ["solana", "sol"],
    "XRP": ["xrp", "ripple"], "BNB": ["bnb", "binance coin"], "DOGE": ["dogecoin", "doge"],
}
MACRO = {"macroeconomics", "central_banks", "inflation", "interest_rates", "geopolitics", "bonds", "currencies"}
HORIZON_MIN = {"hacks": 60, "liquidations": 60, "exchange_incidents": 120, "central_banks": 240, "inflation": 240,
               "interest_rates": 240, "macroeconomics": 360, "geopolitics": 720, "regulation": 1440, "etf": 1440,
               "major_companies": 720, "crypto": 360, "commodities": 720, "equities": 360, "bonds": 720,
               "currencies": 720}

_word_cache: dict[str, re.Pattern] = {}


def _rx(word: str) -> re.Pattern:
    p = _word_cache.get(word)
    if p is None:
        p = _word_cache[word] = re.compile(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])")
    return p


def _hits(text: str, words: list[str]) -> list[str]:
    return [w for w in words if _rx(w).search(text)]


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{3,}", text.lower())}


def novelty(title: str, recent_titles: list[str]) -> float:
    t = tokens(title)
    if not t or not recent_titles:
        return 1.0
    best = max((len(t & tokens(r)) / len(t | tokens(r)) for r in recent_titles if r), default=0.0)
    return round(1.0 - best, 3)


def _clamp(v, lo, hi, default):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default


def normalize_event(raw: dict, item: dict, analyzer: str, nov: float) -> dict:
    cat = str(raw.get("category") or "other").lower().replace(" ", "_")
    affected = raw.get("affected_assets") or []
    if isinstance(affected, str):
        affected = [affected]
    affected = sorted({str(a).upper() for a in affected})[:12]
    return {
        "event": str(raw.get("event") or item["title"])[:300],
        "category": cat,
        "event_type": str(raw.get("event_type") or cat)[:60],
        "asset": (str(raw["asset"]).upper() if raw.get("asset") else (affected[0] if affected else None)),
        "direction": _clamp(raw.get("direction"), -1, 1, 0.0),
        "relevance": _clamp(raw.get("relevance"), 0, 1, 0.0),
        "novelty": _clamp(raw.get("novelty", nov), 0, 1, nov),
        "confidence": _clamp(raw.get("confidence"), 0, 1, 0.0),
        "expected_horizon_min": int(_clamp(raw.get("expected_horizon_min"), 5, 60 * 24 * 14, 360)),
        "affected_assets": affected,
        "summary": str(raw.get("summary") or item.get("summary") or "")[:600],
        "analyzer": analyzer,
    }


class RuleAnalyzer:
    name = "rules-v1"

    def analyze(self, item: dict, recent_titles: list[str]) -> dict:
        text = f"{item['title']} {item.get('summary', '')}".lower()
        cat_hits = {c: _hits(text, ws) for c, ws in CATEGORIES.items()}
        cat_hits = {c: h for c, h in cat_hits.items() if h}
        category = max(cat_hits, key=lambda c: len(cat_hits[c])) if cat_hits else "other"
        assets = sorted(a for a, ws in ASSETS.items() if _hits(text, ws))
        pos, neg = _hits(text, POSITIVE), _hits(text, NEGATIVE)
        direction = (len(pos) - len(neg)) / (len(pos) + len(neg) + 1)
        if assets:
            relevance = 0.9
        elif category in ("crypto", "etf", "hacks", "liquidations", "exchange_incidents"):
            relevance = 0.65
        elif category in MACRO or category == "regulation":
            relevance = 0.45
        elif category in ("commodities", "equities", "major_companies"):
            relevance = 0.3
        else:
            relevance = 0.1
        affected = assets or (["CRYPTO_MARKET"] if category != "other" else [])
        if category in MACRO and "CRYPTO_MARKET" not in affected:
            affected.append("CRYPTO_MARKET")
        n_signal = len(pos) + len(neg) + sum(len(h) for h in cat_hits.values())
        confidence = min(0.6, 0.25 + 0.07 * n_signal) if (pos or neg) else min(0.35, 0.15 + 0.05 * n_signal)
        raw = {"category": category, "direction": direction, "relevance": relevance,
               "confidence": confidence, "expected_horizon_min": HORIZON_MIN.get(category, 360),
               "affected_assets": affected, "asset": assets[0] if assets else None,
               "summary": item.get("summary") or item["title"]}
        return normalize_event(raw, item, self.name, novelty(item["title"], recent_titles))


SYSTEM_PROMPT = (
    "You convert financial news into structured, timestamped market events for a READ-ONLY research system. "
    "Never give trading instructions. Do not invent facts, numbers or prices that are not in the text. "
    "Return ONLY a JSON object {\"events\": [...]} with one event per input item, in input order. Fields: "
    "index (int, input index), event (short factual headline), category (one of: crypto, regulation, "
    "macroeconomics, central_banks, inflation, interest_rates, geopolitics, etf, exchange_incidents, hacks, "
    "liquidations, major_companies, commodities, equities, bonds, currencies, other), asset (main ticker or null), "
    "direction (-1..1, expected price impact on the affected crypto assets; 0 if unclear), relevance (0..1 for "
    "crypto markets), novelty (0..1, is this genuinely new information), confidence (0..1, how certain the "
    "assessment is; lower it for rumours or opinion pieces), expected_horizon_min (int), affected_assets "
    "(tickers, use CRYPTO_MARKET for broad impact), summary (one neutral sentence)."
)


class LLMAnalyzer:
    def __init__(self, settings):
        self.s = settings
        provider = settings.news_llm_provider
        if provider == "auto":
            provider = "anthropic" if settings.anthropic_api_key else ("openai" if settings.openai_api_key and settings.openai_model else "none")
        self.provider = provider
        self.model = settings.anthropic_model if provider == "anthropic" else settings.openai_model
        self.name = f"llm:{provider}:{self.model}"

    @property
    def enabled(self) -> bool:
        if self.provider == "anthropic":
            return bool(self.s.anthropic_api_key)
        if self.provider == "openai":
            return bool(self.s.openai_api_key and self.s.openai_model)
        return False

    def _post(self, url: str, headers: dict, body: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"content-type": "application/json", **headers})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))

    def _complete(self, user: str) -> str:
        if self.provider == "anthropic":
            resp = self._post("https://api.anthropic.com/v1/messages",
                              {"x-api-key": self.s.anthropic_api_key, "anthropic-version": "2023-06-01"},
                              {"model": self.model, "max_tokens": 4000, "system": SYSTEM_PROMPT,
                               "messages": [{"role": "user", "content": user}]})
            return "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")
        resp = self._post(self.s.openai_base_url.rstrip("/") + "/chat/completions",
                          {"authorization": f"Bearer {self.s.openai_api_key}"},
                          {"model": self.model, "response_format": {"type": "json_object"},
                           "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                        {"role": "user", "content": user}]})
        return resp["choices"][0]["message"]["content"]

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text).strip()
        start, end = text.find("{"), text.rfind("}")
        return json.loads(text[start:end + 1])

    def analyze_batch(self, items: list[dict], recent_titles: list[str]) -> list[dict]:
        payload = [{"index": i, "title": it["title"], "summary": it.get("summary", "")[:500],
                    "source": it.get("source"), "published_ms": it.get("published_ms")} for i, it in enumerate(items)]
        data = self._extract_json(self._complete(json.dumps(payload, ensure_ascii=False)))
        events = data.get("events", []) if isinstance(data, dict) else []
        by_index = {int(e.get("index", -1)): e for e in events if isinstance(e, dict)}
        out = []
        for i, it in enumerate(items):
            raw = by_index.get(i)
            if raw is None:
                raise ValueError(f"LLM returned no event for item {i}")
            out.append(normalize_event(raw, it, self.name, novelty(it["title"], recent_titles)))
        return out
