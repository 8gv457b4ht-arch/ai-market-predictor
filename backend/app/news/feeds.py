"""RSS / Atom ingestion with the standard library only."""
from __future__ import annotations

import hashlib
import html
import logging
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

log = logging.getLogger("news")
MAX_BYTES = 3 * 1024 * 1024
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_text(s: str | None, limit: int = 600) -> str:
    if not s:
        return ""
    s = html.unescape(_TAG.sub(" ", s))
    return _WS.sub(" ", s).strip()[:limit]


def parse_date(s: str | None) -> int | None:
    if not s:
        return None
    s = s.strip()
    try:
        dt = parsedate_to_datetime(s)
    except (TypeError, ValueError, IndexError):
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(el, *names) -> str | None:
    for c in el:
        if _local(c.tag) in names and (c.text or "").strip():
            return c.text
    return None


def parse_feed(xml_bytes: bytes, source: str) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    items = []
    for el in root.iter():
        name = _local(el.tag)
        if name not in ("item", "entry"):
            continue
        title = clean_text(_child_text(el, "title"), 300)
        if not title:
            continue
        link = None
        for c in el:
            if _local(c.tag) == "link":
                link = c.get("href") or (c.text or "").strip() or link
                if c.get("rel", "alternate") == "alternate" and c.get("href"):
                    break
        summary = clean_text(_child_text(el, "description", "summary", "content", "encoded"))
        published = parse_date(_child_text(el, "pubdate", "published", "updated", "date"))
        items.append({"title": title, "url": link, "summary": summary, "published_ms": published,
                      "source": source})
    return items


def item_id(item: dict) -> str:
    key = (item.get("url") or "") + "|" + item["title"].lower()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def fetch_feed(url: str, timeout: float = 15) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "AI-Market-Predictor/2.0 (news research)",
                                               "Accept": "application/rss+xml, application/atom+xml, text/xml"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError(f"feed too large: {url}")
    source = urlparse(url).netloc.replace("www.", "")
    items = parse_feed(data, source)
    now = int(time.time() * 1000)
    for it in items:
        # Unknown publication time -> use fetch time but mark it (freshness weighting stays conservative).
        if it["published_ms"] is None:
            it["published_ms"] = now
            it["published_estimated"] = True
        it["published_ms"] = min(it["published_ms"], now)
    return items
