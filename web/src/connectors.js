/* Public market-data connectors that run directly in the user's browser.
 * WebSocket: Binance, Bybit, OKX (trades + order book). REST: candles + 24h tickers.
 * Read-only, no API keys, no order endpoints.
 */
(function (root) {
  "use strict";
  const TF_MS = root.AMP.TF_MS;
  const sym = { binance: (s) => s.replace("/", ""), bybit: (s) => s.replace("/", ""), okx: (s) => s.replace("/", "-") };
  const canon = (ex, raw) => (ex === "okx" ? raw.replace("-", "/") : raw.replace(/(USDT|USDC|USD)$/, "/$1"));
  const BYBIT_IV = { "1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "4h": "240", "1d": "D" };
  const OKX_BAR = { "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1H", "4h": "4H", "1d": "1Dutc" };
  const REST = {
    binance: ["https://data-api.binance.vision", "https://api.binance.com"],
    bybit: ["https://api.bybit.com"],
    okx: ["https://www.okx.com"],
  };
  const WS = {
    binance: ["wss://stream.binance.com:9443/stream", "wss://data-stream.binance.vision/stream"],
    bybit: ["wss://stream.bybit.com/v5/public/spot"],
    okx: ["wss://ws.okx.com:8443/ws/v5/public", "wss://ws.okx.com/ws/v5/public"],
  };

  async function getJson(url, timeoutMs) {
    const ctl = new AbortController(), t = setTimeout(() => ctl.abort(), timeoutMs || 12000);
    try {
      const r = await fetch(url, { signal: ctl.signal, cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.json();
    } finally { clearTimeout(t); }
  }
  const validCandle = (c) => c.low > 0 && c.high >= Math.max(c.open, c.close) - 1e-12 && c.low <= Math.min(c.open, c.close) + 1e-12 && c.volume >= 0;

  // ------------------------------------------------------------- REST
  const restHost = {};
  async function klinesPage(ex, symbol, tf, endMs) {
    const now = Date.now(), tfms = TF_MS[tf];
    let rows;
    if (ex === "binance") {
      const hosts = restHost.binance ? [restHost.binance] : REST.binance;
      let lastErr;
      for (const h of hosts) {
        try {
          const q = new URLSearchParams({ symbol: sym.binance(symbol), interval: tf, limit: "1000" });
          if (endMs) q.set("endTime", String(endMs));
          const d = await getJson(`${h}/api/v3/klines?${q}`);
          restHost.binance = h;
          rows = d.map((r) => ({ open_ts: +r[0], open: +r[1], high: +r[2], low: +r[3], close: +r[4], volume: +r[5],
            taker_buy_volume: +r[9], trades: +r[8], closed: +r[6] < now }));
          break;
        } catch (e) { lastErr = e; }
      }
      if (!rows) throw lastErr;
    } else if (ex === "bybit") {
      const q = new URLSearchParams({ category: "spot", symbol: sym.bybit(symbol), interval: BYBIT_IV[tf], limit: "1000" });
      if (endMs) q.set("end", String(endMs));
      const d = await getJson(`${REST.bybit[0]}/v5/market/kline?${q}`);
      if (d.retCode !== 0) throw new Error(d.retMsg || "bybit error");
      rows = d.result.list.map((r) => ({ open_ts: +r[0], open: +r[1], high: +r[2], low: +r[3], close: +r[4], volume: +r[5], closed: +r[0] + tfms <= now }));
    } else {
      const q = new URLSearchParams({ instId: sym.okx(symbol), bar: OKX_BAR[tf], limit: "300" });
      if (endMs) q.set("after", String(endMs + 1));
      let d = await getJson(`${REST.okx[0]}/api/v5/market/candles?${q}`);
      if (String(d.code) !== "0") throw new Error(d.msg || "okx error");
      if (!d.data.length && endMs) { q.set("limit", "100"); d = await getJson(`${REST.okx[0]}/api/v5/market/history-candles?${q}`); }
      rows = d.data.map((r) => ({ open_ts: +r[0], open: +r[1], high: +r[2], low: +r[3], close: +r[4], volume: +r[5], closed: r[8] === "1" }));
    }
    return rows.filter(validCandle).sort((a, b) => a.open_ts - b.open_ts);
  }
  async function history(ex, symbol, tf, bars) {
    const got = new Map(); let end = null, stalls = 0;
    while (got.size < bars) {
      const page = await klinesPage(ex, symbol, tf, end);
      if (!page.length) break;
      const before = got.size; page.forEach((c) => got.set(c.open_ts, c));
      if (got.size === before && ++stalls >= 2) break;
      end = page[0].open_ts - 1;
    }
    return [...got.values()].sort((a, b) => a.open_ts - b.open_ts).slice(-bars);
  }
  async function ticker24h(ex, symbol) {
    if (ex === "binance") {
      const h = restHost.binance || REST.binance[0];
      const d = await getJson(`${h}/api/v3/ticker/24hr?symbol=${sym.binance(symbol)}`);
      return { last: +d.lastPrice, changePct: +d.priceChangePercent, volume: +d.volume, quoteVolume: +d.quoteVolume };
    }
    if (ex === "bybit") {
      const d = await getJson(`${REST.bybit[0]}/v5/market/tickers?category=spot&symbol=${sym.bybit(symbol)}`);
      const t = d.result.list[0]; return { last: +t.lastPrice, changePct: +t.price24hPcnt * 100, volume: +t.volume24h, quoteVolume: +t.turnover24h };
    }
    const d = await getJson(`${REST.okx[0]}/api/v5/market/ticker?instId=${sym.okx(symbol)}`);
    const t = d.data[0]; return { last: +t.last, changePct: (+t.last / +t.open24h - 1) * 100, volume: +t.vol24h, quoteVolume: +t.volCcy24h };
  }

  // ------------------------------------------------------- order book
  class Book {
    constructor() { this.bids = new Map(); this.asks = new Map(); this.ready = false; this.lastId = null; this.ts = 0; }
    snapshot(b, a, id, ts) { this.bids = new Map(b.map((x) => [+x[0], +x[1]]).filter((x) => x[1] > 0)); this.asks = new Map(a.map((x) => [+x[0], +x[1]]).filter((x) => x[1] > 0)); this.ready = true; this.lastId = id; this.ts = ts; }
    delta(b, a, id, ts) {
      if (!this.ready) return "gap";
      if (id != null && this.lastId != null && id <= this.lastId) return "stale";
      for (const [side, lv] of [[this.bids, b], [this.asks, a]]) for (const [p, q] of lv) { if (+q <= 0) side.delete(+p); else side.set(+p, +q); }
      this.lastId = id ?? this.lastId; this.ts = ts;
      const m = this.metrics(1); if (m.bestBid != null && m.bestBid >= m.bestAsk) { this.ready = false; return "gap"; }
      return "ok";
    }
    top(n) { return { bids: [...this.bids].sort((x, y) => y[0] - x[0]).slice(0, n), asks: [...this.asks].sort((x, y) => x[0] - y[0]).slice(0, n) }; }
    metrics(levels) {
      const { bids, asks } = this.top(levels || 10);
      if (!bids.length || !asks.length) return { bestBid: null, bestAsk: null, mid: null, spreadBps: null, imbalance: null, bidDepth: 0, askDepth: 0 };
      const bb = bids[0][0], ba = asks[0][0], mid = (bb + ba) / 2;
      const bd = bids.reduce((s, x) => s + x[1], 0), ad = asks.reduce((s, x) => s + x[1], 0);
      return { bestBid: bb, bestAsk: ba, mid, spreadBps: ((ba - bb) / mid) * 1e4, imbalance: (bd - ad) / (bd + ad), bidDepth: bd, askDepth: ad };
    }
  }

  // ------------------------------------------------------ parsers
  function parse(ex, raw) {
    if (raw === "pong") return [];
    const m = JSON.parse(raw), out = [];
    if (ex === "binance") {
      const d = m.data || m, st = m.stream || "";
      if (d.e === "trade") out.push({ k: "trade", s: canon(ex, d.s), id: +d.t, ts: +d.T, p: +d.p, q: +d.q, side: d.m ? "sell" : "buy" });
      else if (st.includes("@depth") && d.bids) out.push({ k: "book", s: canon(ex, st.split("@")[0].toUpperCase()), type: "snapshot", b: d.bids, a: d.asks, id: +d.lastUpdateId, ts: Date.now(), exTs: null });
    } else if (ex === "bybit") {
      const t = m.topic || "";
      if (m.success === false) out.push({ k: "error", msg: m.ret_msg || "subscribe failed" });
      if (t.startsWith("publicTrade.")) for (const d of m.data) out.push({ k: "trade", s: canon(ex, d.s), id: String(d.i), ts: +d.T, p: +d.p, q: +d.v, side: d.S === "Buy" ? "buy" : "sell" });
      else if (t.startsWith("orderbook.")) { const d = m.data; out.push({ k: "book", s: canon(ex, d.s), type: m.type === "snapshot" || +d.u === 1 ? "snapshot" : "delta", b: d.b || [], a: d.a || [], id: +d.u, ts: +(m.cts || m.ts), exTs: +(m.cts || m.ts) }); }
    } else {
      if (m.event === "error") out.push({ k: "error", msg: m.msg || "okx error" });
      const ch = m.arg && m.arg.channel;
      for (const d of m.data || []) {
        if (ch === "trades") out.push({ k: "trade", s: canon(ex, d.instId), id: +d.tradeId, ts: +d.ts, p: +d.px, q: +d.sz, side: d.side });
        else if (ch === "books5") out.push({ k: "book", s: canon(ex, m.arg.instId), type: "snapshot", b: d.bids.map((x) => x.slice(0, 2)), a: d.asks.map((x) => x.slice(0, 2)), id: d.seqId != null ? +d.seqId : null, ts: +d.ts, exTs: +d.ts });
      }
    }
    return out;
  }
  function subscribe(ex, symbols) {
    if (ex === "binance") return { suffix: "?streams=" + symbols.flatMap((s) => { const x = sym.binance(s).toLowerCase(); return [`${x}@trade`, `${x}@depth20@100ms`]; }).join("/"), msgs: [], ping: null };
    if (ex === "bybit") return { suffix: "", msgs: [JSON.stringify({ op: "subscribe", args: symbols.flatMap((s) => [`publicTrade.${sym.bybit(s)}`, `orderbook.50.${sym.bybit(s)}`]) })], ping: JSON.stringify({ op: "ping" }) };
    return { suffix: "", msgs: [JSON.stringify({ op: "subscribe", args: symbols.flatMap((s) => [{ channel: "trades", instId: sym.okx(s) }, { channel: "books5", instId: sym.okx(s) }]) })], ping: "ping" };
  }

  // ------------------------------------------------- live stream
  class ExchangeStream {
    constructor(ex, symbols, onUpdate, opts) {
      this.ex = ex; this.symbols = symbols; this.onUpdate = onUpdate || (() => {});
      this.staleMs = (opts && opts.staleMs) || 30000;
      this.state = "connecting"; this.error = null; this.lastMsg = null; this.connectedAt = null;
      this.reconnects = 0; this.failures = 0; this.gaps = 0; this.duplicates = 0; this.invalid = 0; this.urlIdx = 0;
      this.books = {}; this.flow = {}; this.lastPrice = {}; this.lastTradeTs = {}; this.lastId = {}; this.seen = {};
      for (const s of symbols) { this.books[s] = new Book(); this.flow[s] = { cvd: 0, bars: new Map() }; this.seen[s] = new Set(); }
      this.stopped = false;
    }
    start() { this.stopped = false; this._connect(); this._watch = setInterval(() => this._checkStale(), 2000); }
    stop() { this.stopped = true; clearInterval(this._watch); clearInterval(this._ping); try { this.ws && this.ws.close(); } catch (e) { /* closing */ } }
    _connect() {
      if (this.stopped) return;
      const sub = subscribe(this.ex, this.symbols), url = WS[this.ex][this.urlIdx % WS[this.ex].length] + sub.suffix;
      let ws;
      try { ws = new WebSocket(url); } catch (e) { this._fail(e.message); return; }
      this.ws = ws; this.state = this.state === "unavailable" ? "unavailable" : "connecting"; this._gotData = false;
      ws.onopen = () => {
        this.connectedAt = Date.now(); this.lastMsg = Date.now();
        sub.msgs.forEach((m) => ws.send(m));
        clearInterval(this._ping);
        if (sub.ping) this._ping = setInterval(() => { try { ws.send(sub.ping); } catch (e) { /* socket closing */ } }, 20000);
      };
      ws.onmessage = (e) => this._message(e.data);
      ws.onerror = () => { this.error = `cannot connect to ${url.split("?")[0]}`; };
      ws.onclose = (e) => { clearInterval(this._ping); if (!this.stopped) this._fail(this.error || `connection closed (code ${e.code})`); };
    }
    _fail(msg) {
      this.error = msg; this.failures++; this.reconnects++;
      if (!this._gotData) this.urlIdx++; // try the alternative endpoint next time
      this.state = this.failures >= 3 && !this._gotData ? "unavailable" : "reconnecting";
      const delay = Math.min(60000, 1000 * 2 ** Math.min(this.failures, 6)) * (0.5 + Math.random() / 2);
      clearTimeout(this._retry); this._retry = setTimeout(() => this._connect(), delay);
      this.onUpdate(this);
    }
    _checkStale() {
      if (this.ws && this.ws.readyState === 1 && this.lastMsg && Date.now() - this.lastMsg > this.staleMs) {
        this.state = "stale"; this.error = `no data for ${Math.round(this.staleMs / 1000)} s`;
        try { this.ws.close(); } catch (e) { /* reconnect follows */ }
      }
    }
    _message(raw) {
      let evs;
      try { evs = parse(this.ex, typeof raw === "string" ? raw : ""); } catch (e) { this.invalid++; return; }
      this.lastMsg = Date.now();
      for (const ev of evs) {
        if (ev.k === "error") { this.error = ev.msg; continue; }
        if (!this.books[ev.s]) continue;
        const ok = ev.k === "trade" ? this._trade(ev) : this._book(ev);
        // LIVE only after an event whose exchange timestamp is plausible (not stale, not in the future)
        if (ok === "verified") {
          if (!this._gotData) { this._gotData = true; this.failures = 0; }
          this.verifiedAt = Date.now(); this.state = "live"; this.error = null;
        }
      }
      this.onUpdate(this);
    }
    _trade(t) {
      const now = Date.now();
      if (!(t.p > 0) || !(t.q > 0) || !(t.ts <= now + 5000) || !(t.ts >= now - 300000)) {
        this.invalid++; this.error = `trade timestamp ${new Date(t.ts).toISOString()} rejected (clock skew or stale data)`; return "invalid";
      }
      this.skewMs = now - t.ts;
      const seen = this.seen[t.s];
      if (seen.has(t.id)) { this.duplicates++; return "duplicate"; }
      seen.add(t.id); if (seen.size > 5000) seen.delete(seen.values().next().value);
      if (this.ex === "binance" && this.lastId[t.s] != null && t.id > this.lastId[t.s] + 1) this.gaps++;
      if (typeof t.id === "number") this.lastId[t.s] = Math.max(this.lastId[t.s] ?? t.id, t.id);
      const f = this.flow[t.s], m = t.ts - (t.ts % 60000);
      let b = f.bars.get(m); if (!b) { b = { open_ts: m, buy: 0, sell: 0, delta: 0, cvd: 0, trades: 0, last: t.p }; f.bars.set(m, b); }
      if (t.side === "buy") b.buy += t.q; else b.sell += t.q;
      f.cvd += t.side === "buy" ? t.q : -t.q; b.delta = b.buy - b.sell; b.cvd = f.cvd; b.trades++; b.last = t.p;
      if (f.bars.size > 720) f.bars.delete(f.bars.keys().next().value);
      this.lastPrice[t.s] = t.p; this.lastTradeTs[t.s] = Date.now();
      return "verified";
    }
    _book(ev) {
      const bk = this.books[ev.s], now = Date.now();
      if (ev.exTs != null && !(ev.exTs <= now + 5000 && ev.exTs >= now - 300000)) { this.invalid++; return "invalid"; }
      if (ev.type === "snapshot") bk.snapshot(ev.b, ev.a, ev.id, ev.ts);
      else {
        const r = bk.delta(ev.b, ev.a, ev.id, ev.ts);
        if (r === "gap") { this.gaps++; try { this.ws.close(); } catch (e) { /* resubscribe */ } }
        else if (r === "stale") this.duplicates++;
      }
      bk.recv = Date.now();
      return ev.exTs != null && bk.ready ? "verified" : "unverified"; // Binance partial depth has no exchange timestamp
    }
    status() {
      const age = this.verifiedAt ? (Date.now() - this.verifiedAt) / 1000 : null;
      if (this.state === "live" && (age === null || age > this.staleMs / 1000)) this.state = "stale";
      return { exchange: this.ex, state: this.state, skewMs: this.skewMs ?? null, lastMsgAgeSec: age, reconnects: this.reconnects, gaps: this.gaps,
        duplicates: this.duplicates, invalid: this.invalid, error: this.state === "live" ? null : this.error };
    }
  }

  // --------------------------------------------------- connection test
  const REST_PROBE = {
    binance: (h) => `${h}/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=3`,
    bybit: (h) => `${h}/v5/market/kline?category=spot&symbol=BTCUSDT&interval=1&limit=3`,
    okx: (h) => `${h}/api/v5/market/candles?instId=BTC-USDT&bar=1m&limit=3`,
  };
  const lastTs = (ex, d) => (ex === "binance" ? +d[d.length - 1][0] : ex === "bybit" ? +d.result.list[0][0] : +d.data[0][0]);
  async function probeRestHost(ex, host) {
    const url = REST_PROBE[ex](host), t0 = performance.now();
    try {
      const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), 10000);
      let r;
      try { r = await fetch(url, { signal: ctl.signal, cache: "no-store" }); } finally { clearTimeout(timer); }
      const ms = Math.round(performance.now() - t0);
      if (r.status === 451) return { ok: false, host, kind: "region_blocked", error: "HTTP 451: blocked for this country/region", ms };
      if (r.status === 403) return { ok: false, host, kind: "region_blocked", error: "HTTP 403: access denied (often a region block)", ms };
      if (r.status === 429 || r.status === 418) return { ok: false, host, kind: "rate_limited", error: `HTTP ${r.status}: rate limit`, ms };
      if (!r.ok) return { ok: false, host, kind: "http_error", error: `HTTP ${r.status}`, ms };
      const d = await r.json(), ts = lastTs(ex, d), age = (Date.now() - ts) / 1000;
      if (!(age > -60 && age < 600)) return { ok: false, host, kind: "stale_data", error: `newest candle is ${Math.round(age)} s old`, ms };
      return { ok: true, host, ms, candleAgeSec: Math.round(age) };
    } catch (e) {
      const ms = Math.round(performance.now() - t0);
      if (e.name === "AbortError") return { ok: false, host, kind: "timeout", error: "no answer within 10 s", ms };
      // fetch() hides the reason. A no-cors request succeeding means the server is reachable but
      // does not allow browser access (CORS); failing too means a network/DNS/firewall problem.
      try { await fetch(url, { mode: "no-cors", cache: "no-store" }); return { ok: false, host, kind: "cors", error: "server reachable but blocks browser requests (CORS)", ms }; }
      catch (e2) { return { ok: false, host, kind: "network", error: "network, DNS or firewall blocks the connection", ms }; }
    }
  }
  async function probeRest(ex) {
    const tries = [];
    for (const h of REST[ex]) { const r = await probeRestHost(ex, h); tries.push(r); if (r.ok) return { ...r, tries }; }
    return { ...tries[0], tries };
  }
  function probeWs(ex, timeoutMs) {
    return new Promise((resolve) => {
      const st = new ExchangeStream(ex, ["BTC/USDT"], null, { staleMs: 60000 }); const t0 = Date.now(); let done = false;
      const finish = (res) => { if (done) return; done = true; clearInterval(iv); st.stop(); resolve({ ...res, url: WS[ex][st.urlIdx % WS[ex].length] }); };
      const iv = setInterval(() => {
        if (st.state === "live") finish({ ok: true, ms: Date.now() - t0, skewMs: st.skewMs ?? null });
        else if (Date.now() - t0 > timeoutMs) finish({ ok: false, kind: st.failures ? "network" : "no_data", error: st.error || "connected but no timestamp-verified data", ms: Date.now() - t0 });
      }, 200);
      st.start();
    });
  }

  root.AMPConnectors = { probeRest, probeWs,  history, klinesPage, ticker24h, ExchangeStream, Book, parse, subscribe, REST, WS, getJson };
})(typeof self !== "undefined" ? self : globalThis);
