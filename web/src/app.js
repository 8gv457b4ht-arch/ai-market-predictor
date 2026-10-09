/* AI Market Predictor - browser application.
 * Server mode (published site): predictions, ledger, models, learning and news come from the 24/7 backend
 *   (scheduled GitHub Actions job, JSON in the `data` branch). The browser adds live prices/order books.
 * Local mode (file opened directly, or backend unreachable): everything runs in this browser.
 */
(function () {
  "use strict";
  const A = window.AMP, C = window.AMPConnectors;
  const SYMBOLS = ["BTC/USDT", "ETH/USDT"], EXCHANGES = ["binance", "bybit", "okx"], CONTEXT_TFS = ["15m", "1h", "4h", "1d"];
  const HISTORY = { "15m": 3000, "1h": 2000, "4h": 1000, "1d": 600 };
  const DEFAULTS = { started: false, primary: "auto", predictTfs: ["15m", "1h"], horizon: 4, threshold: 0.55, minEdge: 0.1, atrMult: 0.3,
    feeBps: 10, slippageBps: 2, spreadBps: 1, latencyMs: 500, minNewLabels: 300, minHoldout: 150, holdoutFraction: 0.5,
    minLoglossGain: 0.002, bootstrapConfidence: 0.9, maxSpreadBps: 15, maxDivergenceBps: 40, minQuality: 0.7, newsHalfLife: 180,
    anthropicKey: "", anthropicModel: "claude-haiku-5-5", newsKey: "", notifyEnabled: false, notifySignals: true, notifyAll: false, notifyOutages: true };
  const $ = (id) => document.getElementById(id);
  const S = { mode: "detecting", server: null, serverErr: null, serverLedger: [], serverCandles: {}, lastServerFetch: 0, cfg: null,
    settings: { ...DEFAULTS }, primary: null, candles: {}, tickers: {}, streams: {}, registry: { versions: [], production: {} },
    models: {}, ledger: new Map(), news: [], newsStatus: { state: "NOT STARTED" }, jobs: [], training: null, learning: { lastRun: null, status: {} },
    lastPoll: null, pollError: null, historyErrors: {}, tab: "price", symbol: "BTC/USDT", tf: "15m", bookEx: "binance", dbOk: false,
    prevGaps: {}, probe: null, ledgerFilter: "all", ledgerLimit: 50, seenPredictionIds: null, prevSourceOk: {} };

  // ------------------------------------------------------------- helpers
  const isNum = (v) => typeof v === "number" && Number.isFinite(v);
  const pct = (x, d = 1) => (isNum(x) ? (x * 100).toFixed(d) + "%" : "—");
  const num = (x, d = 2) => (isNum(x) ? Number(x).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }) : "—");
  const priceFmt = (x) => (isNum(x) ? Number(x).toLocaleString(undefined, { maximumFractionDigits: x >= 100 ? 2 : 4, minimumFractionDigits: x >= 100 ? 2 : 0 }) : "—");
  const when = (ms) => (ms ? new Date(ms).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—");
  function ago(ms) { if (!ms) return "never"; const s = (Date.now() - ms) / 1000; if (s < 90) return Math.round(s) + " s ago"; if (s < 5400) return Math.round(s / 60) + " min ago"; if (s < 172800) return Math.round(s / 3600) + " h ago"; return Math.round(s / 86400) + " d ago"; }
  function el(tag, attrs, ...kids) { const e = document.createElement(tag); for (const [k, v] of Object.entries(attrs || {})) { if (k === "class") e.className = v; else if (k === "text") e.textContent = v; else e.setAttribute(k, v); } kids.forEach((k) => k != null && e.append(k)); return e; }
  function toast(msg) { const t = $("toast"); t.textContent = msg; t.hidden = false; clearTimeout(toast.t); toast.t = setTimeout(() => (t.hidden = true), 4500); }
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const keyOf = (sym, tf) => `${sym}|${tf}|h${S.settings.horizon}`;
  const higherTfs = (tf) => CONTEXT_TFS.filter((t) => A.TF_MS[t] >= A.TF_MS[tf]);
  const server = () => S.mode === "server" && S.server;
  const REASON_TEXT = { confidence_below_threshold: "confidence below threshold", edge_below_min_edge: "UP and DOWN too close",
    flat_more_likely: "FLAT more likely than the direction", low_confidence: "confidence below threshold", data_quality: "data quality check failed",
    abnormal_market: "abnormal market", low_liquidity: "low liquidity", news_conflict: "fresh news contradicts the model",
    late_decision: "decided too late after the candle close (delayed scheduled run)" };
  const KIND_TEXT = { region_blocked: "blocked in this region", cors: "browser access blocked (CORS)", network: "network/DNS/firewall",
    timeout: "timeout", rate_limited: "rate limited", http_error: "HTTP error", stale_data: "data too old", no_data: "no verified data",
    stale: "stream went silent", gap: "sequence gap", bad_response: "bad response", exchange_error: "exchange error" };

  // ------------------------------------------------------------- storage
  const Store = {
    db: null, mem: new Map(),
    open() {
      return new Promise((res) => {
        try {
          const r = indexedDB.open("ai-market-predictor", 1);
          r.onupgradeneeded = () => { r.result.createObjectStore("kv"); r.result.createObjectStore("ledger", { keyPath: "id" }); };
          r.onsuccess = () => { this.db = r.result; S.dbOk = true; res(true); };
          r.onerror = () => res(false);
        } catch (e) { res(false); }
      });
    },
    tx(store, mode, fn) {
      if (!this.db) return Promise.resolve(null);
      return new Promise((res, rej) => { const t = this.db.transaction(store, mode), st = t.objectStore(store); const r = fn(st); t.oncomplete = () => res(r && "result" in r ? r.result : r); t.onerror = () => rej(t.error); });
    },
    async get(k) { if (!this.db) return this.mem.get(k); return this.tx("kv", "readonly", (st) => st.get(k)); },
    async set(k, v) { if (!this.db) { this.mem.set(k, v); return; } return this.tx("kv", "readwrite", (st) => st.put(v, k)); },
    async putLedger(r) { if (!this.db) return; return this.tx("ledger", "readwrite", (st) => st.put(r)); },
    async allLedger() { if (!this.db) return []; return this.tx("ledger", "readonly", (st) => st.getAll()); },
    async clear() { if (!this.db) { this.mem.clear(); return; } await this.tx("kv", "readwrite", (st) => st.clear()); await this.tx("ledger", "readwrite", (st) => st.clear()); },
  };

  // ------------------------------------------------------------- server (24/7 backend) data
  async function detectServer() {
    if (!/^https?:$/.test(location.protocol)) return false;
    try {
      const r = await fetch("config.json", { cache: "no-store" });
      if (!r.ok) return false;
      S.cfg = await r.json();
      return !!S.cfg.state_url;
    } catch (e) { return false; }
  }
  async function fetchServer(force) {
    if (!S.cfg || (!force && Date.now() - S.lastServerFetch < 55000)) return;
    S.lastServerFetch = Date.now();
    const bust = `?t=${Math.floor(Date.now() / 60000)}`;
    try {
      const r = await fetch(S.cfg.state_url + "state.json" + bust, { cache: "no-store" });
      if (r.status === 404) { S.serverErr = "The backend has not published data yet (first run pending)."; S.server = null; render(); return; }
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const first = !S.server;
      S.server = await r.json(); S.serverErr = null;
      if (first && S.server.primary_exchange) { S.bookEx = S.server.primary_exchange; $("bookEx").value = S.bookEx; }
      const l = await fetch(S.cfg.state_url + "ledger.json" + bust, { cache: "no-store" });
      if (l.ok) { const prev = S.serverLedger; S.serverLedger = (await l.json()).predictions || []; notifyNew(prev); }
      for (const sym of S.server.settings.symbols) for (const tf of S.server.settings.predict_timeframes) {
        const c = await fetch(`${S.cfg.state_url}candles_${sym.replace("/", "-")}_${tf}.json${bust}`, { cache: "no-store" });
        if (c.ok) S.serverCandles[`${sym}|${tf}`] = await c.json();
      }
      notifyOutages();
    } catch (e) { S.serverErr = `Backend data not reachable: ${e.message}`; }
    render();
  }

  // ------------------------------------------------------------- notifications (browser)
  function canNotify() { return "Notification" in window && Notification.permission === "granted" && S.settings.notifyEnabled; }
  function notify(title, body) { try { if (canNotify()) new Notification(title, { body, tag: title }); } catch (e) { /* ignore */ } }
  function notifyNew(prev) {
    const prevIds = new Set(prev.map((p) => p.prediction_id));
    if (!prev.length) return; // first load: do not replay history
    for (const p of S.serverLedger) {
      if (prevIds.has(p.prediction_id)) continue;
      if (p.prediction === "NO TRADE" ? !S.settings.notifyAll : !(S.settings.notifySignals || S.settings.notifyAll)) continue;
      notify(`${p.symbol} ${p.timeframe}: ${p.prediction}`, `Price ${priceFmt(p.price)} · UP ${pct(p.p_up, 0)} DOWN ${pct(p.p_down, 0)} FLAT ${pct(p.p_flat, 0)} · confidence ${pct(p.confidence, 0)} · research only, not advice`);
    }
  }
  function notifyOutages() {
    if (!S.settings.notifyOutages || !server()) return;
    const pr = (S.server.probes && S.server.probes.probes) || {};
    for (const [ex, p] of Object.entries(pr)) {
      const ok = !!(p.rest && p.rest.ok && p.ws && p.ws.verified_live);
      if (ex in S.prevSourceOk && S.prevSourceOk[ex] !== ok) notify(`${ex}: ${ok ? "data source back" : "DATA SOURCE UNAVAILABLE"}`, ok ? "Verified again by the backend." : (p.rest && p.rest.error) || (p.ws && p.ws.error) || "");
      S.prevSourceOk[ex] = ok;
    }
  }

  // ------------------------------------------------------------- worker (local mode training)
  let worker = null, jobSeq = 0; const pending = new Map();
  function makeWorker() {
    try {
      const src = $("amp-core").textContent + "\nself.onmessage=function(e){var d=e.data;try{var r=d.kind==='baseline'?AMP.trainBaseline(d.job):AMP.challengerCycle(d.job);self.postMessage({id:d.id,ok:true,res:r});}catch(err){self.postMessage({id:d.id,ok:false,error:String(err&&err.message||err)});}};";
      const w = new Worker(URL.createObjectURL(new Blob([src], { type: "text/javascript" })));
      w.onmessage = (e) => { const p = pending.get(e.data.id); if (!p) return; pending.delete(e.data.id); e.data.ok ? p.res(e.data.res) : p.rej(new Error(e.data.error)); };
      w.onerror = (e) => { for (const p of pending.values()) p.rej(new Error(e.message || "worker error")); pending.clear(); worker = null; };
      return w;
    } catch (e) { return null; }
  }
  function runJob(kind, job) {
    if (!worker) worker = makeWorker();
    if (worker) return new Promise((res, rej) => { const id = ++jobSeq; pending.set(id, { res, rej }); worker.postMessage({ id, kind, job }); });
    return new Promise((res, rej) => setTimeout(() => { try { res(kind === "baseline" ? A.trainBaseline(job) : A.challengerCycle(job)); } catch (e) { rej(e); } }, 50));
  }

  // ------------------------------------------------------------- market data (browser)
  const closed = (arr) => (arr || []).filter((c) => c.closed);
  function contextCandles(sym, tf, limit) {
    const out = {};
    for (const t of higherTfs(tf)) { const c = closed(S.candles[sym] && S.candles[sym][t]); if (c.length) out[t] = limit ? c.slice(-(t === tf ? limit : Math.ceil(limit / 2))) : c; }
    return out;
  }
  async function detectPrimary() {
    const order = S.settings.primary === "auto" ? ["binance", "okx", "bybit"] : [S.settings.primary];
    for (const ex of order) {
      try { const p = await C.klinesPage(ex, "BTC/USDT", "1h", null); if (p.length > 10) { S.historyErrors[ex] = null; return ex; } }
      catch (e) { S.historyErrors[ex] = e.message || String(e); }
    }
    return null;
  }
  async function loadHistory() {
    setPhase("Downloading market history…");
    S.primary = await detectPrimary();
    if (!S.primary) { setPhase(""); render(); return false; }
    const tfs = [...new Set(S.settings.predictTfs.flatMap(higherTfs))];
    for (const sym of SYMBOLS) {
      S.candles[sym] = S.candles[sym] || {};
      for (const tf of tfs) {
        setPhase(`Downloading ${sym} ${tf} history from ${S.primary}…`);
        try { S.candles[sym][tf] = await C.history(S.primary, sym, tf, HISTORY[tf]); } catch (e) { S.historyErrors[`${sym} ${tf}`] = e.message; }
      }
    }
    S.lastPoll = Date.now(); setPhase(""); render();
    return true;
  }
  async function pollLocal() {
    if (!S.primary) { if (await loadHistory()) await afterData(); return; }
    let errors = 0;
    for (const sym of SYMBOLS) for (const tf of Object.keys(S.candles[sym] || {})) {
      try {
        const fresh = await C.klinesPage(S.primary, sym, tf, null), map = new Map(S.candles[sym][tf].map((c) => [c.open_ts, c]));
        fresh.slice(-5).forEach((c) => map.set(c.open_ts, c));
        S.candles[sym][tf] = [...map.values()].sort((a, b) => a.open_ts - b.open_ts).slice(-(HISTORY[tf] + 500));
      } catch (e) { errors++; S.pollError = `${S.primary}: ${e.message}`; }
    }
    for (const sym of SYMBOLS) for (const ex of EXCHANGES) {
      try { S.tickers[`${ex}|${sym}`] = { ...(await C.ticker24h(ex, sym)), ts: Date.now() }; } catch (e) { S.tickers[`${ex}|${sym}`] = { error: e.message, ts: Date.now() }; }
    }
    if (!errors) { S.lastPoll = Date.now(); S.pollError = null; }
    await afterData();
  }
  async function afterData() {
    for (const sym of SYMBOLS) for (const tf of S.settings.predictTfs) { try { await predictLocal(sym, tf); await resolveLocal(sym, tf); } catch (e) { console.error(e); } }
    ensureModels(); render();
  }

  // ------------------------------------------------------------- local models / learning
  function production(key) { const v = S.registry.production[key]; return v ? S.registry.versions.find((x) => x.version === v) : null; }
  async function modelOf(version) { if (!S.models[version]) S.models[version] = await Store.get(`model:${version}`); return S.models[version]; }
  function jobBase(sym, tf) {
    const s = S.settings;
    return { candlesByTf: contextCandles(sym, tf), tf, horizon: s.horizon, atrMult: s.atrMult, costBps: 2 * s.feeBps + 2 * s.slippageBps + s.spreadBps,
      folds: 3, minTrain: 600, threshold: s.threshold, minEdge: s.minEdge,
      costs: { feeBps: s.feeBps, slippageBps: s.slippageBps, spreadBps: s.spreadBps, latencyMs: s.latencyMs, threshold: s.threshold, minEdge: s.minEdge } };
  }
  function ensureModels() {
    if (S.mode !== "local") return;
    for (const sym of SYMBOLS) for (const tf of S.settings.predictTfs) {
      const key = keyOf(sym, tf), p = production(key);
      if ((!p || p.feature_version !== A.FEATURE_VERSION) && !S.jobs.some((j) => j.key === key) && (!S.training || S.training.key !== key)) S.jobs.push({ kind: "baseline", key, sym, tf });
    }
    pump();
  }
  const newVersion = (sym, tf) => `${sym.replace("/", "")}-${tf}-${new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14)}`;
  async function pump() {
    if (S.training || !S.jobs.length) return;
    const j = S.jobs.shift(); S.training = j; render();
    const t0 = Date.now();
    try {
      const base = jobBase(j.sym, j.tf), rows = (base.candlesByTf[j.tf] || []).length;
      if (rows < 1000) throw new Error(`only ${rows} closed ${j.tf} candles; at least 1000 are needed`);
      if (j.kind === "baseline") {
        const r = await runJob("baseline", base), version = newVersion(j.sym, j.tf);
        await Store.set(`model:${version}`, r.model); S.models[version] = r.model; await Store.set(`oos:${version}`, r.oos);
        const old = production(j.key); if (old) old.status = "archived";
        S.registry.versions.unshift({ version, key: j.key, status: "production", created: Date.now(), feature_version: A.FEATURE_VERSION,
          train_start: r.train_start_ts, train_end: r.train_end_ts, n_train: r.n_train, metrics: r.metrics, reason: old ? "baseline retrained" : "initial baseline", train_sec: (Date.now() - t0) / 1000 });
        S.registry.production[j.key] = version; await Store.set("registry", S.registry);
        toast(`Model ready: ${j.sym} ${j.tf}`);
      } else {
        const prod = production(j.key), model = await modelOf(prod.version), s = S.settings;
        const r = await runJob("challenger", { ...base, production: { model, train_end_ts: prod.train_end }, minNewLabels: s.minNewLabels,
          minHoldout: s.minHoldout, holdoutFraction: s.holdoutFraction, minLoglossGain: s.minLoglossGain, bootstrapConfidence: s.bootstrapConfidence });
        S.learning.status[j.key] = { status: r.status, ts: Date.now(), new_labels: r.new_labels, required: r.required };
        if (r.status === "promoted" || r.status === "rejected") {
          const version = newVersion(j.sym, j.tf);
          S.registry.versions.unshift({ version, key: j.key, status: r.status === "promoted" ? "production" : "rejected", created: Date.now(), feature_version: A.FEATURE_VERSION,
            train_end: r.train_end_ts, n_train: r.n_train, metrics: r.metrics, comparison: r.comparison, reason: r.status });
          if (r.status === "promoted") { await Store.set(`model:${version}`, r.model); S.models[version] = r.model; prod.status = "archived"; S.registry.production[j.key] = version; }
          await Store.set("registry", S.registry);
        }
        await Store.set("learning", S.learning);
      }
    } catch (e) { S.learning.status[j.key] = { status: "error", error: e.message, ts: Date.now() }; }
    finally { S.training = null; render(); setTimeout(pump, 200); }
  }
  function learningCycleLocal() {
    S.learning.lastRun = Date.now();
    for (const sym of SYMBOLS) for (const tf of S.settings.predictTfs) { const key = keyOf(sym, tf); if (production(key) && !S.jobs.some((j) => j.key === key)) S.jobs.push({ kind: "challenger", key, sym, tf }); }
    pump(); render();
  }
  function liveBook(ex, sym) {
    const st = S.streams[ex]; if (!st) return null;
    const bk = st.books[sym]; if (!bk || !bk.ready) return null;
    const age = (Date.now() - (bk.recv || 0)) / 1000; if (age > 30) return null;
    return { ...bk.metrics(10), ageSec: age, exchange: ex };
  }
  function exchangePrices(sym) { const out = {}; for (const ex of EXCHANGES) { const b = liveBook(ex, sym); if (b && b.mid) out[ex] = b.mid; } return out; }
  function qualityLocal(sym, tf) {
    const st = S.streams[S.primary], s = S.settings, key = `${S.primary}|${sym}`;
    const gapsNow = st ? st.gaps : 0, newGaps = Math.max(0, gapsNow - (S.prevGaps[key] ?? gapsNow)); S.prevGaps[key] = gapsNow;
    return A.evaluateQuality({ candles: closed(S.candles[sym] && S.candles[sym][tf]), tf, now: Date.now(), primary: S.primary,
      stream: st ? { state: st.status().state, lastMsg: st.verifiedAt, newGaps } : null, book: liveBook(S.primary, sym), exchangePrices: exchangePrices(sym),
      maxSpreadBps: s.maxSpreadBps, maxDivergenceBps: s.maxDivergenceBps, minScore: s.minQuality });
  }
  async function predictLocal(sym, tf) {
    const key = keyOf(sym, tf), prod = production(key); if (!prod) return;
    const base = closed(S.candles[sym] && S.candles[sym][tf]); if (base.length < 300) return;
    const last = base[base.length - 1], id = `${sym}|${tf}|${S.settings.horizon}|${last.open_ts}`;
    if (S.ledger.has(id) || Date.now() - (last.open_ts + A.TF_MS[tf]) > 2 * A.TF_MS[tf]) return;
    const model = await modelOf(prod.version); if (!model) return;
    const frame = A.buildFrame(contextCandles(sym, tf, 800), tf), row = frame.rows[frame.rows.length - 1];
    const x = model.features.map((f) => (isNum(row[f]) ? row[f] : NaN));
    const [pDown, pFlat, pUp] = A.predictEnsemble(model, [x])[0];
    const s = S.settings, q = qualityLocal(sym, tf), book = liveBook(S.primary, sym), nf = A.newsFeatures(S.news, sym, last.open_ts + A.TF_MS[tf], s.newsHalfLife);
    const g = A.gate(pDown, pFlat, pUp, s.threshold, s.minEdge);
    let regime = row.regime;
    if (!q.ok) regime = "abnormal"; else if (book && isNum(book.spreadBps) && book.spreadBps > s.maxSpreadBps) regime = "low_liquidity";
    const reasons = [];
    if (g.signal === "NO TRADE") reasons.push(g.confidence < s.threshold ? "confidence_below_threshold" : Math.abs(pUp - pDown) < s.minEdge ? "edge_below_min_edge" : "flat_more_likely");
    if (!q.ok) reasons.push("data_quality");
    if (regime === "abnormal") reasons.push("abnormal_market");
    if (regime === "low_liquidity") reasons.push("low_liquidity");
    if (g.signal !== "NO TRADE" && nf.news_relevance >= 0.5 && ((g.signal === "UP" && nf.news_impact <= -0.35) || (g.signal === "DOWN" && nf.news_impact >= 0.35))) reasons.push("news_conflict");
    const rec = { id, prediction_id: id, created_ms: Date.now(), candle_ts: last.open_ts, target_ts: last.open_ts + s.horizon * A.TF_MS[tf], symbol: sym, exchange: S.primary,
      timeframe: tf, horizon_bars: s.horizon, price: last.close, prediction: reasons.length ? "NO TRADE" : g.signal, model_direction: g.modelDir, p_up: pUp, p_down: pDown, p_flat: pFlat,
      confidence: g.confidence, label_threshold: A.labelThreshold(row.atr_pct, s.horizon, s.atrMult, 2 * s.feeBps + 2 * s.slippageBps + s.spreadBps),
      gate_reasons: reasons, regime, quality_score: q.score, news_impact: nf.news_impact, model_version: prod.version, features_version: A.FEATURE_VERSION,
      gate: { confidence: g.confidence, threshold: s.threshold, edge: Math.abs(pUp - pDown), min_edge: s.minEdge, p_flat: pFlat, quality_score: q.score,
        quality_issues: q.issues.map((i) => `${i.code}: ${i.detail}`), decision_delay_sec: Math.round((Date.now() - last.open_ts - A.TF_MS[tf]) / 1000) },
      features: Object.fromEntries(model.features.map((f, i) => [f, isNum(x[i]) ? x[i] : null])) };
    S.ledger.set(id, rec); await Store.putLedger(rec);
  }
  async function resolveLocal(sym, tf) {
    const idx = new Map(closed(S.candles[sym] && S.candles[sym][tf]).map((c) => [c.open_ts, c]));
    for (const r of S.ledger.values()) {
      if (r.symbol !== sym || r.timeframe !== tf || r.resolved_ms) continue;
      const c = idx.get(r.target_ts); if (!c) continue;
      const ret = c.close / r.price - 1, actual = A.directionOf(ret, r.label_threshold), probs = { UP: r.p_up, DOWN: r.p_down, FLAT: r.p_flat };
      Object.assign(r, { resolved_ms: Date.now(), actual_price: c.close, actual_return: ret, actual_direction: actual, error: 1 - probs[actual] });
      if (r.prediction === "NO TRADE") { r.result = "no_trade"; r.error_class = actual === "FLAT" ? "correct_abstain" : r.model_direction === actual ? "gated_would_be_correct" : "missed_move"; }
      else { r.result = r.prediction === actual ? "correct" : "wrong"; r.error_class = r.prediction === actual ? "correct" : actual === "FLAT" ? "false_move" : "wrong_direction"; }
      await Store.putLedger(r);
    }
  }
  async function fetchNewsLocal(force) {
    const now = Date.now();
    if (!force && S.newsStatus.ts && now - S.newsStatus.ts < 10 * 60000) return;
    const recent = S.news.map((e) => e.event), errors = []; let items = [];
    for (const [name, url] of [["CoinDesk", "https://data-api.coindesk.com/news/v1/article/list?lang=EN&limit=50" + (S.settings.newsKey ? `&api_key=${encodeURIComponent(S.settings.newsKey)}` : "")],
      ["CryptoCompare", "https://min-api.cryptocompare.com/data/v2/news/?lang=EN" + (S.settings.newsKey ? `&api_key=${encodeURIComponent(S.settings.newsKey)}` : "")]]) {
      try {
        const d = await C.getJson(url, 15000), list = d.Data || d.data || [], arr = Array.isArray(list) ? list : list.LIST || [];
        items = arr.map((a) => ({ title: a.TITLE || a.title, url: a.URL || a.url, summary: String(a.BODY || a.body || "").slice(0, 400), published_ms: 1000 * (a.PUBLISHED_ON || a.published_on || 0),
          source: (a.SOURCE_DATA && a.SOURCE_DATA.NAME) || (a.source_info && a.source_info.name) || a.source || name })).filter((a) => a.title && a.published_ms > 0 && a.published_ms <= now + 60000);
        if (items.length) break; errors.push(`${name}: empty response`);
      } catch (e) { errors.push(`${name}: ${e.message}`); }
    }
    const fresh = items.filter((it) => !S.news.some((e) => e.url === it.url || e.event === it.title)).map((it) => { const ev = A.analyzeNewsRules(it, recent); recent.push(it.title); return ev; });
    S.news = [...fresh, ...S.news].filter((e) => now - e.timestamp < 3 * 86400e3).sort((a, b) => b.timestamp - a.timestamp).slice(0, 200);
    await Store.set("news", S.news);
    S.newsStatus = { ts: now, state: items.length ? "LIVE" : "DATA SOURCE UNAVAILABLE", analyzer: "rules-v1", errors };
    render();
  }

  // ------------------------------------------------------------- unified view model
  function view() {
    const sv = server(), sym = S.symbol, tf = S.tf;
    if (sv) {
      const key = `${sym}|${tf}|h${sv.settings.horizon_bars}`, m = sv.markets[sym] || {};
      const ledger = S.serverLedger.filter((p) => p.symbol === sym && p.timeframe === tf);
      const cand = S.serverCandles[`${sym}|${tf}`];
      return { key, horizon: sv.settings.horizon_bars, threshold: sv.settings.confidence_threshold, minEdge: sv.settings.min_edge,
        latest: sv.latest_predictions[key], ledger, model: sv.models[key], ticker: m.ticker, serverBook: m.book, serverFlow: m.flow_bars || [],
        primary: sv.primary_exchange, candles: cand ? rowsToCandles(cand.rows) : null };
    }
    const key = keyOf(sym, tf), p = production(key);
    const ledger = [...S.ledger.values()].filter((r) => r.symbol === sym && r.timeframe === tf).sort((a, b) => b.candle_ts - a.candle_ts);
    const tk = S.primary && S.tickers[`${S.primary}|${sym}`];
    return { key, horizon: S.settings.horizon, threshold: S.settings.threshold, minEdge: S.settings.minEdge, latest: ledger[0] || null, ledger,
      model: p ? { production: { version: p.version, created_ms: p.created, n_train: p.n_train, reason: p.reason, metrics: p.metrics }, versions: S.registry.versions.filter((v) => v.key === key).map((v) => ({ ...v, created_ms: v.created })), learning_status: S.learning.status[key] } : null,
      ticker: tk && isNum(tk.last) ? { last: tk.last, change_24h_pct: tk.changePct, ts_ms: tk.ts } : null, primary: S.primary,
      candles: S.candles[sym] && S.candles[sym][tf] ? S.candles[sym][tf].slice(-450) : null };
  }
  function rowsToCandles(rows) { return rows.open_ts.map((t, i) => ({ open_ts: t, open: rows.open[i], high: rows.high[i], low: rows.low[i], close: rows.close[i], volume: rows.volume[i], closed: rows.closed[i], cvd: rows.cvd ? rows.cvd[i] : null })); }

  // ------------------------------------------------------------- status
  function statuses() {
    const sv = server(), now = Date.now(), out = [];
    const browserLive = EXCHANGES.filter((ex) => S.streams[ex] && S.streams[ex].status().state === "live");
    if (S.mode === "server") {
      if (!sv) out.push(["Backend (24/7)", "OFFLINE", "bad", S.serverErr || "waiting for data"]);
      else {
        const age = (now - sv.generated_ms) / 60000, lc = sv.last_cycle || {};
        const st = age > 45 ? "STALE" : lc.ok ? "RUNNING" : "DEGRADED";
        out.push(["Backend (24/7)", st, st === "RUNNING" ? "good" : "warn", `last run ${ago(sv.generated_ms)}${lc.errors && lc.errors.length ? " · errors: " + lc.errors.map((e) => e.step).join(", ") : ""}`]);
        const pr = (sv.probes && sv.probes.probes) || {}, primary = sv.primary_exchange, pp = pr[primary] || {};
        const candleAges = Object.values((pp.rest && pp.rest.candles) || {}).map((c) => c.last_close_age_sec).filter(isNum);
        const verified = pp.rest && pp.rest.ok;
        out.push(["Market data", verified ? `VERIFIED ${ago(sv.probes.ts_ms)}` : "DATA SOURCE UNAVAILABLE", verified ? (age > 45 ? "warn" : "good") : "bad",
          primary ? `${primary} candles via backend${candleAges.length ? ", newest closed candle " + Math.round(Math.min(...candleAges)) + " s old at check" : ""}` : "no exchange reachable from the backend"]);
        const wsOk = Object.entries(pr).filter(([, p]) => p.ws && p.ws.verified_live).map(([ex]) => ex);
        const wsPart = Object.entries(pr).filter(([, p]) => p.ws && !p.ws.verified_live && (p.ws.verified_symbols || []).length).map(([ex, p]) => `${ex} (${p.ws.verified_symbols.join(", ")} only)`);
        const bookState = browserLive.length ? "LIVE" : wsOk.length ? `VERIFIED ${ago(sv.probes.ts_ms)}` : wsPart.length ? `PARTIAL ${ago(sv.probes.ts_ms)}` : "DATA SOURCE UNAVAILABLE";
        out.push(["Order book", bookState, browserLive.length ? "good" : wsOk.length || wsPart.length ? "warn" : "bad",
          browserLive.length ? `live in this browser: ${browserLive.join(", ")}` : wsOk.length || wsPart.length ? `backend sample: ${wsOk.concat(wsPart).join(", ")}` : "no exchange stream verified"]);
        const ns = (sv.news && sv.news.status) || {};
        out.push(["News AI", ns.feeds_ok ? "LIVE" : ns.ts_ms ? "DATA SOURCE UNAVAILABLE" : "NOT STARTED", ns.feeds_ok ? "good" : "bad", ns.analyzer ? `${ns.analyzer} · ${ns.feeds_ok}/${ns.feeds_total} feeds · ${ago(ns.ts_ms)}${ns.llm_error ? " · LLM error" : ""}` : ""]);
        const keys = Object.keys(sv.models), ready = keys.filter((k) => sv.models[k].production);
        out.push(["Prediction", ready.length === keys.length && keys.length ? "READY" : ready.length ? "PARTIAL" : "MODEL NOT READY", ready.length === keys.length && keys.length ? "good" : "warn", `${ready.length}/${keys.length} models in production`]);
        const lr = sv.learning && sv.learning.last_run;
        out.push(["Learning", lr ? "ACTIVE" : "WAITING", lr ? "good" : "warn", lr ? `last cycle ${ago(lr.ts_ms)}` : "first cycle pending"]);
        const lb = sv.last_backup;
        out.push(["Database", sv.ledger_summary ? "OK" : "UNKNOWN", sv.ledger_summary ? "good" : "warn", `${sv.ledger_summary ? sv.ledger_summary.total : 0} predictions stored${lb ? " · backup " + ago(lb.ts_ms) : ""}`]);
      }
      return out;
    }
    const v = view(), base = closed(v.candles || []), lastClose = base.length ? base[base.length - 1].open_ts + A.TF_MS[S.tf] : null;
    const market = !S.primary ? "DATA SOURCE UNAVAILABLE" : lastClose && now - lastClose < A.TF_MS[S.tf] + 300000 && S.lastPoll && now - S.lastPoll < 180000 ? "LIVE" : base.length ? "STALE" : "DATA SOURCE UNAVAILABLE";
    out.push(["Mode", "LOCAL", "warn", "runs only while this page is open"]);
    out.push(["Market data", market, market === "LIVE" ? "good" : market === "STALE" ? "warn" : "bad", S.primary ? `${S.primary} · polled ${ago(S.lastPoll)}${S.pollError ? " · " + S.pollError : ""}` : primaryHint()]);
    out.push(["Order book", browserLive.length ? "LIVE" : S.started ? "DATA SOURCE UNAVAILABLE" : "STOPPED", browserLive.length ? "good" : "bad", EXCHANGES.map((ex) => `${ex} ${S.streams[ex] ? S.streams[ex].status().state : "off"}`).join(", ")]);
    out.push(["News AI", S.newsStatus.state, S.newsStatus.state === "LIVE" ? "good" : "bad", S.newsStatus.analyzer || ""]);
    const p = production(v.key);
    out.push(["Prediction", S.training && S.training.key === v.key ? "TRAINING" : p ? "READY" : "MODEL NOT READY", p ? "good" : "warn", S.training ? `training ${S.training.sym} ${S.training.tf}` : ""]);
    out.push(["Learning", S.training ? "TRAINING" : !S.started ? "STOPPED" : Object.keys(S.registry.production).length ? "ACTIVE" : "WAITING FOR DATA", S.started ? "good" : "warn", S.learning.lastRun ? `last cycle ${ago(S.learning.lastRun)}` : ""]);
    out.push(["Database", S.dbOk ? "OK" : "MEMORY ONLY", S.dbOk ? "good" : "warn", S.dbOk ? `browser storage · ${S.ledger.size} predictions` : "storage blocked"]);
    return out;
  }
  function primaryHint() {
    const errs = Object.entries(S.historyErrors).filter(([, v]) => v).map(([k, v]) => `${k}: ${v}`).join("; ");
    return S.started ? `No exchange reachable from this browser.${errs ? " " + errs : ""} Run the connection test below.` : "Press Start to connect to the exchanges.";
  }

  // ------------------------------------------------------------- rendering
  function setPhase(t) { const p = $("phase"); p.textContent = t; p.hidden = !t; }
  function render() { if (render.q) return; render.q = requestAnimationFrame(() => { render.q = null; try { renderAll(); } catch (e) { console.error(e); } }); }
  function renderAll() {
    const v = view(), sv = server();
    $("welcome").hidden = S.mode !== "local" || S.started;
    $("modeNote").textContent = S.mode === "server" ? (sv ? `Data from the 24/7 backend, updated ${ago(sv.generated_ms)}. ${sv.schedule_note}` : S.serverErr || "Connecting to the backend…")
      : S.mode === "local" ? "Local mode: this page does the work itself and stops when it is closed." : "";
    $("assetName").textContent = `${S.symbol} · ${S.tf}`;
    // price: live browser trade > backend/REST ticker > last closed candle
    const st = S.streams[v.primary], lp = st && st.lastPrice[S.symbol], lpFresh = st && Date.now() - (st.lastTradeTs[S.symbol] || 0) < 30000;
    const base = closed(v.candles || []);
    let price = null, src = "";
    if (lp && lpFresh) { price = lp; src = `live trades on ${v.primary}, this browser`; }
    else if (v.ticker && isNum(v.ticker.last)) { price = v.ticker.last; src = `${v.primary} ticker, ${sv ? "backend check " : ""}${ago(v.ticker.ts_ms)}`; }
    else if (base.length) { price = base[base.length - 1].close; src = "last closed candle"; }
    $("price").textContent = price ? "$" + priceFmt(price) : "DATA SOURCE UNAVAILABLE"; $("price").classList.toggle("unavail", !price);
    $("priceSource").textContent = price ? `Source: ${src}` : S.mode === "server" ? (S.serverErr || "The backend has no price yet.") : primaryHint();
    const ch = v.ticker && v.ticker.change_24h_pct; $("change").textContent = isNum(ch) ? `${ch >= 0 ? "+" : ""}${ch.toFixed(2)}% 24h` : ""; $("change").dataset.dir = ch > 0 ? "up" : ch < 0 ? "down" : "";
    renderPrediction(v); renderStatus(); renderExchanges(v); renderModel(v); renderLedger(v); renderNews(); drawChart(v); renderProbe();
  }
  function reasonDetail(p) {
    const g = p.gate || {}, parts = [];
    for (const r of p.gate_reasons || []) {
      if (r === "confidence_below_threshold" || r === "low_confidence") parts.push(`confidence ${pct(g.confidence ?? p.confidence)} < threshold ${pct(g.threshold, 0)}`);
      else if (r === "edge_below_min_edge") parts.push(`|UP−DOWN| ${pct(g.edge)} < ${pct(g.min_edge, 0)}`);
      else if (r === "flat_more_likely") parts.push(`FLAT ${pct(g.p_flat ?? p.p_flat)} ≥ direction ${pct(g.confidence ?? p.confidence)}`);
      else if (r === "data_quality") parts.push(`data quality ${pct(g.quality_score ?? p.quality_score, 0)}: ${(g.quality_issues || []).join("; ") || "failed"}`);
      else parts.push(REASON_TEXT[r] || r);
    }
    return parts.join(" · ");
  }
  function renderPrediction(v) {
    const p = v.latest, tfms = A.TF_MS[S.tf];
    const fresh = p && Date.now() - (p.candle_ts + tfms) < 2 * tfms + (server() ? 30 * 60000 : 0);
    $("threshold").style.left = `calc(${(v.threshold * 100).toFixed(1)}% - 1px)`; $("threshold").dataset.label = `threshold ${Math.round(v.threshold * 100)}%`;
    $("horizon").textContent = `next ${v.horizon} × ${S.tf}`;
    const sig = $("signal");
    const notReady = !(v.model && v.model.production);
    if (!p || !fresh) {
      sig.textContent = notReady ? (S.training && S.training.key === v.key ? "TRAINING" : "MODEL NOT READY") : "NO TRADE"; sig.dataset.state = notReady ? "none" : "NO TRADE";
      ["segDown", "segFlat", "segUp"].forEach((i) => ($(i).style.width = "0")); ["pDown", "pFlat", "pUp", "confidence", "regime", "quality", "newsImpact"].forEach((i) => ($(i).textContent = "—"));
      $("reasons").textContent = notReady ? (S.mode === "server" ? "The backend trains the first model after downloading history (first runs)." : "The model trains automatically once history is downloaded.")
        : p ? `No current prediction: the latest one is for the candle of ${when(p.candle_ts)}, which is too old.` : "No prediction yet: one is made at each candle close.";
      $("predMeta").textContent = "";
      return;
    }
    sig.textContent = p.prediction; sig.dataset.state = p.prediction;
    $("segDown").style.width = (p.p_down * 100).toFixed(2) + "%"; $("segFlat").style.width = (p.p_flat * 100).toFixed(2) + "%"; $("segUp").style.width = (p.p_up * 100).toFixed(2) + "%";
    $("pDown").textContent = pct(p.p_down); $("pFlat").textContent = pct(p.p_flat); $("pUp").textContent = pct(p.p_up); $("confidence").textContent = pct(p.confidence);
    $("regime").textContent = (p.regime || "—").replace("_", " ").toUpperCase();
    const qs = p.quality_score ?? (p.gate && p.gate.quality_score);
    $("quality").textContent = isNum(qs) ? `${qs >= 0.9 ? "GOOD" : qs >= 0.7 ? "FAIR" : "BAD"} (${Math.round(qs * 100)}%)` : "—";
    $("newsImpact").textContent = isNum(p.news_impact) ? (p.news_impact > 0 ? "+" : "") + p.news_impact.toFixed(2) : "—";
    $("reasons").textContent = p.prediction === "NO TRADE" ? "No trade because: " + reasonDetail(p) + "." : `Signal passed every gate. Model leans ${String(p.model_direction).toLowerCase()}.`;
    $("predMeta").textContent = `Candle ${when(p.candle_ts)} · price at prediction ${priceFmt(p.price)} on ${p.exchange} · decided ${p.gate && isNum(p.gate.decision_delay_sec) ? Math.round(p.gate.decision_delay_sec / 60) + " min after close" + (p.gate.late ? " (late)" : "") : when(p.created_ms)} · model ${p.model_version}`;
    const b = liveBook(v.primary, S.symbol) || (v.serverBook && { spreadBps: v.serverBook.spread_bps, imbalance: v.serverBook.imbalance, ageSec: (Date.now() - v.serverBook.recv_ms) / 1000 });
    $("spread").textContent = b && isNum(b.spreadBps) ? `${num(b.spreadBps, 2)} bps${b.ageSec > 30 ? " (" + ago(Date.now() - b.ageSec * 1000) + ")" : ""}` : "DATA SOURCE UNAVAILABLE";
    $("imbalance").textContent = b && isNum(b.imbalance) ? (b.imbalance > 0 ? "+" : "") + num(b.imbalance, 2) : "—";
  }
  function dl(target, items) { target.replaceChildren(...items.flatMap(([k, v, c, d]) => [el("dt", { text: k }), el("dd", { class: "st-" + c }, el("span", { text: "● " + v }), d ? el("span", { class: "detail", text: d }) : null)])); }
  function renderStatus() {
    const st = statuses(); dl($("statusGrid"), st);
    const head = st.find((x) => x[0] === "Market data") || st[0];
    $("liveDot").className = "live-dot st-" + head[2]; $("liveText").textContent = `${S.mode === "server" ? "24/7 backend" : "local"} · market ${head[1]}`;
  }
  function renderExchanges(v) {
    const sv = server(), pr = (sv && sv.probes && sv.probes.probes) || {}, rows = [];
    for (const ex of EXCHANGES) {
      const s = S.streams[ex], b = liveBook(ex, S.symbol), st = s ? s.status() : null, p = pr[ex];
      const rest = p ? (p.rest.ok ? "ok" : p.rest.partial ? "partial" : `${KIND_TEXT[p.rest.kind] || p.rest.kind || "failed"}`) : "—";
      const ws = p ? (p.ws.verified_live ? `ok (${p.ws.events} msgs)` : KIND_TEXT[p.ws.kind] || p.ws.kind || "no data") : "—";
      rows.push(el("tr", {}, el("td", { text: ex + (ex === v.primary ? " (model)" : "") }),
        el("td", { class: "num", text: s && s.lastPrice[S.symbol] && Date.now() - (s.lastTradeTs[S.symbol] || 0) < 60000 ? priceFmt(s.lastPrice[S.symbol]) : "—" }),
        el("td", { class: "num", text: b ? num(b.spreadBps, 2) : "—" }),
        el("td", { class: st && st.state === "live" ? "st-good" : "st-bad", text: st ? (st.state === "live" ? "LIVE" : st.state === "unavailable" ? "UNAVAILABLE" : st.state) : "off" }),
        el("td", { class: p && p.rest.ok ? "st-good" : p ? "st-bad" : "", text: rest }),
        el("td", { class: p && p.ws.verified_live ? "st-good" : p ? "st-bad" : "", text: ws }),
        el("td", { class: "small muted", text: ((st && st.error) || (p && !p.rest.ok && p.rest.error) || (p && !p.ws.verified_live && p.ws.error) || "").slice(0, 90) })));
    }
    $("exTable").replaceChildren(el("tr", {}, ...["Exchange", "Live price", "Spread bps", "This browser", sv ? `Backend REST (${ago(sv.probes && sv.probes.ts_ms)})` : "Backend REST", "Backend WS", "Last error"].map((h) => el("th", { text: h }))), ...rows);
  }
  function kv(target, pairs) { target.replaceChildren(...pairs.flatMap(([k, val]) => [el("dt", { text: k }), el("dd", { text: val == null ? "—" : String(val) })])); }
  function renderModel(v) {
    const m = v.model, p = m && m.production;
    if (!p) { kv($("modelKv"), [["Status", "MODEL NOT READY"], ["Learning", m && m.learning_status ? JSON.stringify(m.learning_status).slice(0, 160) : "waiting for enough real history"]]); $("whyNoTrade").replaceChildren(); }
    else {
      const mt = p.metrics || {}, b = mt.baseline_prior || {}, sg = mt.signals || {}, ls = m.learning_status, lr = m.ledger_review;
      kv($("modelKv"), [["Model version", p.version], ["Status", `production · ${p.reason || ""}`],
        ["Accuracy (out-of-sample)", `${pct(mt.accuracy)} (naive ${pct(b.accuracy)})`], ["Log loss", `${num(mt.log_loss, 4)} (naive ${num(b.log_loss, 4)})`],
        ["Brier", num(mt.brier, 4)], ["Calibration error (ECE)", num(mt.ece, 3)], ["F1 macro", num(mt.f1_macro, 3)],
        ["Signals in validation", isNum(sg.coverage) ? `${sg.signals} (${pct(sg.coverage)}), hit ${pct(sg.hit_rate)}` : "—"],
        ["Validation", mt.method || "—"], ["Last training", `${when(p.created_ms)} · ${p.n_train} rows`],
        ["Last validation", ls ? `${ls.status}${ls.new_labels != null ? ` (${ls.new_labels}/${ls.required} new labels)` : ""} · ${ago(ls.ts_ms || ls.ts)}` : "not yet"],
        ["Live ledger", lr && lr.resolved ? `${lr.resolved} resolved, ${lr.signals} signals, hit ${pct(lr.signal_hit_rate)}` : "no resolved predictions yet"],
        ["Stored backtest", mt.backtest && mt.backtest.trades ? `${mt.backtest.trades} trades, net ${num(mt.backtest.avg_net_bps, 1)} bps/trade after costs` : "no signal passed the gate"]]);
      renderWhy(mt, v);
    }
    const vs = (m && m.versions) || [];
    $("versions").replaceChildren(el("tr", {}, ...["Version", "Status", "Created", "Log loss", "Accuracy", "Gain", "P(better)", "Reason"].map((h) => el("th", { text: h }))),
      ...vs.slice(0, 15).map((x) => el("tr", {}, el("td", { text: x.version }), el("td", { text: x.status }), el("td", { text: when(x.created_ms) }),
        el("td", { class: "num", text: num(x.log_loss ?? (x.metrics && x.metrics.log_loss), 4) }), el("td", { class: "num", text: pct(x.accuracy ?? (x.metrics && x.metrics.accuracy)) }),
        el("td", { class: "num", text: x.comparison ? num(x.comparison.logloss_gain, 4) : "—" }), el("td", { class: "num", text: x.comparison ? pct(x.comparison.bootstrap_p_better, 0) : "—" }),
        el("td", { text: x.reason || "" }))));
  }
  function renderWhy(mt, v) {
    const g = mt.gate_diagnostics, ld = mt.label_diagnostics, out = [];
    if (g) {
      const f = g.first_failing_condition, n = g.n || 1;
      out.push(el("p", { class: "small", text: `On ${g.n} out-of-sample predictions: ${f.passed} passed (${pct(f.passed / n)}); blocked by confidence < ${pct(g.threshold, 0)}: ${f.confidence_below_threshold}; UP≈DOWN: ${f.edge_below_min_edge}; FLAT more likely: ${f.flat_more_likely}. Confidence median ${pct(g.confidence_quantiles.p50)}, 90th pct ${pct(g.confidence_quantiles.p90)}, max ${pct(g.confidence_quantiles.max)}.` }));
      const cal = g.per_class_calibration;
      if (cal) out.push(el("p", { class: "small", text: "Calibration by class (predicted vs actual): " + Object.entries(cal).map(([k, c]) => `${k} ${pct(c.mean_predicted)} vs ${pct(c.actual_frequency)}`).join(", ") + "." }));
      if (ld) out.push(el("p", { class: "small", text: `A move counts as UP/DOWN above ${num(ld.median_label_threshold_bps, 1)} bps (median); the median move over the horizon is ${num(ld.median_abs_forward_return_bps, 1)} bps, so ${pct(ld.share_moves_above_threshold)} of moves qualify.` }));
      const sw = g.threshold_sweep_informational || [];
      out.push(el("div", { class: "table-scroll" }, el("table", { class: "tbl" }, el("tr", {}, ...["Threshold (info only)", "Signals", "Coverage", "Hit rate", "Avg gross bps"].map((h) => el("th", { text: h }))),
        ...sw.map((r) => el("tr", { class: Math.abs(r.threshold - g.threshold) < 1e-9 ? "current" : "" }, el("td", { text: pct(r.threshold, 0) + (Math.abs(r.threshold - g.threshold) < 1e-9 ? " (current)" : "") }),
          el("td", { class: "num", text: r.signals }), el("td", { class: "num", text: pct(r.coverage) }), el("td", { class: "num", text: pct(r.hit_rate) }), el("td", { class: "num", text: num(r.avg_gross_bps, 1) }))))));
      out.push(el("p", { class: "small muted", text: "The table shows what other thresholds would have done on the same unseen data. The threshold is never lowered automatically." }));
    }
    const sv = server(), reasons = sv && sv.ledger_summary && sv.ledger_summary.no_trade_reasons;
    if (reasons && Object.keys(reasons).length) out.push(el("p", { class: "small", text: "Live NO TRADE reasons so far: " + Object.entries(reasons).map(([k, c]) => `${REASON_TEXT[k] || k}: ${c}`).join(", ") + "." }));
    $("whyNoTrade").replaceChildren(...out);
  }
  function renderLedger(v) {
    let rows = v.ledger.slice().sort((a, b) => b.candle_ts - a.candle_ts);
    const f = S.ledgerFilter;
    if (f === "signals") rows = rows.filter((r) => r.prediction !== "NO TRADE"); else if (f === "resolved") rows = rows.filter((r) => r.resolved_ms); else if (f === "notrade") rows = rows.filter((r) => r.prediction === "NO TRADE");
    const sig = v.ledger.filter((r) => r.resolved_ms && r.prediction !== "NO TRADE"), hits = sig.filter((r) => r.result === "correct").length;
    $("ledgerStats").textContent = `${v.ledger.length} predictions · ${sig.length} resolved signals · hit ${sig.length ? pct(hits / sig.length) : "—"}`;
    const t = $("ledger");
    if (!rows.length) { t.replaceChildren(el("tr", {}, el("td", { class: "muted", text: v.ledger.length ? "Nothing matches this filter." : "No predictions recorded yet. One is made at each candle close." }))); $("moreBtn").hidden = true; return; }
    t.replaceChildren(el("tr", {}, ...["Candle", "Price", "Prediction", "Up", "Down", "Flat", "Conf.", "Quality", "Regime", "Actual", "Return", "Result", "Why / model"].map((h) => el("th", { text: h }))),
      ...rows.slice(0, S.ledgerLimit).map((r) => el("tr", {}, el("td", { text: when(r.candle_ts) }), el("td", { class: "num", text: priceFmt(r.price) }), el("td", { class: "pred-" + r.prediction.replace(" ", ""), text: r.prediction }),
        el("td", { class: "num", text: pct(r.p_up) }), el("td", { class: "num", text: pct(r.p_down) }), el("td", { class: "num", text: pct(r.p_flat) }), el("td", { class: "num", text: pct(r.confidence) }),
        el("td", { class: "num", text: pct(r.quality_score, 0) }), el("td", { text: (r.regime || "").replace("_", " ") }), el("td", { text: r.actual_direction || "pending" }),
        el("td", { class: "num", text: isNum(r.actual_return) ? pct(r.actual_return, 2) : "—" }), el("td", { class: "res-" + (r.result || ""), text: (r.error_class || "").replaceAll("_", " ") }),
        el("td", { class: "small muted wrap", text: (r.prediction === "NO TRADE" ? reasonDetail(r) : "passed all gates") + ` · ${r.model_version}` }))));
    $("moreBtn").hidden = rows.length <= S.ledgerLimit;
  }
  function renderNews() {
    const sv = server();
    let list, status;
    if (sv) {
      const ns = sv.news.status || {};
      status = ns.ts_ms ? `${ns.feeds_ok}/${ns.feeds_total} feeds · ${ns.analyzer} · updated ${ago(ns.ts_ms)}${ns.llm_error ? " · LLM error: " + ns.llm_error : ""}` : "The backend has not read news yet.";
      list = (sv.news.events || []).map((e) => ({ event: e.title, url: e.url, timestamp: e.published_ms, source: e.source, category: e.category || "other", direction: e.direction, relevance: e.relevance, confidence: e.confidence, analyzer: e.analyzer, affected_assets: e.affected_assets || [] }));
    } else { const ns = S.newsStatus; status = ns.ts ? `${ns.state} · ${ns.analyzer} · ${ago(ns.ts)}${ns.errors && ns.errors.length && ns.state !== "LIVE" ? " · " + ns.errors.join("; ") : ""}` : "News loads after Start."; list = S.news; }
    $("newsStatus").textContent = status;
    const base = S.symbol.split("/")[0], rel = list.filter((e) => e.affected_assets.includes(base) || e.affected_assets.includes("CRYPTO_MARKET") || e.relevance >= 0.4).slice(0, 25);
    $("newsList").replaceChildren(...(rel.length ? rel.map((e) => {
      const d = e.direction > 0.1 ? ["dir-up", "positive"] : e.direction < -0.1 ? ["dir-down", "negative"] : ["", "neutral"];
      const title = e.url ? el("a", { href: e.url, target: "_blank", rel: "noopener noreferrer", text: e.event }) : el("span", { text: e.event });
      return el("li", {}, title, el("div", { class: "news-meta" }, el("span", { text: when(e.timestamp) }), el("span", { text: e.source || "" }), el("span", { text: e.category.replaceAll("_", " ") }),
        el("span", { class: d[0], text: `${d[1]} ${e.direction > 0 ? "+" : ""}${num(e.direction, 2)}` }), el("span", { text: `relevance ${Math.round(e.relevance * 100)}%` }),
        el("span", { text: `confidence ${Math.round(e.confidence * 100)}%` }), el("span", { text: e.analyzer || "" })));
    }) : [el("li", { class: "muted", text: list.length ? "No relevant events right now." : "DATA SOURCE UNAVAILABLE" })]));
  }
  function renderProbe() {
    const p = S.probe, t = $("probeTable");
    if (!p) { t.replaceChildren(el("tr", {}, el("td", { class: "muted", text: "Not run yet. The test checks REST candles and live WebSocket data from this device." }))); return; }
    t.replaceChildren(el("tr", {}, ...["Exchange", "REST candles", "Latency", "WebSocket", "Clock skew", "Diagnosis"].map((h) => el("th", { text: h }))),
      ...EXCHANGES.map((ex) => { const r = p[ex] || {}, rs = r.rest, w = r.ws;
        return el("tr", {}, el("td", { text: ex }),
          el("td", { class: rs ? (rs.ok ? "st-good" : "st-bad") : "", text: rs ? (rs.ok ? `ok, candle ${rs.candleAgeSec} s old` : KIND_TEXT[rs.kind] || rs.kind) : "testing…" }),
          el("td", { class: "num", text: rs ? rs.ms + " ms" : "" }),
          el("td", { class: w ? (w.ok ? "st-good" : "st-bad") : "", text: w ? (w.ok ? `verified in ${(w.ms / 1000).toFixed(1)} s` : KIND_TEXT[w.kind] || w.kind) : "testing…" }),
          el("td", { class: "num", text: w && isNum(w.skewMs) ? `${w.skewMs} ms` : "" }),
          el("td", { class: "small muted wrap", text: [rs && !rs.ok ? `REST ${rs.host}: ${rs.error}` : "", w && !w.ok ? `WS: ${w.error}` : ""].filter(Boolean).join(" · ") }));
      }));
  }
  async function runProbe() {
    S.probe = {}; renderProbe(); $("probeBtn").disabled = true;
    await Promise.all(EXCHANGES.map(async (ex) => {
      S.probe[ex] = {}; S.probe[ex].rest = await C.probeRest(ex); renderProbe();
      S.probe[ex].ws = await C.probeWs(ex, 12000); renderProbe();
    }));
    $("probeBtn").disabled = false; S.probe.ts = Date.now();
    const ok = EXCHANGES.filter((ex) => S.probe[ex].rest.ok && S.probe[ex].ws.ok);
    toast(ok.length ? `Working from this device: ${ok.join(", ")}` : "No exchange fully reachable from this device");
  }

  // ------------------------------------------------------------- charts
  function canvas() { const c = $("chart"), r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1; c.width = Math.max(1, Math.round(r.width * dpr)); c.height = Math.max(1, Math.round(r.height * dpr)); const ctx = c.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, r.width, r.height); ctx.font = "11px " + css("--font"); return { ctx, w: r.width, h: r.height }; }
  function scaleOf(vals, lo, hi, pad = 0.06) { let mn = Infinity, mx = -Infinity; for (const v of vals) if (isNum(v)) { mn = Math.min(mn, v); mx = Math.max(mx, v); } if (!isFinite(mn)) { mn = 0; mx = 1; } if (mn === mx) { mn -= 1; mx += 1; } const sp = mx - mn; mn -= sp * pad; mx += sp * pad; return { mn, mx, y: (v) => hi - ((v - mn) / (mx - mn)) * (hi - lo) }; }
  function axis(ctx, sc, x0, x1, top, bottom, fmt) { ctx.strokeStyle = css("--grid"); ctx.fillStyle = css("--ink-2"); ctx.lineWidth = 1; ctx.textAlign = "left"; for (let i = 0; i <= 4; i++) { const v = sc.mn + ((sc.mx - sc.mn) * i) / 4, y = sc.y(v); if (y < top - 1 || y > bottom + 1) continue; ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); ctx.fillText(fmt(v), x1 + 6, y + 4); } }
  function line(ctx, xs, ys, sc, color, w = 1.5) { ctx.strokeStyle = color; ctx.lineWidth = w; ctx.beginPath(); let on = false; ys.forEach((v, i) => { if (!isNum(v)) { on = false; return; } on ? ctx.lineTo(xs[i], sc.y(v)) : ctx.moveTo(xs[i], sc.y(v)); on = true; }); ctx.stroke(); }
  function times(ctx, xs, ts, bottom) { ctx.fillStyle = css("--ink-2"); ctx.textAlign = "center"; const st = Math.max(1, Math.floor(ts.length / 5)); for (let i = st; i < ts.length; i += st) ctx.fillText(when(ts[i]), xs[i], bottom + 14); }
  function empty(msg) { $("chartEmpty").textContent = msg; $("chartEmpty").hidden = !msg; }
  function hover(xs, text) { const c = $("chart"), tip = $("chartTip"); const show = (cx) => { const x = cx - c.getBoundingClientRect().left; let b = 0; xs.forEach((v, i) => { if (Math.abs(v - x) < Math.abs(xs[b] - x)) b = i; }); tip.textContent = text(b); tip.hidden = false; }; c.onmousemove = (e) => show(e.clientX); c.onmouseleave = () => (tip.hidden = true); c.ontouchmove = (e) => e.touches[0] && show(e.touches[0].clientX); }
  function drawChart(v) {
    $("chartTip").hidden = true; $("chart").onmousemove = null;
    const { ctx, w, h } = canvas(), x0 = 6, x1 = w - (w < 520 ? 54 : 66), top = 10, bottom = h - 22;
    const up = css("--up"), down = css("--down"), ink2 = css("--ink-2"), accent = css("--accent");
    if (S.tab === "book") {
      const st = S.streams[S.bookEx], bk = st && st.books[S.symbol]; let bids, asks, note;
      if (bk && bk.ready && Date.now() - (bk.recv || 0) < 30000) { ({ bids, asks } = bk.top(20)); note = `${S.bookEx} live order book in this browser.`; }
      else if (server() && v.serverBook && S.bookEx === v.primary && v.serverBook.bids && v.serverBook.bids.length) { bids = v.serverBook.bids; asks = v.serverBook.asks; note = `${v.primary} order book sampled by the backend ${ago(v.serverBook.recv_ms)} (not live).`; }
      if (!bids) { empty(`DATA SOURCE UNAVAILABLE: no order book from ${S.bookEx}${st && st.error ? " (" + st.error + ")" : ""}`); $("chartNote").textContent = ""; return; }
      empty(""); let cb = 0, ca = 0; const B = bids.map(([p, q]) => [p, (cb += q)]), Aa = asks.map(([p, q]) => [p, (ca += q)]), px = B.concat(Aa).map((x) => x[0]);
      const pmin = Math.min(...px), pmax = Math.max(...px), X = (p) => x0 + ((p - pmin) / (pmax - pmin || 1)) * (x1 - x0), sc = scaleOf([0, cb, ca], top, bottom, 0.02);
      axis(ctx, sc, x0, x1, top, bottom, (x) => num(x, 2));
      for (const [pts, col] of [[B, up], [Aa, down]]) { if (!pts.length) continue; ctx.fillStyle = col; ctx.globalAlpha = 0.25; ctx.beginPath(); ctx.moveTo(X(pts[0][0]), sc.y(0)); pts.forEach(([p, c]) => ctx.lineTo(X(p), sc.y(c))); ctx.lineTo(X(pts[pts.length - 1][0]), sc.y(0)); ctx.fill(); ctx.globalAlpha = 1; ctx.strokeStyle = col; ctx.beginPath(); pts.forEach(([p, c], i) => (i ? ctx.lineTo(X(p), sc.y(c)) : ctx.moveTo(X(p), sc.y(c)))); ctx.stroke(); }
      ctx.fillStyle = ink2; ctx.textAlign = "center"; [pmin, (pmin + pmax) / 2, pmax].forEach((p) => ctx.fillText(priceFmt(p), Math.min(Math.max(X(p), 40), x1 - 30), bottom + 14));
      $("chartNote").textContent = note; return;
    }
    if (S.tab === "cvd") {
      const st = S.streams[S.bookEx], f = st && st.flow[S.symbol], bars = f ? [...f.bars.values()] : []; let pts, note;
      if (bars.length > 1) { pts = bars.map((b) => ({ t: b.open_ts, v: b.cvd })); note = `Live CVD from ${S.bookEx} trades in this browser (1-minute bars).`; }
      else { pts = (v.candles || []).filter((c) => isNum(c.cvd)).map((c) => ({ t: c.open_ts, v: c.cvd })); if (!pts.length && v.candles && v.candles.length > 30) { const ind = A.computeIndicators(v.candles); pts = ind.open_ts.map((t, i) => ({ t, v: ind.cvd[i] })).filter((p) => isNum(p.v)); } note = pts.length ? `CVD from ${v.primary} taker-buy volume per ${S.tf} candle.` : ""; }
      $("chartNote").textContent = note;
      if (pts.length < 2) { empty("DATA SOURCE UNAVAILABLE: no trade-flow data for CVD."); return; }
      empty(""); pts = pts.slice(-220); const xs = pts.map((_, i) => x0 + ((x1 - x0) / pts.length) * (i + 0.5)), sc = scaleOf(pts.map((p) => p.v), top, bottom);
      axis(ctx, sc, x0, x1, top, bottom, (x) => num(x, Math.abs(x) > 100 ? 0 : 2)); line(ctx, xs, pts.map((p) => p.v), sc, accent, 1.8); times(ctx, xs, pts.map((p) => p.t), bottom); hover(xs, (i) => `${when(pts[i].t)} CVD ${num(pts[i].v, 2)}`); return;
    }
    if (!v.candles || v.candles.length < 30) { empty(`DATA SOURCE UNAVAILABLE: no candles. ${S.mode === "server" ? S.serverErr || "The backend has not exported candles yet." : primaryHint()}`); $("chartNote").textContent = ""; return; }
    empty(""); const full = A.computeIndicators(v.candles), k = Math.max(0, full.open_ts.length - 200), ind = {};
    for (const [n, a] of Object.entries(full)) ind[n] = Array.isArray(a) ? a.slice(k) : a;
    const n = ind.open_ts.length, bw = (x1 - x0) / n, xs = ind.open_ts.map((_, i) => x0 + bw * (i + 0.5)), ts = ind.open_ts;
    if (S.tab === "price") {
      const sc = scaleOf(ind.low.concat(ind.high, ind.bb_upper, ind.bb_lower), top, bottom); axis(ctx, sc, x0, x1, top, bottom, priceFmt);
      ctx.fillStyle = css("--grid"); ctx.globalAlpha = 0.7; ctx.beginPath(); ind.bb_upper.forEach((y, i) => isNum(y) && ctx.lineTo(xs[i], sc.y(y))); for (let i = n - 1; i >= 0; i--) if (isNum(ind.bb_lower[i])) ctx.lineTo(xs[i], sc.y(ind.bb_lower[i])); ctx.fill(); ctx.globalAlpha = 1;
      for (let i = 0; i < n; i++) { const col = ind.close[i] >= ind.open[i] ? up : down; ctx.strokeStyle = col; ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(xs[i], sc.y(ind.high[i])); ctx.lineTo(xs[i], sc.y(ind.low[i])); ctx.stroke(); const a = sc.y(ind.open[i]), b = sc.y(ind.close[i]), ww = Math.max(1, bw * 0.65); ctx.fillRect(xs[i] - ww / 2, Math.min(a, b), ww, Math.max(1, Math.abs(b - a))); }
      line(ctx, xs, ind.ema_21, sc, accent, 1.4); line(ctx, xs, ind.ema_55, sc, ink2, 1.2);
      const idx = new Map(ts.map((t, i) => [t, i]));
      for (const r of v.ledger) { if (r.prediction === "NO TRADE") continue; const i = idx.get(r.candle_ts); if (i === undefined) continue; const u = r.prediction === "UP", y = u ? sc.y(ind.low[i]) + 10 : sc.y(ind.high[i]) - 10; ctx.fillStyle = u ? up : down; ctx.beginPath(); ctx.moveTo(xs[i], y + (u ? -6 : 6)); ctx.lineTo(xs[i] - 5, y + (u ? 3 : -3)); ctx.lineTo(xs[i] + 5, y + (u ? 3 : -3)); ctx.fill(); }
      times(ctx, xs, ts, bottom); $("chartNote").textContent = `${v.primary} ${S.tf} candles${server() ? " from the backend" : ""}. Blue EMA 21, grey EMA 55, band Bollinger 20/2, triangles = recorded UP/DOWN predictions.`;
      hover(xs, (i) => `${when(ts[i])} O ${priceFmt(ind.open[i])} H ${priceFmt(ind.high[i])} L ${priceFmt(ind.low[i])} C ${priceFmt(ind.close[i])}`);
    } else if (S.tab === "volume") {
      const sc = scaleOf(ind.volume.concat([0]), top, bottom, 0.02); axis(ctx, sc, x0, x1, top, bottom, (x) => num(x, x > 100 ? 0 : 2));
      ind.volume.forEach((vv, i) => { ctx.fillStyle = ind.close[i] >= ind.open[i] ? up : down; const y = sc.y(vv), ww = Math.max(1, bw * 0.7); ctx.fillRect(xs[i] - ww / 2, y, ww, sc.y(0) - y); });
      times(ctx, xs, ts, bottom); $("chartNote").textContent = `Volume per ${S.tf} candle in base currency.`; hover(xs, (i) => `${when(ts[i])} volume ${num(ind.volume[i], 2)}`);
    } else {
      const mid = top + (bottom - top) / 2, rs = scaleOf([0, 100], top, mid - 10, 0); axis(ctx, rs, x0, x1, top, mid - 10, (x) => x.toFixed(0));
      ctx.setLineDash([4, 4]); ctx.strokeStyle = ink2; [30, 70].forEach((l) => { ctx.beginPath(); ctx.moveTo(x0, rs.y(l)); ctx.lineTo(x1, rs.y(l)); ctx.stroke(); }); ctx.setLineDash([]);
      line(ctx, xs, ind.rsi_14, rs, accent); line(ctx, xs, ind.adx_14, rs, ink2, 1.1);
      const ms = scaleOf(ind.macd_hist.concat([0]), mid + 6, bottom); axis(ctx, ms, x0, x1, mid + 6, bottom, (x) => (x * 1e4).toFixed(1));
      ind.macd_hist.forEach((vv, i) => { if (!isNum(vv)) return; ctx.fillStyle = vv >= 0 ? up : down; const y = ms.y(vv), y0 = ms.y(0), ww = Math.max(1, bw * 0.6); ctx.fillRect(xs[i] - ww / 2, Math.min(y, y0), ww, Math.abs(y0 - y)); });
      times(ctx, xs, ts, bottom); $("chartNote").textContent = "Top: RSI 14 (blue), ADX 14 (grey), lines at 30/70. Bottom: MACD histogram in basis points.";
      hover(xs, (i) => `${when(ts[i])} RSI ${num(ind.rsi_14[i], 1)} ADX ${num(ind.adx_14[i], 1)} MACD ${num((ind.macd_hist[i] || 0) * 1e4, 2)} bps`);
    }
  }

  // ------------------------------------------------------------- settings & actions
  const FIELDS = [["setAnthropicKey", "anthropicKey", "s"], ["setAnthropicModel", "anthropicModel", "s"], ["setNewsKey", "newsKey", "s"], ["setPrimary", "primary", "s"],
    ["setThreshold", "threshold", "n"], ["setEdge", "minEdge", "n"], ["setHorizon", "horizon", "i"], ["setMinLabels", "minNewLabels", "i"], ["setFee", "feeBps", "n"], ["setSlip", "slippageBps", "n"],
    ["setNotify", "notifyEnabled", "b"], ["setNotifySignals", "notifySignals", "b"], ["setNotifyAll", "notifyAll", "b"], ["setNotifyOutages", "notifyOutages", "b"]];
  function fillSettings() { for (const [id, k, t] of FIELDS) { if (t === "b") $(id).checked = !!S.settings[k]; else $(id).value = S.settings[k]; } $("localOnly").hidden = S.mode === "server"; }
  async function saveSettings() {
    const before = { ...S.settings };
    for (const [id, k, t] of FIELDS) { if (t === "b") { S.settings[k] = $(id).checked; continue; } const v = $(id).value.trim(); S.settings[k] = t === "s" ? v : t === "i" ? parseInt(v, 10) : parseFloat(v); if (t !== "s" && !isNum(S.settings[k])) S.settings[k] = before[k]; }
    S.settings.threshold = Math.min(0.95, Math.max(0.34, S.settings.threshold)); S.settings.horizon = Math.min(24, Math.max(1, S.settings.horizon));
    if (S.settings.notifyEnabled && "Notification" in window && Notification.permission !== "granted") {
      const perm = await Notification.requestPermission().catch(() => "denied");
      if (perm !== "granted") { S.settings.notifyEnabled = false; toast("Notifications were not allowed by the browser."); }
    }
    await Store.set("settings", S.settings); $("settingsDialog").close();
    if (S.mode === "local" && before.primary !== S.settings.primary) { S.primary = null; S.candles = {}; await pollLocal(); }
    if (S.mode === "local" && before.horizon !== S.settings.horizon) ensureModels();
    toast(S.settings.notifyEnabled ? "Settings saved. Browser notifications are on while this page is open." : "Settings saved.");
    render();
  }
  function exportCsv() {
    const v = view(), rows = v.ledger.slice().sort((a, b) => a.candle_ts - b.candle_ts);
    const cols = ["prediction_id", "symbol", "timeframe", "exchange", "candle_ts", "created_ms", "price", "prediction", "model_direction", "p_up", "p_down", "p_flat", "confidence", "quality_score", "regime", "model_version", "features_version", "gate_reasons", "resolved_ms", "actual_price", "actual_return", "actual_direction", "error", "result", "error_class"];
    const csv = [cols.join(",")].concat(rows.map((r) => cols.map((c) => JSON.stringify(Array.isArray(r[c]) ? r[c].join("|") : r[c] ?? "")).join(","))).join("\n");
    const a = el("a", { href: URL.createObjectURL(new Blob([csv], { type: "text/csv" })), download: `prediction_ledger_${S.symbol.replace("/", "")}_${S.tf}_${new Date().toISOString().slice(0, 10)}.csv` });
    document.body.append(a); a.click(); a.remove(); toast(`Exported ${rows.length} predictions`);
  }
  async function runBacktest() {
    const out = $("btResult");
    if (S.mode === "server") { const m = view().model, bt = m && m.production && m.production.metrics.backtest; kv(out, bt && bt.trades ? [["Trades", bt.trades], ["Win rate", pct(bt.win_rate)], ["Avg gross / net", `${num(bt.avg_gross_bps, 1)} / ${num(bt.avg_net_bps, 1)} bps`], ["Total net return", pct(bt.total_net_return, 2)], ["Max drawdown", pct(bt.max_drawdown, 2)], ["Costs", `${num(bt.costs && bt.costs.round_trip_cost_bps, 1)} bps round trip`]] : [["Result", "No signal passed the gate in validation."]]); return; }
    const p = production(keyOf(S.symbol, S.tf)); if (!p) { kv(out, [["Result", "MODEL NOT READY"]]); return; }
    const oos = await Store.get(`oos:${p.version}`); if (!oos) { kv(out, [["Result", "No stored out-of-sample predictions for this version."]]); return; }
    const r = A.backtest(oos, S.tf, S.settings.horizon, { threshold: +$("btThr").value, minEdge: S.settings.minEdge, feeBps: +$("btFee").value, slippageBps: +$("btSlip").value, spreadBps: +$("btSpread").value, latencyMs: +$("btLat").value });
    kv(out, r.trades ? [["Trades", r.trades], ["Win rate", pct(r.win_rate)], ["Avg gross / net", `${num(r.avg_gross_bps, 1)} / ${num(r.avg_net_bps, 1)} bps`], ["Total net return", pct(r.total_net_return, 2)], ["Max drawdown", pct(r.max_drawdown, 2)], ["Round-trip cost", `${num(r.round_trip_cost_bps, 1)} bps`]] : [["Trades", 0], ["Note", "No signal passed the confidence gate."]]);
  }
  function confirmInPage(msg, onYes) { $("confirmText").textContent = msg; $("confirmBox").hidden = false; $("confirmYes").onclick = () => { $("confirmBox").hidden = true; onYes(); }; $("confirmNo").onclick = () => ($("confirmBox").hidden = true); }
  function startStreams() { for (const ex of EXCHANGES) { if (S.streams[ex]) continue; const st = new C.ExchangeStream(ex, SYMBOLS, () => render()); S.streams[ex] = st; st.start(); } }
  async function startLocal() {
    S.started = true; S.settings.started = true; await Store.set("settings", S.settings);
    startStreams(); render(); await pollLocal(); fetchNewsLocal(false);
    setInterval(pollLocal, 20000); setInterval(() => fetchNewsLocal(false), 5 * 60000); setInterval(learningCycleLocal, 10 * 60000); setTimeout(learningCycleLocal, 60000);
  }

  // ------------------------------------------------------------- boot
  async function boot() {
    await Store.open();
    try {
      S.settings = { ...DEFAULTS, ...((await Store.get("settings")) || {}) };
      S.registry = (await Store.get("registry")) || S.registry; S.learning = { ...S.learning, ...((await Store.get("learning")) || {}) };
      S.news = (await Store.get("news")) || []; for (const r of await Store.allLedger()) if (r.timeframe) S.ledger.set(r.id, r);
    } catch (e) { console.error(e); }
    const hasServer = await detectServer();
    S.mode = hasServer ? "server" : "local";
    const tfs = S.settings.predictTfs;
    S.tf = tfs[0];
    $("symbol").replaceChildren(...SYMBOLS.map((s) => el("option", { value: s, text: s })));
    $("tf").replaceChildren(...tfs.map((t) => el("option", { value: t, text: t })));
    $("bookEx").replaceChildren(...EXCHANGES.map((e) => el("option", { value: e, text: e })));
    $("symbol").onchange = () => { S.symbol = $("symbol").value; render(); };
    $("tf").onchange = () => { S.tf = $("tf").value; render(); };
    $("bookEx").onchange = () => { S.bookEx = $("bookEx").value; render(); };
    $("startBtn").onclick = () => startLocal();
    $("refreshBtn").onclick = async () => { $("refreshBtn").disabled = true; if (S.mode === "server") await fetchServer(true); else await pollLocal(); $("refreshBtn").disabled = false; toast("Refreshed"); };
    $("settingsBtn").onclick = $("welcomeSettings").onclick = () => { fillSettings(); $("settingsDialog").showModal(); };
    $("settingsForm").onsubmit = (e) => { e.preventDefault(); saveSettings(); };
    $("settingsCancel").onclick = () => $("settingsDialog").close();
    $("probeBtn").onclick = runProbe;
    $("newsBtn").onclick = () => (S.mode === "server" ? fetchServer(true) : fetchNewsLocal(true));
    $("newsBtn").textContent = hasServer ? "Reload" : "Refresh news";
    $("learnBtn").hidden = hasServer; $("retrainBtn").hidden = hasServer;
    $("learnBtn").onclick = () => { learningCycleLocal(); toast("Learning cycle started"); };
    $("retrainBtn").onclick = () => confirmInPage(`Retrain the ${S.symbol} ${S.tf} baseline from scratch? The current model is archived.`, () => { const k = keyOf(S.symbol, S.tf); if (!S.jobs.some((j) => j.key === k)) S.jobs.push({ kind: "baseline", key: k, sym: S.symbol, tf: S.tf }); pump(); toast("Retraining started"); });
    $("exportBtn").onclick = exportCsv;
    $("ledgerFilter").onchange = () => { S.ledgerFilter = $("ledgerFilter").value; S.ledgerLimit = 50; render(); };
    $("moreBtn").onclick = () => { S.ledgerLimit += 100; render(); };
    $("btForm").onsubmit = (e) => { e.preventDefault(); runBacktest(); };
    $("resetBtn").onclick = () => { $("settingsDialog").close(); confirmInPage("Delete all models, predictions, news and settings stored in this browser?", async () => { await Store.clear(); location.reload(); }); };
    for (const b of $("chartTabs").querySelectorAll("button")) b.onclick = () => { S.tab = b.dataset.chart; $("chartTabs").querySelectorAll("button").forEach((x) => x.setAttribute("aria-selected", String(x === b))); render(); };
    $("btFee").value = S.settings.feeBps; $("btSlip").value = S.settings.slippageBps; $("btSpread").value = S.settings.spreadBps; $("btLat").value = S.settings.latencyMs; $("btThr").value = S.settings.threshold;
    $("btForm").hidden = hasServer;
    window.addEventListener("resize", render);
    setInterval(render, 2000);
    if (hasServer) { await fetchServer(true); startStreams(); setInterval(() => fetchServer(false), 60000); }
    else if (S.settings.started) startLocal();
    render();
  }
  window.AMPApp = { state: S };
  boot();
})();
