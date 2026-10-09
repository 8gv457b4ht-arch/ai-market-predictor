/* AI Market Predictor - browser application.
 * Server mode (published site): predictions, ledger, models, learning and news come from the backend
 *   (scheduled GitHub Actions job or a continuous server, JSON in the `data` branch). The browser adds live prices/order books.
 * Local mode (file opened directly, or backend unreachable): everything runs in this browser.
 * Every visible text comes from the translation dictionary (i18n.js); machine codes, tickers, API names and raw
 * exchange messages are shown as they are, next to a translated explanation.
 */
(function () {
  "use strict";
  const A = window.AMP, C = window.AMPConnectors, I = window.AMPI18n, ST = window.AMPStatus;
  const t = (k, p) => I.t(k, p);
  const SYMBOLS = ["BTC/USDT", "ETH/USDT"], EXCHANGES = ["binance", "bybit", "okx"], CONTEXT_TFS = ["15m", "1h", "4h", "1d"];
  const HISTORY = { "15m": 3000, "1h": 2000, "4h": 1000, "1d": 600 };
  const DEFAULTS = { started: false, primary: "auto", predictTfs: ["15m", "1h"], horizon: 4, threshold: 0.55, minEdge: 0.1, atrMult: 0.3,
    feeBps: 10, slippageBps: 2, spreadBps: 1, latencyMs: 500, minNewLabels: 300, minHoldout: 150, holdoutFraction: 0.5,
    minLoglossGain: 0.002, bootstrapConfidence: 0.9, baselinePBetter: 0.95, maxSpreadBps: 15, maxDivergenceBps: 40, minQuality: 0.7, newsHalfLife: 180,
    anthropicKey: "", anthropicModel: "claude-haiku-5-5", newsKey: "", notifyEnabled: false, notifySignals: true, notifyAll: false, notifyOutages: true,
    notifyMinConf: 0, notifyMaxAgeMin: 10 };
  const $ = (id) => document.getElementById(id);
  const S = { mode: "detecting", server: null, serverErr: null, serverLedger: [], serverCandles: {}, serverForecasts: {}, lastServerFetch: 0, cfg: null,
    settings: { ...DEFAULTS }, primary: null, candles: {}, tickers: {}, streams: {}, registry: { versions: [], production: {} },
    models: {}, ledger: new Map(), news: [], newsStatus: { state: "NOT STARTED" }, jobs: [], training: null, learning: { lastRun: null, status: {} },
    lastPoll: null, pollError: null, historyErrors: {}, tab: "price", symbol: "BTC/USDT", tf: "15m", bookEx: "binance", dbOk: false,
    prevGaps: {}, probe: null, ledgerLimit: 50, prevSourceOk: {},
    filter: { type: "all", symbol: "current", tf: "current", model: "all", period: "all" } };

  // ------------------------------------------------------------- helpers
  const isNum = (v) => typeof v === "number" && Number.isFinite(v);
  const loc = () => I.locale();
  const pct = (x, d = 1) => (isNum(x) ? (x * 100).toLocaleString(loc(), { minimumFractionDigits: d, maximumFractionDigits: d }) + "%" : "—");
  const num = (x, d = 2) => (isNum(x) ? Number(x).toLocaleString(loc(), { minimumFractionDigits: d, maximumFractionDigits: d }) : "—");
  const priceFmt = (x) => (isNum(x) ? Number(x).toLocaleString(loc(), { maximumFractionDigits: x >= 100 ? 2 : 4, minimumFractionDigits: x >= 100 ? 2 : 0 }) : "—");
  const when = (ms) => (ms ? new Date(ms).toLocaleString(loc(), { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—");
  const clock = (ms) => (ms ? new Date(ms).toLocaleTimeString(loc(), { hour: "2-digit", minute: "2-digit" }) : "—");
  function ago(ms) {
    if (!ms) return t("time.never");
    const s = (Date.now() - ms) / 1000;
    if (s < 0) return t("time.in", { v: inDur(-s) });
    if (s < 90) return t("time.sec_ago", { n: Math.round(s) });
    if (s < 5400) return t("time.min_ago", { n: Math.round(s / 60) });
    if (s < 172800) return t("time.h_ago", { n: Math.round(s / 3600) });
    return t("time.d_ago", { n: Math.round(s / 86400) });
  }
  function inDur(s) { return s < 5400 ? t("time.min", { n: Math.max(1, Math.round(s / 60)) }) : t("time.h", { n: Math.round(s / 3600) }); }
  function el(tag, attrs, ...kids) { const e = document.createElement(tag); for (const [k, v] of Object.entries(attrs || {})) { if (k === "class") e.className = v; else if (k === "text") e.textContent = v; else if (k.startsWith("on")) e[k] = v; else e.setAttribute(k, v); } kids.forEach((k) => k != null && e.append(k)); return e; }
  function toast(msg) { const tt = $("toast"); tt.textContent = msg; tt.hidden = false; clearTimeout(toast.t); toast.t = setTimeout(() => (tt.hidden = true), 4500); }
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const keyOf = (sym, tf) => `${sym}|${tf}|h${S.settings.horizon}`;
  const higherTfs = (tf) => CONTEXT_TFS.filter((x) => A.TF_MS[x] >= A.TF_MS[tf]);
  const server = () => S.mode === "server" && S.server;
  const dirText = (d) => (d ? t("dir." + d.replace(" ", "_")) : "—");
  const regimeText = (r) => (r ? t("regime." + r) : "—");

  // NO TRADE: machine reason (ledger) -> user-facing code (kept in English, like an error code) + translated explanation
  const CODE_OF = { confidence_below_threshold: "LOW_CONFIDENCE", low_confidence: "LOW_CONFIDENCE", edge_below_min_edge: "LOW_CONFIDENCE",
    flat_more_likely: "LOW_CONFIDENCE", stale_data: "STALE_DATA", late_decision: "STALE_DATA", data_stale: "DATA_STALE", insufficient_history: "INSUFFICIENT_HISTORY",
    model_not_ready: "MODEL_NOT_READY", low_liquidity: "HIGH_SPREAD", model_not_better_than_baseline: "MODEL_NOT_BETTER_THAN_BASELINE",
    no_edge_after_costs: "NO_EDGE_AFTER_COSTS", model_not_validated: "MODEL_NOT_VALIDATED", insufficient_data: "INSUFFICIENT_DATA",
    data_quality: "DATA_QUALITY", abnormal_market: "ABNORMAL_MARKET", news_conflict: "NEWS_CONFLICT" };
  const codeLabel = (c) => c.replaceAll("_", " ");
  function reasonCodes(p) { return [...new Set((p.gate_reasons || []).map((r) => CODE_OF[r] || r.toUpperCase()))]; }
  function reasonDetail(p) {
    const g = p.gate || {}, parts = [];
    for (const r of p.gate_reasons || []) {
      const P = { conf: pct(g.confidence ?? p.confidence), thr: pct(g.threshold, 0), edge: pct(g.edge), min: pct(g.min_edge, 0),
        flat: pct(g.p_flat ?? p.p_flat), q: pct(g.quality_score ?? p.quality_score, 0), delay: isNum(g.decision_delay_sec) ? Math.round(g.decision_delay_sec / 60) : "—",
        issues: (g.quality_issues || []).map(issueText).join("; ") || "—", miss: pct(g.missing_feature_share, 0),
        gain: g.baseline && isNum(g.baseline.gain) ? num(g.baseline.gain, 4) : "—", pb: g.baseline ? pct(g.baseline.p_better, 0) : "—",
        req: g.baseline ? pct(g.baseline.p_required, 0) : "—" };
      parts.push(`${codeLabel(CODE_OF[r] || r)} — ${t("reason." + r, P)}`);
    }
    return parts.join(" · ");
  }
  function issueText(s) { const code = String(s).split(":")[0].trim(); const k = "issue." + code; return I.has(k) ? `${code}: ${t(k)}` : s; }
  const kindText = (k) => (k ? (I.has("kind." + k) ? t("kind." + k) : k) : "");

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
      return new Promise((res, rej) => { const tr = this.db.transaction(store, mode), st = tr.objectStore(store); const r = fn(st); tr.oncomplete = () => res(r && "result" in r ? r.result : r); tr.onerror = () => rej(tr.error); });
    },
    async get(k) { if (!this.db) return this.mem.get(k); return this.tx("kv", "readonly", (st) => st.get(k)); },
    async set(k, v) { if (!this.db) { this.mem.set(k, v); return; } return this.tx("kv", "readwrite", (st) => st.put(v, k)); },
    async putLedger(r) { if (!this.db) return; return this.tx("ledger", "readwrite", (st) => st.put(r)); },
    async allLedger() { if (!this.db) return []; return this.tx("ledger", "readonly", (st) => st.getAll()); },
    async clear() { if (!this.db) { this.mem.clear(); return; } await this.tx("kv", "readwrite", (st) => st.clear()); await this.tx("ledger", "readwrite", (st) => st.clear()); },
  };

  // ------------------------------------------------------------- server (backend) data
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
      if (r.status === 404) { S.serverErr = t("server.not_published"); S.server = null; render(); return; }
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const first = !S.server;
      S.server = await r.json(); S.serverErr = null;
      if (first && S.server.primary_exchange) { S.bookEx = S.server.primary_exchange; $("bookEx").value = S.bookEx; }
      if (first) fillTfOptions(S.server.settings.predict_timeframes);
      const l = await fetch(S.cfg.state_url + "ledger.json" + bust, { cache: "no-store" });
      if (l.ok) { const prev = S.serverLedger; S.serverLedger = (await l.json()).predictions || []; notifyNew(prev); }
      const fr = await fetch(S.cfg.state_url + "forecasts.json" + bust, { cache: "no-store" });
      if (fr.ok) S.serverForecasts = (await fr.json()).recent || {};
      for (const sym of S.server.settings.symbols) for (const tf of S.server.settings.predict_timeframes) {
        const c = await fetch(`${S.cfg.state_url}candles_${sym.replace("/", "-")}_${tf}.json${bust}`, { cache: "no-store" });
        if (c.ok) S.serverCandles[`${sym}|${tf}`] = await c.json();
      }
      notifyOutages();
    } catch (e) { S.serverErr = t("server.unreachable", { error: e.message }); }
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
      const sig = p.prediction !== "NO TRADE";
      if (sig ? !(S.settings.notifySignals || S.settings.notifyAll) : !S.settings.notifyAll) continue;
      if (sig && p.confidence < (S.settings.notifyMinConf || 0)) continue;
      if (Date.now() - (p.candle_ts + A.TF_MS[p.timeframe]) > (S.settings.notifyMaxAgeMin || 10) * 60000) continue; // never a stale signal
      notify(`${p.symbol} ${p.timeframe}: ${p.prediction}`, t("notify.body", { price: priceFmt(p.price), up: pct(p.p_up, 0), down: pct(p.p_down, 0), flat: pct(p.p_flat, 0),
        conf: pct(p.confidence, 0), q: pct(p.quality_score, 0), why: sig ? t("notify.passed") : reasonCodes(p).map(codeLabel).join(", ") }));
    }
  }
  function notifyOutages() {
    if (!S.settings.notifyOutages || !server()) return;
    const pr = (S.server.probes && S.server.probes.probes) || {};
    for (const [ex, p] of Object.entries(pr)) {
      const ok = !!(p.rest && p.rest.ok && p.ws && p.ws.verified_live);
      if (ex in S.prevSourceOk && S.prevSourceOk[ex] !== ok) notify(ok ? t("notify.source_back", { ex }) : t("notify.source_down", { ex }), ok ? t("notify.source_back_body") : (p.rest && p.rest.error) || (p.ws && p.ws.error) || "");
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
    for (const x of higherTfs(tf)) { const c = closed(S.candles[sym] && S.candles[sym][x]); if (c.length) out[x] = limit ? c.slice(-(x === tf ? limit : Math.ceil(limit / 2))) : c; }
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
    setPhase(t("phase.history"));
    S.primary = await detectPrimary();
    if (!S.primary) { setPhase(""); render(); return false; }
    const tfs = [...new Set(S.settings.predictTfs.flatMap(higherTfs))];
    for (const sym of SYMBOLS) {
      S.candles[sym] = S.candles[sym] || {};
      for (const tf of tfs) {
        setPhase(t("phase.history_item", { sym, tf, ex: S.primary }));
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
      folds: 3, minTrain: 600, threshold: s.threshold, minEdge: s.minEdge, baselinePBetter: s.baselinePBetter,
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
      if (rows < 1000) throw new Error(t("local.need_rows", { rows, tf: j.tf }));
      if (j.kind === "baseline") {
        const r = await runJob("baseline", base), version = newVersion(j.sym, j.tf);
        await Store.set(`model:${version}`, r.model); S.models[version] = r.model; await Store.set(`oos:${version}`, r.oos);
        const old = production(j.key); if (old) old.status = "archived";
        S.registry.versions.unshift({ version, key: j.key, status: "production", created: Date.now(), feature_version: A.FEATURE_VERSION,
          train_start: r.train_start_ts, train_end: r.train_end_ts, n_train: r.n_train, metrics: r.metrics, reason: old ? "baseline retrained" : "initial baseline", train_sec: (Date.now() - t0) / 1000 });
        S.registry.production[j.key] = version; await Store.set("registry", S.registry);
        toast(t("toast.model_ready", { sym: j.sym, tf: j.tf }));
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
    const a = (Date.now() - (bk.recv || 0)) / 1000; if (a > 30) return null;
    return { ...bk.metrics(10), ageSec: a, exchange: ex };
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
    if (q.issues.some((i) => i.code === "stale_candles" || i.code === "no_candles")) reasons.push("stale_data");
    if (regime === "abnormal") reasons.push("abnormal_market");
    if (regime === "low_liquidity") reasons.push("low_liquidity");
    const missing = x.filter((v) => !isNum(v)).length / Math.max(1, x.length);
    if (missing > 0.2) reasons.push("insufficient_history");
    const bt = prod.metrics && prod.metrics.baseline_test;
    if (!(bt && bt.passed)) reasons.push("model_not_better_than_baseline");
    const obt = (prod.metrics && prod.metrics.backtest) || {};
    if (g.signal !== "NO TRADE" && !((obt.trades || 0) >= 30 && (obt.avg_net_bps || 0) > 0)) reasons.push("no_edge_after_costs");
    if (g.signal !== "NO TRADE" && nf.news_relevance >= 0.5 && ((g.signal === "UP" && nf.news_impact <= -0.35) || (g.signal === "DOWN" && nf.news_impact >= 0.35))) reasons.push("news_conflict");
    const rt = 2 * s.feeBps + 2 * s.slippageBps + Math.max(s.spreadBps, (book && book.spreadBps) || 0), btm = (prod.metrics && prod.metrics.backtest) || {};
    const rec = { id, prediction_id: id, created_ms: Date.now(), candle_ts: last.open_ts, target_ts: last.open_ts + s.horizon * A.TF_MS[tf], symbol: sym, exchange: S.primary,
      timeframe: tf, horizon_bars: s.horizon, price: last.close, prediction: reasons.length ? "NO TRADE" : g.signal, model_direction: g.modelDir, p_up: pUp, p_down: pDown, p_flat: pFlat,
      confidence: g.confidence, label_threshold: A.labelThreshold(row.atr_pct, s.horizon, s.atrMult, 2 * s.feeBps + 2 * s.slippageBps + s.spreadBps),
      gate_reasons: reasons, regime, quality_score: q.score, news_impact: nf.news_impact, model_version: prod.version, features_version: A.FEATURE_VERSION,
      gate: { confidence: g.confidence, threshold: s.threshold, edge: Math.abs(pUp - pDown), min_edge: s.minEdge, p_flat: pFlat, quality_score: q.score,
        quality_issues: q.issues.map((i) => `${i.code}: ${i.detail}`), decision_delay_sec: Math.round((Date.now() - last.open_ts - A.TF_MS[tf]) / 1000),
        missing_feature_share: missing, baseline: bt || null, check_ts: last.open_ts + (s.horizon + 1) * A.TF_MS[tf],
        costs: { round_trip_bps: rt, fee_bps_per_side: s.feeBps, slippage_bps_per_side: s.slippageBps, spread_bps: Math.max(s.spreadBps, (book && book.spreadBps) || 0),
          oos_signals: btm.trades, oos_avg_net_bps: btm.avg_net_bps, oos_win_rate: btm.win_rate } },
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
  function allLedger() {
    if (server()) return S.serverLedger;
    return [...S.ledger.values()];
  }
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
      model: p ? { production: { version: p.version, created_ms: p.created, n_train: p.n_train, reason: p.reason, metrics: p.metrics }, baseline_test: p.metrics && p.metrics.baseline_test,
        versions: S.registry.versions.filter((v) => v.key === key).map((v) => ({ ...v, created_ms: v.created, log_loss: v.metrics && v.metrics.log_loss, accuracy: v.metrics && v.metrics.accuracy })), learning_status: S.learning.status[key] } : null,
      ticker: tk && isNum(tk.last) ? { last: tk.last, change_24h_pct: tk.changePct, ts_ms: tk.ts } : null, primary: S.primary,
      candles: S.candles[sym] && S.candles[sym][tf] ? S.candles[sym][tf].slice(-450) : null };
  }
  function rowsToCandles(rows) { return rows.open_ts.map((x, i) => ({ open_ts: x, open: rows.open[i], high: rows.high[i], low: rows.low[i], close: rows.close[i], volume: rows.volume[i], closed: rows.closed[i], cvd: rows.cvd ? rows.cvd[i] : null })); }

  // ------------------------------------------------------------- status
  function browserLive() { const o = {}; for (const ex of EXCHANGES) o[ex] = !!(S.streams[ex] && S.streams[ex].status().state === "live"); return o; }
  function statuses() {
    const now = Date.now();
    if (S.mode === "server") return ST.computeServer(server() || null, now, { browserLive: browserLive(), fetchError: S.serverErr });
    const v = view(), base = closed(v.candles || []), lastClose = base.length ? base[base.length - 1].open_ts + A.TF_MS[S.tf] : null;
    const preds = [...S.ledger.values()].map((r) => r.created_ms);
    return ST.computeLocal({ exchanges: EXCHANGES, streams: Object.fromEntries(EXCHANGES.map((ex) => [ex, S.streams[ex] ? S.streams[ex].status().state : null])),
      primary: S.primary, lastPoll: S.lastPoll, lastClose, tfMs: A.TF_MS[S.tf], hasCandles: base.length > 0, dbOk: S.dbOk, ledgerSize: S.ledger.size,
      training: !!S.training, modelReady: !!production(v.key), newsLive: S.newsStatus.state === "LIVE", newsTs: S.newsStatus.ts, newsAnalyzer: S.newsStatus.analyzer,
      lastPrediction: preds.length ? Math.max(...preds) : null, started: S.started }, now);
  }
  function statusDetail(s) {
    const p = { ...(s.params || {}) };
    if (["binance", "bybit", "okx"].includes(s.id)) {
      const bits = [];
      if (s.code === "browser_live") bits.push(t("st.ex.browser_live"));
      if (s.code !== "not_checked" && S.mode === "server") {
        bits.push("REST: " + (s.restOk ? "OK" : `${s.restKind || "—"}${s.restKind ? " (" + kindText(s.restKind) + ")" : ""}`));
        bits.push("WS: " + (s.wsOk ? "OK" : `${s.wsKind || "—"}${s.wsKind ? " (" + kindText(s.wsKind) + ")" : ""}`));
        if (isNum(s.restErrorRate) || isNum(s.wsErrorRate)) bits.push(t("st.ex.error_rate", { rest: pct(s.restErrorRate, 0), ws: pct(s.wsErrorRate, 0) }));
        if (s.primary) bits.push(t("st.ex.primary"));
      } else if (s.code !== "browser_live") bits.push(I.has("st.code." + s.code) ? t("st.code." + s.code) : s.code);
      return bits.join(" · ");
    }
    const key = `st.${s.id}.${s.code}`;
    return I.has(key) ? t(key, p) : s.code;
  }
  function primaryHint() {
    const errs = Object.entries(S.historyErrors).filter(([, v]) => v).map(([k, v]) => `${k}: ${v}`).join("; ");
    return S.started ? t("local.no_exchange", { errors: errs }) : t("local.press_start");
  }

  // ------------------------------------------------------------- rendering
  function setPhase(x) { const p = $("phase"); p.textContent = x; p.hidden = !x; }
  function render() { if (render.q) return; render.q = requestAnimationFrame(() => { render.q = null; try { renderAll(); } catch (e) { console.error(e); } }); }
  function renderAll() {
    const v = view(), sv = server();
    $("welcome").hidden = S.mode !== "local" || S.started;
    $("modeNote").textContent = S.mode === "server"
      ? (sv ? t((sv.health && sv.health.backend_mode) === "continuous" ? "mode.server_continuous" : "mode.server_scheduled", { ago: ago(sv.generated_ms) }) : S.serverErr || t("mode.connecting"))
      : S.mode === "local" ? t("mode.local") : "";
    $("assetName").textContent = `${S.symbol} · ${S.tf}`;
    const st = S.streams[v.primary], lp = st && st.lastPrice[S.symbol], lpFresh = st && Date.now() - (st.lastTradeTs[S.symbol] || 0) < 30000;
    const base = closed(v.candles || []);
    let price = null, src = "";
    if (lp && lpFresh) { price = lp; src = t("price.src_live", { ex: v.primary }); }
    else if (v.ticker && isNum(v.ticker.last)) { price = v.ticker.last; src = t(sv ? "price.src_backend" : "price.src_ticker", { ex: v.primary, ago: ago(v.ticker.ts_ms) }); }
    else if (base.length) { price = base[base.length - 1].close; src = t("price.src_candle"); }
    $("price").textContent = price ? "$" + priceFmt(price) : t("common.unavailable"); $("price").classList.toggle("unavail", !price);
    $("priceSource").textContent = price ? t("price.source", { src }) : S.mode === "server" ? (S.serverErr || t("price.no_backend_price")) : primaryHint();
    const ch = v.ticker && v.ticker.change_24h_pct; $("change").textContent = isNum(ch) ? `${ch >= 0 ? "+" : ""}${num(ch, 2)}% ${t("price.24h")}` : ""; $("change").dataset.dir = ch > 0 ? "up" : ch < 0 ? "down" : "";
    renderPrediction(v); renderStatus(); renderExchanges(v); renderModel(v); renderChanges(v); renderLedger(); renderNews(); drawChart(v); renderProbe(); renderNotifyInfo();
    renderForecasts();
  }
  function renderPrediction(v) {
    const p = v.latest, tfms = A.TF_MS[S.tf];
    const fresh = p && Date.now() - (p.candle_ts + tfms) < 2 * tfms + (server() ? 30 * 60000 : 0);
    $("threshold").style.left = `calc(${(v.threshold * 100).toFixed(1)}% - 1px)`; $("threshold").dataset.label = t("pred.threshold_label", { v: Math.round(v.threshold * 100) });
    $("horizon").textContent = t("pred.horizon", { n: v.horizon, tf: S.tf });
    const sig = $("signal");
    const notReady = !(v.model && v.model.production);
    $("predDetails").replaceChildren();
    if (!p || !fresh) {
      sig.textContent = notReady ? (S.training && S.training.key === v.key ? t("pred.training") : "MODEL NOT READY") : "NO TRADE"; sig.dataset.state = notReady ? "none" : "NO TRADE";
      ["segDown", "segFlat", "segUp"].forEach((i) => ($(i).style.width = "0")); ["pDown", "pFlat", "pUp", "confidence", "regime", "quality", "newsImpact"].forEach((i) => ($(i).textContent = "—"));
      $("reasons").textContent = notReady ? `MODEL NOT READY — ${t(S.mode === "server" ? "pred.not_ready_server" : "pred.not_ready_local")}`
        : p ? `STALE DATA — ${t("pred.too_old", { when: when(p.candle_ts) })}` : t("pred.none_yet");
      $("predMeta").textContent = "";
      return;
    }
    sig.textContent = p.prediction; sig.dataset.state = p.prediction;
    $("segDown").style.width = (p.p_down * 100).toFixed(2) + "%"; $("segFlat").style.width = (p.p_flat * 100).toFixed(2) + "%"; $("segUp").style.width = (p.p_up * 100).toFixed(2) + "%";
    $("pDown").textContent = pct(p.p_down); $("pFlat").textContent = pct(p.p_flat); $("pUp").textContent = pct(p.p_up); $("confidence").textContent = pct(p.confidence);
    $("regime").textContent = regimeText(p.regime);
    const qs = p.quality_score ?? (p.gate && p.gate.quality_score);
    $("quality").textContent = isNum(qs) ? `${t(qs >= 0.9 ? "quality.good" : qs >= 0.7 ? "quality.fair" : "quality.bad")} (${Math.round(qs * 100)}%)` : "—";
    $("newsImpact").textContent = isNum(p.news_impact) ? (p.news_impact > 0 ? "+" : "") + num(p.news_impact, 2) : "—";
    $("reasons").textContent = p.prediction === "NO TRADE" ? t("pred.no_trade_because", { reasons: reasonDetail(p) }) : t("pred.passed", { dir: dirText(p.model_direction) });
    const g = p.gate || {};
    $("predMeta").textContent = t("pred.meta", { candle: when(p.candle_ts), price: priceFmt(p.price), ex: p.exchange,
      decided: isNum(g.decision_delay_sec) ? (g.decision_delay_sec < 120 ? t("pred.decided_after_sec", { sec: Math.round(g.decision_delay_sec) }) : t("pred.decided_after", { min: Math.round(g.decision_delay_sec / 60) })) + (g.late ? ` (${t("pred.late")})` : "") : when(p.created_ms),
      model: p.model_version });
    $("predDetails").replaceChildren(...detailBlocks(p, v.horizon));
    const b = liveBook(v.primary, S.symbol) || (v.serverBook && { spreadBps: v.serverBook.spread_bps, imbalance: v.serverBook.imbalance, ageSec: (Date.now() - v.serverBook.recv_ms) / 1000 });
    $("spread").textContent = b && isNum(b.spreadBps) ? `${num(b.spreadBps, b.spreadBps < 0.1 ? 4 : 2)} ${t("unit.bps")}${b.ageSec > 30 ? " (" + ago(Date.now() - b.ageSec * 1000) + ")" : ""}` : t("common.unavailable");
    $("imbalance").textContent = b && isNum(b.imbalance) ? (b.imbalance > 0 ? "+" : "") + num(b.imbalance, 2) : "—";
  }
  // details shared by the readout and the prediction dialog: direction vs after-cost result, factors, check time, outcome
  function detailBlocks(p, horizon) {
    const g = p.gate || {}, c = g.costs || {}, out = [], thrBps = isNum(p.label_threshold) ? p.label_threshold * 1e4 : null;
    const rows = [];
    rows.push([t("det.direction"), t("det.direction_val", { up: pct(p.p_up), down: pct(p.p_down), flat: pct(p.p_flat), thr: num(thrBps, 1) })]);
    if (isNum(c.round_trip_bps)) rows.push([t("det.costs"), t("det.costs_val", { rt: num(c.round_trip_bps, 1), fee: num(c.fee_bps_per_side, 1), slip: num(c.slippage_bps_per_side, 1), spread: num(c.spread_bps, 1) })]);
    rows.push([t("det.after_costs"), isNum(c.oos_signals) && c.oos_signals > 0 ? t("det.after_costs_val", { n: c.oos_signals, net: num(c.oos_avg_net_bps, 1), win: pct(c.oos_win_rate, 0) }) : t("det.after_costs_none")]);
    const bt = g.baseline;
    rows.push([t("det.baseline"), bt ? t(bt.passed ? "det.baseline_pass" : "det.baseline_fail", { gain: num(bt.gain, 4), lo: bt.gain_ci95 ? num(bt.gain_ci95[0], 4) : "—", hi: bt.gain_ci95 ? num(bt.gain_ci95[1], 4) : "—", p: pct(bt.p_better, 0), req: pct(bt.p_required, 0) }) : t("det.baseline_unknown")]);
    rows.push([t("det.freshness"), t("det.freshness_val", { q: pct(p.quality_score ?? g.quality_score, 0), delay: isNum(g.decision_delay_sec) ? Math.round(g.decision_delay_sec) : "—", issues: (g.quality_issues || []).map(issueText).join("; ") || t("det.no_issues") })]);
    const check = g.check_ts || (p.target_ts && p.target_ts + A.TF_MS[p.timeframe]);
    rows.push([t("det.check_time"), `${when(check)}${check > Date.now() ? " (" + t("time.in", { v: inDur((check - Date.now()) / 1000) }) + ")" : ""}`]);
    rows.push([t("det.outcome"), p.actual_direction ? t("det.outcome_val", { dir: dirText(p.actual_direction), ret: pct(p.actual_return, 2), price: priceFmt(p.actual_price), cls: t("cls." + String(p.error_class || "").replace("_late", "")) }) : t("det.pending")]);
    rows.push([t("det.model"), `${p.model_version} · ${p.features_version || ""}`]);
    out.push(el("dl", { class: "kv detail-kv" }, ...rows.flatMap(([k, val]) => [el("dt", { text: k }), el("dd", { text: val })])));
    const f = g.factors || [];
    if (f.length) {
      out.push(el("p", { class: "small muted", text: t("det.factors_intro", { dir: dirText(p.model_direction) }) }));
      out.push(el("ul", { class: "factors" }, ...f.map((x) => el("li", {}, el("code", { text: x.feature }), el("span", { class: "muted", text: ` ${featureGroup(x.feature)} · ${t("det.value")} ${num(x.value, 4)} (${t("det.median")} ${num(x.median, 4)}) → ` }),
        el("b", { class: x.effect > 0 ? "dir-up" : "dir-down", text: `${x.effect > 0 ? "+" : ""}${num(x.effect * 100, 1)} ${t("unit.pp")}` })))));
    }
    return out;
  }
  function featureGroup(f) {
    const n = f.replace(/^(15m|1h|4h|1d)_/, "");
    const g = /rsi|macd|roc|ret_|stoch|di_diff/.test(n) ? "momentum" : /ema|adx|trend|close_vs/.test(n) ? "trend" : /atr|bb_|vol_z|range|volat/.test(n) ? "volatility"
      : /vol|taker|cvd|flow|obv/.test(n) ? "volume" : /regime/.test(n) ? "regime" : /news/.test(n) ? "news" : "other";
    return `${t("fg." + g)}${/^(1h|4h|1d)_/.test(f) ? " · " + t("fg.higher_tf", { tf: f.split("_")[0] }) : ""}`;
  }
  function renderStatus() {
    const st = statuses();
    $("statusGrid").replaceChildren(...st.flatMap((s) => {
      const tone = ST.TONE[s.state] === "good" && s.partial ? "info" : ST.TONE[s.state];
      return [el("dt", { text: t("st.name." + s.id) }), el("dd", { class: "st-" + tone },
        el("span", { class: "badge", text: s.state }), el("span", { class: "detail", text: [statusDetail(s), s.ts ? ago(s.ts) : ""].filter(Boolean).join(" · ") }))];
    }));
    const data = st.filter((s) => ["binance", "bybit", "okx", "history", "websocket"].includes(s.id));
    const order = ["LIVE", "DELAYED", "STALE", "ERROR", "OFFLINE"], best = order.find((o) => data.some((s) => s.state === o)) || "OFFLINE";
    $("liveDot").className = "live-dot st-" + ST.TONE[best]; $("liveText").textContent = t("top.data_state", { state: best });
  }
  function renderExchanges(v) {
    const sv = server(), pr = (sv && sv.probes && sv.probes.probes) || {}, stats = (sv && sv.source_stats) || {}, rows = [];
    for (const ex of EXCHANGES) {
      const s = S.streams[ex], b = liveBook(ex, S.symbol), st = s ? s.status() : null, p = pr[ex], es = stats[ex] || {};
      const rest = p ? (p.rest.ok ? "OK" : p.rest.partial ? "partial" : (p.rest.kind || "error")) : "—";
      const ws = p ? (p.ws.verified_live ? `OK (${p.ws.events})` : p.ws.kind || "no_data") : "—";
      const rate = (ch) => (es[ch] && es[ch]["24h"] ? `${pct(es[ch]["24h"].error_rate, 0)} / ${es[ch]["24h"].checks}` : "—");
      const raw = ((st && st.state !== "live" && st.error) || (p && !p.rest.ok && p.rest.error) || (p && !p.ws.verified_live && p.ws.error) || "").slice(0, 120);
      const kind = (p && !p.rest.ok && p.rest.kind) || (p && !p.ws.verified_live && p.ws.kind) || (st && st.state !== "live" && st.state);
      rows.push(el("tr", {}, el("td", { "data-label": t("ex.col.exchange"), text: ex + (ex === v.primary ? ` (${t("ex.model")})` : "") }),
        el("td", { class: "num", "data-label": t("ex.col.price"), text: s && s.lastPrice[S.symbol] && Date.now() - (s.lastTradeTs[S.symbol] || 0) < 60000 ? priceFmt(s.lastPrice[S.symbol]) : "—" }),
        el("td", { class: "num", "data-label": t("ex.col.spread"), text: b ? num(b.spreadBps, 2) : "—" }),
        el("td", { class: st && st.state === "live" ? "st-good" : "st-bad", "data-label": t("ex.col.browser"), text: st ? (st.state === "live" ? "LIVE" : st.state.toUpperCase()) : t("ex.off") }),
        el("td", { class: p && p.rest.ok ? "st-good" : p ? "st-bad" : "", "data-label": t("ex.col.rest"), text: rest }),
        el("td", { class: p && p.ws.verified_live ? "st-good" : p ? "st-bad" : "", "data-label": t("ex.col.ws"), text: ws }),
        el("td", { class: "num", "data-label": t("ex.col.err_rest"), text: rate("rest") }), el("td", { class: "num", "data-label": t("ex.col.err_ws"), text: rate("ws") }),
        el("td", { class: "small wrap", "data-label": t("ex.col.error") }, kind && kind !== "live" ? el("span", { text: kindText(kind) + (raw ? " — " : "") }) : null, raw ? el("code", { class: "raw", text: raw }) : null)));
    }
    const head = [t("ex.col.exchange"), t("ex.col.price"), t("ex.col.spread"), t("ex.col.browser"), sv ? t("ex.col.rest_at", { ago: ago(sv.probes && sv.probes.ts_ms) }) : t("ex.col.rest"), t("ex.col.ws"), t("ex.col.err_rest"), t("ex.col.err_ws"), t("ex.col.error")];
    $("exTable").replaceChildren(el("thead", {}, el("tr", {}, ...head.map((h) => el("th", { text: h })))), el("tbody", {}, ...rows));
  }
  function kv(target, pairs) { target.replaceChildren(...pairs.flatMap(([k, val]) => [el("dt", { text: k }), el("dd", { text: val == null ? "—" : String(val) })])); }
  function renderModel(v) {
    const m = v.model, p = m && m.production;
    if (!p) { kv($("modelKv"), [[t("model.status"), "MODEL NOT READY"], [t("model.learning"), m && m.learning_status ? learningText(m.learning_status) : t("model.waiting_history")]]); $("whyNoTrade").replaceChildren(); }
    else {
      const mt = p.metrics || {}, b = mt.baseline_prior || {}, sg = mt.signals || {}, ls = m.learning_status, lr = m.ledger_review, bt = m.baseline_test || mt.baseline_test;
      kv($("modelKv"), [[t("model.version"), p.version], [t("model.status"), `production · ${p.reason || ""}`],
        [t("model.vs_baseline"), bt && isNum(bt.gain) ? t(bt.passed ? "det.baseline_pass" : "det.baseline_fail", { gain: num(bt.gain, 4), lo: bt.gain_ci95 ? num(bt.gain_ci95[0], 4) : "—", hi: bt.gain_ci95 ? num(bt.gain_ci95[1], 4) : "—", p: pct(bt.p_better, 0), req: pct(bt.p_required, 0) }) : t("det.baseline_unknown")],
        [t("model.logloss"), t("model.vs_naive", { m: num(mt.log_loss, 4), n: num(b.log_loss, 4) })], [t("model.brier"), num(mt.brier, 4)],
        [t("model.accuracy"), t("model.vs_naive", { m: pct(mt.accuracy), n: pct(b.accuracy) })], [t("model.ece"), num(mt.ece, 3)],
        [t("model.precision_recall"), `${num(mt.precision_macro, 3)} / ${num(mt.recall_macro, 3)}`],
        [t("model.signals"), isNum(sg.coverage) ? t("model.signals_val", { n: sg.signals, cov: pct(sg.coverage), hit: pct(sg.hit_rate) }) : "—"],
        [t("model.validation"), mt.method || "—"], [t("model.last_training"), t("model.rows", { when: when(p.created_ms), n: p.n_train })],
        [t("model.last_validation"), ls ? `${learningText(ls)} · ${ago(ls.ts_ms || ls.ts)}` : t("model.not_yet")],
        [t("model.live"), lr && lr.resolved ? t("model.live_val", { n: lr.resolved, s: lr.signals, hit: pct(lr.signal_hit_rate) }) : t("model.no_resolved")],
        [t("model.backtest"), mt.backtest && mt.backtest.trades ? t("model.backtest_val", { n: mt.backtest.trades, net: num(mt.backtest.avg_net_bps, 1) }) : t("model.no_signal_passed")],
        [t("model.gaps"), m.gaps ? t("model.gaps_val", { d: m.gaps.last_24h, n: m.gaps.total }) : "—"]]);
      renderWhy(mt, v);
      renderRegimes(mt);
    }
    const vs = (m && m.versions) || [];
    $("versions").replaceChildren(el("thead", {}, el("tr", {}, ...[t("ver.version"), t("ver.status"), t("ver.created"), t("ver.logloss"), t("ver.naive"), t("ver.accuracy"), t("ver.gain"), "P(better)", t("ver.reason")].map((h) => el("th", { text: h })))),
      el("tbody", {}, ...vs.slice(0, 15).map((x) => el("tr", {}, el("td", { "data-label": t("ver.version"), text: x.version }), el("td", { "data-label": t("ver.status"), text: x.status }), el("td", { "data-label": t("ver.created"), text: when(x.created_ms) }),
        el("td", { class: "num", "data-label": t("ver.logloss"), text: num(x.log_loss, 4) }), el("td", { class: "num", "data-label": t("ver.naive"), text: num(x.naive_log_loss, 4) }),
        el("td", { class: "num", "data-label": t("ver.accuracy"), text: pct(x.accuracy) }),
        el("td", { class: "num", "data-label": t("ver.gain"), text: x.comparison ? num(x.comparison.logloss_gain, 4) : "—" }), el("td", { class: "num", "data-label": "P(better)", text: x.comparison ? pct(x.comparison.bootstrap_p_better, 0) : "—" }),
        el("td", { class: "wrap", "data-label": t("ver.reason"), text: x.reason || "" })))));
  }
  function learningText(ls) {
    const k = "learn." + (ls.status || "unknown");
    let s = I.has(k) ? t(k) : ls.status;
    if (ls.new_labels != null) s += ` (${t("learn.labels", { n: ls.new_labels, need: ls.required })})`;
    if (ls.reason) s += ` · ${ls.reason}`;
    if (ls.error) s += ` · ${ls.error}`;
    return s;
  }
  function renderRegimes(mt) {
    const by = mt.by_regime || {};
    const ent = Object.entries(by);
    $("regimeTable").replaceChildren(...(ent.length ? [el("thead", {}, el("tr", {}, ...[t("reg.regime"), t("reg.n"), t("ver.logloss"), t("ver.accuracy"), t("reg.signals"), t("reg.hit")].map((h) => el("th", { text: h })))),
      el("tbody", {}, ...ent.map(([r, x]) => el("tr", {}, el("td", { "data-label": t("reg.regime"), text: regimeText(r) }), el("td", { class: "num", "data-label": t("reg.n"), text: x.n }),
        el("td", { class: "num", "data-label": t("ver.logloss"), text: num(x.log_loss, 4) }), el("td", { class: "num", "data-label": t("ver.accuracy"), text: pct(x.accuracy) }),
        el("td", { class: "num", "data-label": t("reg.signals"), text: x.signals ? x.signals.signals : "—" }), el("td", { class: "num", "data-label": t("reg.hit"), text: x.signals ? pct(x.signals.hit_rate) : "—" }))))] : []));
  }
  function renderWhy(mt, v) {
    const g = mt.gate_diagnostics, ld = mt.label_diagnostics, out = [];
    if (g) {
      const f = g.first_failing_condition, n = g.n || 1;
      out.push(el("p", { class: "small", text: t("why.summary", { n: g.n, passed: f.passed, share: pct(f.passed / n), thr: pct(g.threshold, 0), c: f.confidence_below_threshold, e: f.edge_below_min_edge, fl: f.flat_more_likely,
        p50: pct(g.confidence_quantiles.p50), p90: pct(g.confidence_quantiles.p90), max: pct(g.confidence_quantiles.max) }) }));
      const cal = g.per_class_calibration;
      if (cal) out.push(el("p", { class: "small", text: t("why.calibration", { list: Object.entries(cal).map(([k, c]) => `${dirText(k)} ${pct(c.mean_predicted)} / ${pct(c.actual_frequency)}`).join(", ") }) }));
      if (ld) out.push(el("p", { class: "small", text: t("why.labels", { thr: num(ld.median_label_threshold_bps, 1), med: num(ld.median_abs_forward_return_bps, 1), share: pct(ld.share_moves_above_threshold) }) }));
      const sw = g.threshold_sweep_informational || [];
      out.push(el("div", { class: "table-scroll" }, el("table", { class: "tbl cards" }, el("thead", {}, el("tr", {}, ...[t("why.col.threshold"), t("why.col.signals"), t("why.col.coverage"), t("why.col.hit"), t("why.col.gross")].map((h) => el("th", { text: h })))),
        el("tbody", {}, ...sw.map((r) => el("tr", { class: Math.abs(r.threshold - g.threshold) < 1e-9 ? "current" : "" }, el("td", { "data-label": t("why.col.threshold"), text: pct(r.threshold, 0) + (Math.abs(r.threshold - g.threshold) < 1e-9 ? ` (${t("why.current")})` : "") }),
          el("td", { class: "num", "data-label": t("why.col.signals"), text: r.signals }), el("td", { class: "num", "data-label": t("why.col.coverage"), text: pct(r.coverage) }), el("td", { class: "num", "data-label": t("why.col.hit"), text: pct(r.hit_rate) }), el("td", { class: "num", "data-label": t("why.col.gross"), text: num(r.avg_gross_bps, 1) })))))));
      out.push(el("p", { class: "small muted", text: t("why.note") }));
    }
    const sv = server(), reasons = sv && sv.ledger_summary && sv.ledger_summary.no_trade_reasons;
    if (reasons && Object.keys(reasons).length) {
      const byCode = {}; for (const [k, c] of Object.entries(reasons)) { const code = CODE_OF[k] || k; byCode[code] = (byCode[code] || 0) + c; }
      out.push(el("p", { class: "small", text: t("why.live_reasons", { list: Object.entries(byCode).map(([k, c]) => `${codeLabel(k)}: ${c}`).join(", ") }) }));
    }
    $("whyNoTrade").replaceChildren(...out);
  }
  // model change report: what changed between versions, overall and per market regime
  function renderChanges(v) {
    const sv = server(), box = $("changes");
    const key = v.key, events = sv ? (sv.model_changes || []).filter((e) => !e.model_key || e.model_key === key) : [];
    const vs = (v.model && v.model.versions) || [];
    const items = [];
    for (const x of vs.filter((y) => y.comparison).slice(0, 8)) {
      const c = x.comparison, checks = c.checks || {};
      const failed = Object.entries(checks).filter(([, ok]) => !ok).map(([k]) => t("chk." + k));
      const li = el("li", {}, el("div", {}, el("b", { text: `${when(x.created_ms)} · ${t("chg.status." + (x.status || "unknown"))}` }), el("span", { class: "muted", text: ` ${x.version}` })),
        el("div", { class: "small", text: t("chg.summary", { n: c.n_holdout, before: num(c.logloss_production, 4), after: num(c.logloss_challenger, 4), gain: num(c.logloss_gain, 4), p: pct(c.bootstrap_p_better, 0),
          bb: num(c.brier_production, 4), ba: num(c.brier_challenger, 4) }) }),
        c.halves_gain ? el("div", { class: "small", text: t("chg.halves", { a: num(c.halves_gain[0], 4), b: num(c.halves_gain[1], 4) }) }) : null,
        failed.length ? el("div", { class: "small st-bad", text: t("chg.rejected_because", { list: failed.join(", ") }) }) : el("div", { class: "small st-good", text: t("chg.all_checks") }));
      const br = c.by_regime || {};
      if (Object.keys(br).length) li.append(el("div", { class: "table-scroll" }, el("table", { class: "tbl cards" }, el("thead", {}, el("tr", {}, ...[t("reg.regime"), t("reg.n"), t("chg.before"), t("chg.after"), t("ver.gain")].map((h) => el("th", { text: h })))),
        el("tbody", {}, ...Object.entries(br).map(([r, z]) => el("tr", {}, el("td", { "data-label": t("reg.regime"), text: regimeText(r) }), el("td", { class: "num", "data-label": t("reg.n"), text: z.n }),
          el("td", { class: "num", "data-label": t("chg.before"), text: num(z.logloss_production, 4) }), el("td", { class: "num", "data-label": t("chg.after"), text: num(z.logloss_challenger, 4) }),
          el("td", { class: "num " + (z.gain > 0 ? "st-good" : "st-bad"), "data-label": t("ver.gain"), text: num(z.gain, 4) })))))));
      items.push(li);
    }
    for (const e of events.filter((x) => x.event_type === "model_rollback" || x.event_type === "database_recovery" || x.event_type === "operator_command").slice(0, 6)) {
      const pl = e.payload || {};
      items.push(el("li", {}, el("b", { text: `${when(e.ts_ms)} · ${t("chg.ev." + e.event_type)}` }),
        el("div", { class: "small", text: e.event_type === "model_rollback" ? t("chg.rollback", { from: (pl.from || {}).version || "—", to: (pl.to || {}).version || "—", reason: pl.reason || "" })
          : e.event_type === "database_recovery" ? t("chg.recovery", { backup: pl.backup || "—", problem: pl.problem || "" }) : `${pl.action || ""} ${pl.id || ""}` })));
    }
    box.replaceChildren(...(items.length ? items : [el("li", { class: "muted", text: t("chg.none") })]));
  }
  function ledgerRows() {
    const f = S.filter, now = Date.now();
    let rows = allLedger().slice();
    const sym = f.symbol === "current" ? S.symbol : f.symbol, tf = f.tf === "current" ? S.tf : f.tf;
    if (sym !== "all") rows = rows.filter((r) => r.symbol === sym);
    if (tf !== "all") rows = rows.filter((r) => r.timeframe === tf);
    if (f.model !== "all") rows = rows.filter((r) => r.model_version === f.model);
    const span = { "24h": 86400e3, "7d": 7 * 86400e3, "30d": 30 * 86400e3 }[f.period];
    if (span) rows = rows.filter((r) => now - r.candle_ts <= span);
    if (f.type === "signals") rows = rows.filter((r) => r.prediction !== "NO TRADE"); else if (f.type === "resolved") rows = rows.filter((r) => r.resolved_ms); else if (f.type === "notrade") rows = rows.filter((r) => r.prediction === "NO TRADE");
    return rows.sort((a, b) => b.candle_ts - a.candle_ts || a.symbol.localeCompare(b.symbol));
  }
  function fillModelFilter() {
    const sel = $("fModel"), cur = S.filter.model, models = [...new Set(allLedger().map((r) => r.model_version))].sort().reverse();
    const opts = [el("option", { value: "all", text: t("filter.all_models") }), ...models.map((m) => el("option", { value: m, text: m }))];
    if (sel.options.length !== opts.length || [...sel.options].some((o, i) => o.value !== opts[i].value)) { sel.replaceChildren(...opts); sel.value = models.includes(cur) ? cur : "all"; }
    else sel.options[0].textContent = t("filter.all_models");
  }
  function renderLedger() {
    fillModelFilter();
    const rows = ledgerRows(), all = allLedger();
    const sig = rows.filter((r) => r.resolved_ms && r.prediction !== "NO TRADE"), hits = sig.filter((r) => r.result === "correct").length;
    const resolved = rows.filter((r) => r.resolved_ms), span = rows.length ? [Math.min(...rows.map((r) => r.candle_ts)), Math.max(...rows.map((r) => r.candle_ts))] : null;
    $("ledgerStats").textContent = t("ledger.stats", { n: rows.length, total: all.length, resolved: resolved.length, sig: sig.length, hit: sig.length ? pct(hits / sig.length) : "—",
      from: span ? when(span[0]) : "—", to: span ? when(span[1]) : "—" });
    const tb = $("ledger");
    if (!rows.length) { tb.replaceChildren(el("tbody", {}, el("tr", {}, el("td", { class: "muted", text: all.length ? t("ledger.nothing") : t("ledger.empty") })))); $("moreBtn").hidden = true; return; }
    const head = [t("col.candle"), t("col.asset"), t("col.price"), t("col.prediction"), t("col.up"), t("col.down"), t("col.flat"), t("col.conf"), t("col.quality"), t("col.regime"), t("col.check"), t("col.actual"), t("col.return"), t("col.result"), t("col.why")];
    tb.replaceChildren(el("thead", {}, el("tr", {}, ...head.map((h) => el("th", { text: h })))), el("tbody", {}, ...rows.slice(0, S.ledgerLimit).map((r) => {
      const g = r.gate || {}, check = g.check_ts || (r.target_ts + A.TF_MS[r.timeframe]);
      const tr = el("tr", { class: "clickable", tabindex: "0", role: "button", "aria-label": t("ledger.open_details") },
        el("td", { "data-label": t("col.candle"), text: when(r.candle_ts) }), el("td", { "data-label": t("col.asset"), text: `${r.symbol} ${r.timeframe} · ${r.exchange}` }),
        el("td", { class: "num", "data-label": t("col.price"), text: priceFmt(r.price) }), el("td", { class: "pred-" + r.prediction.replace(" ", ""), "data-label": t("col.prediction"), text: r.prediction }),
        el("td", { class: "num", "data-label": t("col.up"), text: pct(r.p_up) }), el("td", { class: "num", "data-label": t("col.down"), text: pct(r.p_down) }), el("td", { class: "num", "data-label": t("col.flat"), text: pct(r.p_flat) }),
        el("td", { class: "num", "data-label": t("col.conf"), text: pct(r.confidence) }), el("td", { class: "num", "data-label": t("col.quality"), text: pct(r.quality_score, 0) }),
        el("td", { "data-label": t("col.regime"), text: regimeText(r.regime) }), el("td", { "data-label": t("col.check"), text: clock(check) }),
        el("td", { "data-label": t("col.actual"), text: r.actual_direction ? dirText(r.actual_direction) : t("det.pending") }),
        el("td", { class: "num", "data-label": t("col.return"), text: isNum(r.actual_return) ? pct(r.actual_return, 2) : "—" }),
        el("td", { class: "res-" + (r.result || ""), "data-label": t("col.result"), text: r.error_class ? t("cls." + r.error_class.replace("_late", "")) + (r.error_class.endsWith("_late") ? ` (${t("cls.late")})` : "") : "—" }),
        el("td", { class: "small muted wrap", "data-label": t("col.why"), text: (r.prediction === "NO TRADE" ? reasonCodes(r).map(codeLabel).join(", ") : t("ledger.passed")) + ` · ${r.model_version}` }));
      tr.onclick = () => openPrediction(r); tr.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openPrediction(r); } };
      return tr;
    })));
    $("moreBtn").hidden = rows.length <= S.ledgerLimit;
  }
  // ------------------------------------------------------------- forward-looking forecasts per horizon
  const fcodes = (rs) => (rs || []).map((r) => codeLabel(CODE_OF[r] || r.toUpperCase()));
  const bps = (x, d = 1) => (isNum(x) ? `${x > 0 ? "+" : ""}${num(x, d)} ${t("unit.bps")}` : "—");
  const durLabel = (sec) => (sec < 60 ? `${sec} ${t("unit.s")}` : sec < 3600 ? t("time.min", { n: sec / 60 }) : sec < 86400 ? t("time.h", { n: sec / 3600 }) : t("unit.day"));
  function renderForecasts() {
    const sv = server(), tb = $("fcTable");
    if (!sv || !sv.forecasts) { $("fcNote").textContent = t(S.mode === "server" ? "fc.not_yet" : "fc.local"); tb.replaceChildren(); return; }
    const fc = sv.forecasts, now = Date.now();
    $("fcNote").textContent = t("fc.note", { ago: ago(sv.generated_ms) });
    const head = [t("fc.col.horizon"), t("fc.col.status"), t("fc.col.made"), t("fc.col.decision"), t("fc.col.probs"), t("fc.col.range"),
      t("fc.col.target"), t("fc.col.reason"), t("fc.col.checked"), t("fc.col.dir"), t("fc.col.mae"), t("fc.col.brier"), t("fc.col.cover"), t("fc.col.net")];
    const rows = fc.horizons.map((h) => {
      const key = `${S.symbol}|f${h.seconds}`, st = fc.stats[key] || {}, f = fc.latest[key], lv = st.live || {}, ho = st.holdout || {};
      const status = st.status || "not_trained";
      const statusText = t("fst." + status) + (status === "collecting" && st.training && st.training.n_independent_total != null ? ` (${st.training.n_independent_total})` : "");
      const made = f ? `${clock(f.created_ms)} · ${ago(f.created_ms)}` : (st.not_issued && st.not_issued.last ? t("fc.not_issued." + st.not_issued.last) : "—");
      const probs = f ? `↑${pct(f.p_up, 0)} ↓${pct(f.p_down, 0)} ·${pct(f.p_flat, 0)}` : "—";
      const range = f ? `${bps(f.q10_bps)} … ${bps(f.q90_bps)}` + (f.detail && f.detail.range_price ? ` (${priceFmt(f.detail.range_price[0])}–${priceFmt(f.detail.range_price[1])})` : "") : "—";
      const target = f ? `${clock(f.target_ts)}${f.target_ts > now ? " (" + t("time.in", { v: inDur((f.target_ts - now) / 1000) }) + ")" : ""}` : "—";
      const brier = isNum(lv.brier_gain) ? `${num(lv.brier_gain, 4)} [${num(lv.brier_gain_ci95[0], 4)}…${num(lv.brier_gain_ci95[1], 4)}]` : "—";
      const tr = el("tr", { class: "clickable", tabindex: "0", role: "button", "aria-label": t("fc.open") },
        el("td", { "data-label": t("fc.col.horizon"), text: h.label }),
        el("td", { class: "fst-" + status, "data-label": t("fc.col.status"), text: statusText }),
        el("td", { "data-label": t("fc.col.made"), text: made }),
        el("td", { class: "dec-" + (f ? f.decision.replace(" ", "") : ""), "data-label": t("fc.col.decision"), text: f ? f.decision : "—" }),
        el("td", { "data-label": t("fc.col.probs"), text: probs }), el("td", { "data-label": t("fc.col.range"), text: range }),
        el("td", { "data-label": t("fc.col.target"), text: target }),
        el("td", { class: "small wrap", "data-label": t("fc.col.reason"), text: f ? (fcodes(f.reasons).join(", ") || t("ledger.passed")) : "—" }),
        el("td", { class: "num", "data-label": t("fc.col.checked"), text: lv.n != null ? `${lv.n} (${lv.n_independent})${lv.sufficient ? "" : " · " + t("fc.too_few")}` : `0` }),
        el("td", { class: "num", "data-label": t("fc.col.dir"), text: pct(lv.direction_hit, 0) }),
        el("td", { class: "num", "data-label": t("fc.col.mae"), text: isNum(lv.mae_bps) ? `${num(lv.mae_bps, 1)} / ${num(lv.mae_rw_bps, 1)}` : "—" }),
        el("td", { class: "num", "data-label": t("fc.col.brier"), text: brier }),
        el("td", { class: "num", "data-label": t("fc.col.cover"), text: pct(lv.coverage_10_90, 0) }),
        el("td", { class: "num", "data-label": t("fc.col.net"), text: lv.signals ? `${lv.signals}: ${bps(lv.avg_net_bps)}` : "—" }));
      tr.onclick = () => openForecast(h, key); tr.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openForecast(h, key); } };
      return tr;
    });
    tb.replaceChildren(el("thead", {}, el("tr", {}, ...head.map((x) => el("th", { text: x })))), el("tbody", {}, ...rows));
  }
  function openForecast(h, key) {
    const sv = server(), fc = sv.forecasts, st = fc.stats[key] || {}, f = fc.latest[key], ho = st.holdout || {}, lv = st.live || {}, tr = st.training || {};
    $("fcDialogTitle").textContent = `${S.symbol} · ${h.label} · ${t("fst." + (st.status || "not_trained"))}`;
    const out = [el("p", { class: "small", text: t("fc.dlg.setup", { h: durLabel(h.seconds), r: durLabel(h.refresh_sec), grid: h.grid }) })];
    if (f) {
      out.push(el("h4", { text: t("fc.dlg.latest") }));
      kvInto(out, [[t("fc.dlg.id"), f.forecast_id], [t("fc.dlg.made"), `${when(f.created_ms)} (${t("fc.dlg.data_age", { s: num(f.detail.data_age_sec, 1) })})`],
        [t("fc.dlg.price"), `${priceFmt(f.ref_price)} · ${f.exchange} · ${f.ref_source}`], [t("fc.col.decision"), f.decision],
        [t("fc.col.probs"), t("fc.dlg.probs", { up: pct(f.p_up), down: pct(f.p_down), flat: pct(f.p_flat), c: num(f.cost_bps, 0) })],
        [t("fc.col.range"), `${bps(f.q10_bps)} … ${bps(f.q90_bps)}, ${t("fc.dlg.median")} ${bps(f.q50_bps)}`],
        [t("fc.dlg.uncertainty"), `${num(f.uncertainty, 1)} ${t("unit.bps")} · ${t("fc.dlg.entropy")} ${num(f.detail.entropy, 2)}`],
        [t("fc.col.target"), when(f.target_ts)], [t("fc.col.reason"), (f.reasons || []).map((r) => `${codeLabel(CODE_OF[r] || r.toUpperCase())} — ${t("fr." + r)}`).join("; ") || t("ledger.passed")],
        [t("fc.dlg.baseline"), `↑${pct(f.base_p_up, 0)} ↓${pct(f.base_p_down, 0)} ·${pct(f.base_p_flat, 0)}`],
        [t("det.model"), `${f.model_version} · ${f.input_hash}`]]);
    }
    out.push(el("h4", { text: t("fc.dlg.holdout") }));
    kvInto(out, ho.n ? [[t("fc.dlg.n"), `${ho.n} (${t("fc.dlg.independent")} ${ho.n_independent})`],
      [t("fc.col.brier"), `${num(ho.brier, 4)} / ${num(ho.brier_base, 4)}`], [t("fc.dlg.gain"), `${num(ho.gain, 4)} [${num(ho.gain_ci95[0], 4)}…${num(ho.gain_ci95[1], 4)}], P ${pct(ho.p_better, 0)}`],
      [t("fc.col.dir"), pct(ho.direction_hit, 1)], [t("fc.col.mae"), `${num(ho.mae_bps, 1)} / ${num(ho.mae_rw_bps, 1)} ${t("unit.bps")}`],
      [t("fc.col.cover"), pct(ho.coverage_10_90, 0)], [t("fc.col.net"), ho.signals ? `${ho.signals}: ${bps(ho.avg_net_bps)}` : t("model.no_signal_passed")],
      [t("fc.dlg.labels"), ho.label_share ? `↓${pct(ho.label_share.DOWN, 0)} ·${pct(ho.label_share.FLAT, 0)} ↑${pct(ho.label_share.UP, 0)}` : "—"]]
      : [[t("fc.col.status"), tr.reason || t("fst." + (st.status || "not_trained"))]]);
    out.push(el("h4", { text: t("fc.dlg.live") }));
    kvInto(out, [[t("fc.dlg.issued"), `${st.issued || 0}`], [t("fc.dlg.resolved"), `${st.resolved || 0} · ${t("fc.dlg.missing")} ${st.missing_outcome || 0} · ${t("fc.dlg.pending")} ${st.pending || 0} · ${t("fc.dlg.stale")} ${st.stale || 0}`],
      [t("fc.col.brier"), isNum(lv.brier) ? `${num(lv.brier, 4)} / ${num(lv.brier_base, 4)}` : "—"], [t("fc.col.dir"), pct(lv.direction_hit, 1)],
      [t("fc.dlg.drift"), st.drift ? `PSI ${num(st.drift.max_psi, 2)}${st.drift.shifted ? " — " + t("fc.dlg.shifted") : ""} (${st.drift.top.map((x) => x[0]).join(", ")})` : "—"]]);
    const recent = (S.serverForecasts[key] || []).filter((r) => r.resolved_ms).slice(0, 12);
    if (recent.length) out.push(el("div", { class: "table-scroll" }, el("table", { class: "tbl cards" },
      el("thead", {}, el("tr", {}, ...[t("fc.col.made"), t("fc.col.decision"), t("fc.dlg.median"), t("fc.dlg.actual"), t("fc.dlg.source")].map((x) => el("th", { text: x })))),
      el("tbody", {}, ...recent.map((r) => el("tr", {}, el("td", { "data-label": t("fc.col.made"), text: when(r.created_ms) }), el("td", { "data-label": t("fc.col.decision"), text: r.decision }),
        el("td", { class: "num", "data-label": t("fc.dlg.median"), text: bps(r.q50_bps) }), el("td", { class: "num", "data-label": t("fc.dlg.actual"), text: r.resolution_source === "missing" ? t("fc.dlg.no_outcome") : bps(r.actual_bps) }),
        el("td", { "data-label": t("fc.dlg.source"), text: r.resolution_source || "" })))))));
    $("fcDialogBody").replaceChildren(...out);
    $("fcDialog").showModal();
  }
  function kvInto(out, pairs) { out.push(el("dl", { class: "kv detail-kv" }, ...pairs.flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v == null ? "—" : String(v) })]))); }

  function openPrediction(r) {
    $("predDialogTitle").textContent = `${r.symbol} ${r.timeframe} · ${r.prediction}`;
    const hb = r.horizon_bars || 4;
    const head = el("p", { class: "small", text: t("dlg.head", { candle: when(r.candle_ts), ex: r.exchange, h: hb, tf: r.timeframe, price: priceFmt(r.price), regime: regimeText(r.regime) }) });
    const why = el("p", { class: "small", text: r.prediction === "NO TRADE" ? t("pred.no_trade_because", { reasons: reasonDetail(r) }) : t("pred.passed", { dir: dirText(r.model_direction) }) });
    $("predDialogBody").replaceChildren(head, why, ...detailBlocks(r, hb));
    $("predDialog").showModal();
  }
  function renderNews() {
    const sv = server();
    let list, status;
    if (sv) {
      const ns = sv.news.status || {};
      status = ns.ts_ms ? t("news.status", { ok: ns.feeds_ok, total: ns.feeds_total, analyzer: ns.analyzer, ago: ago(ns.ts_ms) }) + (ns.llm_error ? ` · ${t("news.llm_error")}: ${ns.llm_error}` : "") : t("news.not_read");
      list = (sv.news.events || []).map((e) => ({ event: e.title, url: e.url, timestamp: e.published_ms, source: e.source, category: e.category || "other", direction: e.direction, relevance: e.relevance, confidence: e.confidence, analyzer: e.analyzer, affected_assets: e.affected_assets || [] }));
    } else { const ns = S.newsStatus; status = ns.ts ? `${ns.state === "LIVE" ? "DELAYED" : "OFFLINE"} · ${ns.analyzer} · ${ago(ns.ts)}${ns.errors && ns.errors.length && ns.state !== "LIVE" ? " · " + ns.errors.join("; ") : ""}` : t("news.after_start"); list = S.news; }
    $("newsStatus").textContent = status;
    const base = S.symbol.split("/")[0], rel = list.filter((e) => e.affected_assets.includes(base) || e.affected_assets.includes("CRYPTO_MARKET") || e.relevance >= 0.4).slice(0, 25);
    $("newsList").replaceChildren(...(rel.length ? rel.map((e) => {
      const d = e.direction > 0.1 ? ["dir-up", "news.positive"] : e.direction < -0.1 ? ["dir-down", "news.negative"] : ["", "news.neutral"];
      const title = e.url ? el("a", { href: e.url, target: "_blank", rel: "noopener noreferrer", text: e.event }) : el("span", { text: e.event });
      const cat = I.has("cat." + e.category) ? t("cat." + e.category) : e.category.replaceAll("_", " ");
      return el("li", {}, title, el("div", { class: "news-meta" }, el("span", { text: when(e.timestamp) }), el("span", { text: e.source || "" }), el("span", { text: cat }),
        el("span", { class: d[0], text: `${t(d[1])} ${e.direction > 0 ? "+" : ""}${num(e.direction, 2)}` }), el("span", { text: t("news.relevance", { v: Math.round(e.relevance * 100) }) }),
        el("span", { text: t("news.confidence", { v: Math.round(e.confidence * 100) }) }), el("span", { text: e.analyzer || "" })));
    }) : [el("li", { class: "muted", text: list.length ? t("news.none_relevant") : t("common.unavailable") })]));
  }
  function renderProbe() {
    const p = S.probe, tb = $("probeTable");
    if (!p) { tb.replaceChildren(el("tbody", {}, el("tr", {}, el("td", { class: "muted", text: t("probe.not_run") })))); return; }
    tb.replaceChildren(el("thead", {}, el("tr", {}, ...[t("ex.col.exchange"), t("probe.rest"), t("probe.latency"), "WebSocket", t("probe.skew"), t("probe.diagnosis")].map((h) => el("th", { text: h })))),
      el("tbody", {}, ...EXCHANGES.map((ex) => { const r = p[ex] || {}, rs = r.rest, w = r.ws;
        return el("tr", {}, el("td", { "data-label": t("ex.col.exchange"), text: ex }),
          el("td", { class: rs ? (rs.ok ? "st-good" : "st-bad") : "", "data-label": t("probe.rest"), text: rs ? (rs.ok ? t("probe.rest_ok", { s: rs.candleAgeSec }) : `${rs.kind} (${kindText(rs.kind)})`) : t("probe.testing") }),
          el("td", { class: "num", "data-label": t("probe.latency"), text: rs ? rs.ms + " " + t("unit.ms") : "" }),
          el("td", { class: w ? (w.ok ? "st-good" : "st-bad") : "", "data-label": "WebSocket", text: w ? (w.ok ? t("probe.ws_ok", { s: num(w.ms / 1000, 1) }) : `${w.kind} (${kindText(w.kind)})`) : t("probe.testing") }),
          el("td", { class: "num", "data-label": t("probe.skew"), text: w && isNum(w.skewMs) ? `${w.skewMs} ${t("unit.ms")}` : "" }),
          el("td", { class: "small wrap", "data-label": t("probe.diagnosis") }, el("code", { class: "raw", text: [rs && !rs.ok ? `REST ${rs.host}: ${rs.error}` : "", w && !w.ok ? `WS: ${w.error}` : ""].filter(Boolean).join(" · ") })));
      })));
  }
  async function runProbe() {
    S.probe = {}; renderProbe(); $("probeBtn").disabled = true;
    await Promise.all(EXCHANGES.map(async (ex) => {
      S.probe[ex] = {}; S.probe[ex].rest = await C.probeRest(ex); renderProbe();
      S.probe[ex].ws = await C.probeWs(ex, 12000); renderProbe();
    }));
    $("probeBtn").disabled = false; S.probe.ts = Date.now();
    const ok = EXCHANGES.filter((ex) => S.probe[ex].rest.ok && S.probe[ex].ws.ok);
    toast(ok.length ? t("probe.working", { list: ok.join(", ") }) : t("probe.none"));
  }
  function renderNotifyInfo() {
    const sv = server(), n = sv && sv.health && sv.health.notifications;
    if (!n) { $("backendNotify").textContent = t(S.mode === "server" ? "notif.backend_unknown" : "notif.backend_local"); return; }
    const ch = Object.entries(n.channels || {}).filter(([, on]) => on).map(([k]) => k);
    $("backendNotify").textContent = t("notif.backend_info", { state: n.enabled && ch.length ? t("notif.on") : t("notif.off"), channels: ch.join(", ") || t("notif.no_channels"),
      conf: pct(n.min_confidence, 0), q: pct(n.min_quality, 0), tfs: (n.timeframes || []).join(", ") || t("filter.all"), age: Math.round((n.max_age_sec || 0) / 60), lang: n.language || "ru" });
  }

  // ------------------------------------------------------------- charts
  function canvas() { const c = $("chart"), r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1; c.width = Math.max(1, Math.round(r.width * dpr)); c.height = Math.max(1, Math.round(r.height * dpr)); const ctx = c.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, r.width, r.height); ctx.font = "11px " + css("--font"); return { ctx, w: r.width, h: r.height }; }
  function scaleOf(vals, lo, hi, pad = 0.06) { let mn = Infinity, mx = -Infinity; for (const v of vals) if (isNum(v)) { mn = Math.min(mn, v); mx = Math.max(mx, v); } if (!isFinite(mn)) { mn = 0; mx = 1; } if (mn === mx) { mn -= 1; mx += 1; } const sp = mx - mn; mn -= sp * pad; mx += sp * pad; return { mn, mx, y: (v) => hi - ((v - mn) / (mx - mn)) * (hi - lo) }; }
  function axis(ctx, sc, x0, x1, top, bottom, fmt) { ctx.strokeStyle = css("--grid"); ctx.fillStyle = css("--ink-2"); ctx.lineWidth = 1; ctx.textAlign = "left"; for (let i = 0; i <= 4; i++) { const v = sc.mn + ((sc.mx - sc.mn) * i) / 4, y = sc.y(v); if (y < top - 1 || y > bottom + 1) continue; ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); ctx.fillText(fmt(v), x1 + 6, y + 4); } }
  function line(ctx, xs, ys, sc, color, w = 1.5) { ctx.strokeStyle = color; ctx.lineWidth = w; ctx.beginPath(); let on = false; ys.forEach((v, i) => { if (!isNum(v)) { on = false; return; } on ? ctx.lineTo(xs[i], sc.y(v)) : ctx.moveTo(xs[i], sc.y(v)); on = true; }); ctx.stroke(); }
  const shortTime = (ms) => new Date(ms).toLocaleString(loc(), { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
  function times(ctx, xs, ts, bottom) {
    ctx.fillStyle = css("--ink-2"); ctx.textAlign = "center";
    const width = (xs[xs.length - 1] || 0) - (xs[0] || 0), labels = Math.max(2, Math.min(5, Math.floor(width / 120)));
    const st = Math.max(1, Math.floor(ts.length / (labels + 1)));
    for (let i = st; i < ts.length - st / 2; i += st) ctx.fillText(shortTime(ts[i]), xs[i], bottom + 14);
  }
  function empty(msg) { $("chartEmpty").textContent = msg; $("chartEmpty").hidden = !msg; }
  function hover(xs, text) { const c = $("chart"), tip = $("chartTip"); const show = (cx) => { const x = cx - c.getBoundingClientRect().left; let b = 0; xs.forEach((v, i) => { if (Math.abs(v - x) < Math.abs(xs[b] - x)) b = i; }); tip.textContent = text(b); tip.hidden = false; }; c.onmousemove = (e) => show(e.clientX); c.onmouseleave = () => (tip.hidden = true); c.ontouchmove = (e) => e.touches[0] && show(e.touches[0].clientX); }
  function drawChart(v) {
    $("chartTip").hidden = true; $("chart").onmousemove = null;
    const { ctx, w, h } = canvas(), x0 = 6, x1 = w - (w < 520 ? 54 : 66), top = 10, bottom = h - 22;
    const up = css("--up"), down = css("--down"), ink2 = css("--ink-2"), accent = css("--accent");
    if (S.tab === "book") {
      const st = S.streams[S.bookEx], bk = st && st.books[S.symbol]; let bids, asks, note;
      if (bk && bk.ready && Date.now() - (bk.recv || 0) < 30000) { ({ bids, asks } = bk.top(20)); note = t("chart.book_live", { ex: S.bookEx }); }
      else if (server() && v.serverBook && S.bookEx === v.primary && v.serverBook.bids && v.serverBook.bids.length) { bids = v.serverBook.bids; asks = v.serverBook.asks; note = t("chart.book_backend", { ex: v.primary, ago: ago(v.serverBook.recv_ms) }); }
      if (!bids) { empty(t("chart.no_book", { ex: S.bookEx }) + (st && st.error ? ` (${st.error})` : "")); $("chartNote").textContent = ""; return; }
      empty(""); let cb = 0, ca = 0; const B = bids.map(([p, q]) => [p, (cb += q)]), Aa = asks.map(([p, q]) => [p, (ca += q)]), px = B.concat(Aa).map((x) => x[0]);
      const pmin = Math.min(...px), pmax = Math.max(...px), X = (p) => x0 + ((p - pmin) / (pmax - pmin || 1)) * (x1 - x0), sc = scaleOf([0, cb, ca], top, bottom, 0.02);
      axis(ctx, sc, x0, x1, top, bottom, (x) => num(x, 2));
      for (const [pts, col] of [[B, up], [Aa, down]]) { if (!pts.length) continue; ctx.fillStyle = col; ctx.globalAlpha = 0.25; ctx.beginPath(); ctx.moveTo(X(pts[0][0]), sc.y(0)); pts.forEach(([p, c]) => ctx.lineTo(X(p), sc.y(c))); ctx.lineTo(X(pts[pts.length - 1][0]), sc.y(0)); ctx.fill(); ctx.globalAlpha = 1; ctx.strokeStyle = col; ctx.beginPath(); pts.forEach(([p, c], i) => (i ? ctx.lineTo(X(p), sc.y(c)) : ctx.moveTo(X(p), sc.y(c)))); ctx.stroke(); }
      ctx.fillStyle = ink2; ctx.textAlign = "center"; [pmin, (pmin + pmax) / 2, pmax].forEach((p) => ctx.fillText(priceFmt(p), Math.min(Math.max(X(p), 40), x1 - 30), bottom + 14));
      $("chartNote").textContent = note + " " + t("chart.book_axes"); return;
    }
    if (S.tab === "cvd") {
      const st = S.streams[S.bookEx], f = st && st.flow[S.symbol], bars = f ? [...f.bars.values()] : []; let pts, note;
      if (bars.length > 1) { pts = bars.map((b) => ({ t: b.open_ts, v: b.cvd })); note = t("chart.cvd_live", { ex: S.bookEx }); }
      else { pts = (v.candles || []).filter((c) => isNum(c.cvd)).map((c) => ({ t: c.open_ts, v: c.cvd })); if (!pts.length && v.candles && v.candles.length > 30) { const ind = A.computeIndicators(v.candles); pts = ind.open_ts.map((x, i) => ({ t: x, v: ind.cvd[i] })).filter((p) => isNum(p.v)); } note = pts.length ? t("chart.cvd_candles", { ex: v.primary, tf: S.tf }) : ""; }
      $("chartNote").textContent = note;
      if (pts.length < 2) { empty(t("chart.no_cvd")); return; }
      empty(""); pts = pts.slice(-220); const xs = pts.map((_, i) => x0 + ((x1 - x0) / pts.length) * (i + 0.5)), sc = scaleOf(pts.map((p) => p.v), top, bottom);
      axis(ctx, sc, x0, x1, top, bottom, (x) => num(x, Math.abs(x) > 100 ? 0 : 2)); line(ctx, xs, pts.map((p) => p.v), sc, accent, 1.8); times(ctx, xs, pts.map((p) => p.t), bottom); hover(xs, (i) => `${when(pts[i].t)} CVD ${num(pts[i].v, 2)}`); return;
    }
    if (!v.candles || v.candles.length < 30) { empty(t("chart.no_candles") + " " + (S.mode === "server" ? S.serverErr || t("chart.not_exported") : primaryHint())); $("chartNote").textContent = ""; return; }
    empty(""); const full = A.computeIndicators(v.candles), k = Math.max(0, full.open_ts.length - 200), ind = {};
    for (const [n, a] of Object.entries(full)) ind[n] = Array.isArray(a) ? a.slice(k) : a;
    const n = ind.open_ts.length, bw = (x1 - x0) / n, xs = ind.open_ts.map((_, i) => x0 + bw * (i + 0.5)), ts = ind.open_ts;
    if (S.tab === "price") {
      const sc = scaleOf(ind.low.concat(ind.high, ind.bb_upper, ind.bb_lower), top, bottom); axis(ctx, sc, x0, x1, top, bottom, priceFmt);
      ctx.fillStyle = css("--grid"); ctx.globalAlpha = 0.7; ctx.beginPath(); ind.bb_upper.forEach((y, i) => isNum(y) && ctx.lineTo(xs[i], sc.y(y))); for (let i = n - 1; i >= 0; i--) if (isNum(ind.bb_lower[i])) ctx.lineTo(xs[i], sc.y(ind.bb_lower[i])); ctx.fill(); ctx.globalAlpha = 1;
      for (let i = 0; i < n; i++) { const col = ind.close[i] >= ind.open[i] ? up : down; ctx.strokeStyle = col; ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(xs[i], sc.y(ind.high[i])); ctx.lineTo(xs[i], sc.y(ind.low[i])); ctx.stroke(); const a = sc.y(ind.open[i]), b = sc.y(ind.close[i]), ww = Math.max(1, bw * 0.65); ctx.fillRect(xs[i] - ww / 2, Math.min(a, b), ww, Math.max(1, Math.abs(b - a))); }
      line(ctx, xs, ind.ema_21, sc, accent, 1.4); line(ctx, xs, ind.ema_55, sc, ink2, 1.2);
      const idx = new Map(ts.map((x, i) => [x, i]));
      for (const r of v.ledger) { if (r.prediction === "NO TRADE") continue; const i = idx.get(r.candle_ts); if (i === undefined) continue; const u = r.prediction === "UP", y = u ? sc.y(ind.low[i]) + 10 : sc.y(ind.high[i]) - 10; ctx.fillStyle = u ? up : down; ctx.beginPath(); ctx.moveTo(xs[i], y + (u ? -6 : 6)); ctx.lineTo(xs[i] - 5, y + (u ? 3 : -3)); ctx.lineTo(xs[i] + 5, y + (u ? 3 : -3)); ctx.fill(); }
      times(ctx, xs, ts, bottom); $("chartNote").textContent = t("chart.price_note", { ex: v.primary, tf: S.tf, src: server() ? t("chart.from_backend") : "" });
      hover(xs, (i) => `${when(ts[i])} ${t("chart.o")} ${priceFmt(ind.open[i])} ${t("chart.h")} ${priceFmt(ind.high[i])} ${t("chart.l")} ${priceFmt(ind.low[i])} ${t("chart.c")} ${priceFmt(ind.close[i])}`);
    } else if (S.tab === "volume") {
      const sc = scaleOf(ind.volume.concat([0]), top, bottom, 0.02); axis(ctx, sc, x0, x1, top, bottom, (x) => num(x, x > 100 ? 0 : 2));
      ind.volume.forEach((vv, i) => { ctx.fillStyle = ind.close[i] >= ind.open[i] ? up : down; const y = sc.y(vv), ww = Math.max(1, bw * 0.7); ctx.fillRect(xs[i] - ww / 2, y, ww, sc.y(0) - y); });
      times(ctx, xs, ts, bottom); $("chartNote").textContent = t("chart.volume_note", { tf: S.tf }); hover(xs, (i) => `${when(ts[i])} ${t("chart.volume")} ${num(ind.volume[i], 2)}`);
    } else {
      const mid = top + (bottom - top) / 2, rs = scaleOf([0, 100], top, mid - 10, 0); axis(ctx, rs, x0, x1, top, mid - 10, (x) => x.toFixed(0));
      ctx.setLineDash([4, 4]); ctx.strokeStyle = ink2; [30, 70].forEach((l) => { ctx.beginPath(); ctx.moveTo(x0, rs.y(l)); ctx.lineTo(x1, rs.y(l)); ctx.stroke(); }); ctx.setLineDash([]);
      line(ctx, xs, ind.rsi_14, rs, accent); line(ctx, xs, ind.adx_14, rs, ink2, 1.1);
      const ms = scaleOf(ind.macd_hist.concat([0]), mid + 6, bottom); axis(ctx, ms, x0, x1, mid + 6, bottom, (x) => (x * 1e4).toFixed(1));
      ind.macd_hist.forEach((vv, i) => { if (!isNum(vv)) return; ctx.fillStyle = vv >= 0 ? up : down; const y = ms.y(vv), y0 = ms.y(0), ww = Math.max(1, bw * 0.6); ctx.fillRect(xs[i] - ww / 2, Math.min(y, y0), ww, Math.abs(y0 - y)); });
      times(ctx, xs, ts, bottom); $("chartNote").textContent = t("chart.ind_note");
      hover(xs, (i) => `${when(ts[i])} RSI ${num(ind.rsi_14[i], 1)} ADX ${num(ind.adx_14[i], 1)} MACD ${num((ind.macd_hist[i] || 0) * 1e4, 2)} ${t("unit.bps")}`);
    }
  }

  // ------------------------------------------------------------- settings & actions
  const FIELDS = [["setAnthropicKey", "anthropicKey", "s"], ["setAnthropicModel", "anthropicModel", "s"], ["setNewsKey", "newsKey", "s"], ["setPrimary", "primary", "s"],
    ["setThreshold", "threshold", "n"], ["setEdge", "minEdge", "n"], ["setHorizon", "horizon", "i"], ["setMinLabels", "minNewLabels", "i"], ["setFee", "feeBps", "n"], ["setSlip", "slippageBps", "n"],
    ["setNotify", "notifyEnabled", "b"], ["setNotifySignals", "notifySignals", "b"], ["setNotifyAll", "notifyAll", "b"], ["setNotifyOutages", "notifyOutages", "b"],
    ["setNotifyMinConf", "notifyMinConf", "n"], ["setNotifyMaxAge", "notifyMaxAgeMin", "n"]];
  function fillSettings() { for (const [id, k, ty] of FIELDS) { if (ty === "b") $(id).checked = !!S.settings[k]; else $(id).value = S.settings[k]; } $("localOnly").hidden = S.mode === "server"; $("setLang").value = I.lang(); }
  async function saveSettings() {
    const before = { ...S.settings };
    for (const [id, k, ty] of FIELDS) { if (ty === "b") { S.settings[k] = $(id).checked; continue; } const v = $(id).value.trim(); S.settings[k] = ty === "s" ? v : ty === "i" ? parseInt(v, 10) : parseFloat(v); if (ty !== "s" && !isNum(S.settings[k])) S.settings[k] = before[k]; }
    S.settings.threshold = Math.min(0.95, Math.max(0.34, S.settings.threshold)); S.settings.horizon = Math.min(24, Math.max(1, S.settings.horizon));
    S.settings.notifyMinConf = Math.min(0.99, Math.max(0, S.settings.notifyMinConf)); S.settings.notifyMaxAgeMin = Math.min(240, Math.max(1, S.settings.notifyMaxAgeMin));
    if (S.settings.notifyEnabled && "Notification" in window && Notification.permission !== "granted") {
      const perm = await Notification.requestPermission().catch(() => "denied");
      if (perm !== "granted") { S.settings.notifyEnabled = false; toast(t("toast.notif_denied")); }
    }
    await Store.set("settings", S.settings); $("settingsDialog").close();
    if ($("setLang").value !== I.lang()) setLanguage($("setLang").value);
    if (S.mode === "local" && before.primary !== S.settings.primary) { S.primary = null; S.candles = {}; await pollLocal(); }
    if (S.mode === "local" && before.horizon !== S.settings.horizon) ensureModels();
    toast(S.settings.notifyEnabled ? t("toast.saved_notif") : t("toast.saved"));
    render();
  }
  // switching language re-renders texts only: models, ledger and streams stay untouched
  function setLanguage(l) {
    I.setLang(l);
    $("lang").value = I.lang();
    I.apply(document);
    fillTfOptions(server() ? S.server.settings.predict_timeframes : S.settings.predictTfs);
    render();
  }
  function exportCsv() {
    const rows = ledgerRows().slice().sort((a, b) => a.candle_ts - b.candle_ts);
    const cols = ["prediction_id", "symbol", "timeframe", "exchange", "candle_ts", "created_ms", "price", "prediction", "model_direction", "p_up", "p_down", "p_flat", "confidence", "label_threshold", "quality_score", "regime", "model_version", "features_version", "gate_reasons", "resolved_ms", "actual_price", "actual_return", "actual_direction", "error", "result", "error_class"];
    const csv = [cols.join(",")].concat(rows.map((r) => cols.map((c) => JSON.stringify(Array.isArray(r[c]) ? r[c].join("|") : r[c] ?? "")).join(","))).join("\n");
    const a = el("a", { href: URL.createObjectURL(new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8" })), download: `prediction_ledger_${new Date().toISOString().slice(0, 10)}.csv` });
    document.body.append(a); a.click(); a.remove(); toast(t("toast.exported", { n: rows.length }));
  }
  async function runBacktest() {
    const out = $("btResult");
    if (S.mode === "server") { const m = view().model, bt = m && m.production && m.production.metrics.backtest; kv(out, bt && bt.trades ? [[t("bt.trades"), bt.trades], [t("bt.win"), pct(bt.win_rate)], [t("bt.gross_net"), `${num(bt.avg_gross_bps, 1)} / ${num(bt.avg_net_bps, 1)} ${t("unit.bps")}`], [t("bt.total"), pct(bt.total_net_return, 2)], [t("bt.dd"), pct(bt.max_drawdown, 2)], [t("bt.costs"), `${num(bt.costs && bt.costs.round_trip_cost_bps, 1)} ${t("unit.bps")}`]] : [[t("bt.result"), t("model.no_signal_passed")]]); return; }
    const p = production(keyOf(S.symbol, S.tf)); if (!p) { kv(out, [[t("bt.result"), "MODEL NOT READY"]]); return; }
    const oos = await Store.get(`oos:${p.version}`); if (!oos) { kv(out, [[t("bt.result"), t("bt.no_oos")]]); return; }
    const r = A.backtest(oos, S.tf, S.settings.horizon, { threshold: +$("btThr").value, minEdge: S.settings.minEdge, feeBps: +$("btFee").value, slippageBps: +$("btSlip").value, spreadBps: +$("btSpread").value, latencyMs: +$("btLat").value });
    kv(out, r.trades ? [[t("bt.trades"), r.trades], [t("bt.win"), pct(r.win_rate)], [t("bt.gross_net"), `${num(r.avg_gross_bps, 1)} / ${num(r.avg_net_bps, 1)} ${t("unit.bps")}`], [t("bt.total"), pct(r.total_net_return, 2)], [t("bt.dd"), pct(r.max_drawdown, 2)], [t("bt.costs"), `${num(r.round_trip_cost_bps, 1)} ${t("unit.bps")}`]] : [[t("bt.trades"), 0], [t("bt.note"), t("model.no_signal_passed")]]);
  }
  function confirmInPage(msg, onYes) { $("confirmText").textContent = msg; $("confirmBox").hidden = false; $("confirmYes").onclick = () => { $("confirmBox").hidden = true; onYes(); }; $("confirmNo").onclick = () => ($("confirmBox").hidden = true); }
  function startStreams() { for (const ex of EXCHANGES) { if (S.streams[ex]) continue; const st = new C.ExchangeStream(ex, SYMBOLS, () => render()); S.streams[ex] = st; st.start(); } }
  async function startLocal() {
    S.started = true; S.settings.started = true; await Store.set("settings", S.settings);
    startStreams(); render(); await pollLocal(); fetchNewsLocal(false);
    setInterval(pollLocal, 20000); setInterval(() => fetchNewsLocal(false), 5 * 60000); setInterval(learningCycleLocal, 10 * 60000); setTimeout(learningCycleLocal, 60000);
  }
  function fillTfOptions(tfs) {
    const cur = S.tf;
    $("tf").replaceChildren(...tfs.map((x) => el("option", { value: x, text: x })));
    S.tf = tfs.includes(cur) ? cur : tfs[0]; $("tf").value = S.tf;
    const sel = $("fTf"), v = S.filter.tf;
    sel.replaceChildren(el("option", { value: "current", text: t("filter.current") }), el("option", { value: "all", text: t("filter.all") }), ...tfs.map((x) => el("option", { value: x, text: x })));
    sel.value = v;
  }

  // ------------------------------------------------------------- boot
  async function boot() {
    I.apply(document);
    await Store.open();
    try {
      S.settings = { ...DEFAULTS, ...((await Store.get("settings")) || {}) };
      S.registry = (await Store.get("registry")) || S.registry; S.learning = { ...S.learning, ...((await Store.get("learning")) || {}) };
      S.news = (await Store.get("news")) || []; for (const r of await Store.allLedger()) if (r.timeframe) S.ledger.set(r.id, r);
    } catch (e) { console.error(e); }
    const hasServer = await detectServer();
    S.mode = hasServer ? "server" : "local";
    $("symbol").replaceChildren(...SYMBOLS.map((s) => el("option", { value: s, text: s })));
    $("fSymbol").replaceChildren(el("option", { value: "current", "data-i18n": "filter.current", text: t("filter.current") }), el("option", { value: "all", "data-i18n": "filter.all", text: t("filter.all") }), ...SYMBOLS.map((s) => el("option", { value: s, text: s })));
    fillTfOptions(S.settings.predictTfs);
    $("bookEx").replaceChildren(...EXCHANGES.map((e) => el("option", { value: e, text: e })));
    $("lang").value = I.lang();
    $("lang").onchange = () => setLanguage($("lang").value);
    $("symbol").onchange = () => { S.symbol = $("symbol").value; render(); };
    $("tf").onchange = () => { S.tf = $("tf").value; render(); };
    $("bookEx").onchange = () => { S.bookEx = $("bookEx").value; render(); };
    $("startBtn").onclick = () => startLocal();
    $("refreshBtn").onclick = async () => { $("refreshBtn").disabled = true; if (S.mode === "server") await fetchServer(true); else await pollLocal(); $("refreshBtn").disabled = false; toast(t("toast.refreshed")); };
    $("settingsBtn").onclick = $("welcomeSettings").onclick = () => { fillSettings(); $("settingsDialog").showModal(); };
    $("settingsForm").onsubmit = (e) => { e.preventDefault(); saveSettings(); };
    $("settingsCancel").onclick = () => $("settingsDialog").close();
    $("predDialogClose").onclick = () => $("predDialog").close();
    $("fcDialogClose").onclick = () => $("fcDialog").close();
    $("probeBtn").onclick = runProbe;
    $("newsBtn").onclick = () => (S.mode === "server" ? fetchServer(true) : fetchNewsLocal(true));
    $("learnBtn").hidden = hasServer; $("retrainBtn").hidden = hasServer;
    $("learnBtn").onclick = () => { learningCycleLocal(); toast(t("toast.learning_started")); };
    $("retrainBtn").onclick = () => confirmInPage(t("confirm.retrain", { sym: S.symbol, tf: S.tf }), () => { const k = keyOf(S.symbol, S.tf); if (!S.jobs.some((j) => j.key === k)) S.jobs.push({ kind: "baseline", key: k, sym: S.symbol, tf: S.tf }); pump(); toast(t("toast.retrain_started")); });
    $("exportBtn").onclick = exportCsv;
    for (const [id, f] of [["ledgerFilter", "type"], ["fSymbol", "symbol"], ["fTf", "tf"], ["fModel", "model"], ["fPeriod", "period"]]) $(id).onchange = () => { S.filter[f] = $(id).value; S.ledgerLimit = 50; render(); };
    $("moreBtn").onclick = () => { S.ledgerLimit += 100; render(); };
    $("btForm").onsubmit = (e) => { e.preventDefault(); runBacktest(); };
    $("resetBtn").onclick = () => { $("settingsDialog").close(); confirmInPage(t("confirm.reset"), async () => { await Store.clear(); location.reload(); }); };
    for (const b of $("chartTabs").querySelectorAll("button")) b.onclick = () => { S.tab = b.dataset.chart; $("chartTabs").querySelectorAll("button").forEach((x) => x.setAttribute("aria-selected", String(x === b))); render(); };
    $("btFee").value = S.settings.feeBps; $("btSlip").value = S.settings.slippageBps; $("btSpread").value = S.settings.spreadBps; $("btLat").value = S.settings.latencyMs; $("btThr").value = S.settings.threshold;
    $("btForm").hidden = hasServer;
    window.addEventListener("resize", render);
    setInterval(render, 2000);
    if (hasServer) { await fetchServer(true); startStreams(); setInterval(() => fetchServer(false), 60000); }
    else if (S.settings.started) startLocal();
    render();
  }
  window.AMPApp = { state: S, setLanguage };
  boot();
})();
