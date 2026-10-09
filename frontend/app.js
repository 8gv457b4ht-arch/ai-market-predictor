"use strict";
// AI Market Predictor dashboard. Plain JS, no external libraries (works offline / under strict CSP).

const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { const v = localStorage.getItem("amp:" + k); return v === null ? d : v; } catch { return d; } },
  set(k, v) { try { localStorage.setItem("amp:" + k, v); } catch { /* storage unavailable */ } },
  del(k) { try { localStorage.removeItem("amp:" + k); } catch { /* storage unavailable */ } },
};
const S = { config: null, tab: store.get("tab", "price"), data: {}, timer: null, keyPrompted: false };

// ------------------------------------------------------------------ helpers
const pct = (x, d = 1) => (x === null || x === undefined ? "—" : (x * 100).toFixed(d) + "%");
const num = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? "—" : Number(x).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }));
const priceFmt = (x) => (x === null || x === undefined ? "—" : Number(x).toLocaleString(undefined, { maximumFractionDigits: x >= 100 ? 2 : x >= 1 ? 4 : 6, minimumFractionDigits: x >= 100 ? 2 : 0 }));
function ago(sec) {
  if (sec === null || sec === undefined) return "—";
  if (sec < 90) return Math.round(sec) + " s ago";
  if (sec < 5400) return Math.round(sec / 60) + " min ago";
  if (sec < 172800) return Math.round(sec / 3600) + " h ago";
  return Math.round(sec / 86400) + " d ago";
}
const when = (ms) => (ms ? new Date(ms).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—");
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k === "text") e.textContent = v; else e.setAttribute(k, v);
  }
  for (const k of kids) if (k !== null && k !== undefined) e.append(k);
  return e;
}
function toast(msg) {
  const t = $("toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), 4000);
}
function banner(msg) { const b = $("banner"); b.textContent = msg || ""; b.hidden = !msg; }
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

async function api(path, opts = {}) {
  const headers = { Accept: "application/json", ...(opts.headers || {}) };
  const key = store.get("key", "");
  if (key) headers["X-API-Key"] = key;
  const r = await fetch(path, { ...opts, headers });
  let body = null;
  try { body = await r.json(); } catch { body = null; }
  if (r.status === 401) {
    banner("The server requires an API key. Use the API key button to enter it.");
    if (!S.keyPrompted) { S.keyPrompted = true; openKeyDialog(); }
    throw new Error("unauthorized");
  }
  if (r.status === 429) throw new Error("Too many requests; slowing down.");
  if (!r.ok && r.status !== 202 && r.status !== 503 && r.status !== 404) throw new Error((body && (body.detail || body.error)) || "HTTP " + r.status);
  return { status: r.status, body };
}
const q = (extra = {}) => new URLSearchParams({ symbol: $("symbol").value, tf: $("tf").value, ...extra }).toString();

// --------------------------------------------------------------------- init
async function init() {
  const { body: cfg } = await api("/api/config");
  S.config = cfg;
  const fill = (sel, items, saved) => {
    sel.replaceChildren(...items.map((x) => el("option", { value: x, text: x })));
    if (saved && items.includes(saved)) sel.value = saved;
  };
  fill($("symbol"), cfg.symbols, store.get("symbol"));
  fill($("tf"), cfg.predict_timeframes, store.get("tf"));
  fill($("bookEx"), cfg.exchanges, store.get("bookEx", cfg.primary_exchange));
  $("btFee").value = cfg.costs.fee_bps; $("btSlip").value = cfg.costs.slippage_bps;
  $("btSpread").value = cfg.costs.spread_bps; $("btLat").value = cfg.costs.latency_ms; $("btThr").value = cfg.confidence_threshold;
  for (const id of ["symbol", "tf", "bookEx"]) $(id).addEventListener("change", () => { store.set(id, $(id).value); refresh(); });
  $("refreshBtn").addEventListener("click", () => refresh(true));
  $("autoRefresh").checked = store.get("auto", "1") === "1";
  $("autoRefresh").addEventListener("change", () => { store.set("auto", $("autoRefresh").checked ? "1" : "0"); schedule(); });
  $("keyBtn").addEventListener("click", openKeyDialog);
  $("learnBtn").addEventListener("click", () => trigger("/api/learning/run", $("learnBtn"), "Learning cycle queued"));
  $("newsBtn").addEventListener("click", () => trigger("/api/news/refresh", $("newsBtn"), "News refresh queued"));
  $("backupBtn").addEventListener("click", () => trigger("/api/backup/run", $("backupBtn"), "Backup queued"));
  $("btForm").addEventListener("submit", (e) => { e.preventDefault(); runBacktest(); });
  for (const b of $("chartTabs").querySelectorAll("button")) b.addEventListener("click", () => setTab(b.dataset.chart));
  setTab(S.tab, false);
  window.addEventListener("resize", () => drawChart());
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => drawChart());
  if (cfg.auth_required && !store.get("key", "")) openKeyDialog();
  await refresh();
  schedule();
}
function schedule() {
  clearInterval(S.timer);
  if ($("autoRefresh").checked) S.timer = setInterval(() => { if (!document.hidden) refresh(); }, 10000);
}
document.addEventListener("visibilitychange", () => { if (!document.hidden && $("autoRefresh").checked) refresh(); });

async function trigger(path, btn, msg) {
  btn.disabled = true;
  try {
    const { status, body } = await api(path, { method: "POST" });
    toast(status === 202 && body.status === "already_queued" ? "Already queued; the worker will pick it up shortly" : msg);
  } catch (e) { toast("Request failed: " + e.message); }
  finally { setTimeout(() => (btn.disabled = false), 1500); }
}

// ------------------------------------------------------------------ refresh
async function refresh(manual = false) {
  if (!S.config) return;
  const btn = $("refreshBtn");
  btn.disabled = true;
  const tasks = {
    overview: api("/api/overview?" + q()), candles: api("/api/candles?" + q({ limit: 240, exchange: S.config.primary_exchange })),
    news: api("/api/news?" + q({ limit: 25 })), model: api("/api/model?" + q()), ledger: api("/api/predictions?" + q({ limit: 40 })),
    status: api("/api/status"),
  };
  if (S.tab === "book") tasks.book = api("/api/orderbook?" + q({ exchange: $("bookEx").value }));
  if (S.tab === "cvd") tasks.flow = api("/api/orderflow?" + q({ exchange: $("bookEx").value, minutes: 360 }));
  const names = Object.keys(tasks);
  const res = await Promise.allSettled(Object.values(tasks));
  let failed = 0, unauthorized = false;
  res.forEach((r, i) => {
    if (r.status === "fulfilled") S.data[names[i]] = r.value.body;
    else { failed++; if (r.reason && r.reason.message === "unauthorized") unauthorized = true; }
  });
  if (!unauthorized) banner(failed === names.length ? "The API is not reachable. Check that the api service is running." : "");
  try { renderOverview(); } catch (e) { console.error(e); }
  try { renderNews(); } catch (e) { console.error(e); }
  try { renderModel(); } catch (e) { console.error(e); }
  try { renderLedger(); } catch (e) { console.error(e); }
  try { renderStatus(); } catch (e) { console.error(e); }
  drawChart();
  btn.disabled = false;
  if (manual && !failed) toast("Updated");
}

// ------------------------------------------------------------------ readout
function renderOverview() {
  const o = S.data.overview; if (!o) return;
  $("assetName").textContent = `${o.symbol} · ${o.timeframe}`;
  $("price").textContent = o.price === null ? "—" : "$" + priceFmt(o.price);
  $("priceSource").textContent = o.price_source ? `Source: ${o.price_source.replaceAll("_", " ")} on ${o.exchange}` : "No market data yet";
  const ch = o.ticker && o.ticker.change_24h_pct;
  $("change").textContent = ch === null || ch === undefined ? "" : (ch >= 0 ? "+" : "") + ch.toFixed(2) + "% 24h";
  $("change").dataset.dir = ch > 0 ? "up" : ch < 0 ? "down" : "";
  const p = o.prediction;
  const sig = $("signal");
  const thr = o.confidence_threshold;
  $("threshold").style.left = `calc(${(thr * 100).toFixed(1)}% - 1px)`;
  $("threshold").dataset.label = "threshold " + Math.round(thr * 100) + "%";
  $("horizon").textContent = `next ${o.horizon_bars} × ${o.timeframe}`;
  if (!p) {
    sig.textContent = "Waiting for model"; sig.dataset.state = "none";
    for (const id of ["segDown", "segFlat", "segUp"]) $(id).style.width = "0";
    for (const id of ["pDown", "pFlat", "pUp", "confidence"]) $(id).textContent = "—";
    $("reasons").textContent = "No prediction yet. The learner trains the first model once enough history is collected.";
  } else {
    sig.textContent = p.prediction; sig.dataset.state = p.prediction;
    $("segDown").style.width = (p.p_down * 100).toFixed(2) + "%";
    $("segFlat").style.width = (p.p_flat * 100).toFixed(2) + "%";
    $("segUp").style.width = (p.p_up * 100).toFixed(2) + "%";
    $("pDown").textContent = pct(p.p_down); $("pFlat").textContent = pct(p.p_flat); $("pUp").textContent = pct(p.p_up);
    $("confidence").textContent = pct(p.confidence);
    const why = { confidence_below_threshold: "model confidence below threshold", edge_below_min_edge: "UP and DOWN too close", flat_more_likely: "FLAT is more likely than the direction", data_quality: "data quality check failed",
      abnormal_market: "abnormal market conditions", low_liquidity: "low liquidity", news_conflict: "fresh news contradicts the model" };
    const reasons = (p.gate_reasons || []).map((r) => why[r] || r);
    $("reasons").textContent = (reasons.length ? "No trade because: " + reasons.join("; ") + ". " : "") +
      `Model ${p.model_direction.toLowerCase()} · candle ${when(p.candle_ts)} · made ${ago(p.age_sec)}`;
  }
  $("regime").textContent = (o.regime || "—").replaceAll("_", " ");
  const qd = o.quality;
  $("quality").textContent = qd ? `${Math.round(qd.score * 100)}%${qd.ok ? "" : " (blocked)"}` : "—";
  $("quality").title = qd ? qd.issues.map((i) => i.detail).join("\n") : "";
  const snap = p && p.snapshot;
  $("newsImpact").textContent = snap && snap.news_impact !== null && snap.news_impact !== undefined ? (snap.news_impact > 0 ? "+" : "") + snap.news_impact.toFixed(2) : "—";
  const b = o.book;
  $("spread").textContent = b && b.spread_bps !== null ? b.spread_bps.toFixed(2) + " bps" : "—";
  $("imbalance").textContent = b && b.imbalance !== null ? (b.imbalance > 0 ? "+" : "") + b.imbalance.toFixed(2) : "—";
  $("updated").textContent = b ? "book " + ago(b.age_sec) : "";
}

// -------------------------------------------------------------------- news
function renderNews() {
  const n = S.data.news; if (!n) return;
  const st = n.status;
  $("newsStatus").textContent = st ? `${st.analyzer} · ${st.feeds_ok}/${st.feeds_total} feeds ok · updated ${ago((Date.now() - st.ts_ms) / 1000)}` +
    (st.llm_error ? " · LLM error, rules used" : "") : "The news worker has not run yet.";
  const list = $("newsList");
  if (!n.events.length) { list.replaceChildren(el("li", { class: "muted", text: "No news events stored yet." })); return; }
  list.replaceChildren(...n.events.map((e) => {
    const dir = e.direction > 0.1 ? ["dir-up", "positive"] : e.direction < -0.1 ? ["dir-down", "negative"] : ["", "neutral"];
    const title = e.url ? el("a", { href: e.url, target: "_blank", rel: "noopener noreferrer", text: e.title }) : el("span", { text: e.title });
    return el("li", {}, title, el("div", { class: "news-meta" },
      el("span", { text: when(e.published_ms) }), el("span", { text: e.source || "" }),
      el("span", { text: (e.category || "").replaceAll("_", " ") }),
      el("span", { class: dir[0], text: dir[1] + " " + (e.direction > 0 ? "+" : "") + e.direction.toFixed(2) }),
      el("span", { text: "relevance " + Math.round(e.relevance * 100) + "%" }),
      el("span", { text: "confidence " + Math.round(e.confidence * 100) + "%" }),
      el("span", { text: (e.affected_assets || []).join(", ") })));
  }));
}

// ------------------------------------------------------------------- model
function kv(target, pairs) {
  target.replaceChildren(...pairs.flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v === null || v === undefined ? "—" : String(v) })]));
}
function renderModel() {
  const m = S.data.model; if (!m) return;
  const p = m.production;
  if (!p) {
    const ls = m.learning_status;
    kv($("modelKv"), [["Status", "No production model yet"], ["Learner", ls ? JSON.stringify(ls) : "waiting for history"]]);
  } else {
    const met = p.metrics || {};
    const base = met.baseline_prior || {};
    const sig = met.signals || {};
    const rv = m.ledger_review || {};
    kv($("modelKv"), [
      ["Model version", p.version],
      ["Status", p.status + (p.reason ? " — " + p.reason : "")],
      ["Accuracy (out-of-sample)", pct(met.accuracy) + (base.accuracy !== undefined ? ` (naive ${pct(base.accuracy)})` : "")],
      ["Log loss", num(met.log_loss, 4) + (base.log_loss !== undefined ? ` (naive ${num(base.log_loss, 4)})` : "")],
      ["Brier", num(met.brier, 4)],
      ["F1 macro", num(met.f1_macro, 3)],
      ["Calibration error", num(met.ece, 3)],
      ["Signals passing gate", sig.signals !== undefined ? `${sig.signals} (${pct(sig.coverage)}), hit rate ${pct(sig.hit_rate)}` : "—"],
      ["Validation", met.method || "—"],
      ["Last training", when(p.created_ms)],
      ["Last validation", m.learning_status ? `${m.learning_status.status} ${when(m.learning_status.ts_ms)}` : "—"],
      ["Live ledger", rv.resolved ? `${rv.resolved} resolved, ${rv.signals} signals, hit ${pct(rv.signal_hit_rate)}` : "no resolved predictions yet"],
      ["Backtest (stored)", met.backtest && met.backtest.trades ? `${met.backtest.trades} trades, net ${num(met.backtest.avg_net_bps, 1)} bps/trade` : "no trades passed the gate"],
    ]);
  }
  const t = $("versions");
  t.replaceChildren(el("tr", {}, ...["Version", "Status", "Created", "Log loss", "Accuracy", "Logloss gain", "P(better)", "Reason"].map((h) => el("th", { text: h }))),
    ...m.versions.map((v) => el("tr", {}, el("td", { text: v.version }), el("td", { text: v.status }), el("td", { text: when(v.created_ms) }),
      el("td", { class: "num", text: num(v.log_loss, 4) }), el("td", { class: "num", text: pct(v.accuracy) }),
      el("td", { class: "num", text: v.comparison ? num(v.comparison.logloss_gain, 4) : "—" }),
      el("td", { class: "num", text: v.comparison ? pct(v.comparison.bootstrap_p_better, 0) : "—" }),
      el("td", { text: v.reason || "" }))));
}
async function runBacktest() {
  const params = q({ fee_bps: $("btFee").value, slippage_bps: $("btSlip").value, spread_bps: $("btSpread").value,
    latency_ms: $("btLat").value, threshold: $("btThr").value });
  try {
    const { status, body } = await api("/api/backtest?" + params);
    if (status !== 200) { kv($("btResult"), [["Result", body.error || "not available"]]); return; }
    const r = body.result;
    if (!r.trades) { kv($("btResult"), [["Trades", 0], ["Note", r.note || "no signals"]]); return; }
    kv($("btResult"), [["Trades", `${r.trades} (${r.long} long / ${r.short} short)`], ["Win rate", pct(r.win_rate)],
      ["Avg gross / net", `${num(r.avg_gross_bps, 1)} / ${num(r.avg_net_bps, 1)} bps`], ["Total net return", pct(r.total_net_return, 2)],
      ["Max drawdown", pct(r.max_drawdown, 2)], ["Profit factor", num(r.profit_factor, 2)], ["Buy and hold", pct(r.buy_and_hold_return, 2)],
      ["Round-trip cost", num(r.costs.round_trip_cost_bps, 1) + " bps"], ["Note", r.disclaimer]]);
  } catch (e) { toast("Backtest failed: " + e.message); }
}

// ------------------------------------------------------------------- ledger
function renderLedger() {
  const L = S.data.ledger; if (!L) return;
  const rows = L.predictions;
  const res = rows.filter((r) => r.resolved_ms && r.prediction !== "NO TRADE");
  const hits = res.filter((r) => r.result === "correct").length;
  $("ledgerStats").textContent = rows.length ? `${rows.length} shown · ${res.length} resolved signals · hit ${res.length ? pct(hits / res.length) : "—"}` : "";
  const t = $("ledger");
  if (!rows.length) { t.replaceChildren(el("tr", {}, el("td", { class: "muted", text: "No predictions recorded yet." }))); return; }
  t.replaceChildren(el("tr", {}, ...["Candle", "Price", "Prediction", "Up", "Down", "Flat", "Regime", "Actual", "Return", "Result", "Model"].map((h) => el("th", { text: h }))),
    ...rows.map((r) => el("tr", {}, el("td", { text: when(r.candle_ts) }), el("td", { class: "num", text: priceFmt(r.price) }),
      el("td", { text: r.prediction }), el("td", { class: "num", text: pct(r.p_up) }), el("td", { class: "num", text: pct(r.p_down) }),
      el("td", { class: "num", text: pct(r.p_flat) }), el("td", { text: (r.regime || "").replaceAll("_", " ") }),
      el("td", { text: r.actual_direction || "pending" }), el("td", { class: "num", text: r.actual_return === null ? "—" : pct(r.actual_return, 2) }),
      el("td", { class: "res-" + (r.result || ""), text: (r.error_class || "").replaceAll("_", " ") }),
      el("td", { text: (r.model_version || "").slice(-19) }))));
}

// ------------------------------------------------------------------- status
function renderStatus() {
  const s = S.data.status; if (!s) return;
  const cls = (v, good, warn) => (good.includes(v) ? "st-good" : warn.includes(v) ? "st-warn" : "st-bad");
  const item = (label, value, c, detail) => [el("dt", { text: label }), el("dd", { class: c }, value, detail ? el("span", { class: "detail", text: detail }) : null)];
  const sv = s.services || {};
  $("statusGrid").replaceChildren(
    ...item("WebSocket", s.websocket.state, cls(s.websocket.state, ["CONNECTED"], ["PARTIAL"]),
      Object.entries(s.websocket.exchanges).map(([k, v]) => `${k} ${v.connected ? "on" : "off"}`).join(", ")),
    ...item("Market data", s.market_data.state, cls(s.market_data.state, ["LIVE"], []), s.market_data.last_tick_age_sec !== null ? "last tick " + ago(s.market_data.last_tick_age_sec) : "REST candles only, no live ticks"),
    ...item("News", s.news.state, cls(s.news.state, ["LIVE"], ["DEGRADED", "STALE", "NOT STARTED"]), s.news.analyzer || ""),
    ...item("Model", s.model.state, cls(s.model.state, ["READY"], ["PARTIAL"]), ""),
    ...item("Learning", s.learning.state, cls(s.learning.state, ["ACTIVE"], []), "heartbeat " + ago(s.learning.heartbeat_age_sec)),
    ...item("Database", s.database.state, cls(s.database.state, ["OK"], []), s.database.size_bytes ? (s.database.size_bytes / 1048576).toFixed(1) + " MB " + s.database.dialect : s.database.dialect),
    ...item("Last prediction", s.last_prediction ? ago(s.last_prediction.age_sec) : "none", s.last_prediction ? "st-good" : "st-warn", ""),
    ...item("Last backup", s.last_backup ? ago(s.last_backup.age_sec) : "none", s.last_backup ? "st-good" : "st-warn", ""),
    ...item("API auth", s.auth.startsWith("enabled") ? "ENABLED" : "DISABLED", s.auth.startsWith("enabled") ? "st-good" : "st-bad", ""),
    ...Object.entries(sv).flatMap(([k, v]) => item("Service " + k, v.alive ? "RUNNING" : "DOWN", v.alive ? "st-good" : "st-bad", v.age_sec === null ? "no heartbeat" : "heartbeat " + ago(v.age_sec))));
  const rows = [];
  for (const [ex, v] of Object.entries(s.websocket.exchanges)) for (const [sym, d] of Object.entries(v.symbols)) rows.push([ex, sym, d]);
  $("streams").replaceChildren(el("tr", {}, ...["Exchange", "Symbol", "State", "Last message", "Reconnects", "Gaps", "Duplicates", "Invalid", "Last error"].map((h) => el("th", { text: h }))),
    ...rows.map(([ex, sym, d]) => el("tr", {}, el("td", { text: ex }), el("td", { text: sym }), el("td", { text: d.state }),
      el("td", { text: ago(d.last_msg_age_sec) }), el("td", { class: "num", text: d.reconnects }), el("td", { class: "num", text: d.gaps }),
      el("td", { class: "num", text: d.duplicates }), el("td", { class: "num", text: d.invalid }), el("td", { text: (d.last_error || "").slice(0, 80) }))));
}

// ------------------------------------------------------------------- charts
function setTab(tab, reload = true) {
  S.tab = tab; store.set("tab", tab);
  for (const b of $("chartTabs").querySelectorAll("button")) b.setAttribute("aria-selected", String(b.dataset.chart === tab));
  if (reload) refresh(); else drawChart();
}

function canvasCtx() {
  const c = $("chart");
  const r = c.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  c.width = Math.max(1, Math.round(r.width * dpr)); c.height = Math.max(1, Math.round(r.height * dpr));
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  ctx.font = "11px " + css("--font");
  return { ctx, w: r.width, h: r.height };
}
function empty(msg) { $("chartEmpty").textContent = msg; $("chartEmpty").hidden = !msg; }

function scale(vals, lo, hi, pad = 0.06) {
  let mn = Infinity, mx = -Infinity;
  for (const v of vals) if (v !== null && v !== undefined && Number.isFinite(v)) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
  if (!Number.isFinite(mn)) { mn = 0; mx = 1; }
  if (mn === mx) { mn -= 1; mx += 1; }
  const span = mx - mn; mn -= span * pad; mx += span * pad;
  return { mn, mx, y: (v) => hi - ((v - mn) / (mx - mn)) * (hi - lo) };
}
function axis(ctx, sc, x0, x1, top, bottom, fmt) {
  ctx.strokeStyle = css("--grid"); ctx.fillStyle = css("--ink-2"); ctx.lineWidth = 1; ctx.textAlign = "left";
  for (let i = 0; i <= 4; i++) {
    const v = sc.mn + ((sc.mx - sc.mn) * i) / 4, y = sc.y(v);
    if (y < top - 1 || y > bottom + 1) continue;
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
    ctx.fillText(fmt(v), x1 + 6, y + 4);
  }
}
function line(ctx, xs, ys, sc, color, width = 1.5) {
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath(); let on = false;
  ys.forEach((v, i) => { if (v === null || v === undefined) { on = false; return; } const y = sc.y(v); on ? ctx.lineTo(xs[i], y) : ctx.moveTo(xs[i], y); on = true; });
  ctx.stroke();
}
function timeLabels(ctx, xs, ts, bottom) {
  ctx.fillStyle = css("--ink-2"); ctx.textAlign = "center";
  const n = ts.length, step = Math.max(1, Math.floor(n / 5));
  for (let i = step; i < n; i += step) ctx.fillText(new Date(ts[i]).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }), xs[i], bottom + 14);
}

function drawChart() {
  $("chartTip").hidden = true;
  const tab = S.tab;
  const { ctx, w, h } = canvasCtx();
  const right = w < 520 ? 52 : 64, x0 = 6, x1 = w - right, top = 10, bottom = h - 22;
  $("chart").onmousemove = null; $("chart").ontouchmove = null;
  if (tab === "book") return drawBook(ctx, w, h, x0, x1, top, bottom);
  if (tab === "cvd") return drawCvd(ctx, x0, x1, top, bottom);
  const c = (S.data.candles && S.data.candles.candles) || [];
  if (c.length < 2) { empty("No candles yet. The collector backfills history from the exchange on start."); $("chartNote").textContent = ""; return; }
  empty("");
  const n = c.length, bw = (x1 - x0) / n;
  const xs = c.map((_, i) => x0 + bw * (i + 0.5));
  const ts = c.map((d) => d.open_ts);
  const up = css("--up"), down = css("--down"), ink2 = css("--ink-2"), accent = css("--accent");
  if (tab === "price") {
    const sc = scale(c.flatMap((d) => [d.low, d.high, d.bb_upper, d.bb_lower]), top, bottom);
    axis(ctx, sc, x0, x1, top, bottom, priceFmt);
    ctx.fillStyle = css("--grid");
    ctx.globalAlpha = 0.6; ctx.beginPath();
    c.forEach((d, i) => { if (d.bb_upper !== null) ctx.lineTo(xs[i], sc.y(d.bb_upper)); });
    for (let i = n - 1; i >= 0; i--) if (c[i].bb_lower !== null) ctx.lineTo(xs[i], sc.y(c[i].bb_lower));
    ctx.fill(); ctx.globalAlpha = 1;
    c.forEach((d, i) => {
      const col = d.close >= d.open ? up : down;
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(xs[i], sc.y(d.high)); ctx.lineTo(xs[i], sc.y(d.low)); ctx.stroke();
      const yo = sc.y(d.open), yc = sc.y(d.close), bwid = Math.max(1, bw * 0.65);
      ctx.globalAlpha = d.closed ? 1 : 0.45;
      ctx.fillRect(xs[i] - bwid / 2, Math.min(yo, yc), bwid, Math.max(1, Math.abs(yc - yo)));
      ctx.globalAlpha = 1;
    });
    line(ctx, xs, c.map((d) => d.ema_21), sc, accent, 1.4);
    line(ctx, xs, c.map((d) => d.ema_55), sc, ink2, 1.2);
    // past predictions as markers
    const preds = ((S.data.ledger && S.data.ledger.predictions) || []).filter((p) => p.prediction !== "NO TRADE");
    const idx = new Map(ts.map((t, i) => [t, i]));
    for (const p of preds) {
      const i = idx.get(p.candle_ts); if (i === undefined) continue;
      const isUp = p.prediction === "UP", y = isUp ? sc.y(c[i].low) + 10 : sc.y(c[i].high) - 10;
      ctx.fillStyle = isUp ? up : down; ctx.beginPath();
      ctx.moveTo(xs[i], y + (isUp ? -6 : 6)); ctx.lineTo(xs[i] - 5, y + (isUp ? 3 : -3)); ctx.lineTo(xs[i] + 5, y + (isUp ? 3 : -3)); ctx.fill();
    }
    timeLabels(ctx, xs, ts, bottom);
    $("chartNote").textContent = `Candles from ${S.data.candles.exchange}; blue line EMA 21, grey line EMA 55, shaded band Bollinger 20/2; triangles mark past UP/DOWN predictions.`;
    hover(xs, (i) => `${when(ts[i])} O ${priceFmt(c[i].open)} H ${priceFmt(c[i].high)} L ${priceFmt(c[i].low)} C ${priceFmt(c[i].close)}`);
  } else if (tab === "volume") {
    const sc = scale(c.map((d) => d.volume).concat([0]), top, bottom, 0.02);
    axis(ctx, sc, x0, x1, top, bottom, (v) => num(v, v > 100 ? 0 : 2));
    c.forEach((d, i) => { ctx.fillStyle = d.close >= d.open ? up : down; const y = sc.y(d.volume); ctx.fillRect(xs[i] - Math.max(1, bw * 0.7) / 2, y, Math.max(1, bw * 0.7), sc.y(0) - y); });
    timeLabels(ctx, xs, ts, bottom);
    $("chartNote").textContent = "Volume per candle in base currency, coloured by candle direction.";
    hover(xs, (i) => `${when(ts[i])} volume ${num(c[i].volume, 2)}`);
  } else if (tab === "ind") {
    const mid = top + (bottom - top) * 0.5;
    const rs = scale([0, 100], top, mid - 10, 0);
    axis(ctx, { ...rs, mn: 0, mx: 100, y: rs.y }, x0, x1, top, mid - 10, (v) => v.toFixed(0));
    ctx.setLineDash([4, 4]); ctx.strokeStyle = ink2;
    for (const lv of [30, 70]) { ctx.beginPath(); ctx.moveTo(x0, rs.y(lv)); ctx.lineTo(x1, rs.y(lv)); ctx.stroke(); }
    ctx.setLineDash([]);
    line(ctx, xs, c.map((d) => d.rsi_14), rs, accent, 1.5);
    line(ctx, xs, c.map((d) => d.adx_14), rs, ink2, 1.1);
    const ms = scale(c.map((d) => d.macd_hist).concat([0]), mid + 6, bottom);
    axis(ctx, ms, x0, x1, mid + 6, bottom, (v) => (v * 1e4).toFixed(1));
    c.forEach((d, i) => { if (d.macd_hist === null) return; ctx.fillStyle = d.macd_hist >= 0 ? up : down; const y = ms.y(d.macd_hist), y0 = ms.y(0); ctx.fillRect(xs[i] - Math.max(1, bw * 0.6) / 2, Math.min(y, y0), Math.max(1, bw * 0.6), Math.abs(y0 - y)); });
    timeLabels(ctx, xs, ts, bottom);
    $("chartNote").textContent = "Top: RSI 14 (blue) and ADX 14 (grey), dashed lines at 30 and 70. Bottom: MACD histogram in basis points of price.";
    hover(xs, (i) => `${when(ts[i])} RSI ${num(c[i].rsi_14, 1)} ADX ${num(c[i].adx_14, 1)} MACD ${num((c[i].macd_hist || 0) * 1e4, 2)} bps`);
  }
}

function drawCvd(ctx, x0, x1, top, bottom) {
  const f = S.data.flow && S.data.flow.flow_bars;
  let pts, note;
  if (f && f.length > 1) {
    pts = f.map((b) => ({ t: b.open_ts, v: b.cvd, d: b.delta }));
    note = `Live CVD from the ${S.data.flow.exchange} trade stream (taker buys minus taker sells, 1-minute bars, last 6 h).`;
  } else {
    const c = (S.data.candles && S.data.candles.candles) || [];
    pts = c.filter((d) => d.cvd !== null).map((d) => ({ t: d.open_ts, v: d.cvd, d: null }));
    note = pts.length ? "Live stream has no flow bars yet; showing CVD from exchange taker-buy volume per candle." : "";
  }
  $("chartNote").textContent = note;
  if (pts.length < 2) { empty("No order-flow data yet. CVD appears once the collector receives trades."); return; }
  empty("");
  const n = pts.length, bw = (x1 - x0) / n, xs = pts.map((_, i) => x0 + bw * (i + 0.5));
  const sc = scale(pts.map((p) => p.v), top, bottom);
  axis(ctx, sc, x0, x1, top, bottom, (v) => num(v, Math.abs(v) > 100 ? 0 : 2));
  line(ctx, xs, pts.map((p) => p.v), sc, css("--accent"), 1.8);
  timeLabels(ctx, xs, pts.map((p) => p.t), bottom);
  hover(xs, (i) => `${when(pts[i].t)} CVD ${num(pts[i].v, 2)}${pts[i].d !== null ? " Δ " + num(pts[i].d, 3) : ""}`);
}

function drawBook(ctx, w, h, x0, x1, top, bottom) {
  const b = S.data.book;
  if (!b || !b.available) { empty("No live order book yet for this exchange. It appears once the collector's WebSocket is connected."); $("chartNote").textContent = ""; return; }
  empty("");
  let cb = 0, ca = 0;
  const bids = b.bids.map(([p, q]) => [p, (cb += q)]), asks = b.asks.map(([p, q]) => [p, (ca += q)]);
  const all = bids.concat(asks);
  if (!all.length) { empty("Order book is empty."); return; }
  const px = all.map((x) => x[0]);
  const pmin = Math.min(...px), pmax = Math.max(...px);
  const X = (p) => x0 + ((p - pmin) / (pmax - pmin || 1)) * (x1 - x0);
  const sc = scale([0, cb, ca], top, bottom, 0.02);
  axis(ctx, sc, x0, x1, top, bottom, (v) => num(v, 2));
  const area = (pts, color, dir) => {
    if (!pts.length) return;
    ctx.fillStyle = color; ctx.globalAlpha = 0.25; ctx.beginPath(); ctx.moveTo(X(pts[0][0]), sc.y(0));
    for (const [p, c] of pts) ctx.lineTo(X(p), sc.y(c));
    ctx.lineTo(X(pts[pts.length - 1][0]), sc.y(0)); ctx.fill(); ctx.globalAlpha = 1;
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath();
    pts.forEach(([p, c], i) => (i ? ctx.lineTo(X(p), sc.y(c)) : ctx.moveTo(X(p), sc.y(c)))); ctx.stroke();
  };
  area(bids, css("--up")); area(asks, css("--down"));
  ctx.fillStyle = css("--ink-2"); ctx.textAlign = "center";
  for (const p of [pmin, (pmin + pmax) / 2, pmax]) ctx.fillText(priceFmt(p), Math.min(Math.max(X(p), 40), x1 - 30), bottom + 14);
  $("chartNote").textContent = `${b.exchange} top ${b.bids.length}/${b.asks.length} levels, cumulative size. Best bid ${priceFmt(b.best_bid)}, best ask ${priceFmt(b.best_ask)}, spread ${num(b.spread_bps, 2)} bps, imbalance ${num(b.imbalance, 2)}, updated ${ago(b.age_sec)}.`;
}

function hover(xs, text) {
  const c = $("chart"), tip = $("chartTip");
  const show = (clientX) => {
    const r = c.getBoundingClientRect(), x = clientX - r.left;
    let best = 0;
    for (let i = 1; i < xs.length; i++) if (Math.abs(xs[i] - x) < Math.abs(xs[best] - x)) best = i;
    tip.textContent = text(best); tip.hidden = false;
  };
  c.onmousemove = (e) => show(e.clientX);
  c.onmouseleave = () => (tip.hidden = true);
  c.ontouchmove = (e) => { if (e.touches[0]) show(e.touches[0].clientX); };
}

// ------------------------------------------------------------------ api key
function openKeyDialog() {
  const d = $("keyDialog");
  $("keyInput").value = store.get("key", "");
  if (typeof d.showModal === "function") d.showModal(); else d.setAttribute("open", "");
}
$("keyForm").addEventListener("submit", () => {
  const v = $("keyInput").value.trim();
  if (v) { store.set("key", v); toast("Key saved in this browser"); } else toast("Key is empty; nothing saved");
  S.keyPrompted = false; refresh();
});
$("keyCancel").addEventListener("click", () => $("keyDialog").close());
$("keyClear").addEventListener("click", () => { store.del("key"); $("keyInput").value = ""; $("keyDialog").close(); toast("Key removed from this browser"); });

init().catch((e) => banner("Could not load the dashboard: " + e.message));
