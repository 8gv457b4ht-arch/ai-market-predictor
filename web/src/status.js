/* Component status for the status panel. Pure functions (no DOM, no language): the app translates.
 *
 * Data statuses:   LIVE    - timestamp-verified real-time stream (WebSocket in this browser, or a continuous backend)
 *                  DELAYED - verified data, but obtained periodically (scheduled backend) or with a delay
 *                  STALE   - older than allowed
 *                  OFFLINE - the source is not reachable / never delivered data
 *                  ERROR   - processing failed
 * Service statuses (database, model, run, backup) use OK instead of LIVE/DELAYED.
 * A periodic backend can never produce LIVE.
 */
(function (root) {
  "use strict";
  const MIN = 60000, HOUR = 3600000;
  const isNum = (v) => typeof v === "number" && Number.isFinite(v);
  const age = (now, ts) => (isNum(ts) ? now - ts : Infinity);

  // freshness of something the backend produces every `interval` ms
  function dataState(now, ts, continuous, interval) {
    const a = age(now, ts);
    if (!isNum(ts)) return "OFFLINE";
    if (continuous && a <= Math.max(3 * MIN, 3 * interval)) return "LIVE";
    if (a <= 2 * interval + 20 * MIN) return "DELAYED";
    return "STALE";
  }
  function serviceState(now, ts, maxAge) { return !isNum(ts) ? "OFFLINE" : age(now, ts) <= maxAge ? "OK" : "STALE"; }

  function computeServer(sv, now, ctx) {
    ctx = ctx || {};
    const out = [];
    const browserLive = ctx.browserLive || {};
    const anyBrowserLive = Object.values(browserLive).some(Boolean);
    if (!sv) {
      out.push({ id: "api", state: "OFFLINE", code: ctx.fetchError ? "fetch_failed" : "waiting", params: { error: ctx.fetchError || "" } });
      return out;
    }
    const h = sv.health || {}, continuous = (h.backend_mode || sv.mode) === "continuous";
    const interval = (h.cycle_interval_sec || 900) * 1000;
    const pr = (sv.probes && sv.probes.probes) || {}, probeTs = sv.probes && sv.probes.ts_ms, stats = sv.source_stats || {};
    // 1. API = the published backend data
    out.push({ id: "api", state: dataState(now, sv.generated_ms, continuous, interval), ts: sv.generated_ms,
      code: continuous ? "continuous" : "periodic", params: { min: Math.round(interval / MIN) } });
    // 2-4. exchanges
    for (const ex of (sv.settings && sv.settings.exchanges) || ["binance", "bybit", "okx"]) {
      const p = pr[ex], st = stats[ex] || {};
      const rate = (ch) => st[ch] && st[ch]["24h"] ? st[ch]["24h"].error_rate : null;
      const base = { id: ex, ts: probeTs, restKind: p && p.rest && !p.rest.ok ? (p.rest.kind || (p.rest.partial ? "partial" : "error")) : null,
        wsKind: p && p.ws && !p.ws.verified_live ? (p.ws.kind || "no_data") : null, restOk: !!(p && p.rest && p.rest.ok),
        wsOk: !!(p && p.ws && p.ws.verified_live), restErrorRate: rate("rest"), wsErrorRate: rate("ws"),
        restError: p && p.rest && p.rest.error, wsError: p && p.ws && p.ws.error, primary: sv.primary_exchange === ex };
      if (browserLive[ex]) out.push({ ...base, state: "LIVE", code: "browser_live" });
      else if (!p) out.push({ ...base, state: "OFFLINE", code: "not_checked" });
      else if (base.restOk || base.wsOk) out.push({ ...base, state: dataState(now, probeTs, continuous, interval), code: base.restOk && base.wsOk ? "rest_ws_ok" : "partial", partial: !(base.restOk && base.wsOk) });
      else out.push({ ...base, state: "OFFLINE", code: "down" });
    }
    // 5. historical candles of the model exchange
    const primary = sv.primary_exchange, pp = pr[primary];
    if (!primary || !pp || !pp.rest) out.push({ id: "history", state: "OFFLINE", code: "no_primary" });
    else {
      const cs = Object.entries(pp.rest.candles || {}).filter(([k]) => (sv.settings.predict_timeframes || []).some((tf) => k.endsWith(" " + tf)));
      const gaps = cs.reduce((a, [, c]) => a + (c.gaps_remaining || 0), 0);
      const ok = cs.length && cs.every(([, c]) => c.ok);
      out.push({ id: "history", state: ok ? dataState(now, probeTs, continuous, interval) : cs.some(([, c]) => c.ok) ? dataState(now, probeTs, continuous, interval) : "OFFLINE",
        ts: probeTs, code: ok ? (gaps ? "gaps" : "complete") : "partial", partial: !ok || gaps > 0, params: { exchange: primary, gaps } });
    }
    // 6. WebSocket
    const wsOk = Object.keys(pr).filter((ex) => pr[ex].ws && pr[ex].ws.verified_live);
    if (anyBrowserLive) out.push({ id: "websocket", state: "LIVE", code: "browser_live", params: { list: Object.keys(browserLive).filter((k) => browserLive[k]).join(", ") } });
    else if (wsOk.length) out.push({ id: "websocket", state: continuous ? dataState(now, probeTs, true, interval) : dataState(now, probeTs, false, interval), ts: probeTs,
      code: continuous ? "backend_stream" : "backend_sample", params: { list: wsOk.join(", ") } });
    else out.push({ id: "websocket", state: "OFFLINE", code: "none_verified" });
    // 7. database
    const dbh = h.database || (sv.last_cycle && sv.last_cycle.db);
    if (dbh && dbh.ok === false) out.push({ id: "database", state: "ERROR", code: "integrity_failed", params: { problem: dbh.check } });
    else out.push({ id: "database", state: dbh ? serviceState(now, dbh.ts_ms, 2 * interval + 20 * MIN) : "OK", ts: dbh && dbh.ts_ms,
      code: h.last_recovery && age(now, h.last_recovery.ts_ms) < 24 * HOUR ? "recovered" : "integrity_ok",
      params: { predictions: dbh ? dbh.predictions : sv.ledger_summary && sv.ledger_summary.total } });
    // 8. models
    const keys = Object.keys(sv.models || {}), ready = keys.filter((k) => sv.models[k].production);
    const mc = h.models_check || {}, broken = Object.values(mc).filter((x) => x && x.status === "broken").length;
    const passed = keys.filter((k) => sv.models[k].baseline_test && sv.models[k].baseline_test.passed).length;
    out.push({ id: "model", state: broken ? "ERROR" : !ready.length ? "OFFLINE" : "OK", code: broken ? "broken" : !ready.length ? "not_ready" : "ready",
      partial: ready.length < keys.length || passed < ready.length, params: { ready: ready.length, total: keys.length, passed } });
    // 9. news
    const ns = (sv.news && sv.news.status) || {};
    if (!ns.ts_ms) out.push({ id: "news", state: "OFFLINE", code: "not_started" });
    else if (!ns.feeds_ok) out.push({ id: "news", state: "OFFLINE", code: "feeds_down", ts: ns.ts_ms });
    else out.push({ id: "news", state: dataState(now, ns.ts_ms, continuous, Math.max(interval, 5 * MIN)), ts: ns.ts_ms, code: ns.llm_error ? "llm_error" : "ok",
      partial: !!ns.llm_error, params: { ok: ns.feeds_ok, total: ns.feeds_total, analyzer: ns.analyzer || "" } });
    // 10. last successful prediction (shortest timeframe sets the expected pace)
    const tfMin = Math.min(...((sv.settings && sv.settings.predict_timeframes) || ["15m"]).map((t) => ({ "15m": 15, "1h": 60, "4h": 240 }[t] || 15)));
    out.push({ id: "prediction", state: dataState(now, h.last_prediction_ms, continuous, tfMin * MIN), ts: h.last_prediction_ms, code: h.last_prediction_ms ? "made" : "none" });
    // 11. background run
    const lc = sv.last_cycle || {}, started = h.last_cycle_started_ms || lc.started_ms;
    const errs = (lc.errors || []).map((e) => e.step);
    out.push({ id: "run", state: errs.length ? "ERROR" : serviceState(now, started, 2 * interval + 20 * MIN), ts: started,
      code: errs.length ? "errors" : continuous ? "continuous" : "scheduled", params: { steps: errs.join(", "), sec: lc.duration_sec } });
    // 12. backup
    const lb = sv.last_backup;
    out.push({ id: "backup", state: lb ? serviceState(now, lb.ts_ms, 9 * HOUR) : "OFFLINE", ts: lb && lb.ts_ms, code: lb ? "verified" : "none",
      params: { file: lb && lb.database } });
    return out;
  }

  // Local mode: the browser itself is the backend
  function computeLocal(ls, now) {
    const out = [{ id: "api", state: "OFFLINE", code: "local_mode" }];
    for (const ex of ls.exchanges) {
      const s = ls.streams[ex];
      out.push({ id: ex, state: s === "live" ? "LIVE" : s ? "OFFLINE" : "OFFLINE", code: s === "live" ? "browser_live" : s ? "browser_" + s : "not_started" });
    }
    const fresh = ls.lastPoll && now - ls.lastPoll < 3 * MIN && ls.lastClose && now - ls.lastClose < ls.tfMs + 5 * MIN;
    out.push({ id: "history", state: !ls.primary ? "OFFLINE" : fresh ? "DELAYED" : ls.hasCandles ? "STALE" : "OFFLINE", ts: ls.lastPoll,
      code: ls.primary ? "browser_poll" : "no_primary", params: { exchange: ls.primary || "" } });
    const live = ls.exchanges.filter((ex) => ls.streams[ex] === "live");
    out.push({ id: "websocket", state: live.length ? "LIVE" : "OFFLINE", code: live.length ? "browser_live" : "none_verified", params: { list: live.join(", ") } });
    out.push({ id: "database", state: ls.dbOk ? "OK" : "ERROR", code: ls.dbOk ? "browser_storage" : "memory_only", params: { predictions: ls.ledgerSize } });
    out.push({ id: "model", state: ls.training ? "OK" : ls.modelReady ? "OK" : "OFFLINE", code: ls.training ? "training" : ls.modelReady ? "ready" : "not_ready", params: {} });
    out.push({ id: "news", state: ls.newsLive ? "DELAYED" : "OFFLINE", ts: ls.newsTs, code: ls.newsLive ? "ok" : "feeds_down", params: { analyzer: ls.newsAnalyzer || "" } });
    out.push({ id: "prediction", state: dataState(now, ls.lastPrediction, false, ls.tfMs), ts: ls.lastPrediction, code: ls.lastPrediction ? "made" : "none" });
    out.push({ id: "run", state: ls.started ? "OK" : "OFFLINE", code: ls.started ? "page_open" : "stopped" });
    out.push({ id: "backup", state: "OFFLINE", code: "local_none" });
    return out;
  }

  const TONE = { LIVE: "good", OK: "good", DELAYED: "info", STALE: "warn", OFFLINE: "bad", ERROR: "bad" };
  const api = { computeServer, computeLocal, dataState, serviceState, TONE };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.AMPStatus = api;
})(typeof self !== "undefined" ? self : this);
