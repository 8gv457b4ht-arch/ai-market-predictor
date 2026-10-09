// Translation dictionary completeness and status rules (pure logic, no browser).
const test = require("node:test");
const assert = require("node:assert");
const fs = require("fs");
const path = require("path");
const I = require("../src/i18n.js");
const ST = require("../src/status.js");

const SRC = ["app.js", "index.html"].map((f) => fs.readFileSync(path.join(__dirname, "../src", f), "utf8")).join("\n");

test("all three languages have exactly the same keys and no empty text", () => {
  const ru = Object.keys(I.DICT.ru);
  for (const l of ["uk", "en"]) {
    assert.deepStrictEqual(Object.keys(I.DICT[l]).sort(), ru.slice().sort(), l);
    for (const k of ru) assert.ok(typeof I.DICT[l][k] === "string" && (I.DICT[l][k].length > 0 || k === "cls."), `${l}:${k}`);
  }
});

test("every literal key used by the page exists, placeholders match across languages", () => {
  const used = new Set([...SRC.matchAll(/\bt\("([a-z0-9_.]+)"/g)].map((m) => m[1])
    .concat([...SRC.matchAll(/data-i18n(?:-aria|-ph)?="([a-z0-9_.]+)"/g)].map((m) => m[1])).filter((k) => !k.endsWith(".")));
  const missing = [...used].filter((k) => !(k in I.DICT.ru));
  assert.deepStrictEqual(missing, []);
  const ph = (s) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort().join(",");
  for (const k of Object.keys(I.DICT.ru)) for (const l of ["uk", "en"]) assert.strictEqual(ph(I.DICT[l][k]), ph(I.DICT.ru[k]), `${l}:${k}`);
});

test("dynamic key families are complete (reasons, regimes, statuses, error kinds, outcome classes)", () => {
  const need = {
    "reason.": ["confidence_below_threshold", "edge_below_min_edge", "flat_more_likely", "stale_data", "late_decision", "insufficient_history",
      "model_not_ready", "low_liquidity", "model_not_better_than_baseline", "no_edge_after_costs", "data_quality", "abnormal_market", "news_conflict"],
    "regime.": ["trending", "ranging", "high_volatility", "low_liquidity", "abnormal"],
    "dir.": ["UP", "DOWN", "FLAT", "NO_TRADE"],
    "kind.": ["region_blocked", "cors", "network", "timeout", "rate_limited", "http_error", "stale_data", "no_data", "stale", "gap", "exchange_error"],
    "cls.": ["correct", "wrong_direction", "false_move", "correct_abstain", "gated_would_be_correct", "missed_move", "late"],
    "st.name.": ["api", "binance", "bybit", "okx", "history", "websocket", "database", "model", "news", "prediction", "run", "backup"],
  };
  for (const [p, list] of Object.entries(need)) for (const k of list) assert.ok(I.has(p + k), p + k);
  // every status code the status module can emit has a text
  const src = fs.readFileSync(path.join(__dirname, "../src/status.js"), "utf8");
  for (const m of src.matchAll(/id: "(\w+)"[^}]*?code: ([^,}]+)/g)) {
    for (const c of m[2].matchAll(/"(\w+)"/g)) {
      const id = m[1];
      if (["binance", "bybit", "okx"].includes(id)) continue;
      assert.ok(I.has(`st.${id}.${c[1]}`) || I.has(`st.code.${c[1]}`), `st.${id}.${c[1]}`);
    }
  }
});

test("Russian is the default; switching language changes texts and is remembered in memory", () => {
  assert.strictEqual(I.lang(), "ru");
  assert.strictEqual(I.t("sys.title"), "Состояние системы");
  I.setLang("uk"); assert.strictEqual(I.t("sys.title"), "Стан системи");
  I.setLang("en"); assert.strictEqual(I.t("time.min_ago", { n: 5 }), "5 min ago");
  I.setLang("xx"); assert.strictEqual(I.lang(), "en");
  I.setLang("ru");
});

const NOW = 1_791_550_000_000, MIN = 60_000;
function state(over) {
  return Object.assign({
    generated_ms: NOW - 5 * MIN, mode: "scheduled", primary_exchange: "binance",
    settings: { exchanges: ["binance", "bybit", "okx"], predict_timeframes: ["15m", "1h"] },
    probes: { ts_ms: NOW - 6 * MIN, probes: {
      binance: { rest: { ok: true, candles: { "BTC/USDT 15m": { ok: true, gaps_remaining: 0 } } }, ws: { verified_live: true } },
      bybit: { rest: { ok: false, kind: "region_blocked", error: "HTTP 403" }, ws: { verified_live: true } },
      okx: { rest: { ok: false, kind: "timeout" }, ws: { verified_live: false, kind: "timeout" } } } },
    models: { "BTC/USDT|15m|h4": { production: {}, baseline_test: { passed: false } } },
    news: { status: { ts_ms: NOW - 6 * MIN, feeds_ok: 8, feeds_total: 8, analyzer: "rules" } },
    health: { backend_mode: "scheduled", cycle_interval_sec: 900, last_prediction_ms: NOW - 7 * MIN, last_cycle_started_ms: NOW - 8 * MIN,
      database: { ok: true, ts_ms: NOW - 6 * MIN, predictions: 20 } },
    last_cycle: { errors: [] }, last_backup: { ts_ms: NOW - 2 * 3600e3, database: "market_x.sqlite3.gz" },
  }, over || {});
}
const byId = (arr) => Object.fromEntries(arr.map((s) => [s.id, s]));

test("a scheduled backend is never LIVE; the 12 components are all reported", () => {
  const s = byId(ST.computeServer(state(), NOW, {}));
  assert.deepStrictEqual(Object.keys(s), ["api", "binance", "bybit", "okx", "history", "websocket", "database", "model", "news", "prediction", "run", "backup"]);
  assert.ok(!Object.values(s).some((x) => x.state === "LIVE"));
  assert.strictEqual(s.api.state, "DELAYED");
  assert.strictEqual(s.binance.state, "DELAYED");
  assert.strictEqual(s.bybit.state, "DELAYED"); assert.ok(s.bybit.partial); assert.strictEqual(s.bybit.restKind, "region_blocked");
  assert.strictEqual(s.okx.state, "OFFLINE");
  assert.strictEqual(s.database.state, "OK"); assert.strictEqual(s.backup.state, "OK"); assert.strictEqual(s.model.state, "OK");
  assert.strictEqual(s.model.params.passed, 0);
});

test("LIVE only from timestamp-verified streams in this browser or a fresh continuous backend", () => {
  let s = byId(ST.computeServer(state(), NOW, { browserLive: { okx: true } }));
  assert.strictEqual(s.okx.state, "LIVE"); assert.strictEqual(s.websocket.state, "LIVE"); assert.strictEqual(s.api.state, "DELAYED");
  s = byId(ST.computeServer(state({ mode: "continuous", generated_ms: NOW - MIN, health: { backend_mode: "continuous", cycle_interval_sec: 60, last_prediction_ms: NOW - MIN } ,
    probes: { ts_ms: NOW - MIN, probes: state().probes.probes } }), NOW, {}));
  assert.strictEqual(s.api.state, "LIVE"); assert.strictEqual(s.binance.state, "LIVE");
});

test("old data turns STALE, failures turn ERROR / OFFLINE", () => {
  let s = byId(ST.computeServer(state({ generated_ms: NOW - 3 * 3600e3, probes: { ts_ms: NOW - 3 * 3600e3, probes: state().probes.probes } }), NOW, {}));
  assert.strictEqual(s.api.state, "STALE"); assert.strictEqual(s.binance.state, "STALE");
  s = byId(ST.computeServer(state({ last_cycle: { errors: [{ step: "learning" }] }, health: { ...state().health, database: { ok: false, check: "malformed" } } }), NOW, {}));
  assert.strictEqual(s.run.state, "ERROR"); assert.strictEqual(s.database.state, "ERROR");
  s = byId(ST.computeServer(null, NOW, { fetchError: "HTTP 404" }));
  assert.strictEqual(s.api.state, "OFFLINE");
  s = byId(ST.computeServer(state({ last_backup: null, news: { status: { ts_ms: NOW, feeds_ok: 0, feeds_total: 8 } } }), NOW, {}));
  assert.strictEqual(s.backup.state, "OFFLINE"); assert.strictEqual(s.news.state, "OFFLINE");
});
