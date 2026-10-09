// Node tests for the pure core. Synthetic data here is TEST-ONLY and never shipped in the app.
const test = require("node:test");
const assert = require("node:assert/strict");
const A = require("../src/core.js");

function series(n, kind, seed, tf = "15m", start = 1_700_000_000_000 - (1_700_000_000_000 % 86_400_000)) {
  const r = A.rng(seed); const g = () => { let u = 0; for (let i = 0; i < 6; i++) u += r(); return (u - 3) / Math.sqrt(0.5); };
  let p = 30000, trend = 0; const out = [];
  for (let i = 0; i < n; i++) {
    let ret;
    if (kind === "planted") { trend = 0.97 * trend + 0.0012 * g(); ret = trend + 0.002 * g(); } else ret = 0.004 * g();
    const o = p; p = p * Math.exp(ret); const sp = Math.abs(0.0015 * g()) * p, v = Math.exp(3 + 0.4 * g());
    out.push({ open_ts: start + i * A.TF_MS[tf], open: o, high: Math.max(o, p) + sp, low: Math.min(o, p) - sp, close: p, volume: v,
      taker_buy_volume: v * Math.min(0.95, Math.max(0.05, 0.5 + 0.08 * g())) });
  }
  return out;
}
function resample(base, baseTf, tf) {
  const k = A.TF_MS[tf] / A.TF_MS[baseTf], out = [];
  for (let i = 0; i + k <= base.length; i += k) {
    const c = base.slice(i, i + k);
    out.push({ open_ts: c[0].open_ts, open: c[0].open, close: c[k - 1].close, high: Math.max(...c.map((x) => x.high)),
      low: Math.min(...c.map((x) => x.low)), volume: c.reduce((a, x) => a + x.volume, 0), taker_buy_volume: c.reduce((a, x) => a + x.taker_buy_volume, 0) });
  }
  return out;
}
const job = (base, extra = {}) => ({ candlesByTf: { "15m": base, "1h": resample(base, "15m", "1h"), "4h": resample(base, "15m", "4h") },
  tf: "15m", horizon: 4, atrMult: 0.3, costBps: 25, folds: 3, minTrain: 600, threshold: 0.55, minEdge: 0.1,
  costs: { feeBps: 10, slippageBps: 2, spreadBps: 1, latencyMs: 500, threshold: 0.55, minEdge: 0.1 }, ...extra });

test("indicators are bounded and complete", () => {
  const ind = A.computeIndicators(series(600, "rw", 1));
  const rsi = ind.rsi_14.filter(A.isNum); assert.ok(rsi.length > 500 && rsi.every((v) => v >= 0 && v <= 100));
  assert.ok(ind.adx_14.filter(A.isNum).every((v) => v >= 0 && v <= 100));
  assert.ok(ind.cvd.every(A.isNum));
  for (const f of A.BASE_FEATURES) assert.ok(f in ind, f);
});

test("features are causal: changing the future does not change the past", () => {
  const s = series(800, "rw", 2), a = A.computeIndicators(s);
  const t = s.map((c, i) => (i >= 600 ? { ...c, open: c.open * 1.5, high: c.high * 1.5, low: c.low * 1.5, close: c.close * 1.5, volume: c.volume * 10 } : c));
  const b = A.computeIndicators(t);
  for (const f of A.BASE_FEATURES.concat(["atr_pct_rank", "ret_z"])) for (let i = 0; i < 600; i++) {
    const x = a[f][i], y = b[f][i]; assert.ok((Number.isNaN(x) && Number.isNaN(y)) || x === y, `${f}[${i}]`);
  }
});

test("higher timeframe values come only from closed bars", () => {
  const base = series(1200, "rw", 3), h1 = resample(base, "15m", "1h");
  const fr = A.buildFrame({ "15m": base, "1h": h1 }, "15m"), hi = A.computeIndicators(h1);
  for (const i of [300, 555, 1000]) {
    const row = fr.rows[i]; let j = -1; h1.forEach((c, k) => { if (c.open_ts + 3600e3 <= row.decision_ts) j = k; });
    const v = row["1h_rsi_14"], w = hi.rsi_14[j]; assert.ok((Number.isNaN(v) && Number.isNaN(w)) || v === w);
  }
});

test("labels use the configured threshold and leave the future unlabelled", () => {
  const rows = [100, 101, 99, 100, 103].map((c) => ({ close: c, atr_pct: 0.001 }));
  A.addLabels(rows, 1, 0, 50);
  assert.deepEqual(rows.slice(0, 4).map((r) => r.label), [2, 0, 2, 2]);
  assert.equal(rows[4].label, null);
});

test("gating", () => {
  assert.equal(A.gate(0.2, 0.1, 0.7, 0.55, 0.1).signal, "UP");
  assert.equal(A.gate(0.3, 0.3, 0.4, 0.55, 0.1).signal, "NO TRADE");
  assert.equal(A.gate(0.6, 0.65, 0.0, 0.55, 0.1).signal, "NO TRADE");
  assert.equal(A.gate(0.7, 0.1, 0.2, 0.55, 0.1).signal, "DOWN");
});

test("leakage canary: no edge on a random walk", () => {
  const t0 = Date.now();
  const res = A.trainBaseline(job(series(2600, "rw", 4)));
  const m = res.metrics;
  console.log(`  random walk: OOS logloss ${m.log_loss.toFixed(4)} vs baseline ${m.baseline_prior.log_loss.toFixed(4)}, signals ${m.signals.signals}, ${(Date.now() - t0) / 1000}s`);
  assert.ok(m.log_loss > m.baseline_prior.log_loss - 0.02);
  assert.ok(m.accuracy < 0.45);
});

test("the pipeline learns a real planted signal", () => {
  const res = A.trainBaseline(job(series(2600, "planted", 5)));
  const m = res.metrics;
  console.log(`  planted: OOS logloss ${m.log_loss.toFixed(4)} vs baseline ${m.baseline_prior.log_loss.toFixed(4)}, hit ${m.signals.hit_rate}`);
  assert.ok(m.log_loss < m.baseline_prior.log_loss - 0.1);
  assert.ok(m.signals.signals > 0 && m.signals.hit_rate > 0.6);
  const P = A.predictEnsemble(res.model, A.toMatrix(A.buildFrame(job(series(700, "planted", 6)).candlesByTf, "15m").rows.slice(-5), res.model.features));
  P.forEach((p) => assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-9));
  const json = JSON.parse(JSON.stringify(res.model)); // models must survive storage
  const P2 = A.predictEnsemble(json, A.toMatrix(A.buildFrame(job(series(700, "planted", 6)).candlesByTf, "15m").rows.slice(-5), res.model.features));
  assert.deepEqual(P2, P);
});

test("champion/challenger only promotes on unseen data and real gains", () => {
  const all = series(3300, "planted", 8);
  const first = A.trainBaseline(job(all.slice(0, 2500)));
  const prod = { model: first.model, train_end_ts: first.train_end_ts };
  const wait = A.challengerCycle(job(all.slice(0, 2500), { production: prod, minNewLabels: 200, minHoldout: 100, holdoutFraction: 0.5, minLoglossGain: 0.002, bootstrapConfidence: 0.9 }));
  assert.equal(wait.status, "waiting");
  const r = A.challengerCycle(job(all, { production: prod, minNewLabels: 200, minHoldout: 100, holdoutFraction: 0.5, minLoglossGain: 0.002, bootstrapConfidence: 0.9 }));
  assert.ok(["promoted", "rejected"].includes(r.status));
  assert.ok(r.holdout_start > prod.train_end_ts + 4 * 900e3 && r.train_end_ts < r.holdout_start);
  assert.equal(r.status === "promoted", r.comparison.promote);
  const y = Array.from({ length: 300 }, (_, i) => i % 3), good = y.map((t) => [0, 1, 2].map((k) => (k === t ? 0.7 : 0.15))), flat = y.map(() => [1 / 3, 1 / 3, 1 / 3]);
  const cfg = { minLoglossGain: 0.002, bootstrapConfidence: 0.9 };
  assert.equal(A.compareModels(y, flat, good, cfg, 4).promote, true);
  assert.equal(A.compareModels(y, good, good, cfg, 4).promote, false);
});

test("backtest charges costs", () => {
  const oos = Array.from({ length: 300 }, (_, i) => ({ open_ts: i * 900e3, open: 100 * 1.001 ** i, close: 100 * 1.001 ** (i + 1), p: [0.1, 0.1, 0.8] }));
  const free = A.backtest(oos, "15m", 4, { threshold: 0.55, minEdge: 0.1 });
  const paid = A.backtest(oos, "15m", 4, { threshold: 0.55, minEdge: 0.1, feeBps: 10, slippageBps: 5, spreadBps: 2 });
  assert.equal(free.trades, paid.trades);
  assert.ok(Math.abs(free.avg_net_bps - paid.avg_net_bps - 32) < 1);
});

test("data quality flags stale data, gaps and exchange disagreement", () => {
  const c = series(300, "rw", 9), now = c[299].open_ts + 900e3 + 5000;
  const ok = A.evaluateQuality({ candles: c, tf: "15m", now, stream: { state: "live", lastMsg: now }, exchangePrices: { binance: 100, bybit: 100.01 }, maxDivergenceBps: 40, maxSpreadBps: 15, primary: "binance" });
  assert.equal(ok.label, "GOOD");
  assert.equal(A.evaluateQuality({ candles: c, tf: "15m", now: now + 3600e3 }).ok, false);
  const gap = A.evaluateQuality({ candles: c.filter((_, i) => i < 100 || i > 120), tf: "15m", now });
  assert.ok(gap.issues.some((i) => i.code === "missing_bars") && !gap.ok);
  const dis = A.evaluateQuality({ candles: c, tf: "15m", now, exchangePrices: { binance: 100, bybit: 100, okx: 101.5 }, maxDivergenceBps: 40, primary: "binance" });
  assert.ok(dis.issues.some((i) => i.code === "exchange_disagreement" && i.severity === "warning"));
});

test("news rules and freshness-weighted features", () => {
  const etf = A.analyzeNewsRules({ title: "Bitcoin ETF sees record inflows as price surges", published_ms: 1000 }, []);
  const hack = A.analyzeNewsRules({ title: "Major crypto exchange hacked, $200M drained", published_ms: 1000 }, []);
  assert.equal(etf.category, "etf"); assert.ok(etf.direction > 0 && etf.affected_assets.includes("BTC"));
  assert.equal(hack.category, "hacks"); assert.ok(hack.direction < 0 && hack.expected_horizon_min === 60);
  assert.ok(etf.confidence <= 0.6);
  const now = 10 * 3600e3, ev = (t) => ({ ...etf, timestamp: now - t * 60e3, event: `e${t}` });
  const f = A.newsFeatures([ev(10), ev(600)], "BTC/USDT", now, 180);
  assert.ok(f.news_top[0].weight > 4 * f.news_top[1].weight);
  assert.equal(A.newsFeatures([ev(-30)], "BTC/USDT", now, 180).news_event_count, 0); // future news ignored
});

test("probabilities match base rates on imbalanced, uninformative labels", () => {
  const r = A.rng(11), n = 1400, d = 6;
  const X = Array.from({ length: n }, () => Array.from({ length: d }, () => r()));
  const y = X.map(() => { const u = r(); return u < 0.15 ? 0 : u < 0.85 ? 1 : 2; });
  const m = A.fitEnsemble(X.slice(0, 1000), y.slice(0, 1000), ["a", "b", "c", "d", "e", "f"]);
  const P = A.predictEnsemble(m, X.slice(1000));
  const mean = (k) => P.reduce((s, p) => s + p[k], 0) / P.length;
  assert.ok(Math.abs(mean(1) - 0.7) < 0.08, `FLAT ${mean(1)}`);
  assert.ok(Math.abs(mean(0) - 0.15) < 0.06 && Math.abs(mean(2) - 0.15) < 0.06);
});
