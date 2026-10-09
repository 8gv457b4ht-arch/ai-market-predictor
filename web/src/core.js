/* AI Market Predictor - core (pure computation, no I/O).
 * Runs in the page, in a Web Worker and in Node (tests).
 * Port of backend/app/ml/* and learning/engine.py from the Python project.
 */
(function (root) {
  "use strict";
  const TF_MS = { "1m": 60e3, "5m": 300e3, "15m": 900e3, "30m": 1800e3, "1h": 3600e3, "4h": 14400e3, "1d": 86400e3 };
  const CLASSES = ["DOWN", "FLAT", "UP"];
  const FEATURE_VERSION = "js-fv2"; // fv2: natural class weights + temperature/bias calibration
  const BASE_FEATURES = ["ret_1", "ret_3", "ret_6", "ret_12", "roc_10", "ema_gap_9_21", "ema_gap_21_55", "close_vs_ema200",
    "ema21_slope", "rsi_14", "macd", "macd_signal", "macd_hist", "atr_pct", "adx_14", "di_diff", "bb_width", "bb_pos",
    "rv_20", "rv_ratio", "vol_z", "trend_strength", "range_pct", "body_pct", "buy_ratio", "flow_imbalance_20"];
  const CONTEXT_FEATURES = ["ret_1", "ret_3", "rsi_14", "macd_hist", "adx_14", "di_diff", "atr_pct", "bb_pos",
    "trend_strength", "ema_gap_9_21", "vol_z", "flow_imbalance_20"];
  const REGIMES = ["trending", "ranging", "high_volatility", "low_liquidity", "abnormal"];
  const NaN_ = Number.NaN;
  const isNum = (v) => typeof v === "number" && Number.isFinite(v);

  // ------------------------------------------------------------ random
  function rng(seed) { // mulberry32
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0; let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // -------------------------------------------------------- indicators
  function emaSeries(x, span, minPeriods) {
    const a = 2 / (span + 1), out = new Array(x.length).fill(NaN_);
    let s = NaN_, n = 0;
    for (let i = 0; i < x.length; i++) {
      if (!isNum(x[i])) { if (n >= minPeriods) out[i] = s; continue; }
      s = n === 0 ? x[i] : a * x[i] + (1 - a) * s; n++;
      if (n >= minPeriods) out[i] = s;
    }
    return out;
  }
  function wilder(x, n) {
    const a = 1 / n, out = new Array(x.length).fill(NaN_);
    let s = NaN_, c = 0;
    for (let i = 0; i < x.length; i++) {
      if (!isNum(x[i])) { if (c >= n) out[i] = s; continue; }
      s = c === 0 ? x[i] : a * x[i] + (1 - a) * s; c++;
      if (c >= n) out[i] = s;
    }
    return out;
  }
  function rolling(x, n, fn, minPeriods) {
    const out = new Array(x.length).fill(NaN_), mp = minPeriods || n;
    for (let i = 0; i < x.length; i++) {
      const w = [];
      for (let j = Math.max(0, i - n + 1); j <= i; j++) if (isNum(x[j])) w.push(x[j]);
      if (w.length >= mp) out[i] = fn(w);
    }
    return out;
  }
  const mean = (w) => w.reduce((a, b) => a + b, 0) / w.length;
  const std = (w) => { if (w.length < 2) return NaN_; const m = mean(w); return Math.sqrt(w.reduce((a, b) => a + (b - m) ** 2, 0) / (w.length - 1)); };
  const sum = (w) => w.reduce((a, b) => a + b, 0);
  function rollingFast(x, n, kind) { // O(n) mean/std/sum for full windows (no NaN inside)
    if (kind === "mean") return rolling(x, n, mean);
    if (kind === "std") return rolling(x, n, std);
    return rolling(x, n, sum);
  }
  const diff = (x, k) => x.map((v, i) => (i >= k && isNum(v) && isNum(x[i - k]) ? v - x[i - k] : NaN_));
  const pct = (x, k) => x.map((v, i) => (i >= k && isNum(v) && isNum(x[i - k]) && x[i - k] !== 0 ? v / x[i - k] - 1 : NaN_));
  const zip = (a, b, f) => a.map((v, i) => { const r = f(v, b[i]); return isNum(r) ? r : NaN_; });

  function computeIndicators(c) {
    const n = c.length;
    const close = c.map((r) => r.close), high = c.map((r) => r.high), low = c.map((r) => r.low);
    const open = c.map((r) => r.open), vol = c.map((r) => r.volume);
    const out = { open_ts: c.map((r) => r.open_ts), open, high, low, close, volume: vol };
    const logc = close.map(Math.log);
    out.ret_1 = diff(logc, 1); out.ret_3 = diff(logc, 3); out.ret_6 = diff(logc, 6); out.ret_12 = diff(logc, 12);
    out.roc_10 = pct(close, 10);
    const e = {};
    for (const s of [9, 12, 21, 26, 55, 200]) e[s] = emaSeries(close, s, s);
    out.ema_9 = e[9]; out.ema_21 = e[21]; out.ema_55 = e[55]; out.ema_200 = e[200];
    out.ema_gap_9_21 = close.map((v, i) => (e[9][i] - e[21][i]) / v);
    out.ema_gap_21_55 = close.map((v, i) => (e[21][i] - e[55][i]) / v);
    out.close_vs_ema200 = close.map((v, i) => v / e[200][i] - 1);
    out.ema21_slope = pct(e[21], 5);
    const d = diff(close, 1);
    const gain = wilder(d.map((v) => (isNum(v) ? Math.max(v, 0) : v)), 14);
    const loss = wilder(d.map((v) => (isNum(v) ? Math.max(-v, 0) : v)), 14);
    out.rsi_14 = gain.map((g, i) => (!isNum(g) ? NaN_ : loss[i] === 0 ? 100 : 100 - 100 / (1 + g / loss[i])));
    const macd = e[12].map((v, i) => v - e[26][i]);
    const sig = emaSeries(macd, 9, 9);
    out.macd = macd.map((v, i) => v / close[i]); out.macd_signal = sig.map((v, i) => v / close[i]);
    out.macd_hist = macd.map((v, i) => (v - sig[i]) / close[i]);
    const tr = close.map((_, i) => (i === 0 ? high[i] - low[i] : Math.max(high[i] - low[i], Math.abs(high[i] - close[i - 1]), Math.abs(low[i] - close[i - 1]))));
    const atr = wilder(tr, 14);
    out.atr_14 = atr; out.atr_pct = atr.map((v, i) => v / close[i]);
    const pdm = [], mdm = [];
    for (let i = 0; i < n; i++) {
      if (i === 0) { pdm.push(NaN_); mdm.push(NaN_); continue; }
      const up = high[i] - high[i - 1], dn = low[i - 1] - low[i];
      pdm.push(up > dn && up > 0 ? up : 0); mdm.push(dn > up && dn > 0 ? dn : 0);
    }
    const pdi = wilder(pdm, 14).map((v, i) => (100 * v) / atr[i]);
    const mdi = wilder(mdm, 14).map((v, i) => (100 * v) / atr[i]);
    const dx = pdi.map((v, i) => (v + mdi[i] > 0 ? (100 * Math.abs(v - mdi[i])) / (v + mdi[i]) : NaN_));
    out.di_diff = pdi.map((v, i) => (v - mdi[i]) / 100);
    out.adx_14 = wilder(dx, 14);
    const mid = rollingFast(close, 20, "mean"), sd = rollingFast(close, 20, "std");
    out.bb_mid = mid; out.bb_upper = mid.map((m, i) => m + 2 * sd[i]); out.bb_lower = mid.map((m, i) => m - 2 * sd[i]);
    out.bb_width = mid.map((m, i) => (4 * sd[i]) / m);
    out.bb_pos = close.map((v, i) => (sd[i] > 0 ? (v - out.bb_lower[i]) / (4 * sd[i]) : NaN_));
    out.rv_20 = rollingFast(out.ret_1, 20, "std");
    out.rv_100 = rolling(out.ret_1, 100, std, 50);
    out.rv_ratio = out.rv_20.map((v, i) => v / out.rv_100[i]);
    const lv = vol.map((v) => Math.log1p(v));
    const lvm = rolling(lv, 50, mean, 20), lvs = rolling(lv, 50, std, 20);
    out.vol_z = lv.map((v, i) => (lvs[i] > 0 ? (v - lvm[i]) / lvs[i] : NaN_));
    out.trend_strength = close.map((_, i) => (e[9][i] - e[55][i]) / atr[i]);
    out.range_pct = close.map((v, i) => (high[i] - low[i]) / v);
    out.body_pct = close.map((v, i) => (v - open[i]) / v);
    const hasTbv = c.some((r) => isNum(r.taker_buy_volume));
    if (hasTbv) {
      const tbv = c.map((r) => (isNum(r.taker_buy_volume) ? r.taker_buy_volume : NaN_));
      out.buy_ratio = tbv.map((v, i) => (vol[i] > 0 ? v / vol[i] : NaN_));
      const flow = tbv.map((v, i) => 2 * v - vol[i]);
      out.flow_delta = flow;
      let cum = 0; out.cvd = flow.map((v) => (isNum(v) ? (cum += v) : NaN_));
      const fs = rollingFast(flow, 20, "sum"), vs = rollingFast(vol, 20, "sum");
      out.flow_imbalance_20 = fs.map((v, i) => (vs[i] > 0 ? v / vs[i] : NaN_));
    } else {
      for (const k of ["buy_ratio", "flow_delta", "cvd", "flow_imbalance_20"]) out[k] = new Array(n).fill(NaN_);
    }
    // rolling percentile rank of ATR% (pandas rank(pct=True), average ties)
    out.atr_pct_rank = out.atr_pct.map((v, i) => {
      if (!isNum(v)) return NaN_;
      let less = 0, eq = 0, cnt = 0;
      for (let j = Math.max(0, i - 199); j <= i; j++) { const w = out.atr_pct[j]; if (!isNum(w)) continue; cnt++; if (w < v) less++; else if (w === v) eq++; }
      return cnt >= 50 ? (less + (eq + 1) / 2) / cnt : NaN_;
    });
    out.ret_z = out.ret_1.map((v, i) => v / out.rv_100[i]);
    for (const k of Object.keys(out)) if (Array.isArray(out[k])) out[k] = out[k].map((v) => (typeof v === "number" && !Number.isFinite(v) ? NaN_ : v));
    return out;
  }

  function classifyRegime(adx, atrRank, volZ, retZ, opts) {
    opts = opts || {};
    if (opts.qualityOk === false || (isNum(retZ) && Math.abs(retZ) > 6)) return "abnormal";
    if (isNum(atrRank) && atrRank >= 0.9) return "high_volatility";
    if ((isNum(volZ) && volZ <= -1.5) || (isNum(opts.spreadBps) && isNum(opts.maxSpreadBps) && opts.spreadBps > opts.maxSpreadBps)) return "low_liquidity";
    if (isNum(adx) && adx >= 25) return "trending";
    return "ranging";
  }

  // --------------------------------------------------- feature frame
  function buildFrame(candlesByTf, baseTf) {
    const base = candlesByTf[baseTf];
    if (!base || base.length === 0) return { rows: [], features: [] };
    const bms = TF_MS[baseTf], ind = computeIndicators(base);
    const rows = base.map((r, i) => {
      const row = { open_ts: r.open_ts, decision_ts: r.open_ts + bms, open: r.open, close: r.close, high: r.high,
        low: r.low, volume: r.volume, atr_pct: ind.atr_pct[i] };
      for (const f of BASE_FEATURES) row[`${baseTf}_${f}`] = ind[f][i];
      row.regime = classifyRegime(ind.adx_14[i], ind.atr_pct_rank[i], ind.vol_z[i], ind.ret_z[i]);
      for (const g of REGIMES) row[`regime_${g}`] = row.regime === g ? 1 : 0;
      return row;
    });
    const features = BASE_FEATURES.map((f) => `${baseTf}_${f}`).concat(REGIMES.map((g) => `regime_${g}`));
    for (const tf of Object.keys(candlesByTf)) {
      const df = candlesByTf[tf];
      if (tf === baseTf || TF_MS[tf] <= bms || !df || df.length === 0) continue;
      const hi = computeIndicators(df), hms = TF_MS[tf];
      let j = -1;
      for (const row of rows) { // higher bar becomes visible only once it has closed
        while (j + 1 < df.length && df[j + 1].open_ts + hms <= row.decision_ts) j++;
        for (const f of CONTEXT_FEATURES) row[`${tf}_${f}`] = j >= 0 ? hi[f][j] : NaN_;
      }
      for (const f of CONTEXT_FEATURES) features.push(`${tf}_${f}`);
    }
    // keep only features with information
    const used = features.filter((f) => rows.some((r) => isNum(r[f])));
    return { rows, features: used, indicators: ind };
  }

  function labelThreshold(atrPct, horizon, atrMult, costBps) {
    return Math.max(costBps / 1e4, atrMult * (isNum(atrPct) ? atrPct : 0) * Math.sqrt(horizon));
  }
  function directionOf(ret, thr) { return ret > thr ? "UP" : ret < -thr ? "DOWN" : "FLAT"; }
  function addLabels(rows, horizon, atrMult, costBps) {
    const atrs = rows.map((r) => r.atr_pct).filter(isNum).sort((a, b) => a - b);
    const med = atrs.length ? atrs[Math.floor(atrs.length / 2)] : 0;
    rows.forEach((r, i) => {
      r.label_threshold = labelThreshold(isNum(r.atr_pct) ? r.atr_pct : med, horizon, atrMult, costBps);
      if (i + horizon < rows.length) {
        r.fwd_return = rows[i + horizon].close / r.close - 1;
        r.label = CLASSES.indexOf(directionOf(r.fwd_return, r.label_threshold));
      } else { r.fwd_return = NaN_; r.label = null; }
    });
    return rows;
  }
  function trainingRows(frame, warmup) {
    const rows = frame.rows.slice(warmup || 60);
    return rows.filter((r) => frame.features.filter((f) => isNum(r[f])).length >= 0.8 * frame.features.length);
  }

  // ------------------------------------------------------- matrices
  function toMatrix(rows, features) { return rows.map((r) => features.map((f) => (isNum(r[f]) ? r[f] : NaN_))); }
  function fitImputer(X) {
    const d = X[0].length, med = new Array(d).fill(0);
    for (let j = 0; j < d; j++) {
      const col = X.map((r) => r[j]).filter(isNum).sort((a, b) => a - b);
      med[j] = col.length ? col[Math.floor(col.length / 2)] : 0;
    }
    return med;
  }
  const impute = (X, med) => X.map((r) => r.map((v, j) => (isNum(v) ? v : med[j])));
  function fitScaler(X) {
    const d = X[0].length, mu = new Array(d).fill(0), sd = new Array(d).fill(1);
    for (let j = 0; j < d; j++) {
      const col = X.map((r) => r[j]); const m = mean(col);
      const s = Math.sqrt(col.reduce((a, b) => a + (b - m) ** 2, 0) / col.length);
      mu[j] = m; sd[j] = s > 1e-12 ? s : 1;
    }
    return { mu, sd };
  }
  const scale = (X, sc) => X.map((r) => r.map((v, j) => (v - sc.mu[j]) / sc.sd[j]));
  // Natural class weights (all 1). Balanced weights were removed after real BTC/ETH data showed they
  // inflate P(UP)/P(DOWN) far above the true frequencies. Kept as an option for experiments.
  function classWeights(y, k, balanced) {
    if (!balanced) return new Array(k).fill(1);
    const cnt = new Array(k).fill(0); y.forEach((v) => cnt[v]++);
    return cnt.map((c) => (c > 0 ? y.length / (k * c) : 0));
  }
  function softmaxRow(z) { const m = Math.max(...z); const e = z.map((v) => Math.exp(v - m)); const s = sum(e); return e.map((v) => v / s); }

  // ----------------------------------------------- logistic regression
  function fitLogReg(X, y, opts) {
    opts = opts || {};
    const K = 3, n = X.length, d = X[0].length, lr = opts.lr || 0.2, iters = opts.iters || 250, l2 = opts.l2 || 1e-3;
    const med = fitImputer(X), Xi = impute(X, med), sc = fitScaler(Xi), Z = scale(Xi, sc), cw = classWeights(y, K);
    const W = Array.from({ length: K }, () => new Array(d).fill(0)), b = new Array(K).fill(0);
    const mW = W.map((r) => r.map(() => 0)), vW = W.map((r) => r.map(() => 0)), mb = b.map(() => 0), vb = b.map(() => 0);
    const b1 = 0.9, b2 = 0.999;
    for (let it = 1; it <= iters; it++) {
      const gW = W.map((r) => r.map(() => 0)), gb = new Array(K).fill(0);
      let wsum = 0;
      for (let i = 0; i < n; i++) {
        const z = W.map((w, k) => b[k] + w.reduce((a, wv, j) => a + wv * Z[i][j], 0));
        const p = softmaxRow(z), w = cw[y[i]]; wsum += w;
        for (let k = 0; k < K; k++) {
          const g = w * (p[k] - (y[i] === k ? 1 : 0));
          gb[k] += g; const row = gW[k], zi = Z[i];
          for (let j = 0; j < d; j++) row[j] += g * zi[j];
        }
      }
      for (let k = 0; k < K; k++) { // Adam
        const g0 = gb[k] / wsum; mb[k] = b1 * mb[k] + (1 - b1) * g0; vb[k] = b2 * vb[k] + (1 - b2) * g0 * g0;
        b[k] -= (lr * (mb[k] / (1 - b1 ** it))) / (Math.sqrt(vb[k] / (1 - b2 ** it)) + 1e-8);
        for (let j = 0; j < d; j++) {
          const g = gW[k][j] / wsum + l2 * W[k][j];
          mW[k][j] = b1 * mW[k][j] + (1 - b1) * g; vW[k][j] = b2 * vW[k][j] + (1 - b2) * g * g;
          W[k][j] -= (lr * (mW[k][j] / (1 - b1 ** it))) / (Math.sqrt(vW[k][j] / (1 - b2 ** it)) + 1e-8);
        }
      }
    }
    return { type: "logreg", med, mu: sc.mu, sd: sc.sd, W, b };
  }
  function predictLogReg(m, X) {
    return X.map((r) => {
      const z = r.map((v, j) => ((isNum(v) ? v : m.med[j]) - m.mu[j]) / m.sd[j]);
      return softmaxRow(m.W.map((w, k) => m.b[k] + w.reduce((a, wv, j) => a + wv * z[j], 0)));
    });
  }

  // ----------------------------------------------------------- trees
  function fitBins(X, nb) {
    const d = X[0].length, edges = [];
    for (let j = 0; j < d; j++) {
      const col = X.map((r) => r[j]).sort((a, b) => a - b), e = [];
      for (let q = 1; q < nb; q++) { const v = col[Math.floor((q * col.length) / nb)]; if (!e.length || v > e[e.length - 1]) e.push(v); }
      edges.push(e);
    }
    return edges;
  }
  function binRow(r, edges) {
    return r.map((v, j) => { const e = edges[j]; let lo = 0, hi = e.length; while (lo < hi) { const m = (lo + hi) >> 1; if (v > e[m]) lo = m + 1; else hi = m; } return lo; });
  }
  // generic best-split search on binned data; `stat` aggregates per-bin statistics
  function growTree(B, idx, edges, depth, maxDepth, minLeaf, featSel, leafFn, gainFn, statFn, statDim) {
    const node = { leaf: leafFn(idx) };
    if (depth >= maxDepth || idx.length < 2 * minLeaf) return node;
    let best = null;
    for (const j of featSel()) {
      const nb = edges[j].length + 1;
      const hist = Array.from({ length: nb }, () => new Float64Array(statDim)), cnt = new Int32Array(nb);
      for (const i of idx) { const bi = B[i][j]; statFn(hist[bi], i); cnt[bi]++; }
      const tot = new Float64Array(statDim); hist.forEach((h) => h.forEach((v, k) => (tot[k] += v)));
      const left = new Float64Array(statDim); let nl = 0;
      for (let t = 0; t < nb - 1; t++) {
        hist[t].forEach((v, k) => (left[k] += v)); nl += cnt[t];
        const nr = idx.length - nl;
        if (nl < minLeaf || nr < minLeaf) continue;
        const right = tot.map((v, k) => v - left[k]);
        const g = gainFn(left, right, tot);
        if (g > 1e-12 && (!best || g > best.gain)) best = { gain: g, j, t };
      }
    }
    if (!best) return node;
    const L = [], R = [];
    for (const i of idx) (B[i][best.j] <= best.t ? L : R).push(i);
    node.f = best.j; node.t = best.t;
    node.l = growTree(B, L, edges, depth + 1, maxDepth, minLeaf, featSel, leafFn, gainFn, statFn, statDim);
    node.r = growTree(B, R, edges, depth + 1, maxDepth, minLeaf, featSel, leafFn, gainFn, statFn, statDim);
    delete node.leaf;
    return node;
  }
  function walkTree(node, b) { while (node.leaf === undefined) node = b[node.f] <= node.t ? node.l : node.r; return node.leaf; }

  function fitForest(X, y, opts) {
    opts = opts || {};
    const nTrees = opts.trees || 40, maxDepth = opts.depth || 6, minLeaf = opts.minLeaf || 20, rand = rng(opts.seed || 42);
    const med = fitImputer(X), Xi = impute(X, med), edges = fitBins(Xi, 32), B = Xi.map((r) => binRow(r, edges));
    const d = X[0].length, mtry = Math.max(1, Math.round(Math.sqrt(d))), cw = classWeights(y, 3), trees = [];
    const gini = (s) => { const t = s[0] + s[1] + s[2]; return t > 0 ? t * (1 - (s[0] / t) ** 2 - (s[1] / t) ** 2 - (s[2] / t) ** 2) : 0; };
    for (let k = 0; k < nTrees; k++) {
      const idx = Array.from({ length: X.length }, () => Math.floor(rand() * X.length));
      const featSel = () => { const s = new Set(); while (s.size < mtry) s.add(Math.floor(rand() * d)); return [...s]; };
      const leafFn = (ix) => { const c = [0, 0, 0]; ix.forEach((i) => (c[y[i]] += cw[y[i]])); const t = sum(c) || 1; return c.map((v) => v / t); };
      trees.push(growTree(B, idx, edges, 0, maxDepth, minLeaf, featSel, leafFn,
        (l, r, t) => gini(t) - gini(l) - gini(r), (h, i) => (h[y[i]] += cw[y[i]]), 3));
    }
    return { type: "forest", med, edges, trees };
  }
  function predictForest(m, X) {
    return X.map((r) => {
      const b = binRow(r.map((v, j) => (isNum(v) ? v : m.med[j])), m.edges), p = [0, 0, 0];
      for (const t of m.trees) { const l = walkTree(t, b); p[0] += l[0]; p[1] += l[1]; p[2] += l[2]; }
      return p.map((v) => v / m.trees.length);
    });
  }

  function fitBoost(X, y, opts) {
    opts = opts || {};
    const rounds = opts.rounds || 40, lr = opts.lr || 0.1, maxDepth = opts.depth || 3, minLeaf = opts.minLeaf || 30, lam = 1.0;
    const med = fitImputer(X), Xi = impute(X, med), edges = fitBins(Xi, 32), B = Xi.map((r) => binRow(r, edges));
    const n = X.length, d = X[0].length, cw = classWeights(y, 3);
    const prior = [0, 0, 0]; y.forEach((v) => (prior[v] += cw[v])); const ps = sum(prior);
    const init = prior.map((v) => Math.log(Math.max(v / ps, 1e-6)));
    const F = Array.from({ length: n }, () => init.slice()), all = Array.from({ length: n }, (_, i) => i), feats = Array.from({ length: d }, (_, j) => j);
    const trees = [];
    for (let r = 0; r < rounds; r++) {
      const P = F.map(softmaxRow), round = [];
      for (let k = 0; k < 3; k++) {
        const g = new Float64Array(n), h = new Float64Array(n);
        for (let i = 0; i < n; i++) { const w = cw[y[i]], p = P[i][k]; g[i] = w * (p - (y[i] === k ? 1 : 0)); h[i] = w * Math.max(p * (1 - p), 1e-6); }
        const score = (s) => (s[0] * s[0]) / (s[1] + lam);
        const tree = growTree(B, all, edges, 0, maxDepth, minLeaf, () => feats,
          (ix) => { let G = 0, H = 0; ix.forEach((i) => { G += g[i]; H += h[i]; }); return -G / (H + lam); },
          (l, rr, t) => score(l) + score(rr) - score(t), (hh, i) => { hh[0] += g[i]; hh[1] += h[i]; }, 2);
        for (let i = 0; i < n; i++) F[i][k] += lr * walkTree(tree, B[i]);
        round.push(tree);
      }
      trees.push(round);
    }
    return { type: "boost", med, edges, init, lr, trees };
  }
  function predictBoost(m, X) {
    return X.map((r) => {
      const b = binRow(r.map((v, j) => (isNum(v) ? v : m.med[j])), m.edges), z = m.init.slice();
      for (const round of m.trees) for (let k = 0; k < 3; k++) z[k] += m.lr * walkTree(round[k], b);
      return softmaxRow(z);
    });
  }

  // -------------------------------------------------------- ensemble
  function rawEnsemble(parts, X) {
    const a = predictLogReg(parts.logreg, X), b = predictForest(parts.forest, X), c = predictBoost(parts.boost, X);
    return a.map((p, i) => { const q = p.map((v, k) => Math.max(1e-6, (v + b[i][k] + c[i][k]) / 3)); const s = sum(q); return q.map((v) => v / s); });
  }
  const tempScale = (P, T, bias) => P.map((p) => softmaxRow(p.map((v, k) => Math.log(Math.max(v, 1e-6)) / T + (bias ? bias[k] : 0))));
  function golden(f, a, b, iters) { const gr = (Math.sqrt(5) - 1) / 2; for (let i = 0; i < (iters || 40); i++) { const c = b - gr * (b - a), d = a + gr * (b - a); if (f(c) < f(d)) b = d; else a = c; } return (a + b) / 2; }
  function fitParts(X, y, seed) {
    return { logreg: fitLogReg(X, y), forest: fitForest(X, y, { seed }), boost: fitBoost(X, y) };
  }
  function fitEnsemble(X, y, features, opts) {
    opts = opts || {};
    if (new Set(y).size < 2) throw new Error("training labels contain a single class");
    let T = 1, bias = [0, 0, 0];
    const nCal = Math.floor(y.length * 0.2);
    if (opts.calibrate !== false && nCal >= 100 && new Set(y.slice(0, -nCal)).size >= 2) {
      const early = fitParts(X.slice(0, -nCal), y.slice(0, -nCal), 7);
      const P = rawEnsemble(early, X.slice(-nCal)), yc = y.slice(-nCal);
      const nll = (t, b) => -mean(tempScale(P, t, b).map((p, i) => Math.log(Math.max(p[yc[i]], 1e-9))));
      // temperature + per-class bias (corrects base-rate shift), coordinate search
      for (let round = 0; round < 3; round++) {
        T = golden((t) => nll(t, bias), 0.3, 5);
        for (const k of [0, 1]) bias[k] = golden((v) => { const b = bias.slice(); b[k] = v; return nll(T, b); }, -3, 3);
      }
    }
    const prior = [0, 0, 0]; y.forEach((v) => prior[v]++);
    return { parts: fitParts(X, y, 42), temperature: T, bias, features, classPrior: prior.map((v) => v / y.length), procedure: "js-natural-weights-temp-bias" };
  }
  function predictEnsemble(model, X) { return tempScale(rawEnsemble(model.parts, X), model.temperature, model.bias); }

  // --------------------------------------------------------- gating
  function gate(pDown, pFlat, pUp, threshold, minEdge) {
    const probs = { DOWN: pDown, FLAT: pFlat, UP: pUp };
    const modelDir = Object.keys(probs).reduce((a, b) => (probs[b] > probs[a] ? b : a));
    const dir = pUp >= pDown ? "UP" : "DOWN", conf = Math.max(pUp, pDown);
    const ok = conf >= threshold && Math.abs(pUp - pDown) >= minEdge && conf >= pFlat;
    return { signal: ok ? dir : "NO TRADE", modelDir, confidence: conf };
  }

  // -------------------------------------------------------- metrics
  function metrics(y, P, regimes, threshold, minEdge) {
    const n = y.length, pred = P.map((p) => p.indexOf(Math.max(...p)));
    const cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; y.forEach((t, i) => cm[t][pred[i]]++);
    const prec = [0, 1, 2].map((k) => { const c = cm[0][k] + cm[1][k] + cm[2][k]; return c ? cm[k][k] / c : 0; });
    const rec = [0, 1, 2].map((k) => { const c = sum(cm[k]); return c ? cm[k][k] / c : 0; });
    const f1 = prec.map((p, k) => (p + rec[k] ? (2 * p * rec[k]) / (p + rec[k]) : 0));
    const logLoss = -mean(y.map((t, i) => Math.log(Math.max(P[i][t], 1e-9))));
    const brier = mean(y.map((t, i) => P[i].reduce((a, v, k) => a + (v - (k === t ? 1 : 0)) ** 2, 0)));
    let ece = 0; const rel = [];
    for (let b = 0; b < 10; b++) {
      const ix = []; P.forEach((p, i) => { const c = Math.max(...p); if ((b === 0 ? c >= 0 : c > b / 10) && c <= (b + 1) / 10) ix.push(i); });
      if (!ix.length) continue;
      const conf = mean(ix.map((i) => Math.max(...P[i]))), acc = mean(ix.map((i) => (pred[i] === y[i] ? 1 : 0)));
      ece += (ix.length / n) * Math.abs(acc - conf); rel.push({ bin: b, n: ix.length, confidence: conf, accuracy: acc });
    }
    const sig = signalStats(y, P, threshold, minEdge);
    const out = { n, accuracy: mean(pred.map((p, i) => (p === y[i] ? 1 : 0))), log_loss: logLoss, brier, ece, reliability: rel,
      precision_macro: mean(prec), recall_macro: mean(rec), f1_macro: mean(f1), confusion_matrix: cm, signals: sig,
      label_distribution: [0, 1, 2].map((k) => y.filter((v) => v === k).length) };
    if (regimes) {
      out.by_regime = {};
      for (const g of new Set(regimes)) {
        const ix = regimes.map((r, i) => (r === g ? i : -1)).filter((i) => i >= 0);
        if (ix.length < 10) continue;
        out.by_regime[g] = { n: ix.length, accuracy: mean(ix.map((i) => (pred[i] === y[i] ? 1 : 0))),
          log_loss: -mean(ix.map((i) => Math.log(Math.max(P[i][y[i]], 1e-9)))) };
      }
    }
    return out;
  }
  function signalStats(y, P, threshold, minEdge) {
    let n = 0, hits = 0;
    P.forEach((p, i) => { const g = gate(p[0], p[1], p[2], threshold, minEdge); if (g.signal !== "NO TRADE") { n++; if (CLASSES.indexOf(g.signal) === y[i]) hits++; } });
    return { signals: n, coverage: y.length ? n / y.length : 0, hit_rate: n ? hits / n : null };
  }
  function baseline(yTrain, yTest) {
    const c = [1e-6, 1e-6, 1e-6]; yTrain.forEach((v) => c[v]++); const s = sum(c), p = c.map((v) => v / s);
    return { log_loss: -mean(yTest.map((t) => Math.log(p[t]))), accuracy: mean(yTest.map((t) => (t === p.indexOf(Math.max(...p)) ? 1 : 0))) };
  }

  // --------------------------------------------------- walk-forward
  function walkForward(rows, features, horizon, opts) {
    opts = opts || {};
    const folds = opts.folds || 3, minTrain = opts.minTrain || 600, d = rows.filter((r) => r.label !== null);
    const n = d.length;
    if (n < minTrain + folds * 50) throw new Error(`not enough labelled rows for walk-forward: ${n}`);
    const testTotal = Math.min(Math.floor(n * 0.4), n - minTrain), block = Math.floor(testTotal / folds), start0 = n - block * folds;
    const oos = [], foldInfo = [];
    for (let k = 0; k < folds; k++) {
      const s = start0 + k * block, e = s + block;
      const tr = d.slice(0, Math.max(0, s - horizon)), te = d.slice(s, e); // purge: train labels must not reach into the test block
      const ytr = tr.map((r) => r.label);
      if (new Set(ytr).size < 2) continue;
      const model = fitEnsemble(toMatrix(tr, features), ytr, features);
      const P = predictEnsemble(model, toMatrix(te, features));
      te.forEach((r, i) => oos.push({ open_ts: r.open_ts, open: r.open, close: r.close, label: r.label, regime: r.regime, p: P[i] }));
      const yte = te.map((r) => r.label);
      foldInfo.push({ fold: k, train_rows: tr.length, test_rows: te.length, test_start: te[0].open_ts,
        log_loss: -mean(yte.map((t, i) => Math.log(Math.max(P[i][t], 1e-9)))), baseline: baseline(ytr, yte) });
    }
    if (!oos.length) throw new Error("walk-forward produced no valid folds");
    const y = oos.map((r) => r.label), P = oos.map((r) => r.p);
    const m = metrics(y, P, oos.map((r) => r.regime), opts.threshold || 0.55, opts.minEdge || 0.1);
    m.baseline_prior = baseline(d.slice(0, Math.max(0, start0 - horizon)).map((r) => r.label), y);
    m.folds = foldInfo;
    m.method = `expanding walk-forward, ${foldInfo.length} folds, purge ${horizon} bars, out-of-sample only`;
    return { metrics: m, oos };
  }

  // ------------------------------------------------------- backtest
  function backtest(oos, tf, horizon, p) {
    const lat = Math.floor((p.latencyMs || 0) / TF_MS[tf]), perSide = ((p.spreadBps || 0) / 2 + (p.slippageBps || 0)) / 1e4, fee = (p.feeBps || 0) / 1e4;
    const trades = []; let i = 0;
    while (i < oos.length) {
      const g = gate(oos[i].p[0], oos[i].p[1], oos[i].p[2], p.threshold, p.minEdge);
      const ei = i + 1 + lat, xi = i + horizon;
      if (g.signal === "NO TRADE" || xi >= oos.length || ei > xi || oos[xi].open_ts - oos[i].open_ts !== horizon * TF_MS[tf]) { i++; continue; }
      const side = g.signal === "UP" ? 1 : -1, entry = oos[ei].open * (1 + side * perSide), exit = oos[xi].close * (1 - side * perSide);
      trades.push({ gross: side * (oos[xi].close / oos[ei].open - 1), net: side * (exit / entry - 1) - 2 * fee });
      i = xi + 1;
    }
    if (!trades.length) return { trades: 0 };
    let eq = 1, peak = 1, dd = 0; trades.forEach((t) => { eq *= 1 + t.net; peak = Math.max(peak, eq); dd = Math.min(dd, eq / peak - 1); });
    return { trades: trades.length, win_rate: mean(trades.map((t) => (t.net > 0 ? 1 : 0))), avg_gross_bps: mean(trades.map((t) => t.gross)) * 1e4,
      avg_net_bps: mean(trades.map((t) => t.net)) * 1e4, total_net_return: eq - 1, max_drawdown: dd,
      round_trip_cost_bps: 2 * (p.feeBps || 0) + 2 * (p.slippageBps || 0) + (p.spreadBps || 0) };
  }

  // ----------------------------------------- champion / challenger
  function blockBootstrapProb(diff, block, nBoot, seed) {
    const n = diff.length; if (!n) return 0;
    block = Math.max(1, Math.min(block, n)); const r = rng(seed || 7), nb = Math.ceil(n / block); let wins = 0;
    for (let b = 0; b < (nBoot || 1000); b++) {
      let s = 0, c = 0;
      for (let k = 0; k < nb && c < n; k++) { const st = Math.floor(r() * (n - block + 1)); for (let q = 0; q < block && c < n; q++, c++) s += diff[st + q]; }
      if (s / n > 0) wins++;
    }
    return wins / (nBoot || 1000);
  }
  function compareModels(y, Pprod, Pchal, cfg, horizon) {
    const llp = y.map((t, i) => -Math.log(Math.max(Pprod[i][t], 1e-9))), llc = y.map((t, i) => -Math.log(Math.max(Pchal[i][t], 1e-9)));
    const br = (P) => mean(y.map((t, i) => P[i].reduce((a, v, k) => a + (v - (k === t ? 1 : 0)) ** 2, 0)));
    const acc = (P) => mean(y.map((t, i) => (P[i].indexOf(Math.max(...P[i])) === t ? 1 : 0)));
    const gain = mean(llp.map((v, i) => v - llc[i])), prob = blockBootstrapProb(llp.map((v, i) => v - llc[i]), horizon);
    const checks = { logloss_gain_ok: gain >= cfg.minLoglossGain, bootstrap_ok: prob >= cfg.bootstrapConfidence,
      brier_ok: br(Pchal) <= br(Pprod) + 1e-4, accuracy_ok: acc(Pchal) >= acc(Pprod) - 0.01 };
    return { n_holdout: y.length, logloss_production: mean(llp), logloss_challenger: mean(llc), logloss_gain: gain,
      bootstrap_p_better: prob, brier_production: br(Pprod), brier_challenger: br(Pchal),
      accuracy_production: acc(Pprod), accuracy_challenger: acc(Pchal), checks, promote: Object.values(checks).every(Boolean) };
  }

  // --------------------------------------------------- data quality
  function missingBars(candles, tf) {
    if (candles.length < 2) return 0;
    const span = Math.round((candles[candles.length - 1].open_ts - candles[0].open_ts) / TF_MS[tf]) + 1;
    return Math.max(0, span - candles.length);
  }
  function evaluateQuality(ctx) {
    const issues = []; let score = 1;
    const add = (code, sev, pen, detail) => { issues.push({ code, severity: sev, detail }); score = Math.max(0, score - pen); };
    const { candles, tf, now } = ctx;
    if (!candles || !candles.length) { add("no_candles", "critical", 1, "no closed candles"); }
    else {
      const age = now - (candles[candles.length - 1].open_ts + TF_MS[tf]);
      if (age > TF_MS[tf] + 180e3) add("stale_candles", "critical", 0.6, `last closed candle ended ${Math.round(age / 1000)} s ago`);
      const miss = missingBars(candles.slice(-300), tf);
      if (miss > 0) add("missing_bars", miss > 10 ? "critical" : "warning", Math.min(0.4, 0.02 * miss), `${miss} missing bars in last 300`);
      const bad = candles.slice(-300).filter((c) => !(c.low > 0) || c.high < Math.max(c.open, c.close) || c.low > Math.min(c.open, c.close) || c.volume < 0).length;
      if (bad) add("invalid_candles", "critical", 0.5, `${bad} invalid candles`);
      if (candles[candles.length - 1].open_ts > now) add("future_candle", "critical", 0.5, "candle timestamp in the future");
    }
    if (ctx.stream) {
      const s = ctx.stream, a = s.lastMsg ? (now - s.lastMsg) / 1000 : null;
      if (s.state !== "live" || a === null || a > 30) add("ws_down", "warning", 0.2, `${ctx.primary} live stream ${s.state}`);
      if (s.newGaps) add("ws_gaps", "warning", Math.min(0.3, 0.05 * s.newGaps), `${s.newGaps} new stream gaps`);
    }
    if (ctx.book && isNum(ctx.book.spreadBps) && ctx.book.ageSec < 30 && ctx.book.spreadBps > ctx.maxSpreadBps) add("wide_spread", "warning", 0.2, `spread ${ctx.book.spreadBps.toFixed(1)} bps`);
    const prices = ctx.exchangePrices || {};
    const vals = Object.values(prices);
    if (vals.length >= 2) {
      const sorted = vals.slice().sort((a, b) => a - b), med = sorted.length % 2 ? sorted[(sorted.length - 1) / 2] : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
      for (const [ex, p] of Object.entries(prices)) {
        const dev = Math.abs(p / med - 1) * 1e4;
        if (dev > ctx.maxDivergenceBps) add("exchange_disagreement", ex === ctx.primary ? "critical" : "warning", 0.3, `${ex} deviates ${dev.toFixed(1)} bps`);
      }
    }
    score = Math.round(score * 1000) / 1000;
    const ok = score >= (ctx.minScore || 0.7) && !issues.some((i) => i.severity === "critical");
    return { score, ok, issues, label: !ok ? "BAD" : score >= 0.9 ? "GOOD" : "FAIR" };
  }

  // --------------------------------------------------------- news
  const NEWS_CATEGORIES = {
    crypto: ["bitcoin", "btc", "ethereum", "ether", "crypto", "stablecoin", "solana", "altcoin", "blockchain", "defi", "token", "xrp", "halving"],
    regulation: ["sec", "regulation", "regulator", "lawsuit", "ban", "approval", "approves", "cftc", "mica", "enforcement", "court", "legislation"],
    macroeconomics: ["gdp", "recession", "unemployment", "payrolls", "nonfarm", "jobs report", "retail sales", "pmi", "economy"],
    central_banks: ["fed", "federal reserve", "fomc", "ecb", "boj", "bank of japan", "bank of england", "powell", "lagarde", "central bank"],
    inflation: ["inflation", "cpi", "ppi", "pce", "consumer prices"],
    interest_rates: ["rate hike", "rate cut", "interest rate", "interest rates", "basis points", "rate decision"],
    geopolitics: ["war", "sanctions", "missile", "conflict", "tariff", "tariffs", "invasion", "ceasefire", "military", "election"],
    etf: ["etf", "etfs", "spot etf", "inflows", "outflows", "blackrock", "grayscale"],
    exchange_incidents: ["outage", "halts", "halted", "suspends withdrawals", "downtime", "delist", "insolvency", "bankruptcy"],
    hacks: ["hack", "hacked", "exploit", "exploited", "stolen", "breach", "drained", "attacker"],
    liquidations: ["liquidation", "liquidations", "liquidated", "short squeeze", "long squeeze"],
    major_companies: ["microstrategy", "strategy inc", "tesla", "nvidia", "coinbase", "earnings", "revenue"],
    commodities: ["oil", "crude", "gold", "opec", "natural gas", "copper"],
    equities: ["stocks", "s&p 500", "nasdaq", "dow jones", "equities", "wall street"],
    bonds: ["bond", "bonds", "treasury", "treasuries", "yields"],
    currencies: ["dollar", "dxy", "euro", "yen", "forex", "yuan"],
  };
  const POS = ["surge", "surges", "rally", "rallies", "soar", "soars", "jumps", "approve", "approves", "approved", "approval", "inflows",
    "beats", "record high", "all-time high", "rate cut", "cools", "eases", "gains", "bullish", "adoption", "upgrade", "rebound", "recovers", "ceasefire", "dovish", "buys"];
  const NEG = ["plunge", "plunges", "crash", "crashes", "tumbles", "slumps", "hack", "hacked", "exploit", "ban", "bans", "lawsuit", "sues",
    "outflows", "liquidation", "liquidations", "sanctions", "war", "rate hike", "hotter", "misses", "bearish", "downgrade", "halt", "halts",
    "sell-off", "selloff", "fraud", "charged", "hawkish", "drained", "stolen", "recession", "default", "invasion", "dumps", "bankruptcy"];
  const ASSETS = { BTC: ["bitcoin", "btc"], ETH: ["ethereum", "ether", "eth"], SOL: ["solana"], XRP: ["xrp", "ripple"] };
  const MACRO = ["macroeconomics", "central_banks", "inflation", "interest_rates", "geopolitics", "bonds", "currencies"];
  const HORIZON = { hacks: 60, liquidations: 60, exchange_incidents: 120, central_banks: 240, inflation: 240, interest_rates: 240, regulation: 1440, etf: 1440 };
  const hit = (text, words) => words.filter((w) => new RegExp(`(^|[^a-z0-9])${w.replace(/[.*+?^${}()|[\]\\&]/g, "\\$&")}($|[^a-z0-9])`).test(text));
  const tokens = (s) => new Set((s.toLowerCase().match(/[a-z0-9]{3,}/g) || []));
  function novelty(title, recent) {
    const t = tokens(title); if (!t.size || !recent.length) return 1;
    let best = 0; for (const r of recent) { const u = tokens(r); let inter = 0; t.forEach((x) => u.has(x) && inter++); const j = inter / (t.size + u.size - inter); best = Math.max(best, j); }
    return Math.round((1 - best) * 1000) / 1000;
  }
  function analyzeNewsRules(item, recent) {
    const text = `${item.title} ${item.summary || ""}`.toLowerCase();
    let category = "other", bestN = 0;
    for (const [c, ws] of Object.entries(NEWS_CATEGORIES)) { const h = hit(text, ws).length; if (h > bestN) { bestN = h; category = c; } }
    const assets = Object.keys(ASSETS).filter((a) => hit(text, ASSETS[a]).length);
    const pos = hit(text, POS), neg = hit(text, NEG);
    const direction = (pos.length - neg.length) / (pos.length + neg.length + 1);
    const relevance = assets.length ? 0.9 : ["crypto", "etf", "hacks", "liquidations", "exchange_incidents"].includes(category) ? 0.65
      : MACRO.includes(category) || category === "regulation" ? 0.45 : ["commodities", "equities", "major_companies"].includes(category) ? 0.3 : 0.1;
    const affected = assets.length ? assets.slice() : category !== "other" ? ["CRYPTO_MARKET"] : [];
    if (MACRO.includes(category) && !affected.includes("CRYPTO_MARKET")) affected.push("CRYPTO_MARKET");
    const nSig = pos.length + neg.length + bestN;
    return { event: item.title, timestamp: item.published_ms, category, asset: assets[0] || null, direction, relevance,
      novelty: novelty(item.title, recent), confidence: pos.length || neg.length ? Math.min(0.6, 0.25 + 0.07 * nSig) : Math.min(0.35, 0.15 + 0.05 * nSig),
      expected_horizon_min: HORIZON[category] || 360, affected_assets: affected, summary: item.summary || item.title, analyzer: "rules-v1",
      source: item.source, url: item.url };
  }
  function newsFeatures(events, symbol, now, halfLifeMin) {
    const base = symbol.split("/")[0]; let total = 0, relMax = 0, count = 0; const top = [];
    for (const ev of events) {
      if (ev.timestamp > now || now - ev.timestamp > 48 * 3600e3) continue;
      let rel;
      if ((ev.affected_assets || []).includes(base) || ev.asset === base) rel = ev.relevance;
      else if ((ev.affected_assets || []).includes("CRYPTO_MARKET") || MACRO.includes(ev.category) || ["crypto", "etf", "regulation"].includes(ev.category)) rel = ev.relevance * 0.6;
      else continue;
      const age = (now - ev.timestamp) / 60e3; let decay = 0.5 ** (age / halfLifeMin);
      if (age > 2 * (ev.expected_horizon_min || 360)) decay *= 0.25;
      const w = rel * ev.confidence * Math.sqrt(isNum(ev.novelty) ? ev.novelty : 0.5) * decay;
      if (w <= 0) continue;
      total += w * ev.direction; relMax = Math.max(relMax, rel * decay);
      if (w > 0.02) { count++; top.push({ title: ev.event, weight: w, direction: ev.direction }); }
    }
    top.sort((a, b) => b.weight - a.weight);
    return { news_impact: Math.tanh(total), news_relevance: relMax, news_event_count: count, news_top: top.slice(0, 5) };
  }

  // ----------------------------------------------------- training jobs
  function trainBaseline(job) {
    const frame = buildFrame(job.candlesByTf, job.tf);
    const rows = addLabels(trainingRows(frame), job.horizon, job.atrMult, job.costBps);
    const labelled = rows.filter((r) => r.label !== null);
    const wf = walkForward(labelled, frame.features, job.horizon, { folds: job.folds, minTrain: job.minTrain, threshold: job.threshold, minEdge: job.minEdge });
    wf.metrics.backtest = backtest(wf.oos, job.tf, job.horizon, job.costs);
    const model = fitEnsemble(toMatrix(labelled, frame.features), labelled.map((r) => r.label), frame.features);
    return { model, metrics: wf.metrics, oos: wf.oos.map((r) => ({ open_ts: r.open_ts, open: r.open, close: r.close, p: r.p })),
      train_start_ts: labelled[0].open_ts, train_end_ts: labelled[labelled.length - 1].open_ts, n_train: labelled.length };
  }
  function challengerCycle(job) {
    const frame = buildFrame(job.candlesByTf, job.tf);
    const rows = addLabels(trainingRows(frame), job.horizon, job.atrMult, job.costBps).filter((r) => r.label !== null);
    const bar = TF_MS[job.tf], prod = job.production, h = job.horizon;
    const fresh = rows.filter((r) => r.open_ts > prod.train_end_ts + h * bar); // labels production never saw
    if (fresh.length < job.minNewLabels) return { status: "waiting", new_labels: fresh.length, required: job.minNewLabels };
    const nHold = Math.max(job.minHoldout, Math.floor(fresh.length * job.holdoutFraction));
    const hold = fresh.slice(-nHold), holdStart = hold[0].open_ts;
    const train = rows.filter((r) => r.open_ts < holdStart - h * bar); // purge gap
    const y = hold.map((r) => r.label);
    if (new Set(y).size < 2) return { status: "waiting", reason: "class_diversity" };
    const Pp = predictEnsemble(prod.model, toMatrix(hold, prod.model.features));
    const chal = fitEnsemble(toMatrix(train, frame.features), train.map((r) => r.label), frame.features);
    const Pc = predictEnsemble(chal, toMatrix(hold, frame.features));
    const cmp = compareModels(y, Pp, Pc, job, h);
    const m = metrics(y, Pc, hold.map((r) => r.regime), job.threshold, job.minEdge);
    m.method = "chronological holdout unseen by production and challenger";
    return { status: cmp.promote ? "promoted" : "rejected", model: chal, metrics: m, comparison: cmp,
      train_start_ts: train[0].open_ts, train_end_ts: train[train.length - 1].open_ts, n_train: train.length, holdout_start: holdStart };
  }

  const AMP = { TF_MS, CLASSES, FEATURE_VERSION, BASE_FEATURES, CONTEXT_FEATURES, REGIMES, rng, computeIndicators, classifyRegime,
    buildFrame, addLabels, trainingRows, labelThreshold, directionOf, toMatrix, fitLogReg, predictLogReg, fitForest, predictForest,
    fitBoost, predictBoost, fitEnsemble, predictEnsemble, gate, metrics, baseline, walkForward, backtest, blockBootstrapProb,
    compareModels, missingBars, evaluateQuality, analyzeNewsRules, novelty, newsFeatures, trainBaseline, challengerCycle, isNum };
  root.AMP = AMP;
  if (typeof module !== "undefined" && module.exports) module.exports = AMP;
})(typeof self !== "undefined" ? self : globalThis);
