"""Why do the horizon models not (yet) earn after costs? A measured answer per symbol x horizon, from the
stored independent-check results and live forecasts (read-only on the database).

    python scripts/forecast_diagnostics.py --db state/market.sqlite3 --out docs/FORECAST_DIAGNOSTICS.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


STATUS_RU = {"collecting": "сбор данных", "validated": "проверена, лучше базовой", "not_better": "не лучше базовой",
             "degraded": "ухудшилась", None: "не обучена"}


def binom_ci(p: float, n: int) -> tuple[float, float]:
    if not n:
        return (float("nan"), float("nan"))
    se = math.sqrt(max(p * (1 - p), 1e-12) / n)
    return (p - 1.96 * se, p + 1.96 * se)


def diagnose(st: dict, cost: float) -> list[str]:
    ho = st.get("holdout") or {}
    out = []
    if not ho:
        return ["нет независимой проверки: недостаточно данных"]
    ls = ho.get("label_share") or {}
    beyond = (ls.get("UP") or 0) + (ls.get("DOWN") or 0)
    if beyond < 0.10:
        out.append(f"движения больше издержек ({cost:.0f} б.п.) редки: {beyond:.0%} — прибыль после издержек почти невозможна")
    n_ind = ho.get("n_independent") or 0
    d = ho.get("direction_hit")
    if d is not None:
        lo, hi = binom_ci(d, n_ind)
        if lo <= 0.5 <= hi:
            out.append(f"направление угадывается на уровне случайности: {d:.1%} (95 %: {lo:.1%}…{hi:.1%}, n={n_ind})")
        elif lo > 0.5:
            out.append(f"есть статистически заметное преимущество по направлению: {d:.1%} (95 %: {lo:.1%}…{hi:.1%})")
    if ho.get("mae_bps") is not None and ho.get("mae_rw_bps") is not None and ho["mae_bps"] >= ho["mae_rw_bps"]:
        out.append(f"медианный прогноз ошибается не меньше, чем «цена не изменится» ({ho['mae_bps']:.1f} vs {ho['mae_rw_bps']:.1f} б.п.)")
    if (ho.get("ece") or 0) > 0.05:
        out.append(f"калибровка неточная: ECE {ho['ece']:.3f}")
    cov = ho.get("coverage_10_90")
    if cov is not None and not 0.72 <= cov <= 0.88:
        out.append(f"диапазон 10–90 % накрывает {cov:.0%} исходов вместо ~80 %")
    if ho.get("gain") is not None:
        lo, hi = ho["gain_ci95"]
        out.append(("вероятности лучше базовых частот" if lo > 0 else "вероятности не доказанно лучше базовых частот")
                   + f": выигрыш log loss {ho['gain']:+.4f} (95 %: {lo:+.4f}…{hi:+.4f})")
    sig = ho.get("signals") or 0
    if sig == 0:
        out.append("ни один прогноз не прошёл порог уверенности — сигналов для оценки после издержек нет")
    else:
        out.append(f"сигналов {sig}, в среднем {ho.get('avg_net_bps'):+.1f} б.п. после издержек"
                   + (" — слишком мало для вывода" if sig < 30 else ""))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", default="docs/FORECAST_DIAGNOSTICS.md")
    ap.add_argument("--cost", type=float, default=25.0)
    a = ap.parse_args()
    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    from backend.app.forecast.horizons import HORIZONS
    lines = [f"# Диагностика прогнозов по горизонтам — {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", "",
             "Источник: независимые проверки моделей (данные, не использованные при обучении) и живые прогнозы из базы.",
             f"Издержки на круг: {a.cost:.0f} б.п. «Независимых» — неперекрывающихся исходов.", ""]
    syms = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM forecasts UNION SELECT DISTINCT substr(key, 8, instr(substr(key,8),'|')-1) "
                                      "FROM system_state WHERE key LIKE 'fstate:%'") if r[0]]
    for sym in sorted(set(syms)):
        lines += [f"## {sym}", "", "| Горизонт | Статус | Данных (незав.) | Исходы ↓/·/↑ | Направление | MAE модель / «без изменений» | Brier модель / база | Сигналы, после издержек |",
                  "|---|---|---|---|---|---|---|---|"]
        notes = []
        for h in HORIZONS:
            row = con.execute("SELECT value_json FROM system_state WHERE key=?", (f"fstate:{sym}|f{h.seconds}",)).fetchone()
            st = json.loads(row[0]) if row else {}
            ho = st.get("holdout") or {}
            ls = ho.get("label_share") or {}
            lines.append(f"| {h.label} | {STATUS_RU.get(st.get('status'), st.get('status'))} | {st.get('n_rows', 0)} ({st.get('n_independent_total', 0)}) | "
                         + (f"{ls.get('DOWN', 0):.0%}/{ls.get('FLAT', 0):.0%}/{ls.get('UP', 0):.0%}" if ls else "—") + " | "
                         + (f"{ho['direction_hit']:.1%}" if ho.get("direction_hit") is not None else "—") + " | "
                         + (f"{ho['mae_bps']:.1f} / {ho['mae_rw_bps']:.1f}" if ho.get("mae_bps") is not None else "—") + " | "
                         + (f"{ho['brier']:.4f} / {ho['brier_base']:.4f}" if ho.get("brier") is not None else "—") + " | "
                         + (f"{ho['signals']}: {ho['avg_net_bps']:+.1f}" if ho.get("signals") else ("0" if ho else "—")) + " |")
            notes.append(f"- **{h.label}**: " + "; ".join(diagnose(st, a.cost)))
        lines += ["", *notes, ""]
    Path(a.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
