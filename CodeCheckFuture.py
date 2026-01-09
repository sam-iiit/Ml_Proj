# CodeCheckFuture.py
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from Code import feat, mat, rftune, ridgetune, gbrt, YEAR_COL, prepare_X_for_year, scatter, captt, FEAT_SELECTED

ALL = "data/cities_matched_2018_2024_complete.csv"
OUT = Path("reports/final")
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

def pct(a: np.ndarray) -> np.ndarray:
    return (np.exp(a) - 1.0) * 100.0

def save_forecast(a: pd.DataFrame, b: np.ndarray, c: int, d: str):
    e = a.copy()
    e["pred_dlog"] = b.astype(float)
    e["pred_pct_growth"] = pct(e["pred_dlog"].values)
    e["pred_year"] = c
    f = OUT / d
    e[["city_id", "pred_year", "pred_dlog", "pred_pct_growth"]].to_csv(f, index=False)
    g = OUT / d.replace(".csv", "_top50.csv")
    e.sort_values("pred_dlog", ascending=False).head(50)[["city_id", "pred_year", "pred_dlog", "pred_pct_growth"]].to_csv(g, index=False)

def best_model(a, y, X) -> str:
    d = {"ridge": float(np.sqrt(np.mean((y.values - a["ridge"].predict(X))**2))),
         "rf": float(np.sqrt(np.mean((y.values - a["rf"].predict(X))**2))),
         "gbrt": float(np.sqrt(np.mean((y.values - a["gbrt"].predict(X))**2)))}
    e = min(d.items(), key=lambda x: x[1])[0]
    return e

def main():
    ly_raw = int(pd.read_csv(ALL)[YEAR_COL].max())
    for h in [1, 3, 5]:
        df = feat(ALL, h)
        X, y = mat(df)
        if X.empty or y.empty:
            raise ValueError("training matrix empty")
        feat_list = FEAT_SELECTED[h]
        Xb, meta = prepare_X_for_year(ly_raw, feat_list=feat_list)
        if Xb.empty:
            raise ValueError("inference matrix empty for last year")
        Xw, Xbw, caps = captt(X, Xb, lo=0.01, hi=0.99)
        m_ridge = ridgetune(Xw, y)
        m_rf = rftune(Xw, y)
        m_gbrt = gbrt(Xw, y)
        md = {"ridge": m_ridge, "rf": m_rf, "gbrt": m_gbrt}
        nm = best_model(md, y, Xw)
        yh = md[nm].predict(Xbw)
        py = ly_raw + h
        fn = f"forecast_h{h}_{nm}.csv"
        save_forecast(meta[["city_id"]], yh, py, fn)
    df1 = feat(ALL, 1)
    X1, y1 = mat(df1)
    X1w, X1w2, caps1 = captt(X1, X1, lo=0.01, hi=0.99)
    m1 = {"ridge": ridgetune(X1w, y1), "rf": rftune(X1w, y1), "gbrt": gbrt(X1w, y1)}
    bm = best_model(m1, y1, X1w)
    yh2 = m1[bm].predict(X1w)
    fig1 = FIG / f"rank_scatter_h1_{bm}.png"
    scatter(y1.values, yh2, str(fig1))

if __name__ == "__main__":
    main()