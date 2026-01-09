# CodeCheck.py
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from Code import feat, cut, mat, scatter, dtarget, zbase, rftune, ridgetune, gbrt, compare, evalm, YEAR_COL, POP_COL, tcol, FEATURES, winso, applycaps

ALL = "data/cities_matched_2018_2024_complete.csv"
H = 1
T = tcol(H)
TRAIN_END_YEAR = 2022
TEST_YEARS = [2023]
GOALS = {"rmse_max": 0.0100, "r2_min": 0.05, "spearman_min": 0.35, "beat_zero_rmse": True}
OUT = Path("reports/midterm")
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

def passgo(a: dict, b: dict) -> bool:
    c = a["rmse"] <= GOALS["rmse_max"]
    d = a["r2"] >= GOALS["r2_min"]
    e = a["spearman"]
    f = np.isnan(e)
    g = GOALS["spearman_min"]
    h = (e >= g) if not f else (g <= -1)
    i = True
    if GOALS["beat_zero_rmse"]:
        i = a["rmse"] < b["rmse"]
    return c and d and h and i

def save_compare(a: List[Dict[str, Any]], b: Path):
    c = pd.DataFrame(a)
    d = c.set_index("model")[["rmse", "mae", "r2"]]
    ax = d.plot(kind="bar")
    ax.set_ylabel("score")
    ax.set_title("model comparison")
    plt.tight_layout()
    plt.savefig(b, dpi=160)
    plt.close()

def main():
    a = feat(ALL, H)
    b, c = cut(a, TRAIN_END_YEAR, TEST_YEARS)
    d, e = mat(b)
    f, g = mat(c)
    r = winso(d, list(d.columns), c=0.005, d=0.995)
    d2 = applycaps(d, r)
    f2 = applycaps(f, r)
    h = zbase(a, TRAIN_END_YEAR, TEST_YEARS, H)
    m = []
    i = ridgetune(d2, e)
    j, jp = evalm(i, d2, e, f2, g)
    j["model"] = "ridge"
    m.append(j)
    k = rftune(d2, e)
    l, lp = evalm(k, d2, e, f2, g)
    l["model"] = "rf"
    m.append(l)
    n = gbrt(d2, e)
    o, op = evalm(n, d2, e, f2, g)
    o["model"] = "gbrt"
    m.append(o)
    u = compare(m)
    v = u[["model", "rmse", "r2", "spearman", "train_seconds"]]
    w = min(m, key=lambda z: z["rmse"])
    x = w["model"]
    y = {"ridge": jp, "rf": lp, "gbrt": op}[x]
    z = FIG / "model_compare.png"
    save_compare(m, z)
    aa = FIG / "rank_scatter_best.png"
    scatter(g.values, y, str(aa))
    ad = pd.DataFrame({"city_id": c["city_id"].values, "year": c[YEAR_COL].values})
    ae = pd.Series(y, name="pred_dlog")
    af = pd.Series(g.values, name="true_dlog")
    ag = pd.concat([ad, ae, af], axis=1)
    ah = (np.exp(ag["pred_dlog"]) - 1.0) * 100.0
    ai = (np.exp(ag["true_dlog"]) - 1.0) * 100.0
    ag["pred_pct_growth"] = ah
    ag["true_pct_growth"] = ai
    ag["abs_error_dlog"] = (ag["pred_dlog"] - ag["true_dlog"]).abs()
    ag["abs_error_pct"] = (ag["pred_pct_growth"] - ag["true_pct_growth"]).abs()
    aj = OUT / "test_predictions.csv"
    ag.to_csv(aj, index=False)
    ak = OUT / "top_cities_predictions.csv"
    ag.sort_values("pred_dlog", ascending=False).head(50).to_csv(ak, index=False)
    as_ = passgo(w, h)
    if not as_:
        av = w["rmse"] - GOALS["rmse_max"]
        aw = w["rmse"] - h["rmse"]
        ax = GOALS["r2_min"] - w["r2"]
        ay = GOALS["spearman_min"] - (w["spearman"] if not np.isnan(w["spearman"]) else -1.0)
        _ = (av, aw, ax, ay) 

if __name__ == "__main__":
    main()


