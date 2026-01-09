# Code.py
from pathlib import Path
from typing import List, Tuple, Dict, Any
import time, re, warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

VERBOSE = False
def log(a: str = ""):
    if VERBOSE:
        print(a)

HG = 1
TARGET_COL = f"target_dlogpop_next{HG}"
_ID_PATTERNS = [r"city", r"state", r"msa", r"cbsa", r"fips"]
_YEAR_CANDS = {"year", "yr"}
_POP_CANDS = {"population", "pop"}
CITY_ID_COLS: List[str] = ["City_pretty", "State_pretty"]
YEAR_COL: str = "Year"
POP_COL: str = "Population"
FEATURES: List[str] = []
FEAT_SELECTED: Dict[int, List[str]] = {}
GPATH: str = ""

def tcol(horizon: int) -> str:
    return f"target_dlogpop_next{horizon}"

def readx(df: pd.DataFrame) -> Tuple[List[str], str, str, List[str]]:
    all_cols = list(df.columns)
    potential_ids = []
    
    for col in all_cols:
        if any(re.search(pattern, col.lower()) for pattern in _ID_PATTERNS):
            potential_ids.append(col)
    
    seen = set()
    id_cols = [col for col in potential_ids if not (col in seen or seen.add(col))]
    year_col = next((col for col in all_cols if col.lower() in _YEAR_CANDS), None)
    pop_col = next((col for col in all_cols if col.lower() in _POP_CANDS), None)
    reserved = set(id_cols + [c for c in [year_col, pop_col] if c])
    numeric_features = [
        col for col in all_cols
        if col not in reserved and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    z_feats = [f for f in numeric_features if f.lower().startswith("z_")]
    bls_feats = [f for f in numeric_features if f.lower().startswith("bls_")]
    census_feats = [
        f for f in numeric_features if f.lower().startswith("census_") or f.lower().startswith("pop_")
    ]
    ordered_feats = (
        z_feats
        + bls_feats
        + census_feats
        + [f for f in numeric_features if f not in z_feats + bls_feats + census_feats]
    )
    
    final_ids = id_cols if id_cols else ["City_pretty", "State_pretty"]
    final_year = year_col or "Year"
    final_pop = pop_col or "Population"
    
    return final_ids, final_year, final_pop, ordered_feats

def getd(filepath: str) -> None:
    preview_df = pd.read_csv(filepath, nrows=5)
    ids, year_col, pop_col, feats = readx(preview_df)
    global CITY_ID_COLS, YEAR_COL, POP_COL, FEATURES, GPATH
    CITY_ID_COLS = ids
    YEAR_COL = year_col
    POP_COL = pop_col
    FEATURES = feats
    GPATH = filepath

def cityid(df: pd.DataFrame) -> pd.Series:
    combined = df[CITY_ID_COLS].astype(str)
    return combined.agg(", ".join, axis=1)

def base(filepath: str) -> pd.DataFrame:
    data = pd.read_csv(filepath)
    data["city_id"] = cityid(data)
    return data

def targ(df: pd.DataFrame, horizon: int, dropna: bool = True) -> pd.DataFrame:
    df_sorted = df.sort_values(["city_id", YEAR_COL]).copy()
    df_sorted[POP_COL] = pd.to_numeric(df_sorted[POP_COL], errors="coerce").astype(float)
    df_sorted = df_sorted[df_sorted[POP_COL] >= 1000]
    
    df_sorted["pop_next"] = df_sorted.groupby("city_id")[POP_COL].shift(-horizon)
    df_sorted[tcol(horizon)] = np.log(df_sorted["pop_next"]) - np.log(df_sorted[POP_COL])
    df_sorted[tcol(horizon)] = df_sorted[tcol(horizon)].replace([np.inf, -np.inf], np.nan)
    
    if dropna:
        df_sorted = df_sorted.dropna(subset=[tcol(horizon)]).copy()
    
    lower, upper = df_sorted[tcol(horizon)].quantile([0.01, 0.99], interpolation="linear")
    df_sorted[tcol(horizon)] = df_sorted[tcol(horizon)].clip(lower, upper)
    return df_sorted

def cover(df: pd.DataFrame, candidate_cols: List[str], min_valid: float = 0.8) -> List[str]:
    chosen = []
    n = len(df)
    for col in candidate_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            coverage = 1 - df[col].isna().sum() / max(n, 1)
            if coverage >= min_valid:
                chosen.append(col)
    return chosen
def oldt(df: pd.DataFrame, horizon: int, lag: int = 1) -> pd.DataFrame: #___________________________________________________________________
    df_sorted = df.sort_values(["city_id", YEAR_COL]).copy()
    target_col_name = tcol(horizon)
    df_sorted[f"L{lag}_{target_col_name}"] = df_sorted.groupby("city_id")[target_col_name].shift(lag)
    return df_sorted

def olds(df: pd.DataFrame, horizon: int, lag: int = 1) -> pd.DataFrame:
    df_sorted = df.sort_values(["city_id", YEAR_COL]).copy()
    state_col = CITY_ID_COLS[1] if len(CITY_ID_COLS) > 1 else "State_pretty"
    target_col_name = tcol(horizon)    
    state_avg_target = df_sorted.groupby([state_col, YEAR_COL])[target_col_name].mean().shift(lag)
    lag_col_name = f"state_lag{lag}_target"
    state_avg_df = state_avg_target.reset_index(name=lag_col_name)    
    df_merged = df_sorted.merge(state_avg_df, on=[state_col, YEAR_COL], how="left")
    if lag == 1:
        df_merged["state_prev_target"] = df_merged.groupby("city_id")[lag_col_name].shift(1)
    return df_merged

def lags(df: pd.DataFrame, cols_to_lag: List[str], lag_periods=(1, 2, 3)) -> pd.DataFrame:
    df_sorted = df.sort_values(["city_id", YEAR_COL]).copy()
    grouped_by_city = df_sorted.groupby("city_id")
    new_features_dict = {}
    for col in cols_to_lag:
        if col in df_sorted.columns and pd.api.types.is_numeric_dtype(df_sorted[col]):
            for lag_val in lag_periods:
                new_features_dict[f"L{lag_val}_{col}"] = grouped_by_city[col].shift(lag_val)
            new_features_dict[f"MA2_{col}"] = grouped_by_city[col].rolling(2).mean().reset_index(level=0, drop=True)
            new_features_dict[f"MA3_{col}"] = grouped_by_city[col].rolling(3).mean().reset_index(level=0, drop=True)
    if new_features_dict:
        df_sorted = pd.concat([df_sorted, pd.DataFrame(new_features_dict, index=df_sorted.index)], axis=1)
    return df_sorted

def deltaf(df: pd.DataFrame, feature_cols: List[str], lag: int = 1, prefixes=("z_", "bls_", "census_", "pop_")) -> pd.DataFrame:
    df_sorted = df.sort_values(["city_id", YEAR_COL]).copy()
    grouped_by_city = df_sorted.groupby("city_id")
    new_features_dict = {}
    for col in feature_cols:
        col_lower = col.lower()
        has_prefix = any(col_lower.startswith(p) for p in prefixes)
        if has_prefix and col in df_sorted and pd.api.types.is_numeric_dtype(df_sorted[col]):
            prev_val = grouped_by_city[col].shift(lag)
            prev_val_safe = prev_val.replace(0, np.nan) # Avoid division by zero
            new_features_dict[f"R{lag}_{col}"] = (df_sorted[col] - prev_val) / prev_val_safe
    if new_features_dict:
        df_sorted = pd.concat([df_sorted, pd.DataFrame(new_features_dict, index=df_sorted.index)], axis=1)
    return df_sorted

def roll(df: pd.DataFrame, feature_cols: List[str], windows=[2, 3, 4]) -> pd.DataFrame:
    df_sorted = df.sort_values(["city_id", YEAR_COL]).copy()
    grouped_by_city = df_sorted.groupby("city_id")
    for col in feature_cols:
        if col in df_sorted.columns and pd.api.types.is_numeric_dtype(df_sorted[col]):
            for window_size in windows:
                new_stats_dict = {}
                rolling_obj = grouped_by_city[col].rolling(window=window_size, min_periods=1)
                new_stats_dict[f"ROLL_MEAN{window_size}_{col}"] = rolling_obj.mean().reset_index(level=0, drop=True)
                new_stats_dict[f"ROLL_STD{window_size}_{col}"] = rolling_obj.std().reset_index(level=0, drop=True)
                new_stats_dict[f"ROLL_MIN{window_size}_{col}"] = rolling_obj.min().reset_index(level=0, drop=True)
                new_stats_dict[f"ROLL_MAX{window_size}_{col}"] = rolling_obj.max().reset_index(level=0, drop=True)
                new_stats_dict[f"ROLL_MEDIAN{window_size}_{col}"] = rolling_obj.median().reset_index(level=0, drop=True)
                new_stats_dict[f"ROLL_SKEW{window_size}_{col}"] = rolling_obj.skew().reset_index(level=0, drop=True)
                if window_size > 1:
                    new_stats_dict[f"ROLL_PCTCHG{window_size}_{col}"] = grouped_by_city[col].pct_change(periods=window_size - 1, fill_method=None).reset_index(level=0, drop=True)
                    new_stats_dict[f"ROLL_RANGE{window_size}_{col}"] = new_stats_dict[f"ROLL_MAX{window_size}_{col}"] - new_stats_dict[f"ROLL_MIN{window_size}_{col}"]
                    denominator = pd.Series(new_stats_dict[f"ROLL_MEAN{window_size}_{col}"]).where(lambda x: x != 0, 1e-8)
                    new_stats_dict[f"ROLL_COV{window_size}_{col}"] = new_stats_dict[f"ROLL_STD{window_size}_{col}"] / denominator
                for q in [0.1, 0.25, 0.75, 0.9]:
                    new_stats_dict[f"ROLL_Q{int(q*100)}_{window_size}_{col}"] = rolling_obj.quantile(q).reset_index(level=0, drop=True)
                df_sorted = df_sorted.assign(**new_stats_dict)
    return df_sorted

def pick(df: pd.DataFrame, top_k: int = 40) -> List[str]:
    from sklearn.feature_selection import mutual_info_regression, f_regression
    candidate_features = []
    for col in FEATURES:
        if (col in df.columns 
            and pd.api.types.is_numeric_dtype(df[col]) 
            and df[col].notna().mean() >= 0.80 
            and df[col].nunique() > 1):         
            candidate_features.append(col)
    if not candidate_features:
        return []
    X_all = df[candidate_features].astype(float).replace([np.inf, -np.inf], np.nan)
    y_all = df[TARGET_COL].astype(float)
    
    valid_target_mask = y_all.notna()
    X_valid = X_all.loc[valid_target_mask]
    y_valid = y_all.loc[valid_target_mask]
    median_values = X_valid.median(numeric_only=True)
    X_imputed = X_valid.fillna(median_values)
    mi_scores = mutual_info_regression(X_imputed, y_valid, discrete_features=False, random_state=42, n_neighbors=5)
    f_scores, p_values = f_regression(X_imputed, y_valid)
    p_values_safe = np.nan_to_num(p_values, nan=0.0, posinf=0.0, neginf=0.0) 
    mi_max = mi_scores.max() + 1e-12
    mi_scaled = mi_scores / mi_max
    p_value_max = p_values_safe.max() + 1e-12
    p_value_scaled = p_values_safe / p_value_max
    hybrid_score = 0.5 * mi_scaled + 0.5 * p_value_scaled
    sorted_indices = np.argsort(hybrid_score)[::-1]
    num_to_keep = min(top_k, len(candidate_features))
    top_features = [candidate_features[idx] for idx in sorted_indices[:num_to_keep]]
    lagged_target_feat = f"L1_{TARGET_COL}"
    if lagged_target_feat in df.columns and lagged_target_feat not in top_features:
        top_features[-1] = lagged_target_feat  
    return top_features

def eng(df: pd.DataFrame, base_features: List[str], horizon: int) -> pd.DataFrame:
    df_with_target_lag = oldt(df, horizon=horizon, lag=1)
    df_with_state_lag = olds(df_with_target_lag, horizon=horizon, lag=1)
    covered_features = cover(df_with_state_lag, base_features, 0.80)
    if len(covered_features) == 0:
        covered_features = base_features[:30]
    df_with_lags = lags(df_with_state_lag, covered_features, lag_periods=(1, 2, 3))
    df_with_deltas = deltaf(df_with_lags, covered_features, lag=1)
    df_with_rolling = roll(df_with_deltas, covered_features, windows=[2, 3, 4])
    return df_with_rolling

def feat(a: str, h: int) -> pd.DataFrame: #___________________________________________________________________
    global FEATURES, TARGET_COL, HG, FEAT_SELECTED
    HG = h
    TARGET_COL = tcol(h)
    getd(a)
    b = base(a)
    c = targ(b, h, dropna=True)
    d = eng(c, FEATURES, h)
    e = pick(d, top_k=40)
    FEATURES = e if len(e) > 0 else cover(d, FEATURES, 0.80)[:20]
    FEAT_SELECTED[h] = list(FEATURES)
    f = d.dropna(subset=[TARGET_COL])
    f = f.loc[:, ~f.columns.duplicated()]
    return f

def feat_infer(a: str, h: int, use_features: List[str] | None = None) -> pd.DataFrame:
    global TARGET_COL, HG
    HG = h
    TARGET_COL = tcol(h)
    getd(a)
    b = base(a)
    c = targ(b, h, dropna=False)
    d = eng(c, FEATURES, h)
    d = d.loc[:, ~d.columns.duplicated()]
    return d

def cut(a: pd.DataFrame, b: int, c: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = a[a[YEAR_COL] <= b].copy()
    e = a[a[YEAR_COL].isin(c)].copy()
    return d, e

def mat(a: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    b = [c for c in FEATURES if c in a.columns]
    c = a[b].replace([np.inf, -np.inf], np.nan)
    c = c.loc[:, ~c.columns.duplicated()]
    d = a[TARGET_COL].astype(float).replace([np.inf, -np.inf], np.nan)
    e = d.notna()
    f = c.loc[e].copy()
    g = d.loc[e].copy()
    return f, g

def winso(a: pd.DataFrame, b: List[str], c: float = 0.01, d: float = 0.99) -> Dict[str, Tuple[float, float]]:
    e = {}
    for f in b:
        if f in a.columns and pd.api.types.is_numeric_dtype(a[f]):
            i = a[f].quantile(c)
            j = a[f].quantile(d)
            e[f] = (i, j)
    return e

def applycaps(a: pd.DataFrame, b: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    c = a.copy()
    for d, e in b.items():
        if d in c.columns:
            f, g = e
            c[d] = c[d].clip(f, g)
    return c

def captt(a: pd.DataFrame, b: pd.DataFrame, lo=0.01, hi=0.99):
    c = list(a.columns)
    d = winso(a, c, c=lo, d=hi)
    e = applycaps(a, d)
    f = applycaps(b, d)
    return e, f, d

def metr(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import spearmanr, ConstantInputWarning
    c = float(np.sqrt(mean_squared_error(a, b)))
    d = float(mean_absolute_error(a, b))
    e = float(r2_score(a, b))
    f = np.isfinite(a) & np.isfinite(b)
    g = np.nan
    if f.any():
        h = np.asarray(a, dtype=float)[f]
        i = np.asarray(b, dtype=float)[f]
        if not np.isclose(np.nanstd(h), 0.0) and not np.isclose(np.nanstd(i), 0.0):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConstantInputWarning)
                j, _ = spearmanr(h, i)
            g = float(j) if np.isfinite(j) else np.nan
    return {"rmse": c, "mae": d, "r2": e, "spearman": g}

def scatter(a, b, c: str):
    import matplotlib.pyplot as plt
    d = pd.Series(a).rank(method="average")
    e = pd.Series(b).rank(method="average")
    plt.figure()
    plt.scatter(d, e, s=22, alpha=0.7)
    f = int(np.nanmax(np.maximum(d.values, e.values)))
    plt.plot([1, f], [1, f], linewidth=1)
    g = plt.gca()
    g.set_aspect("equal", adjustable="box")
    g.set_xticks(range(1, f + 1, max(1, f // 10)))
    g.set_yticks(range(1, f + 1, max(1, f // 10)))
    g.grid(True, linewidth=0.5, alpha=0.3)
    plt.xlabel("Actual rank")
    plt.ylabel("Predicted rank")
    plt.title("Ranking agreement")
    Path(c).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(c, dpi=160)
    plt.close()

def ridge(a: pd.DataFrame, b: pd.Series):
    from sklearn.linear_model import Ridge as R
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    c = list(a.columns)
    d = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    e = ColumnTransformer([("num", d, c)], remainder="drop")
    f = Pipeline([("pre", e), ("model", R(alpha=0.3, random_state=0))])
    return f.fit(a, b)

def rf(a: pd.DataFrame, b: pd.Series):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    c = a.select_dtypes(include=["number"]).columns.tolist()
    d = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    e = ColumnTransformer([("num", d, c)])
    f = RandomForestRegressor(n_estimators=700, max_depth=10, min_samples_split=8, min_samples_leaf=3, max_features="sqrt", bootstrap=True, oob_score=False, n_jobs=-1, random_state=0)
    g = Pipeline([("preprocessor", e), ("model", f)])
    return g.fit(a, b)

def gbrt(a: pd.DataFrame, b: pd.Series):
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    c = a.select_dtypes(include=["number"]).columns.tolist()
    d = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    e = ColumnTransformer([("num", d, c)])
    f = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=800, max_depth=4, min_samples_leaf=20, max_bins=64, l2_regularization=0.2, early_stopping=True, validation_fraction=0.2, n_iter_no_change=15, random_state=42)
    g = Pipeline([("preprocessor", e), ("model", f)])
    return g.fit(a, b)

def rftune(a: pd.DataFrame, b: pd.Series):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint
    c = RandomForestRegressor(random_state=42, n_jobs=-1)
    d = {"n_estimators": randint(400, 1200), "max_depth": [None, 8, 10, 12], "min_samples_split": randint(2, 20), "min_samples_leaf": randint(1, 6), "max_features": ["sqrt", "log2", 0.5], "bootstrap": [True]}
    e = RandomizedSearchCV(c, d, n_iter=18, cv=5, verbose=0, random_state=42, n_jobs=-1, scoring="neg_mean_squared_error")
    f = e.fit(a, b)
    return f.best_estimator_

def ridgetune(a: pd.DataFrame, b: pd.Series):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    c = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("ridge", Ridge(random_state=42))])
    d = {"ridge__alpha": [0.05, 0.1, 0.3, 1.0, 3.0], "ridge__fit_intercept": [True, False]}
    e = GridSearchCV(c, d, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=0)
    f = e.fit(a, b)
    return f.best_estimator_

def evalm(a, b: pd.DataFrame, c: pd.Series, d: pd.DataFrame, e: pd.Series) -> Dict[str, Any]:
    f = time.time()
    g = a.fit(b, c)
    h = time.time() - f
    i = g.predict(d)
    j = metr(e.values, i)
    j["train_seconds"] = float(h)
    return j, i

def dtarget(a: pd.DataFrame, b: int, c: list):
    def x(d):
        e = [.01, .05, .5, .95, .99]
        f = d[TARGET_COL].describe(percentiles=e)
        return {"count": int(f["count"]), "mean": float(f["mean"]), "std": float(f["std"]), "p01": float(f["1%"]), "p05": float(f["5%"]), "p50": float(f["50%"]), "p95": float(f["95%"]), "p99": float(f["99%"])}
    g, h = cut(a, b, c)
    return {"overall": x(a), "train": x(g), "test": x(h)}

def zbase(a: pd.DataFrame, b: int, c: list, h: int):
    d, e = cut(a, b, c)
    f, g = mat(e)
    if g.empty:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "spearman": np.nan}
    i = float(g.mean())
    j = np.full_like(g.values, i, dtype=float)
    k = metr(g.values, j)
    return k

def compare(a: List[Dict[str, Any]]) -> pd.DataFrame:
    b = pd.DataFrame(a)
    b["rmse_improvement"] = 1 - b["rmse"] / b["rmse"].max()
    b["mae_improvement"] = 1 - b["mae"] / b["mae"].max()
    return b.sort_values("rmse")

def prepare_X_for_year(year: int, feat_list: List[str] | None = None):
    a = feat_infer(GPATH, HG)
    b = a[a[YEAR_COL] == year].copy()
    want = feat_list if feat_list is not None else FEAT_SELECTED.get(HG, FEATURES)
    want = [d for d in want if d in b.columns]
    X = b[want].replace([np.inf, -np.inf], np.nan)
    M = b[["city_id", YEAR_COL, POP_COL]].copy()
    return X, M