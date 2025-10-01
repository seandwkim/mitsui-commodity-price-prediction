import pandas as pd, polars as pl, numpy as np
import time, os, gc, json, itertools, warnings
warnings.simplefilter('ignore')

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tqdm import tqdm
from math import ceil
from pathlib import Path
from datetime import datetime

class CFG:
    path = "/kaggle/input/mitsui-commodity-prediction-challenge/"
    seed = 42
    targets = [f"target_{i}" for i in range (424)]
    solution_null_filler = 0.0
    train_end = 1826
    test_start = 1827
    test_end = 1956
    out_dir = "/kaggle/working/artifacts_predict_mitsui"

os.makedirs(CFG.out_dir, exist_ok=True)
os.makedirs(f"{CFG.out_dir}/xgb_json", exist_ok=True)

# ==============================================================================
# RANK-BASED SHARPE RATIO METRIC
# ==============================================================================

def rank_correlation_sharpe_ratio(merged_df: pd.DataFrame) -> float:
    prediction_cols = [c for c in merged_df.columns if c.startswith("prediction_")]
    target_cols = [c for c in merged_df.columns if c.startswith("target_")]

    def _compute_rank_correlation(row):
        non_null_targets = [c for c in target_cols if not pd.isnull(row[c])]
        if not non_null_targets:
            return 0.0

        preds_for_targets = [f"prediction_{c.split('_', 1)[1]}" for c in non_null_targets]
        cols_exist = [
            (t, p) for t, p in zip(non_null_targets, preds_for_targets) if p in row.index
        ]
        if not cols_exist:
            return 0.0
        non_null_targets, preds_for_targets = zip(*cols_exist)

        tvals = pd.Series(row[list(non_null_targets)])
        pvals = pd.Series(row[list(preds_for_targets)])

        if tvals.std(ddof=0) == 0 or pvals.std(ddof=0) == 0:
            return 0.0

        return np.corrcoef(
            pvals.rank(method="average"), tvals.rank(method="average")
        )[0, 1]

    daily_rank_corrs = merged_df.apply(_compute_rank_correlation, axis=1)
    std_dev = daily_rank_corrs.std(ddof=0)
    if std_dev == 0:
        return 0.0
    return float(daily_rank_corrs.mean() / std_dev)


# ==============================================================================
# DATA PROCESSING
# ==============================================================================

train = pd.read_csv(CFG.path + "train.csv")
train_labels = pd.read_csv(CFG.path + "train_labels.csv")
target_pairs = pd.read_csv(CFG.path + "target_pairs.csv")

target_constituent_pairs = (
    target_pairs["pair"].dropna().str.strip()
    .str.split(r"\s*-\s*", expand=True)
    .rename(columns={0: "left", 1: "right"})
    .dropna(subset=["left", "right"])
)
target_constituents = pd.Index(target_constituent_pairs["left"]).union(target_constituent_pairs["right"])
target_cols_in_train = target_constituents.intersection(train.columns).tolist()

cols_A = list(dict.fromkeys(target_cols_in_train + ["date_id"]))
train_A = train[cols_A]

cols_B = [c for c in train.columns if c not in set(target_cols_in_train)] + ["date_id"]
cols_B = list(dict.fromkeys(cols_B))
train_B = train[cols_B]

def data_processing(df):
    df = df.sort_values("date_id")
    df = df.ffill()
    spread_pairs = target_constituent_pairs[
        target_constituent_pairs["left"].isin(df.columns)
        & target_constituent_pairs["right"].isin(df.columns)
    ]
    spread_cols = {
        f"{a}_{b}_spread": df[a] - df[b]
        for a, b in spread_pairs[["left", "right"]].itertuples(index=False)
    }
    periods = [1, 5, 10, 20]
    new_cols = {}
    for col in target_constituents.intersection(df.columns):
        for period in periods:
            ratio = df[col] / df[col].shift(period)
            ratio = ratio.replace([np.inf, -np.inf], np.nan)
            new_cols[f"{col}_returns_{period}"] = ratio.fillna(0)
            roll = df[col].rolling(period, min_periods=1)
            mean = roll.mean()
            new_cols[f"{col}_rolling_mean_{period}"] = mean

    lme_fx_values = [val for val in target_constituents if str(val).startswith(("LME", "FX"))]
    lme_fx_pairs = [(a, b) for a, b in itertools.combinations(lme_fx_values, 2)]
    for a, b in lme_fx_pairs:
        if a in df.columns and b in df.columns:
            new_cols[f"{a}_{b}_ratio"] = df[a] / df[b]

    broken_columns = [
        "US_Stock_GOLD_adj_open",
        "US_Stock_GOLD_adj_high",
        "US_Stock_GOLD_adj_low",
        "US_Stock_GOLD_adj_close",
        "US_Stock_GOLD_adj_volume"
    ]
    for col in broken_columns:
        if col in df.columns and df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = pd.concat([df, pd.DataFrame(spread_cols, index=df.index)], axis=1)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].clip(-1e10, 1e10)
    
    return df


# ==============================================================================
# MODELS
# ==============================================================================
train_enhanced_A = data_processing(train_A)
train_enhanced_B = data_processing(train_B)
KEY = "date_id"
features_B = [c for c in train_enhanced_B.columns if c != KEY]
y_train = train_labels.iloc[:CFG.train_end+1][CFG.targets].fillna(CFG.solution_null_filler)
X_train_B = train_enhanced_B.iloc[:CFG.train_end+1][features_B].replace([np.inf, -np.inf], np.nan).fillna(0)

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=CFG.seed,
    n_jobs=-1
)
rf.fit(X_train_B, y_train)

rf_importance = rf.feature_importances_
feat_names_B = np.array(features_B)
k = max(1, ceil(0.05 * len(feat_names_B)))
top_idx = np.argsort(rf_importance)[::-1][:k]
top_features_B = feat_names_B[top_idx].tolist()

B_top = train_enhanced_B[[KEY] + top_features_B]
train_enhanced_A = train_enhanced_A.merge(B_top, on=KEY, how="left")

CFG.features = [c for c in train_enhanced_A.columns if c != KEY]
X_train_A = train_enhanced_A.iloc[:CFG.train_end+1][CFG.features].replace([np.inf, -np.inf], np.nan).fillna(0)

VAL_DAYS = 180
val_start = CFG.train_end + 1 - VAL_DAYS
y_tr  = y_train.iloc[:val_start].copy()
y_val = y_train.iloc[val_start:].copy()
X_tr  = X_train_A.iloc[:val_start][CFG.features].astype(np.float32)
X_val = X_train_A.iloc[val_start:][CFG.features].astype(np.float32)

xgb_models = []
xgb_params = dict(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    min_child_weight=1.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=CFG.seed,
    n_jobs=-1,
    verbosity=0
)

for t in CFG.targets:
    model = XGBRegressor(**xgb_params)
    model.fit(
        X_tr, y_tr[t].values,
        eval_set=[(X_val, y_val[t].values)],
        eval_metric="rmse",
        verbose=False,
        early_stopping_rounds=15
    )
    xgb_models.append(model)

X_test = train_enhanced_A.iloc[CFG.test_start:CFG.test_end+1][CFG.features].replace([np.inf, -np.inf], np.nan).fillna(0)
y_test = train_labels.iloc[CFG.test_start:CFG.test_end+1][CFG.targets].fillna(CFG.solution_null_filler)

X_test_xgb = X_test.astype(np.float32)
xgb_pred = np.column_stack([
    m.predict(
        X_test_xgb,
        iteration_range=(0, m.best_iteration + 1) if hasattr(m, "best_iteration") else None
    )
    for m in xgb_models
])

y_pred_df = pd.DataFrame(xgb_pred, columns=CFG.targets)
y_test_df = y_test.reset_index(drop=True)
solution_df = y_test_df.copy()
submission_df = y_pred_df.rename(columns={c: c.replace('target_', 'prediction_') for c in y_pred_df.columns})
merged_df = pd.concat([solution_df, submission_df], axis=1)
sharpe_ratio = rank_correlation_sharpe_ratio(merged_df)
print(f"{sharpe_ratio:.6f}")

# ==============================================================================
# SAVE ARTIFACTS
# ==============================================================================

OUT = Path(CFG.out_dir)
(OUT / "xgb_models").mkdir(parents=True, exist_ok=True)

target_constituent_pairs.to_json(OUT / "target_constituent_pairs.json", orient="records", indent=2)
with open(OUT / "target_constituents.json", "w") as f:
    json.dump(list(map(str, target_constituents)), f, indent=2)

with open(OUT / "features_full.json", "w") as f:
    json.dump(list(map(str, CFG.features)), f, indent=2)
xgb_feats = list(map(str, (XGB_FEATURES if "XGB_FEATURES" in globals() else CFG.features)))
with open(OUT / "xgb_features.json", "w") as f:
    json.dump(xgb_feats, f, indent=2)

xgb_index = []
base_xgb_params = {k: v for k, v in xgb_params.items() if k != "early_stopping_rounds"}
for target, model in zip(CFG.targets, xgb_models):
    p = OUT / "xgb_models" / f"{target}.json"
    model.save_model(str(p))
    xgb_index.append({
        "target": target,
        "model_path": str(p),
        "params": base_xgb_params,
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "features": xgb_feats,
    })
with open(OUT / "xgb_models_index.json", "w") as f:
    json.dump(xgb_index, f, indent=2)

meta = {
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "seed": CFG.seed,
    "targets": list(map(str, CFG.targets)),
    "num_xgb_models": len(xgb_index),
    "num_features_full": len(CFG.features),
    "num_features_xgb": len(xgb_feats),
    "num_target_pairs": int(len(target_constituent_pairs)),
    "num_target_constituents": int(len(target_constituents)),
    "libraries": {
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": __import__('sklearn').__version__,
        "xgboost": __import__('xgboost').__version__,
    }
}
with open(OUT / "artifacts_manifest.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Artifacts saved to:", OUT)
