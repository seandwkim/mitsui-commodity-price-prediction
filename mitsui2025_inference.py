import os, json
from pathlib import Path
import numpy as np
import itertools
import pandas as pd
import polars as pl
from xgboost import XGBRegressor
import kaggle_evaluation.mitsui_inference_server

TARGET_COUNT = 424
TARGET_COLS  = [f"target_{i}" for i in range(TARGET_COUNT)]

ART = Path("/kaggle/input/artifacts-mitsui-prediction/artifacts_predict_mitsui")

import threading
print(f"[boot] starting; ART={ART}", flush=True)

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


def _to_pd(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df

def _wrap_targets_only(pred_vector: np.ndarray) -> pd.DataFrame:
    arr = np.asarray(pred_vector, dtype=np.float64).reshape(-1)
    out = pd.DataFrame([arr], columns=TARGET_COLS)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.fillna(0.0, inplace=True)
    return out

_STATE = {
    "loaded": False,
    "xgb_features": None,
    "xgb_models": None,
    "xgb_best_iterations": None,
    "buffer_raw": None,
}

target_constituent_pairs = None
target_constituents = None

def _install_target_pairs_globals():
    global target_constituent_pairs, target_constituents
    target_constituent_pairs = pd.DataFrame(json.load(open(ART / "target_constituent_pairs.json")))
    target_constituents = pd.Index(target_constituent_pairs["left"]).union(target_constituent_pairs["right"])

def _load_artifacts_once():
    if _STATE["loaded"]:
        return

    print("[artifacts] loading…", flush=True)
    if "_ping_timer" not in _STATE:
        _STATE["_ping_timer"] = threading.Timer(
            300.0,  # 5 minutes
            lambda: print("[status] still loading artifacts…", flush=True)
        )
        _STATE["_ping_timer"].daemon = True
        _STATE["_ping_timer"].start()

    _install_target_pairs_globals()
    xgb_features = json.load(open(ART / "xgb_features.json")) if (ART / "xgb_features.json").exists() \
                   else json.load(open(ART / "features_full.json"))
    xgb_index = json.load(open(ART / "xgb_models_index.json"))

    xgb_models, best_its = [], []
    for item in xgb_index:
        target = item["target"]
        params = item.get("params", {})
        best_it = item.get("best_iteration", -1)
        model_path = Path(item.get("model_path", ""))
        if not model_path.exists():
            model_path = ART / "xgb_models" / f"{target}.json"
        m = XGBRegressor(**params)
        m.load_model(str(model_path))
        m._best_iteration = int(best_it) if best_it is not None else -1
        xgb_models.append(m)
        best_its.append(m._best_iteration)

    _STATE.update({
        "loaded": True,
        "xgb_features": list(map(str, xgb_features)),
        "xgb_models": xgb_models,
        "xgb_best_iterations": np.array(best_its, dtype=int),
        "buffer_raw": pd.DataFrame(),
    })

    t = _STATE.pop("_ping_timer", None)
    if t: 
        try: t.cancel()
        except: pass
    print(f"[artifacts] loaded {len(xgb_models)} xgb models; {len(xgb_features)} features", flush=True)


def _append_buffer(test_pd_row: pd.DataFrame):
    if _STATE["buffer_raw"].empty:
        _STATE["buffer_raw"] = test_pd_row.copy()
    else:
        _STATE["buffer_raw"] = pd.concat([_STATE["buffer_raw"], test_pd_row], axis=0, ignore_index=True)
    _STATE["buffer_raw"] = _STATE["buffer_raw"].tail(25).reset_index(drop=True)

def _engineer_features_with_data_processing() -> pd.DataFrame:
    engineered = data_processing(_STATE["buffer_raw"].copy())
    return engineered.iloc[[-1]]

def predict(
    test: pl.DataFrame | pd.DataFrame,
    label_lags_1_batch: pl.DataFrame | pd.DataFrame,
    label_lags_2_batch: pl.DataFrame | pd.DataFrame,
    label_lags_3_batch: pl.DataFrame | pd.DataFrame,
    label_lags_4_batch: pl.DataFrame | pd.DataFrame,
) -> pd.DataFrame:
    if not _STATE["loaded"]:
        _load_artifacts_once()
        print("[predict] starting…", flush=True)

    test_pd = _to_pd(test)
    _append_buffer(test_pd)
    _STATE["_calls"] = _STATE.get("_calls", 0) + 1
    if _STATE["_calls"] % 2 == 1:
        print(f"[predict] row {_STATE['_calls']} | buffer={len(_STATE['buffer_raw'])}", flush=True)

    row_processed = _engineer_features_with_data_processing()

    X = row_processed.reindex(columns=_STATE["xgb_features"], fill_value=0.0).astype(np.float32)
    preds = []
    for m in _STATE["xgb_models"]:
        it = getattr(m, "_best_iteration", -1)
        if it is not None and it >= 0:
            preds.append(m.predict(X, iteration_range=(0, it + 1)))
        else:
            preds.append(m.predict(X))

    vec = np.column_stack(preds).astype(np.float64).reshape(-1) 
    return _wrap_targets_only(vec)

_inference_server = kaggle_evaluation.mitsui_inference_server.MitsuiInferenceServer(predict)
if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    _inference_server.serve()
else:
    _inference_server.run_local_gateway(("/kaggle/input/mitsui-commodity-prediction-challenge/",))
