# Mitsui Commodity Price Prediction (2025 Kaggle Challenge)

This repository documents my approach to the **Mitsui & Co. Commodity Price Prediction Challenge (2025)**.  
The task: develop stable and accurate models that predict 424 target time series of commodity price differences, using global financial market data such as LME, JPX, US stocks, and Forex.

---

## Highlights
- End-to-end ML pipeline: ETL → Feature Engineering → Model Training → Inference  
- Custom rank-based Sharpe ratio evaluation (competition metric)  
- Random Forest feature selection + XGBoost regressors per target  
- Stable generalization: 0.20 (submission) vs. 0.19 (validation)

---

## Approach

1. **Data Processing & Feature Engineering**
   - Forward-filling and cleaning noisy series
   - Rolling means & returns (1, 5, 10, 20 days)  
   - Commodity pair spreads and FX/LME ratios  
   - Handling broken numeric columns

2. **Feature Selection**
   - Train a **Random Forest regressor**  
   - Rank features by importance  
   - Retain the top ~5% for efficiency  

3. **Model Training**
   - Train **XGBoost regressors** per target series  
   - Time-based validation (last 180 days as holdout)  
   - Early stopping on RMSE to prevent overfitting  

4. **Evaluation**
   - Compute the **Rank-based Sharpe ratio** (mean Spearman rank correlation (predictions vs. targets) divided by its standard deviation) on validation split  
   - Achieved ~0.20 (submission) vs. ~0.19 (validation) → stable generalization  

5. **Inference**
   - Artifacts (features, models, metadata) are saved in JSON format  
   - `mitsui2025_inference.py` loads these artifacts and serves predictions through Kaggle’s `MitsuiInferenceServer`
   
---

## Repository Structure

```
mitsui-commodity-price-prediction/
│
├── mitsui2025_pipeline.py      # Full pipeline: preprocessing, FE, RF feature selection, XGB training
├── mitsui2025_inference.py     # Inference server: loads artifacts & serves predictions
├── artifacts/                  # Saved models, features, and metadata
└── README.md
```

---

## Usage

### 1. Train & Generate Artifacts
```bash
python mitsui2025_pipeline.py
```

Artifacts are saved under:
```
/artifacts_predict_mitsui/
```

### 2. Run Inference
```bash
python mitsui2025_inference.py
```

- In Kaggle: connects to the evaluation API.  
- Locally: runs a mock inference gateway on sample data.  

---

## Results

- **Submission Sharpe ratio:** ~0.20  
- **Validation Sharpe ratio:** ~0.19  
- The model shows stable performance across phases with robust generalization.  

---

## Requirements

- Python ≥ 3.9  
- Dependencies:  
  - `pandas`, `polars`, `numpy`, `scikit-learn`, `xgboost`, `tqdm`  
  - `kaggle_evaluation` (preinstalled in Kaggle runtime)  

Install with:
```bash
pip install -r requirements.txt
```

---

## Notes

- The code is structured for Kaggle’s runtime (paths may need minor tweaks locally).  
- Pipeline and inference scripts are decoupled for reproducibility.  
- The methodology can be extended with richer FE, ensembling, or deep learning backbones.  

---

## References

- [Competition Overview](https://www.kaggle.com/competitions/mitsui-commodity-price-prediction-challenge)  
