# Early Prediction of Sepsis in ICU — Capstone Project

**Author:** Aagam Katariya

## Summary
Early warning system for sepsis onset using temporal feature engineering 
and gradient boosting (XGBoost), benchmarked against classical ML baselines.

## Key Result

| Model | AUROC | AUPRC |
|---|---|---|
| Logistic Regression + SMOTE | 0.703 | 0.079 |
| Random Forest + SMOTE | 0.739 | 0.088 |
| HistGradientBoosting + SMOTE | 0.733 | 0.088 |
| **Temporal XGBoost (6h)** | **0.87+** | **0.40+** |
| **Temporal XGBoost (12h)** | **0.86+** | **0.38+** |

Temporal XGBoost significantly outperforms all baselines, validating 
the value of engineered lag/rolling/trend features.

## Repository Structure

| Synopsis Section | Code File |
|---|---|
| Dataset & Preprocessing | `Dataset.csv`, `model_training.ipynb` |
| Baseline Models + GroupKFold CV | `baseline_models_and_cv.ipynb` |
| Temporal XGBoost Training | `model_training.ipynb`, `models/` |
| SHAP Explainability | `shap_figures/`, `summary.json` |
| Metrics for Paper | `metrics_for_paper.csv`, `paper_metrics.json` |
| Baseline Comparison | `baseline_comparison_results.csv` |
| Dashboard Deployment | `streamlit_app.py`, `dashboard.py` |
| Deep Learning (Future Work) | `future_work/` |

## How to Run
1. `pip install -r requirements.txt`
2. Run `model_training.ipynb` (trains XGBoost + SHAP)
3. Run `baseline_models_and_cv.ipynb` (baseline comparison)
4. Launch dashboard: `streamlit run streamlit_app.py`