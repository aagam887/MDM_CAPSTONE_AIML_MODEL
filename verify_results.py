"""
Verification script for ICU Sepsis Capstone Project.
Reads every metrics JSON/CSV and prints a consolidated audit.
"""

import json
import os
import pandas as pd

print("=" * 75)
print("COMPLETE RESULTS VERIFICATION — ICU SEPSIS CAPSTONE")
print("=" * 75)

# ============================================================
# 1. paper_metrics.json — Final XGBoost + dataset stats
# ============================================================
print("\n📄 1. paper_metrics.json")
print("-" * 75)
if os.path.exists("paper_metrics.json"):
    with open("paper_metrics.json") as f:
        pm = json.load(f)

    # Dataset
    if "dataset" in pm:
        ds = pm["dataset"]
        print(f"   Dataset samples       : {ds.get('n_samples', 'N/A'):,}")
        print(f"   Patients              : {ds.get('n_patients', 'N/A'):,}")
        print(f"   Features              : {ds.get('n_features', 'N/A')}")
        print(f"   Positive rate         : {ds.get('positive_rate', 0):.4f}")
        print(f"   Imbalance ratio       : {ds.get('class_imbalance_ratio', 0):.2f}:1")

    # XGBoost 6h
    if "xgb_6h" in pm:
        m = pm["xgb_6h"]
        print(f"\n   ── XGBoost 6h ──")
        print(f"      AUROC     : {m.get('AUROC', 'N/A'):.4f}")
        print(f"      AUPRC     : {m.get('AUPRC', 'N/A'):.4f}")
        print(f"      Recall    : {m.get('Recall', 'N/A'):.4f}")
        print(f"      Precision : {m.get('Precision', 'N/A'):.4f}")
        print(f"      F1-Score  : {m.get('F1-Score', 'N/A'):.4f}")
        print(f"      Brier     : {m.get('Brier', 'N/A'):.4f}")
        print(f"      TP / FP   : {m.get('TP', 'N/A')} / {m.get('FP', 'N/A')}")
        print(f"      TN / FN   : {m.get('TN', 'N/A')} / {m.get('FN', 'N/A')}")

    # XGBoost 12h
    if "xgb_12h" in pm:
        m = pm["xgb_12h"]
        print(f"\n   ── XGBoost 12h ──")
        print(f"      AUROC     : {m.get('AUROC', 'N/A'):.4f}")
        print(f"      AUPRC     : {m.get('AUPRC', 'N/A'):.4f}")
        print(f"      Recall    : {m.get('Recall', 'N/A'):.4f}")
        print(f"      Precision : {m.get('Precision', 'N/A'):.4f}")
        print(f"      F1-Score  : {m.get('F1-Score', 'N/A'):.4f}")
        print(f"      Brier     : {m.get('Brier', 'N/A'):.4f}")
        print(f"      TP / FP   : {m.get('TP', 'N/A')} / {m.get('FP', 'N/A')}")
        print(f"      TN / FN   : {m.get('TN', 'N/A')} / {m.get('FN', 'N/A')}")
else:
    print("   ❌ File not found")

# ============================================================
# 2. baseline_metrics.json — Mean metrics for 3 baselines
# ============================================================
print("\n\n📄 2. baseline_metrics.json")
print("-" * 75)
if os.path.exists("baseline_metrics.json"):
    with open("baseline_metrics.json") as f:
        bm = json.load(f)

    for model, metrics in bm.items():
        print(f"\n   ── {model} ──")
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    print(f"      {k:12s}: {v:.4f}")
else:
    print("   ❌ File not found")

# ============================================================
# 3. ablation_results.csv — Feature subset comparison
# ============================================================
print("\n\n📄 3. ablation_results.csv")
print("-" * 75)
if os.path.exists("ablation_results.csv"):
    ab = pd.read_csv("ablation_results.csv")
    print(ab.to_string(index=False))
else:
    print("   ❌ File not found")

# ============================================================
# 4. baseline_comparison_results.csv — Final comparison table
# ============================================================
print("\n\n📄 4. baseline_comparison_results.csv")
print("-" * 75)
if os.path.exists("baseline_comparison_results.csv"):
    bc = pd.read_csv("baseline_comparison_results.csv")
    print(bc.to_string(index=False))
else:
    print("   ❌ File not found")

# ============================================================
# 5. Per-fold CSVs — Cross-validation audit
# ============================================================
print("\n\n📄 5. Per-fold cross-validation CSVs")
print("-" * 75)
for fname in ["baseline_logreg_per_fold.csv",
              "baseline_rf_per_fold.csv",
              "baseline_hgb_per_fold.csv"]:
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        print(f"\n   ── {fname} ──")
        print(df.to_string(index=False))
    else:
        print(f"   ❌ {fname} not found")

# ============================================================
# 6. models/metrics_6h.json and 12h.json
# ============================================================
print("\n\n📄 6. models/metrics_*.json")
print("-" * 75)
for fname in ["models/metrics_6h.json", "models/metrics_12h.json"]:
    if os.path.exists(fname):
        with open(fname) as f:
            m = json.load(f)
        print(f"\n   ── {fname} ──")
        for k, v in m.items():
            print(f"      {k}: {v}")
    else:
        print(f"   ❌ {fname} not found")

# ============================================================
# 7. shap_figures/summary.json
# ============================================================
print("\n\n📄 7. shap_figures/summary.json")
print("-" * 75)
if os.path.exists("shap_figures/summary.json"):
    with open("shap_figures/summary.json") as f:
        sm = json.load(f)

    print(f"   n_features        : {sm.get('n_features', 'N/A')}")
    print(f"   n_shap_samples    : {sm.get('n_shap_samples', 'N/A'):,}")
    print(f"   base_6h           : {sm.get('base_6h', 'N/A'):.4f}")
    print(f"   base_12h          : {sm.get('base_12h', 'N/A'):.4f}")
    print(f"\n   Top 10 features (6h):")
    for i, feat in enumerate(sm.get("top10_features_6h", []), 1):
        print(f"      {i}. {feat}")
    print(f"\n   Top 10 features (12h):")
    for i, feat in enumerate(sm.get("top10_features_12h", []), 1):
        print(f"      {i}. {feat}")
else:
    print("   ❌ File not found")

# ============================================================
# 8. future_work/dl_metrics.json and dl_metrics_option_c.json
# ============================================================
print("\n\n📄 8. future_work DL metrics")
print("-" * 75)
for fname in ["future_work/dl_metrics.json",
              "future_work/dl_metrics_option_c.json"]:
    if os.path.exists(fname):
        with open(fname) as f:
            dl = json.load(f)
        print(f"\n   ── {fname} ──")
        print(json.dumps(dl, indent=6))
    else:
        print(f"   ❌ {fname} not found")

# ============================================================
# 9. patient_predictions.csv — sanity check
# ============================================================
print("\n\n📄 9. patient_predictions.csv — SHAP coverage check")
print("-" * 75)
if os.path.exists("patient_predictions.csv"):
    pp = pd.read_csv("patient_predictions.csv")
    print(f"   Rows                 : {len(pp):,}")
    print(f"   Columns              : {list(pp.columns)}")
    if "Top_Factors_6h" in pp.columns:
        print(f"   Top_Factors_6h non-null : {pp['Top_Factors_6h'].notna().sum():,} / {len(pp):,}")
    if "Top_Factors_12h" in pp.columns:
        print(f"   Top_Factors_12h non-null: {pp['Top_Factors_12h'].notna().sum():,} / {len(pp)}")
    if "Recommendation" in pp.columns:
        print(f"   Recommendations non-null: {pp['Recommendation'].notna().sum():,} / {len(pp):,}")
else:
    print("   ❌ File not found")

# ============================================================
# 10. CROSS-CHECK — consistency check
# ============================================================
print("\n\n🔍 10. CROSS-CHECK — Are numbers consistent?")
print("-" * 75)

# Check XGBoost AUROC consistency between paper_metrics.json and PPT
if os.path.exists("paper_metrics.json"):
    with open("paper_metrics.json") as f:
        pm = json.load(f)

    auroc_6h = pm.get("xgb_6h", {}).get("AUROC", 0)
    auroc_12h = pm.get("xgb_12h", {}).get("AUROC", 0)

    print(f"   XGBoost 6h AUROC (paper_metrics.json) : {auroc_6h:.4f}")
    print(f"   XGBoost 12h AUROC (paper_metrics.json): {auroc_12h:.4f}")

    # Compare to baseline
    if os.path.exists("baseline_metrics.json"):
        with open("baseline_metrics.json") as f:
            bm = json.load(f)
        best_baseline_auroc = max(
            m.get("AUROC", 0) for m in bm.values() if isinstance(m, dict)
        )
        print(f"   Best baseline AUROC                   : {best_baseline_auroc:.4f}")
        print(f"   Improvement over baseline            : +{auroc_6h - best_baseline_auroc:.4f}")

    # Check SHAP factor coverage
    if os.path.exists("patient_predictions.csv"):
        pp = pd.read_csv("patient_predictions.csv")
        coverage = pp["Top_Factors_6h"].notna().sum() / len(pp) * 100
        print(f"   SHAP factor coverage                  : {coverage:.1f}%")

print("\n" + "=" * 75)
print("✅ VERIFICATION COMPLETE")
print("=" * 75)