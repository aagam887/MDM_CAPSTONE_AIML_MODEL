# 🏥 AI-Based ICU Patient (Sepsis) Deterioration & Early Warning System

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.44+-purple.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](#license)

A **temporal machine learning pipeline** for early sepsis prediction in the ICU, combining **engineered lag/rolling/trend features** with **regularized XGBoost** and **SHAP explainability**. Benchmarked against classical ML baselines and deep learning alternatives under rigorous **5-fold GroupKFold** cross-validation.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Results Highlights](#-results-highlights)
- [Explainability (SHAP)](#-explainability-shap)
- [Live Dashboard](#-live-dashboard)
- [Key Findings](#-key-findings)
- [Limitations & Future Work](#-limitations--future-work)
- [References](#-references)
- [Author](#-author)
- [License](#-license)

---

## 🎯 Overview

Sepsis is a **life-threatening organ dysfunction** caused by a dysregulated host response to infection, affecting **~48.9 million people worldwide annually** and contributing to **~11 million deaths**. Every hour of delayed treatment increases mortality by **4–8%**.

This project delivers an **early warning system** that predicts sepsis onset **6–12 hours before clinical recognition**, using engineered temporal features from ICU time-series data. The system is fully deployed with a **live Streamlit dashboard** for bedside clinical decision support.

### 🏆 Headline Result

| Model | AUROC | AUPRC | Recall |
|---|---|---|---|
| **Temporal XGBoost (6h)** | **0.8003** | **0.1122** | **0.5865** |
| **Temporal XGBoost (12h)** | **0.8035** | **0.1153** | **0.6084** |
| Best Classical Baseline (RF) | 0.7391 | 0.0879 | 0.4156 |

**+0.06 AUROC and +0.03 AUPRC** improvement over the best baseline, validating the value of temporal feature engineering.

---

## 📊 Key Results

### Master Model Comparison (5-Fold GroupKFold)

| Family | Model | AUROC | AUPRC | Recall |
|---|---|---|---|---|
| Classical | Logistic Regression + SMOTE | 0.703 | 0.080 | 0.601 |
| Classical | Random Forest + SMOTE | 0.739 | 0.088 | 0.416 |
| Classical | HistGradientBoosting + SMOTE | 0.733 | 0.088 | 0.367 |
| **Advanced** | **Temporal XGBoost (6h)** | **0.800** | **0.112** | **0.587** |
| **Advanced** | **Temporal XGBoost (12h)** | **0.804** | **0.115** | **0.608** |
| Future Work | LSTM (proper prediction) | 0.571 | 0.094 | 0.491 |
| Future Work | GRU (proper prediction) | 0.563 | 0.092 | 0.508 |

### Ablation Study — Feature Subset Impact

| Feature Subset | AUROC | AUPRC |
|---|---|---|
| Full (84 features) | 0.7646 | 0.1079 |
| Dynamic only (77) | 0.6844 | 0.0535 |
| Static + time only (7) | 0.6791 | 0.0732 |
| No ICULOS + HospAdmTime | 0.7587 | 0.1017 |

**Key insight:** Removing dynamic vitals drops AUROC by ~8 points. Removing the strongest static feature (ICULOS) loses only ~0.6 points — proving temporal features carry the real signal.

---

## 📁 Dataset

**Source:** [PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/)

| Attribute | Value |
|---|---|
| Total hourly records | 546,123 |
| Unique patients | 14,057 |
| Clinical variables | 40 (vitals, labs, demographics) |
| Target label | `SepsisLabel` (binary, Sepsis-3) |
| Row-level prevalence | ~2.17% (45:1 imbalance) |
| Patient-level prevalence | ~8.81% |
| Time resolution | 1 hour |

---

## 🔬 Methodology

### Pipeline
