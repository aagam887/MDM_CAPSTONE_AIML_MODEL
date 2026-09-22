"""
Streamlit Web Application: ICU Sepsis Early Warning & Risk Assessment Dashboard
Author: Ayush Bankar (Capstone Project)
Usage: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="Sepsis Early Warning System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; color: #1E3A8A; font-weight: 700; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .high-risk { border-left-color: #EF4444 !important; background-color: #FEF2F2 !important; }
    .mod-risk  { border-left-color: #F59E0B !important; background-color: #FFFBEB !important; }
    .low-risk  { border-left-color: #10B981 !important; background-color: #ECFDF5 !important; }
    .vital-box {
        background-color: #F1F5F9; border-radius: 8px; padding: 12px;
        text-align: center; margin-bottom: 8px;
    }
    .vital-label { font-size: 0.8rem; color: #64748B; }
    .vital-value { font-size: 1.4rem; font-weight: bold; color: #1E293B; }
    .vital-unit  { font-size: 0.75rem; color: #94A3B8; }
    .filter-count {
        background: #DBEAFE; color: #1E40AF; padding: 4px 8px;
        border-radius: 6px; font-size: 0.85rem; text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# Title
# ============================================================
st.markdown('<div class="main-header">🏥 ICU Sepsis Early Warning Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time Bedside Risk Assessment & SHAP Interpretability System</div>', unsafe_allow_html=True)


# ============================================================
# Data loading + normalization
# ============================================================
@st.cache_data
def load_data():
    pred_path = "patient_predictions.csv"
    if not os.path.exists(pred_path):
        st.error(f"Prediction file '{pred_path}' not found!")
        return None

    df = pd.read_csv(pred_path)

    rename_map = {
        "Risk_6h": "Prob_6h",
        "Risk_12h": "Prob_12h",
        "Top_Factors_6h": "SHAP_Factors_6h",
        "Top_Factors_12h": "SHAP_Factors_12h",
        "Recommendation": "Clinical_Recommendation",
    }
    df = df.rename(columns=rename_map)

    if "Age" not in df.columns:
        df["Age"] = np.nan
    if "Gender" not in df.columns:
        df["Gender"] = 1

    for c in ["Prob_6h", "Prob_12h"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        if df[c].max() > 1:
            df[c] /= 100

    df["_max_risk"] = df[["Prob_6h", "Prob_12h"]].max(axis=1)
    df["Risk_Tier"] = df["_max_risk"].apply(
        lambda p: "High Risk (>=30%)" if p >= 0.30
        else "Moderate Risk (10-30%)" if p >= 0.10
        else "Low Risk (<10%)"
    )
    return df


df = load_data()


# ============================================================
# Sidebar — All filters
# ============================================================
if df is not None:

    st.sidebar.header("🔍 Filter & Search")

    # -------- Reset button --------
    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.startswith("filter_"):
                del st.session_state[key]
        st.rerun()

    # ============================================================
    # FILTER 1 — Risk tier
    # ============================================================
    st.sidebar.markdown("##### 1️⃣ Risk Tier")
    risk_tier_filter = st.sidebar.multiselect(
        "Risk Level",
        options=["High Risk (>=30%)", "Moderate Risk (10-30%)", "Low Risk (<10%)"],
        default=["High Risk (>=30%)", "Moderate Risk (10-30%)", "Low Risk (<10%)"],
        key="filter_risk_tier",
        label_visibility="collapsed"
    )

    # ============================================================
    # FILTER 2 — Patient ID search
    # ============================================================
    st.sidebar.markdown("##### 2️⃣ Patient ID Search")
    id_search = st.sidebar.text_input(
        "Type part of a Patient ID",
        key="filter_id_search",
        placeholder="e.g. 21 or 6853",
        label_visibility="collapsed"
    )

    # ============================================================
    # FILTER 3 — Min risk threshold
    # ============================================================
    st.sidebar.markdown("##### 3️⃣ Minimum Risk Threshold")
    min_risk = st.sidebar.slider(
        "Show patients with max risk ≥",
        min_value=0.0, max_value=1.0, value=0.0, step=0.05,
        format="%.0f%%",
        key="filter_min_risk",
        label_visibility="collapsed"
    )

    # ============================================================
    # FILTER 4 — ICU Length of Stay range
    # ============================================================
    st.sidebar.markdown("##### 4️⃣ ICU Length of Stay (hours)")
    if "ICULOS" in df.columns and df["ICULOS"].notna().any():
        los_min_data = float(df["ICULOS"].min(skipna=True))
        los_max_data = float(df["ICULOS"].max(skipna=True))
        los_range = st.sidebar.slider(
            "ICULOS range",
            min_value=float(los_min_data),
            max_value=float(los_max_data),
            value=(float(los_min_data), float(los_max_data)),
            step=1.0,
            key="filter_los",
            label_visibility="collapsed"
        )
    else:
        los_range = (0.0, 10000.0)

    # ============================================================
    # FILTER 5 — Vital sign range sliders (collapsible)
    # ============================================================
    with st.sidebar.expander("5️⃣ Vital Sign Ranges", expanded=False):

        vital_ranges = {}

        vital_meta = [
            ("HR",    "Heart Rate (bpm)",       40.0, 200.0),
            ("O2Sat", "O2 Saturation (%)",      60.0, 100.0),
            ("Temp",  "Temperature (°C)",       30.0, 42.0),
            ("SBP",   "Systolic BP (mmHg)",     50.0, 220.0),
            ("MAP",   "MAP (mmHg)",             30.0, 150.0),
            ("DBP",   "Diastolic BP (mmHg)",    20.0, 140.0),
            ("Resp",  "Respiration (/min)",     5.0,  60.0),
        ]

        for key, label, lo, hi in vital_meta:
            if key in df.columns and df[key].notna().any():
                data_lo = max(float(df[key].min(skipna=True)), float(lo))
                data_hi = min(float(df[key].max(skipna=True)), float(hi))

                # Ensure data_lo <= data_hi
                if data_lo >= data_hi:
                    data_lo, data_hi = float(lo), float(hi)

                vital_ranges[key] = st.slider(
                    label,
                    min_value=float(lo),
                    max_value=float(hi),
                    value=(float(data_lo), float(data_hi)),
                    step=1.0,
                    key=f"filter_vital_{key}"
                )

    # ============================================================
    # FILTER 6 — Missing vitals toggle
    # ============================================================
    st.sidebar.markdown("##### 6️⃣ Data Completeness")
    hide_missing = st.sidebar.checkbox(
        "Hide patients with any missing vitals",
        value=False,
        key="filter_hide_missing"
    )

    # ============================================================
    # FILTER 7 — Sort by
    # ============================================================
    st.sidebar.markdown("##### 7️⃣ Sort By")
    sort_col = st.sidebar.selectbox(
        "Sort patients by",
        options=["Max Risk (High → Low)", "6h Risk", "12h Risk",
                 "ICU Length of Stay", "Heart Rate", "O2 Saturation"],
        index=0,
        key="filter_sort",
        label_visibility="collapsed"
    )


    # ============================================================
    # Apply filters
    # ============================================================
    filtered_df = df.copy()

    # 1. Risk tier
    if risk_tier_filter:
        filtered_df = filtered_df[filtered_df["Risk_Tier"].isin(risk_tier_filter)]

    # 2. ID search
    if id_search:
        filtered_df = filtered_df[
            filtered_df["Patient_ID"].astype(str).str.contains(id_search, case=False, na=False)
        ]

    # 3. Min risk
    if min_risk > 0:
        filtered_df = filtered_df[filtered_df["_max_risk"] >= min_risk]

    # 4. ICU LOS range
    if "ICULOS" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["ICULOS"].isna()) |
            ((filtered_df["ICULOS"] >= los_range[0]) & (filtered_df["ICULOS"] <= los_range[1]))
        ]

    # 5. Vital ranges
    for key, (lo, hi) in vital_ranges.items():
        if key in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df[key].isna()) |
                ((filtered_df[key] >= lo) & (filtered_df[key] <= hi))
            ]

    # 6. Hide missing
    if hide_missing:
        vital_cols = [c for c in ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
                      if c in filtered_df.columns]
        filtered_df = filtered_df.dropna(subset=vital_cols)

    # 7. Sort
    sort_map = {
        "Max Risk (High → Low)": ("_max_risk", False),
        "6h Risk":               ("Prob_6h", False),
        "12h Risk":              ("Prob_12h", False),
        "ICU Length of Stay":    ("ICULOS", False),
        "Heart Rate":            ("HR", False),
        "O2 Saturation":         ("O2Sat", True),
    }
    sort_col_name, sort_asc = sort_map[sort_col]
    if sort_col_name in filtered_df.columns:
        filtered_df = filtered_df.sort_values(sort_col_name, ascending=sort_asc, na_position="last")


    # ============================================================
    # Sidebar filter summary
    # ============================================================
    st.sidebar.markdown("---")
    total_patients = len(df)
    shown_patients = len(filtered_df)
    pct = (shown_patients / total_patients * 100) if total_patients else 0
    st.sidebar.markdown(
        f"<div class='filter-count'>"
        f"<b>{shown_patients:,}</b> / {total_patients:,} patients matched ({pct:.1f}%)"
        f"</div>",
        unsafe_allow_html=True
    )


    # ============================================================
    # Main content
    # ============================================================
    if shown_patients == 0:
        st.warning("⚠️ No patients match the current filters. Try widening the ranges or clicking Reset.")
        st.stop()

    # -------- Patient selector (sorted ascending) --------
    st.sidebar.markdown("##### 8️⃣ Select Patient")

    patient_ids = sorted(
        filtered_df["Patient_ID"].astype(str).unique().tolist(),
        key=lambda x: float(x) if x.replace(".", "", 1).isdigit() else float("inf")
    )

    selected_id = st.sidebar.selectbox(
        "Select Patient ID",
        patient_ids,
        key="filter_patient_select",
        label_visibility="collapsed"
    )

    patient_row = filtered_df[filtered_df["Patient_ID"].astype(str) == selected_id].iloc[0]

    prob_6h  = float(patient_row["Prob_6h"])
    prob_12h = float(patient_row["Prob_12h"])


    # ============================================================
    # Patient Overview
    # ============================================================
    st.markdown("### 📋 Patient Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Patient ID", str(patient_row.get("Patient_ID", selected_id)))
    with col2:
        age_val = patient_row.get("Age", "N/A")
        if age_val != "N/A" and pd.notna(age_val):
            try:
                st.metric("Age", f"{float(age_val):.0f} yrs")
            except (TypeError, ValueError):
                st.metric("Age", "N/A")
        else:
            st.metric("Age", "N/A")
    with col3:
        los_val = patient_row.get("ICULOS", "N/A")
        if pd.notna(los_val):
            try:
                st.metric("ICU Length of Stay", f"{float(los_val):.0f} hrs")
            except (TypeError, ValueError):
                st.metric("ICU Length of Stay", "N/A")
        else:
            st.metric("ICU Length of Stay", "N/A")
    with col4:
        hour_val = patient_row.get("Hour", "N/A")
        st.metric("Observation Hour", f"{hour_val}" if hour_val != "N/A" else "N/A")

    st.markdown("---")

    # ============================================================
    # Vital Signs Panel
    # ============================================================
    st.markdown("### 💓 Current Vital Signs")

    vitals = [
        ("HR",    "Heart Rate",    "bpm"),
        ("O2Sat", "O2 Saturation", "%"),
        ("Temp",  "Temperature",   "°C"),
        ("SBP",   "Systolic BP",   "mmHg"),
        ("MAP",   "MAP",           "mmHg"),
        ("DBP",   "Diastolic BP",  "mmHg"),
        ("Resp",  "Respiration",   "/min"),
    ]

    cols = st.columns(len(vitals))
    for col, (key, label, unit) in zip(cols, vitals):
        val = patient_row.get(key, np.nan)
        display = f"{val:.1f}" if pd.notna(val) else "N/A"
        with col:
            st.markdown(
                f"""
                <div class="vital-box">
                    <div class="vital-label">{label}</div>
                    <div class="vital-value">{display}</div>
                    <div class="vital-unit">{unit}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ============================================================
    # Risk Assessment
    # ============================================================
    st.markdown("### ⚠️ Sepsis Risk Horizons")
    r1, r2 = st.columns(2)

    def get_risk_style(prob):
        if prob >= 0.30:
            return "high-risk", "🔴 HIGH RISK", "#EF4444"
        elif prob >= 0.10:
            return "mod-risk", "🟠 MODERATE RISK", "#F59E0B"
        else:
            return "low-risk", "🟢 LOW RISK", "#10B981"

    _, label_6h, color_6h   = get_risk_style(prob_6h)
    _, label_12h, color_12h = get_risk_style(prob_12h)

    with r1:
        st.subheader("6-Hour Lookahead Risk")
        st.progress(min(prob_6h, 1.0))
        st.markdown(
            f"**Probability:** `{prob_6h * 100:.1f}%` — "
            f"**Tier:** <span style='color:{color_6h};font-weight:bold;'>{label_6h}</span>",
            unsafe_allow_html=True
        )

    with r2:
        st.subheader("12-Hour Lookahead Risk")
        st.progress(min(prob_12h, 1.0))
        st.markdown(
            f"**Probability:** `{prob_12h * 100:.1f}%` — "
            f"**Tier:** <span style='color:{color_12h};font-weight:bold;'>{label_12h}</span>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ============================================================
    # SHAP Explainability
    # ============================================================
    e1, e2 = st.columns(2)

    with e1:
        st.markdown("### 🧬 Top SHAP Driving Factors (6h Horizon)")
        shap_6h = str(patient_row.get("SHAP_Factors_6h", "N/A"))
        st.info(f"**Key Clinical Features Influencing Risk:**\n\n{shap_6h}")
        st.caption("↑ = pushing toward deterioration, ↓ = protective. Explained with SHAP.")

    with e2:
        st.markdown("### 🧬 Top SHAP Driving Factors (12h Horizon)")
        shap_12h = str(patient_row.get("SHAP_Factors_12h", "N/A"))
        st.info(f"**Key Clinical Features Influencing Risk:**\n\n{shap_12h}")
        st.caption("↑ = pushing toward deterioration, ↓ = protective. Explained with SHAP.")

    st.markdown("---")

    # ============================================================
    # Clinical Recommendation
    # ============================================================
    st.markdown("### 🩺 Automated Clinical Recommendation")
    rec_text = str(patient_row.get("Clinical_Recommendation", "Vitals within safe range"))

    if prob_6h >= 0.30 or prob_12h >= 0.30:
        st.error(f"**CRITICAL ACTION REQUIRED:**\n\n{rec_text}")
    elif prob_6h >= 0.10 or prob_12h >= 0.10:
        st.warning(f"**ELEVATED MONITORING:**\n\n{rec_text}")
    else:
        st.success(f"**ROUTINE CARE:**\n\n{rec_text}")

    st.caption("Rule-based note derived from abnormal vitals. Decision-support aid, not a diagnosis.")

    st.markdown("---")

    # ============================================================
    # Cohort Explorer Table
    # ============================================================
    st.markdown("### 📊 Filtered Patient Cohort Explorer")
    st.caption(f"Showing top {min(50, shown_patients)} of {shown_patients:,} filtered patients.")

    display_cols = ["Patient_ID", "Hour", "ICULOS",
                    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
                    "Prob_6h", "Prob_12h", "Risk_Tier"]
    display_cols = [c for c in display_cols if c in filtered_df.columns]

    table_df = filtered_df[display_cols].head(50).copy()

    for c in ["Prob_6h", "Prob_12h"]:
        if c in table_df.columns:
            table_df[c] = (table_df[c] * 100).round(1).astype(str) + "%"

    st.dataframe(table_df, use_container_width=True, height=400)

    # ============================================================
    # Download filtered results
    # ============================================================
    csv_bytes = filtered_df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇️ Download {shown_patients:,} filtered patients as CSV",
        data=csv_bytes,
        file_name="filtered_sepsis_patients.csv",
        mime="text/csv"
    )

    # ============================================================
    # Footer
    # ============================================================
    st.sidebar.markdown("---")
    st.sidebar.info("Designed for ICU Clinical Support. Powered by XGBoost & SHAP.")
    st.sidebar.caption("Research prototype — not a certified medical device.")

#streamlit run streamlit_app.py