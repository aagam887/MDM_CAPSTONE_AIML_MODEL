"""
Streamlit Web Application: ICU Sepsis Early Warning & Risk Assessment Dashboard
Author: Ayush Bankar (Capstone Project)
Usage: streamlit run streamlit_app.py

Features:
  - Real-time risk assessment (6h + 12h lookahead)
  - SHAP interpretability (with graceful fallback)
  - Risk-aware clinical recommendations
  - Latest-recorded vitals with staleness indicators
  - Per-patient vital-sign trend charts
  - Risk horizon comparison chart
  - Full cohort explorer with filters
  - Fast loading via disk cache
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import altair as alt
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
    .vital-stale { font-size: 0.7rem; color: #F59E0B; margin-top: 4px; }
    .vital-fresh { font-size: 0.7rem; color: #10B981; margin-top: 4px; }

    .filter-count {
        background: #DBEAFE; color: #1E40AF; padding: 4px 8px;
        border-radius: 6px; font-size: 0.85rem; text-align: center;
    }

    .risk-header {
        display: flex; align-items: center; justify-content: space-between;
        background: #F8FAFC; padding: 14px 20px; border-radius: 10px;
        margin-bottom: 12px; border-left: 5px solid #1E3A8A;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# Title
# ============================================================
st.markdown('<div class="main-header">🏥 ICU Sepsis Early Warning Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time Bedside Risk Assessment & SHAP Interpretability System</div>', unsafe_allow_html=True)


# ============================================================
# Constants
# ============================================================
VITAL_KEYS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
CACHE_FILE = "patient_vitals_cached.csv"
DATASET_FILE = "Dataset.csv"
PREDICTIONS_FILE = "patient_predictions.csv"


# ============================================================
# Helper: clean Patient ID display (574.0 → "574")
# ============================================================
def clean_pid(x):
    try:
        if pd.isna(x):
            return "N/A"
        f = float(x)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except (TypeError, ValueError):
        return str(x)


# ============================================================
# Helper: is value missing (None / nan / '' / 'nan' / 'None')
# ============================================================
def is_missing(v):
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    s = str(v).strip().lower()
    return s in ("", "nan", "none", "n/a", "<na>")


# ============================================================
# Fast loader: last non-null vitals per patient (vectorized + disk cache)
# ============================================================
@st.cache_data(show_spinner=False)
def get_last_nonnull_vitals(dataset_path=DATASET_FILE):
    """
    Compute the last non-null value of each vital per patient.
    Vectorized for speed. Caches to disk for instant future startup.
    """

    # Disk cache lookup
    if os.path.exists(CACHE_FILE):
        try:
            cached = pd.read_csv(CACHE_FILE)
            cached["Patient_ID"] = cached["Patient_ID"].astype(str)
            return cached
        except Exception:
            pass

    if not os.path.exists(dataset_path):
        return None

    try:
        needed = ["Patient_ID", "Hour"] + VITAL_KEYS
        raw = pd.read_csv(
            dataset_path,
            usecols=lambda c: c in needed,
            low_memory=False,
        )

        if "Unnamed: 0" in raw.columns:
            raw = raw.drop(columns=["Unnamed: 0"])

        if "Patient_ID" not in raw.columns or "Hour" not in raw.columns:
            return None

        raw["Patient_ID"] = raw["Patient_ID"].astype(str)
        raw = raw.sort_values(["Patient_ID", "Hour"])

        results = {}
        for v in VITAL_KEYS:
            if v not in raw.columns:
                continue
            sub = raw.loc[raw[v].notna(), ["Patient_ID", "Hour", v]]
            last = sub.groupby("Patient_ID", sort=False).tail(1).set_index("Patient_ID")
            results[f"{v}_value"] = last[v]
            results[f"{v}_hour"]  = last["Hour"]

        last_hour = raw.groupby("Patient_ID", sort=False)["Hour"].max()

        vitals_df = pd.DataFrame(results)
        vitals_df["_last_hour"] = last_hour
        vitals_df = vitals_df.reset_index()

        # Save disk cache
        try:
            vitals_df.to_csv(CACHE_FILE, index=False)
        except Exception:
            pass

        return vitals_df

    except Exception as e:
        print(f"Warning: could not compute last-non-null vitals — {e}")
        return None


# ============================================================
# NEW: Per-patient time-series loader (for trend charts)
# ============================================================
@st.cache_data(show_spinner=False)
def get_patient_timeseries(patient_id, dataset_path=DATASET_FILE):
    """Load full ICU time-series for one patient (cached per patient)."""
    if not os.path.exists(dataset_path):
        return None
    try:
        needed = ["Patient_ID", "Hour"] + VITAL_KEYS
        raw = pd.read_csv(
            dataset_path,
            usecols=lambda c: c in needed,
            low_memory=False,
        )
        if "Patient_ID" not in raw.columns or "Hour" not in raw.columns:
            return None
        raw["Patient_ID"] = raw["Patient_ID"].astype(str)
        sub = raw[raw["Patient_ID"] == str(patient_id)].sort_values("Hour")
        return sub if len(sub) else None
    except Exception as e:
        print(f"Warning: timeseries load failed — {e}")
        return None


# ============================================================
# Main data loader
# ============================================================
@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(PREDICTIONS_FILE):
        st.error(f"❌ Prediction file '{PREDICTIONS_FILE}' not found!")
        st.info("Run the training pipeline first to generate it.")
        return None

    df = pd.read_csv(PREDICTIONS_FILE)

    # Column name normalization
    rename_map = {
        "Risk_6h": "Prob_6h",
        "Risk_12h": "Prob_12h",
        "Top_Factors_6h": "SHAP_Factors_6h",
        "Top_Factors_12h": "SHAP_Factors_12h",
        "Recommendation": "Clinical_Recommendation",
    }
    df = df.rename(columns=rename_map)

    # Fallbacks for missing columns
    if "Age" not in df.columns:
        df["Age"] = np.nan
    if "Gender" not in df.columns:
        df["Gender"] = 1

    # Normalize probabilities (ensure 0–1 range)
    for c in ["Prob_6h", "Prob_12h"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        if df[c].max() > 1:
            df[c] /= 100

    # Consistent ID type
    df["Patient_ID"] = df["Patient_ID"].astype(str)

    # Merge in last non-null vitals
    last_vitals = get_last_nonnull_vitals(DATASET_FILE)

    if last_vitals is not None:
        last_vitals["Patient_ID"] = last_vitals["Patient_ID"].astype(str)

        # Drop columns that will be replaced to avoid suffixing
        drop_cols = [c for c in last_vitals.columns if c != "Patient_ID" and c in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")

        df = df.merge(last_vitals, on="Patient_ID", how="left")

        # Map to display columns
        for v in VITAL_KEYS:
            val_col = f"{v}_value"
            if val_col in df.columns:
                df[v] = df[val_col]

        # Compute freshness (hours since vital was charted)
        if "_last_hour" in df.columns:
            for v in VITAL_KEYS:
                hour_col = f"{v}_hour"
                if hour_col in df.columns:
                    df[f"{v}_age"] = (
                        (df["_last_hour"] - df[hour_col])
                        .fillna(0)
                        .clip(lower=0)
                        .astype(int)
                    )

    # Risk tier
    df["_max_risk"] = df[["Prob_6h", "Prob_12h"]].max(axis=1)
    df["Risk_Tier"] = df["_max_risk"].apply(
        lambda p: "High Risk (>=30%)" if p >= 0.30
        else "Moderate Risk (10-30%)" if p >= 0.10
        else "Low Risk (<10%)"
    )

    # Clean display ID
    df["Patient_ID_Display"] = df["Patient_ID"].apply(clean_pid)

    return df


# Load everything
with st.spinner("Loading dashboard data… (first run may take 30–60s)"):
    df = load_data()


# ============================================================
# Sidebar filters + main content
# ============================================================
if df is not None:

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------
    st.sidebar.header("🔍 Filter & Search")

    # Reset button
    if st.sidebar.button("🔄 Reset All Filters", width='stretch'):
        for key in list(st.session_state.keys()):
            if key.startswith("filter_"):
                del st.session_state[key]
        st.rerun()

    # Filter 1 — Risk tier
    st.sidebar.markdown("##### 1️⃣ Risk Tier")
    risk_tier_filter = st.sidebar.multiselect(
        "Risk Level",
        options=["High Risk (>=30%)", "Moderate Risk (10-30%)", "Low Risk (<10%)"],
        default=["High Risk (>=30%)", "Moderate Risk (10-30%)", "Low Risk (<10%)"],
        key="filter_risk_tier",
        label_visibility="collapsed"
    )

    # Filter 2 — Patient ID search
    st.sidebar.markdown("##### 2️⃣ Patient ID Search")
    id_search = st.sidebar.text_input(
        "Type part of a Patient ID",
        key="filter_id_search",
        placeholder="e.g. 21 or 6853",
        label_visibility="collapsed"
    )

    # Filter 3 — Min risk threshold
    st.sidebar.markdown("##### 3️⃣ Minimum Risk Threshold")
    min_risk = st.sidebar.slider(
        "Show patients with max risk ≥",
        min_value=0.0, max_value=1.0, value=0.0, step=0.05,
        format="%.0f%%",
        key="filter_min_risk",
        label_visibility="collapsed"
    )

    # Filter 4 — ICU LOS range
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

    # Filter 5 — Vital ranges
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
                if data_lo >= data_hi:
                    data_lo, data_hi = float(lo), float(hi)
                vital_ranges[key] = st.slider(
                    label,
                    min_value=float(lo), max_value=float(hi),
                    value=(float(data_lo), float(data_hi)),
                    step=1.0,
                    key=f"filter_vital_{key}"
                )

    # Filter 6 — Data completeness
    st.sidebar.markdown("##### 6️⃣ Data Completeness")
    hide_missing = st.sidebar.checkbox(
        "Hide patients with missing vitals",
        value=False,
        key="filter_hide_missing"
    )

    # Filter 7 — Sort by
    st.sidebar.markdown("##### 7️⃣ Sort By")
    sort_col = st.sidebar.selectbox(
        "Sort patients by",
        options=["Max Risk (High → Low)", "6h Risk", "12h Risk",
                 "ICU Length of Stay", "Heart Rate", "O2 Saturation"],
        index=0,
        key="filter_sort",
        label_visibility="collapsed"
    )

    # --------------------------------------------------------
    # Apply filters
    # --------------------------------------------------------
    filtered_df = df.copy()

    if risk_tier_filter:
        filtered_df = filtered_df[filtered_df["Risk_Tier"].isin(risk_tier_filter)]

    if id_search:
        filtered_df = filtered_df[
            filtered_df["Patient_ID"].astype(str).str.contains(id_search, case=False, na=False)
        ]

    if min_risk > 0:
        filtered_df = filtered_df[filtered_df["_max_risk"] >= min_risk]

    if "ICULOS" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["ICULOS"].isna()) |
            ((filtered_df["ICULOS"] >= los_range[0]) & (filtered_df["ICULOS"] <= los_range[1]))
        ]

    for key, (lo, hi) in vital_ranges.items():
        if key in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df[key].isna()) |
                ((filtered_df[key] >= lo) & (filtered_df[key] <= hi))
            ]

    if hide_missing:
        vital_cols = [c for c in VITAL_KEYS if c in filtered_df.columns]
        filtered_df = filtered_df.dropna(subset=vital_cols)

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

    # --------------------------------------------------------
    # Sidebar summary
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Empty state
    # --------------------------------------------------------
    if shown_patients == 0:
        st.warning("⚠️ No patients match the current filters. Try widening the ranges or clicking Reset.")
        st.stop()

    # --------------------------------------------------------
    # Patient selector
    # --------------------------------------------------------
    st.sidebar.markdown("##### 8️⃣ Select Patient")
    patient_ids = sorted(
        filtered_df["Patient_ID"].unique().tolist(),
        key=lambda x: float(x) if x.replace(".", "", 1).isdigit() else float("inf")
    )

    id_display_map = {pid: clean_pid(pid) for pid in patient_ids}

    selected_id = st.sidebar.selectbox(
        "Select Patient ID",
        options=patient_ids,
        format_func=lambda x: id_display_map.get(x, x),
        key="filter_patient_select",
        label_visibility="collapsed"
    )

    patient_row = filtered_df[filtered_df["Patient_ID"] == selected_id].iloc[0]
    prob_6h  = float(patient_row["Prob_6h"])
    prob_12h = float(patient_row["Prob_12h"])

    # ============================================================
    # MAIN CONTENT
    # ============================================================

    # --- Patient Overview ---
    st.markdown("### 📋 Patient Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Patient ID", clean_pid(patient_row.get("Patient_ID", selected_id)))

    with col2:
        age_val = patient_row.get("Age", np.nan)
        if pd.notna(age_val):
            try:
                st.metric("Age", f"{float(age_val):.0f} yrs")
            except (TypeError, ValueError):
                st.metric("Age", "N/A")
        else:
            st.metric("Age", "N/A")

    with col3:
        los_val = patient_row.get("ICULOS", np.nan)
        if pd.notna(los_val):
            try:
                st.metric("ICU Length of Stay", f"{float(los_val):.0f} hrs")
            except (TypeError, ValueError):
                st.metric("ICU Length of Stay", "N/A")
        else:
            st.metric("ICU Length of Stay", "N/A")

    with col4:
        hour_val = patient_row.get("Hour", np.nan)
        st.metric("Observation Hour", f"{hour_val:.0f}" if pd.notna(hour_val) else "N/A")

    st.markdown("---")

    # --- Vital Signs Panel ---
    st.markdown("### 💓 Latest Recorded Vital Signs")
    st.caption(
        "Values shown are the **most recent non-null measurements**. "
        "Staleness indicator shows how many hours ago the value was charted."
    )

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

        age_key = f"{key}_age"
        if age_key in patient_row.index and pd.notna(patient_row[age_key]):
            age = int(patient_row[age_key])
            if age <= 2:
                staleness = f'<div class="vital-fresh">✓ measured {age}h ago</div>'
            elif age <= 8:
                staleness = f'<div class="vital-stale">⚠ {age}h ago</div>'
            else:
                staleness = f'<div class="vital-stale">⚠ {age}h ago (stale)</div>'
        else:
            staleness = ''

        with col:
            st.markdown(
                f"""
                <div class="vital-box">
                    <div class="vital-label">{label}</div>
                    <div class="vital-value">{display}</div>
                    <div class="vital-unit">{unit}</div>
                    {staleness}
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ============================================================
    # NEW: Vital Signs Trend Charts
    # ============================================================
    st.markdown("### 📈 Vital Sign Trends Over ICU Stay")
    st.caption(
        "Each tab shows the patient's charted values across their ICU stay. "
        "Gaps reflect missing charting; the y-axis auto-scales per vital."
    )

    ts_df = get_patient_timeseries(selected_id)

    if ts_df is not None and len(ts_df) > 0:
        available_vitals = [
            (k, label, unit)
            for (k, label, unit) in vitals
            if k in ts_df.columns and ts_df[k].notna().any()
        ]

        if available_vitals:
            tabs = st.tabs([f"{label}" for _, label, _ in available_vitals])
            for tab, (key, label, unit) in zip(tabs, available_vitals):
                with tab:
                    series = ts_df[["Hour", key]].dropna().set_index("Hour")
                    if len(series) >= 2:
                        st.line_chart(series, height=280, use_container_width=True)
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Observations", f"{len(series)}")
                        m2.metric("Min", f"{series[key].min():.1f} {unit}")
                        m3.metric("Max", f"{series[key].max():.1f} {unit}")
                    elif len(series) == 1:
                        st.info("Only 1 observation recorded — can't plot a trend.")
                        st.metric(label, f"{series[key].iloc[0]:.1f} {unit}")
                    else:
                        st.info(f"No charted {label} values for this patient.")
        else:
            st.info("No vital sign time-series available for this patient.")
    else:
        st.info(
            f"Full ICU time-series not available for Patient {clean_pid(selected_id)}. "
            f"Ensure `{DATASET_FILE}` is present in the working directory."
        )

    st.markdown("---")

    # --- Risk Assessment ---
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

        # --- Risk horizon comparison chart (explicit order + tier colors) ---
    st.markdown("#### 🔎 Risk Horizon Comparison")

    def _tier_color(p):
        if p >= 0.30: return "#EF4444"   # red
        if p >= 0.10: return "#F59E0B"   # amber
        return "#10B981"                 # green

    risk_chart_df = pd.DataFrame({
        "Horizon":   ["6-Hour Risk", "12-Hour Risk"],
        "Risk (%)":  [round(prob_6h * 100, 1), round(prob_12h * 100, 1)],
        "Color":     [_tier_color(prob_6h), _tier_color(prob_12h)],
    })

    _order = ["6-Hour Risk", "12-Hour Risk"]

    _bars = (
        alt.Chart(risk_chart_df)
        .mark_bar(size=90, cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X(
                "Horizon:N",
                sort=_order,
                axis=alt.Axis(labelAngle=0, title=None, labelFontSize=13, labelPadding=8),
            ),
            y=alt.Y(
                "Risk (%):Q",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(title="Predicted Risk (%)", grid=True),
            ),
            color=alt.Color("Color:N", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("Horizon:N", title="Horizon"),
                alt.Tooltip("Risk (%):Q", title="Risk", format=".1f"),
            ],
        )
    )

    _labels = (
        alt.Chart(risk_chart_df)
        .mark_text(dy=-10, fontSize=14, fontWeight="bold", color="#1E293B")
        .encode(
            x=alt.X("Horizon:N", sort=_order),
            y=alt.Y("Risk (%):Q"),
            text=alt.Text("Risk (%):Q", format=".1f"),
        )
    )

    st.altair_chart((_bars + _labels).properties(height=280), use_container_width=True)

    st.markdown("---")

    # --- SHAP Explainability ---
    e1, e2 = st.columns(2)

    shap_6h_raw  = patient_row.get("SHAP_Factors_6h", None)
    shap_12h_raw = patient_row.get("SHAP_Factors_12h", None)

    shap_6h_ok  = not is_missing(shap_6h_raw)
    shap_12h_ok = not is_missing(shap_12h_raw)

    with e1:
        st.markdown("### 🧬 Top SHAP Driving Factors (6h Horizon)")
        if shap_6h_ok:
            st.info(f"**Key Clinical Features Influencing Risk:**\n\n{shap_6h_raw}")
            st.caption("↑ = pushing toward deterioration, ↓ = protective. Explained with SHAP.")
        else:
            st.warning(
                "**SHAP attribution not available for this patient.**\n\n"
                "SHAP values were pre-computed on a subset of the cohort. "
                "This patient was outside that subset. The risk score above "
                "is still valid — it comes from the trained XGBoost model."
            )

    with e2:
        st.markdown("### 🧬 Top SHAP Driving Factors (12h Horizon)")
        if shap_12h_ok:
            st.info(f"**Key Clinical Features Influencing Risk:**\n\n{shap_12h_raw}")
            st.caption("↑ = pushing toward deterioration, ↓ = protective. Explained with SHAP.")
        else:
            st.warning(
                "**SHAP attribution not available for this patient.**\n\n"
                "SHAP values were pre-computed on a subset of the cohort. "
                "This patient was outside that subset. The risk score above "
                "is still valid — it comes from the trained XGBoost model."
            )

    st.markdown("---")

    # --- Risk-Aware Clinical Recommendation ---
    st.markdown("### 🩺 Automated Clinical Recommendation")

    vitals_rec = str(patient_row.get("Clinical_Recommendation", "")).strip()
    vitals_rec_missing = is_missing(vitals_rec) or vitals_rec == "Vitals within safe range"

    if prob_6h >= 0.30 or prob_12h >= 0.30:
        header = "🔴 CRITICAL ACTION REQUIRED"
        action = (
            "**Immediate clinical review** — assess for infection source, "
            "order blood cultures + lactate, consider sepsis bundle initiation. "
            "Re-check vitals every 30 minutes."
        )
        if not vitals_rec_missing:
            action += f"\n\n**Detected abnormalities:** {vitals_rec}"
        st.error(f"{header}\n\n{action}")

    elif prob_6h >= 0.10 or prob_12h >= 0.10:
        header = "🟠 ELEVATED MONITORING"
        action = (
            "**Increased surveillance** — reassess vitals and labs within 1 hour, "
            "review recent trends for deterioration signals."
        )
        if not vitals_rec_missing:
            action += f"\n\n**Detected abnormalities:** {vitals_rec}"
        st.warning(f"{header}\n\n{action}")

    else:
        st.success(f"🟢 ROUTINE CARE\n\n{vitals_rec if not vitals_rec_missing else 'Vitals within safe range'}")

    st.caption("Risk-tier-aware recommendation. Decision-support aid, not a diagnosis.")

    st.markdown("---")

    # --- Cohort Explorer ---
    st.markdown("### 📊 Filtered Patient Cohort Explorer")
    st.caption(f"Showing top {min(50, shown_patients)} of {shown_patients:,} filtered patients.")

    display_cols = ["Patient_ID_Display", "Hour", "ICULOS",
                    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
                    "Prob_6h", "Prob_12h", "Risk_Tier"]
    display_cols = [c for c in display_cols if c in filtered_df.columns]

    table_df = filtered_df[display_cols].head(50).copy()
    table_df = table_df.rename(columns={"Patient_ID_Display": "Patient_ID"})

    for c in ["Prob_6h", "Prob_12h"]:
        if c in table_df.columns:
            table_df[c] = (table_df[c] * 100).round(1).astype(str) + "%"

    st.dataframe(table_df, width='stretch', height=400)

    # --- Download ---
    csv_bytes = filtered_df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇️ Download {shown_patients:,} filtered patients as CSV",
        data=csv_bytes,
        file_name="filtered_sepsis_patients.csv",
        mime="text/csv"
    )

    # --- Footer ---
    st.sidebar.markdown("---")
    st.sidebar.info("Designed for ICU Clinical Support. Powered by XGBoost & SHAP.")
    st.sidebar.caption("Research prototype — not a certified medical device.")

#streamlit run streamlit_app.py