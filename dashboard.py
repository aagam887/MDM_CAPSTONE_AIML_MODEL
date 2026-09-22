# ============================================================
# FINAL SEPSIS EARLY WARNING SYSTEM - HTML DASHBOARD
# ============================================================

import json
import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. LOAD PATIENT PREDICTIONS
# ------------------------------------------------------------

CSV_FILE = "patient_predictions.csv"

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(
        f"{CSV_FILE} not found.\n"
        f"Current folder: {os.getcwd()}\n"
        "Make sure patient_predictions.csv is in the same folder."
    )

df = pd.read_csv(CSV_FILE)

print("CSV loaded successfully")
print("Shape:", df.shape)
print("Columns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 2. AUTOMATICALLY FIND IMPORTANT COLUMNS
# ------------------------------------------------------------


def find_column(possible_names):
    lower_map = {str(c).lower(): c for c in df.columns}

    for name in possible_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    for c in df.columns:
        cl = str(c).lower()
        for name in possible_names:
            if name.lower() in cl:
                return c

    return None


patient_col = find_column(["Patient_ID", "patient_id", "PatientID", "patient"])

risk6_col = find_column(
    [
        "Risk_6h",
        "Risk6h",
        "Probability_6h",
        "Prob_6h",
        "Prediction_6h",
        "Target_6h_Probability",
        "6h_probability",
    ]
)

risk12_col = find_column(
    [
        "Risk_12h",
        "Risk12h",
        "Probability_12h",
        "Prob_12h",
        "Prediction_12h",
        "Target_12h_Probability",
        "12h_probability",
    ]
)

factors6_col = find_column(["Top_Factors_6h", "TopFactors6h", "Factors_6h"])
factors12_col = find_column(["Top_Factors_12h", "TopFactors12h", "Factors_12h"])
recommendation_col = find_column(["Recommendation", "Recommended_Attention", "Clinical_Note"])


# If prediction columns are not found, search by keywords
if risk6_col is None:
    for c in df.columns:
        if "6" in str(c).lower() and (
            "prob" in str(c).lower()
            or "risk" in str(c).lower()
            or "pred" in str(c).lower()
        ):
            risk6_col = c
            break

if risk12_col is None:
    for c in df.columns:
        if "12" in str(c).lower() and (
            "prob" in str(c).lower()
            or "risk" in str(c).lower()
            or "pred" in str(c).lower()
        ):
            risk12_col = c
            break


if patient_col is None:
    raise ValueError(
        "Patient ID column could not be found.\n"
        f"Available columns: {df.columns.tolist()}"
    )

if risk6_col is None or risk12_col is None:
    raise ValueError(
        "6-hour or 12-hour prediction column could not be found.\n"
        f"Available columns: {df.columns.tolist()}"
    )


print("\nDetected columns:")
print("Patient ID :", patient_col)
print("6h Risk    :", risk6_col)
print("12h Risk   :", risk12_col)
print("6h Factors     :", factors6_col or "NOT FOUND (SHAP card hidden)")
print("12h Factors    :", factors12_col or "NOT FOUND (SHAP card hidden)")
print("Recommendation :", recommendation_col or "NOT FOUND (Rec card hidden)")


# ------------------------------------------------------------
# 3. KEEP ONE LATEST RECORD PER PATIENT
# ------------------------------------------------------------

df = df.copy()

if "Hour" in df.columns:
    df = df.sort_values([patient_col, "Hour"])

patient_df = df.groupby(patient_col, as_index=False).tail(1).copy()

print("\nPatients available:", patient_df[patient_col].nunique())


# ------------------------------------------------------------
# 4. CLEAN RISK VALUES
# ------------------------------------------------------------

patient_df[risk6_col] = pd.to_numeric(
    patient_df[risk6_col], errors="coerce"
).fillna(0)

patient_df[risk12_col] = pd.to_numeric(
    patient_df[risk12_col], errors="coerce"
).fillna(0)

# Convert percentage values such as 35 -> 0.35
if patient_df[risk6_col].max() > 1:
    patient_df[risk6_col] /= 100

if patient_df[risk12_col].max() > 1:
    patient_df[risk12_col] /= 100


# ------------------------------------------------------------
# 5. RISK CLASSIFICATION
# ------------------------------------------------------------


def risk_level(p):
    if p >= 0.30:
        return "HIGH"
    elif p >= 0.10:
        return "MODERATE"
    else:
        return "LOW"


patient_df["Risk6_Level"] = patient_df[risk6_col].apply(risk_level)
patient_df["Risk12_Level"] = patient_df[risk12_col].apply(risk_level)


# ------------------------------------------------------------
# 6. CONVERT DATA TO JSON
# ------------------------------------------------------------

records = patient_df.to_dict(orient="records")


def clean_value(v):
    if pd.isna(v):
        return None
    if isinstance(
        v,
        (np.integer,),
    ):
        return int(v)
    if isinstance(
        v,
        (np.floating,),
    ):
        return float(v)
    return v


clean_records = []

for row in records:
    clean_row = {}
    for k, v in row.items():
        clean_row[str(k)] = clean_value(v)
    clean_records.append(clean_row)

json_data = json.dumps(clean_records)


# ------------------------------------------------------------
# 7. HTML DASHBOARD
# ------------------------------------------------------------


html = r"""
<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sepsis Early Warning System</title>

<style>

* { box-sizing: border-box; }

body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    background: #f4f7fb;
    color: #1f2937;
}

.header {
    background: linear-gradient(135deg, #111827, #1f3c88);
    color: white;
    padding: 20px 40px;
}
.header h1 { margin: 0; font-size: 24px; }
.header p  { margin: 6px 0 0; opacity: 0.85; font-size: 14px; }

.container {
    max-width: 1400px;
    margin: auto;
    padding: 20px;
}

/* ---------- STICKY TOOLBAR ---------- */
.toolbar {
    position: sticky;
    top: 0;
    z-index: 100;
    background: white;
    padding: 16px 20px;
    border-radius: 14px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
    display: grid;
    grid-template-columns: 1fr auto auto;
    gap: 15px;
    align-items: center;
}
@media(max-width: 900px) { .toolbar { grid-template-columns: 1fr; } }

.search-wrap { position: relative; }
.search-input {
    width: 100%;
    padding: 12px 40px 12px 14px;
    border: 2px solid #d1d5db;
    border-radius: 10px;
    font-size: 15px;
    background: #fff;
}
.search-input:focus {
    outline: none;
    border-color: #1f3c88;
    box-shadow: 0 0 0 3px rgba(31,60,136,0.1);
}

.filters { display: flex; gap: 8px; }
.filter-btn {
    padding: 10px 16px;
    border: 2px solid #e5e7eb;
    background: white;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    color: #4b5563;
    white-space: nowrap;
}
.filter-btn.active { border-color: #1f3c88; background: #1f3c88; color: white; }
.filter-btn .count {
    display: inline-block;
    margin-left: 6px;
    padding: 1px 7px;
    border-radius: 10px;
    background: rgba(0,0,0,0.08);
    font-size: 11px;
}
.filter-btn.active .count { background: rgba(255,255,255,0.25); }

.toggle-btn {
    padding: 10px 16px;
    border: 2px solid #e5e7eb;
    background: white;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    color: #4b5563;
    white-space: nowrap;
}

/* ---------- TILE GRID ---------- */
.tiles-section {
    background: white;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 20px;
    box-shadow: 0 3px 15px rgba(0,0,0,0.06);
}

.tiles-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
    font-size: 13px;
    color: #6b7280;
}

.tiles-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 8px;
    max-height: 380px;
    overflow-y: auto;
    padding-right: 5px;
}

.tile {
    background: #fafbfc;
    padding: 10px 12px;
    border-radius: 8px;
    cursor: pointer;
    border-left: 5px solid #9ca3af;
    transition: transform 0.12s, background 0.12s, box-shadow 0.12s;
}
.tile:hover {
    transform: translateY(-1px);
    background: white;
    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
}
.tile.selectedTile {
    outline: 2px solid #1f3c88;
    outline-offset: 1px;
    background: white;
}
.tile.risk-low      { border-left-color: #16a34a; }
.tile.risk-moderate { border-left-color: #f59e0b; }
.tile.risk-high     { border-left-color: #dc2626; }

.tile-id { font-size: 13px; font-weight: 700; color: #1f2937; margin-bottom: 3px; }
.tile-risk { font-size: 10px; color: #6b7280; line-height: 1.3; }

.pagination {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 8px;
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px solid #f3f4f6;
}
.page-btn {
    padding: 6px 12px;
    border: 1px solid #e5e7eb;
    background: white;
    border-radius: 6px;
    font-size: 12px;
    cursor: pointer;
    color: #4b5563;
}
.page-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.page-info { font-size: 12px; color: #6b7280; padding: 0 10px; }

/* ---------- DETAIL VIEW ---------- */
.patient-header {
    background: white;
    padding: 18px 22px;
    border-radius: 14px;
    margin-bottom: 16px;
    box-shadow: 0 3px 15px rgba(0,0,0,0.06);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.patient-header h2 { margin: 0; font-size: 20px; }
.status { margin-top: 4px; font-size: 13px; color: #6b7280; }

.back-btn {
    padding: 8px 14px;
    border: 2px solid #e5e7eb;
    background: white;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    color: #4b5563;
    cursor: pointer;
}

.vitals {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
}
.vital {
    background: white;
    padding: 14px;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.vital-title { font-size: 12px; color: #6b7280; }
.vital-value { font-size: 20px; font-weight: 700; margin-top: 4px; }

.risk-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
}
.risk-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 3px 15px rgba(0,0,0,0.06);
}
.risk-card h3 { margin: 0 0 10px; font-size: 14px; color: #6b7280; font-weight: 600; }
.risk-number { font-size: 38px; font-weight: 800; margin: 6px 0 10px; }
.risk-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 16px;
    font-weight: 700;
    font-size: 12px;
}
.low       { background: #dcfce7; color: #166534; }
.moderate  { background: #fef3c7; color: #92400e; }
.high      { background: #fee2e2; color: #991b1b; }

.warning {
    margin-top: 10px;
    padding: 12px;
    border-radius: 8px;
    background: #f8fafc;
    line-height: 1.5;
    font-size: 13px;
}

.info {
    margin-top: 16px;
    background: white;
    padding: 18px 22px;
    border-radius: 14px;
    box-shadow: 0 3px 15px rgba(0,0,0,0.06);
}
.info h3 { margin-top: 0; font-size: 15px; }

/* SHAP card */
.shap-card {
    margin-top: 16px;
    background: white;
    padding: 18px 22px;
    border-radius: 14px;
    box-shadow: 0 3px 15px rgba(0,0,0,0.06);
    border-left: 5px solid #6b7280;
}
.shap-card h3 { margin-top: 0; margin-bottom: 10px; font-size: 15px; }
.shap-factors {
    font-size: 14px;
    line-height: 1.7;
    color: #1f2937;
    font-family: 'Consolas', 'Courier New', monospace;
    background: #f9fafb;
    padding: 10px 14px;
    border-radius: 8px;
}
.shap-note {
    font-size: 11px;
    color: #6b7280;
    margin-top: 8px;
    font-style: italic;
}

/* Recommendation card */
.rec-card {
    margin-top: 16px;
    background: #f0f9ff;
    padding: 18px 22px;
    border-radius: 14px;
    border-left: 5px solid #1f3c88;
    box-shadow: 0 3px 15px rgba(0,0,0,0.06);
}
.rec-card h3 { margin-top: 0; color: #1f3c88; font-size: 15px; }
.rec-card p  { font-size: 14px; line-height: 1.6; margin: 0; }

.footer { text-align: center; padding: 25px; color: #6b7280; font-size: 12px; }

@media(max-width: 800px) {
    .risk-grid { grid-template-columns: 1fr; }
    .header { padding: 16px 20px; }
    .container { padding: 12px; }
    .tiles-grid { grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); }
}
</style>
</head>
<body>

<div class="header">
    <h1>🩺 Sepsis Early Warning System</h1>
    <p>Patient deterioration prediction for the next 6–12 hours with explainable AI</p>
</div>

<div class="container">

    <div class="toolbar">
        <div class="search-wrap">
            <input type="text" id="searchInput" class="search-input"
                   placeholder="🔎  Search patient ID or type to filter…"
                   autocomplete="off" oninput="onSearchChange()">
        </div>
        <div class="filters">
            <button class="filter-btn active" data-filter="all" onclick="setFilter('all')">
                All <span class="count" id="count-all">0</span>
            </button>
            <button class="filter-btn" data-filter="high" onclick="setFilter('high')">
                High <span class="count" id="count-high">0</span>
            </button>
            <button class="filter-btn" data-filter="moderate" onclick="setFilter('moderate')">
                Mod <span class="count" id="count-mod">0</span>
            </button>
            <button class="filter-btn" data-filter="low" onclick="setFilter('low')">
                Low <span class="count" id="count-low">0</span>
            </button>
        </div>
        <button class="toggle-btn" onclick="toggleTiles()" id="toggleBtn">▲ Hide List</button>
    </div>

    <div class="tiles-section" id="tilesSection">
        <div class="tiles-header">
            <span>
                <b id="visibleCount">0</b> patients shown
                <span id="searchInfo" style="color:#1f3c88;"></span>
            </span>
        </div>
        <div id="tiles" class="tiles-grid"></div>
        <div class="pagination" id="pagination">
            <button class="page-btn" id="prevBtn" onclick="prevPage()">← Prev</button>
            <span class="page-info" id="pageInfo">Page 1 of 1</span>
            <button class="page-btn" id="nextBtn" onclick="nextPage()">Next →</button>
        </div>
    </div>

    <div id="dashboard" class="detail-section"></div>

</div>

<div class="footer">
    AI-based research prototype • Predictions are decision-support estimates and not a clinical diagnosis.
</div>

<script>

const data = __DATA__;

const patientColumn   = "__PATIENT__";
const risk6Column     = "__RISK6__";
const risk12Column    = "__RISK12__";
const factors6Column  = "__FACTORS6__";
const factors12Column = "__FACTORS12__";
const recColumn       = "__RECOMMENDATION__";

const PAGE_SIZE = 24;

let currentFilter = "all";
let currentPage   = 0;
let searchQuery   = "";
let filteredData  = data.slice();


function safe(value) {
    if (value === null || value === undefined || value === "") return "N/A";
    return value;
}

function percentage(value) {
    value = Number(value);
    if (isNaN(value)) return "N/A";
    return (value * 100).toFixed(1) + "%";
}

function level(value) {
    value = Number(value);
    if (value >= 0.30) return "HIGH";
    if (value >= 0.10) return "MODERATE";
    return "LOW";
}

function levelClass(value) { return level(value).toLowerCase(); }

function maxRisk(p) {
    return Math.max(Number(p[risk6Column]) || 0, Number(p[risk12Column]) || 0);
}


function onSearchChange() {
    searchQuery = document.getElementById("searchInput").value.trim().toLowerCase();
    currentPage = 0;
    applyFiltersAndRender();
}

function setFilter(f) {
    currentFilter = f;
    currentPage = 0;
    document.querySelectorAll(".filter-btn").forEach(b => {
        b.classList.toggle("active", b.dataset.filter === f);
    });
    applyFiltersAndRender();
}

function applyFiltersAndRender() {

    filteredData = data.filter(p => {
        const lvl = level(maxRisk(p));
        if (currentFilter === "all")      return true;
        if (currentFilter === "high")     return lvl === "HIGH";
        if (currentFilter === "moderate") return lvl === "MODERATE";
        if (currentFilter === "low")      return lvl === "LOW";
        return true;
    });

    if (searchQuery) {
        filteredData = filteredData.filter(p =>
            String(p[patientColumn]).toLowerCase().includes(searchQuery)
        );
    }

    document.getElementById("count-all").textContent  = data.length;
    document.getElementById("count-high").textContent = data.filter(p => level(maxRisk(p)) === "HIGH").length;
    document.getElementById("count-mod").textContent  = data.filter(p => level(maxRisk(p)) === "MODERATE").length;
    document.getElementById("count-low").textContent  = data.filter(p => level(maxRisk(p)) === "LOW").length;

    document.getElementById("visibleCount").textContent = filteredData.length;
    document.getElementById("searchInfo").textContent =
        searchQuery ? ` (matching "${searchQuery}")` : "";

    renderTiles();
}

function renderTiles() {

    const tilesContainer = document.getElementById("tiles");
    tilesContainer.innerHTML = "";

    const totalPages = Math.max(1, Math.ceil(filteredData.length / PAGE_SIZE));
    if (currentPage >= totalPages) currentPage = totalPages - 1;
    if (currentPage < 0) currentPage = 0;

    const start = currentPage * PAGE_SIZE;
    const end   = Math.min(start + PAGE_SIZE, filteredData.length);
    const pageItems = filteredData.slice(start, end);

    if (pageItems.length === 0) {
        tilesContainer.innerHTML = `
            <div style="grid-column:1/-1;text-align:center;padding:30px;color:#9ca3af;">
                No patients match the current filter / search.
            </div>
        `;
    } else {
        pageItems.forEach(patient => {
            const risk6 = Number(patient[risk6Column]);
            const risk12 = Number(patient[risk12Column]);
            const mr = maxRisk(patient);
            const lvl = level(mr).toLowerCase();

            const tile = document.createElement("div");
            tile.className = `tile risk-${lvl}`;
            tile.dataset.id = patient[patientColumn];

            tile.innerHTML = `
                <div class="tile-id">${safe(patient[patientColumn])}</div>
                <div class="tile-risk">6h: ${percentage(risk6)}</div>
                <div class="tile-risk">12h: ${percentage(risk12)}</div>
            `;

            tile.addEventListener("click", () => selectPatient(patient[patientColumn]));
            tilesContainer.appendChild(tile);
        });
    }

    document.getElementById("pageInfo").textContent =
        `Page ${currentPage + 1} of ${totalPages}`;
    document.getElementById("prevBtn").disabled = currentPage === 0;
    document.getElementById("nextBtn").disabled = currentPage >= totalPages - 1;

    const selectedId = document.getElementById("searchInput").dataset.selected;
    if (selectedId) {
        const active = document.querySelector(`.tile[data-id="${selectedId}"]`);
        if (active) active.classList.add("selectedTile");
    }
}

function prevPage() {
    if (currentPage > 0) { currentPage--; renderTiles(); }
}

function nextPage() {
    const totalPages = Math.ceil(filteredData.length / PAGE_SIZE);
    if (currentPage < totalPages - 1) { currentPage++; renderTiles(); }
}

function toggleTiles() {
    const section = document.getElementById("tilesSection");
    const btn = document.getElementById("toggleBtn");
    if (section.style.display === "none") {
        section.style.display = "";
        btn.textContent = "▲ Hide List";
    } else {
        section.style.display = "none";
        btn.textContent = "▼ Show List";
    }
}

function selectPatient(id) {

    document.getElementById("searchInput").dataset.selected = id;

    document.querySelectorAll(".tile").forEach(t => t.classList.remove("selectedTile"));
    const activeTile = document.querySelector(`.tile[data-id="${id}"]`);
    if (activeTile) {
        activeTile.classList.add("selectedTile");
        activeTile.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    const patient = data.find(x => String(x[patientColumn]) === String(id));
    if (!patient) return;

    const risk6  = Number(patient[risk6Column]);
    const risk12 = Number(patient[risk12Column]);

    let warning = "";
    if (risk6 >= 0.30 || risk12 >= 0.30) {
        warning = `<div class="warning high"><b>HIGH RISK:</b> The model indicates an increased probability of deterioration within the prediction window. Clinical review is recommended.</div>`;
    } else if (risk6 >= 0.10 || risk12 >= 0.10) {
        warning = `<div class="warning moderate"><b>MODERATE RISK:</b> The model detects potential deterioration. Continue close monitoring.</div>`;
    } else {
        warning = `<div class="warning low"><b>LOW RISK:</b> No high deterioration probability detected.</div>`;
    }

    let factors6HTML  = "";
    let factors12HTML = "";
    let recHTML       = "";

    if (factors6Column !== "NONE" && patient[factors6Column]) {
        factors6HTML = `
            <div class="shap-card" style="border-left-color:#ef4444;">
                <h3>🔍 Top Contributing Factors — 6h Risk</h3>
                <div class="shap-factors">${safe(patient[factors6Column])}</div>
                <div class="shap-note">Explained with SHAP. ↑ = pushing toward deterioration, ↓ = protective.</div>
            </div>`;
    }
    if (factors12Column !== "NONE" && patient[factors12Column]) {
        factors12HTML = `
            <div class="shap-card" style="border-left-color:#f59e0b;">
                <h3>🔍 Top Contributing Factors — 12h Risk</h3>
                <div class="shap-factors">${safe(patient[factors12Column])}</div>
                <div class="shap-note">Explained with SHAP. ↑ = pushing toward deterioration, ↓ = protective.</div>
            </div>`;
    }
    if (recColumn !== "NONE" && patient[recColumn]) {
        recHTML = `
            <div class="rec-card">
                <h3>🩺 Recommended Attention</h3>
                <p>${safe(patient[recColumn])}</p>
                <div class="shap-note">Rule-based note from abnormal vitals. Decision-support aid, not a diagnosis.</div>
            </div>`;
    }

    document.getElementById("dashboard").innerHTML = `

        <div class="patient-header">
            <div>
                <h2>Patient ${safe(patient[patientColumn])}</h2>
                <div class="status">Current patient monitoring status</div>
            </div>
            <button class="back-btn" onclick="scrollToList()">↑ Back to List</button>
        </div>

        <div class="vitals">
            ${vitalCard("Heart Rate",    safe(patient["HR"]),     "bpm")}
            ${vitalCard("O₂ Saturation", safe(patient["O2Sat"]),  "%")}
            ${vitalCard("Temperature",   safe(patient["Temp"]),   "°C")}
            ${vitalCard("Systolic BP",   safe(patient["SBP"]),    "mmHg")}
            ${vitalCard("MAP",           safe(patient["MAP"]),    "mmHg")}
            ${vitalCard("Diastolic BP",  safe(patient["DBP"]),    "mmHg")}
            ${vitalCard("Respiration",   safe(patient["Resp"]),   "/min")}
            ${vitalCard("ICU Hours",     safe(patient["ICULOS"]), "h")}
        </div>

        <div class="risk-grid">
            <div class="risk-card">
                <h3>6-HOUR DETERIORATION RISK</h3>
                <div class="risk-number">${percentage(risk6)}</div>
                <span class="risk-badge ${levelClass(risk6)}">${level(risk6)} RISK</span>
                <div class="warning">Prediction window: <b>Next 6 hours</b></div>
            </div>
            <div class="risk-card">
                <h3>12-HOUR DETERIORATION RISK</h3>
                <div class="risk-number">${percentage(risk12)}</div>
                <span class="risk-badge ${levelClass(risk12)}">${level(risk12)} RISK</span>
                <div class="warning">Prediction window: <b>Next 12 hours</b></div>
            </div>
        </div>

        ${factors6HTML}
        ${factors12HTML}
        ${recHTML}

        <div class="info">
            <h3>Clinical Decision Support</h3>
            ${warning}
            <p style="font-size:12px;color:#6b7280;">Research prototype — decision-support only. Not a clinical diagnosis.</p>
        </div>
    `;

    document.getElementById("dashboard").scrollIntoView({ behavior: "smooth", block: "start" });
}

function vitalCard(title, value, unit) {
    return `
        <div class="vital">
            <div class="vital-title">${title}</div>
            <div class="vital-value">
                ${safe(value)}
                <span style="font-size:12px;font-weight:400;color:#6b7280;">${unit}</span>
            </div>
        </div>
    `;
}

function scrollToList() {
    document.getElementById("tilesSection").scrollIntoView({ behavior: "smooth", block: "start" });
}

applyFiltersAndRender();

if (data.length > 0) {
    selectPatient(data[0][patientColumn]);
}

</script>

</body>
</html>
"""

html = html.replace("__DATA__", json_data)
html = html.replace("__PATIENT__", str(patient_col))
html = html.replace("__RISK6__", str(risk6_col))
html = html.replace("__RISK12__", str(risk12_col))
html = html.replace("__FACTORS6__",  str(factors6_col) if factors6_col else "NONE")
html = html.replace("__FACTORS12__", str(factors12_col) if factors12_col else "NONE")
html = html.replace("__RECOMMENDATION__", str(recommendation_col) if recommendation_col else "NONE")


# ------------------------------------------------------------
# 9. SAVE HTML
# ------------------------------------------------------------

HTML_FILE = "sepsis_early_warning_dashboard.html"

with open(HTML_FILE, "w", encoding="utf-8") as f:
    f.write(html)


print("\n" + "=" * 60)
print("FINAL DASHBOARD CREATED SUCCESSFULLY")
print("=" * 60)

print("HTML file:", os.path.abspath(HTML_FILE))
print("Patients :", len(patient_df))
print("Status   : READY")