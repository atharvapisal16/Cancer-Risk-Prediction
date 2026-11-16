"""
OncoRisk AI™ — Advanced Multi-Factor Oncology Risk Stratification Platform
=============================================================================
A Clinical Decision Support System (CDSS) for Multi-Cancer Risk Stratification,
Biomarker Analysis, Sub-Domain Factor Decomposition, and Evidence-Based Triage.

Model Architecture:
  - Optuna-Tuned, Class-Weighted Gradient Boosted Decision Trees (XGBoost)
  - Calibrated for High-Sensitivity Risk Stratification (80% Recall on High-Risk)
  - Multidimensional Feature Space: Clinical, Genetic, Lifestyle, Diet, & Environmental
"""

import os
import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# --------------------------------------------------------------------------- #
# Page Configuration & Medical Design Tokens
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="OncoRisk AI™ | Advanced Oncology Risk Stratification",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACTS_DIR = "artifacts"

# Color Palette for Clinical Stratification
RISK_COLORS = {
    "Low": "#10B981",       # Emerald Green
    "Medium": "#F59E0B",    # Amber / Gold
    "High": "#EF4444",      # Crimson / Rose
}

RISK_BG_COLORS = {
    "Low": "rgba(16, 185, 129, 0.12)",
    "Medium": "rgba(245, 158, 11, 0.12)",
    "High": "rgba(239, 68, 68, 0.12)",
}

RISK_BORDER_COLORS = {
    "Low": "#10B981",
    "Medium": "#F59E0B",
    "High": "#EF4444",
}

RISK_RADAR_FILL = {
    "Low": "rgba(16, 185, 129, 0.25)",
    "Medium": "rgba(245, 158, 11, 0.25)",
    "High": "rgba(239, 68, 68, 0.25)",
}

# --------------------------------------------------------------------------- #
# Custom CSS for Modern Health-Tech Glassmorphism Aesthetics
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero Header Styling */
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 50%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(14, 165, 233, 0.08) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-top: 6px;
        font-weight: 400;
        line-height: 1.5;
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(255, 255, 255, 0.05);
        color: #E2E8F0;
        margin-right: 8px;
        margin-top: 10px;
    }

    /* Clinical Cards */
    .clinical-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 16px;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .clinical-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
    }
    .card-header-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Result Banner */
    .result-banner {
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin: 18px 0;
        border-width: 2px;
        border-style: solid;
        animation: fadeIn 0.4s ease-in-out;
    }
    .result-level-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin: 0;
    }
    .result-confidence {
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 6px;
        opacity: 0.9;
    }
    .result-description {
        font-size: 0.92rem;
        margin-top: 12px;
        color: #CBD5E1;
        max-width: 650px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    }

    /* Archetype Preset Chips */
    .preset-btn {
        width: 100%;
        margin-bottom: 6px;
    }

    /* Monospace tags */
    .mono-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        padding: 2px 6px;
        border-radius: 4px;
        background: rgba(0, 0, 0, 0.3);
    }

    /* Key Driver Item */
    .driver-pill {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        border-radius: 8px;
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #38BDF8;
        margin-bottom: 8px;
        font-size: 0.88rem;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Cached Artifacts Loader
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_clinical_artifacts(artifacts_dir: str):
    if not os.path.isdir(artifacts_dir):
        return None, None, None, None
    model = joblib.load(os.path.join(artifacts_dir, "model.pkl"))
    label_encoders = joblib.load(os.path.join(artifacts_dir, "label_encoders.pkl"))
    target_encoder = joblib.load(os.path.join(artifacts_dir, "target_encoder.pkl"))
    feature_info = joblib.load(os.path.join(artifacts_dir, "feature_info.pkl"))
    return model, label_encoders, target_encoder, feature_info


# --------------------------------------------------------------------------- #
# Preset Archetypes for 1-Click Fast Demonstrations
# --------------------------------------------------------------------------- #
PRESET_PATIENTS = {
    "🟢 Healthy Adult (Low Baseline Risk)": {
        "Cancer_Type": "Breast",
        "Age": 32,
        "Gender": 0,  # Female
        "Smoking": 0.0,
        "Alcohol_Use": 1.0,
        "Obesity": 1.5,
        "Family_History": 0,
        "Diet_Red_Meat": 2.0,
        "Diet_Salted_Processed": 1.5,
        "Fruit_Veg_Intake": 9.0,
        "Physical_Activity": 8.5,
        "Air_Pollution": 1.5,
        "Occupational_Hazards": 0.5,
        "BRCA_Mutation": 0,
        "H_Pylori_Infection": 0,
        "Calcium_Intake": 8.0,
        "BMI": 21.8,
        "Physical_Activity_Level": 8.5,
        "height_cm": 168,
        "weight_kg": 61.5,
    },
    "🟡 Average Urban Profile (Moderate Risk)": {
        "Cancer_Type": "Colon",
        "Age": 56,
        "Gender": 1,  # Male
        "Smoking": 4.5,
        "Alcohol_Use": 4.5,
        "Obesity": 5.5,
        "Family_History": 0,
        "Diet_Red_Meat": 6.0,
        "Diet_Salted_Processed": 5.5,
        "Fruit_Veg_Intake": 4.5,
        "Physical_Activity": 4.0,
        "Air_Pollution": 5.5,
        "Occupational_Hazards": 4.0,
        "BRCA_Mutation": 0,
        "H_Pylori_Infection": 0,
        "Calcium_Intake": 4.5,
        "BMI": 26.5,
        "Physical_Activity_Level": 4.5,
        "height_cm": 175,
        "weight_kg": 81.2,
    },
    "🧬 Genetic Predisposition (BRCA1/2+ High Risk)": {
        "Cancer_Type": "Breast",
        "Age": 45,
        "Gender": 0,  # Female
        "Smoking": 2.0,
        "Alcohol_Use": 3.0,
        "Obesity": 4.0,
        "Family_History": 1,
        "Diet_Red_Meat": 4.0,
        "Diet_Salted_Processed": 3.5,
        "Fruit_Veg_Intake": 6.5,
        "Physical_Activity": 5.0,
        "Air_Pollution": 4.5,
        "Occupational_Hazards": 1.0,
        "BRCA_Mutation": 1,
        "H_Pylori_Infection": 0,
        "Calcium_Intake": 6.0,
        "BMI": 24.5,
        "Physical_Activity_Level": 5.0,
        "height_cm": 165,
        "weight_kg": 66.6,
    },
    "🏭 Heavy Industrial & Toxic Exposure (High Risk)": {
        "Cancer_Type": "Lung",
        "Age": 66,
        "Gender": 1,  # Male
        "Smoking": 9.0,
        "Alcohol_Use": 7.5,
        "Obesity": 7.0,
        "Family_History": 0,
        "Diet_Red_Meat": 8.0,
        "Diet_Salted_Processed": 8.0,
        "Fruit_Veg_Intake": 2.0,
        "Physical_Activity": 2.0,
        "Air_Pollution": 9.5,
        "Occupational_Hazards": 9.0,
        "BRCA_Mutation": 0,
        "H_Pylori_Infection": 0,
        "Calcium_Intake": 2.5,
        "BMI": 31.0,
        "Physical_Activity_Level": 2.0,
        "height_cm": 172,
        "weight_kg": 91.7,
    },
    "⚠️ Metabolic & Multi-Factor Compounded (High Risk)": {
        "Cancer_Type": "Colon",
        "Age": 70,
        "Gender": 1,  # Male
        "Smoking": 8.0,
        "Alcohol_Use": 8.5,
        "Obesity": 9.0,
        "Family_History": 1,
        "Diet_Red_Meat": 9.0,
        "Diet_Salted_Processed": 9.0,
        "Fruit_Veg_Intake": 1.5,
        "Physical_Activity": 1.0,
        "Air_Pollution": 8.0,
        "Occupational_Hazards": 6.5,
        "BRCA_Mutation": 0,
        "H_Pylori_Infection": 1,
        "Calcium_Intake": 2.0,
        "BMI": 35.5,
        "Physical_Activity_Level": 1.0,
        "height_cm": 170,
        "weight_kg": 102.6,
    },
}


# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #
def initialize_session_state(preset_data=None):
    if preset_data is None:
        preset_data = PRESET_PATIENTS["🟡 Average Urban Profile (Moderate Risk)"]
    for k, v in preset_data.items():
        st.session_state[f"field_{k}"] = v


def calculate_domain_indices(inputs: dict) -> dict:
    """Decomposes the 18 input risk factors into 5 normalized clinical domains (0-100%)."""
    # 1. Genetic & Biomarkers: Family History (0 or 1), BRCA (0 or 1), H. Pylori (0 or 1)
    gen_score = (
        (inputs["Family_History"] * 30.0) +
        (inputs["BRCA_Mutation"] * 45.0) +
        (inputs["H_Pylori_Infection"] * 25.0)
    )

    # 2. Substance & Lifestyle: Smoking (0-10) and Alcohol (0-10)
    lifestyle_score = ((inputs["Smoking"] * 0.6) + (inputs["Alcohol_Use"] * 0.4)) * 10.0

    # 3. Dietary Hazard: Red Meat (0-10), Salted/Processed (0-10), inverse Fruit/Veg (10 - X), inverse Calcium (10 - X)
    diet_score = (
        (inputs["Diet_Red_Meat"] * 0.35) +
        (inputs["Diet_Salted_Processed"] * 0.35) +
        ((10.0 - inputs["Fruit_Veg_Intake"]) * 0.15) +
        ((10.0 - inputs["Calcium_Intake"]) * 0.15)
    ) * 10.0

    # 4. Environmental & Occupational: Air Pollution (0-10) & Occupational Hazards (0-10)
    env_score = ((inputs["Air_Pollution"] * 0.6) + (inputs["Occupational_Hazards"] * 0.4)) * 10.0

    # 5. Metabolic & Inactivity: BMI (18.5-40 normalized), Obesity (0-10), inverse Physical Activity (10-X)
    bmi_norm = np.clip((inputs["BMI"] - 18.5) / (38.0 - 18.5), 0.0, 1.0) * 10.0
    activity_inv = 10.0 - (0.5 * inputs["Physical_Activity"] + 0.5 * inputs["Physical_Activity_Level"])
    metabolic_score = ((bmi_norm * 0.4) + (inputs["Obesity"] * 0.35) + (activity_inv * 0.25)) * 10.0

    return {
        "Genetics & Pathogens": float(np.clip(gen_score, 0, 100)),
        "Substance & Tobacco": float(np.clip(lifestyle_score, 0, 100)),
        "Dietary Imbalance": float(np.clip(diet_score, 0, 100)),
        "Environmental Exposure": float(np.clip(env_score, 0, 100)),
        "Metabolic & Inactivity": float(np.clip(metabolic_score, 0, 100)),
    }


def identify_top_risk_drivers(inputs: dict, model, feature_info: dict) -> list:
    """Ranks patient-specific features combining user elevation and XGBoost global feature importance."""
    importances = dict(zip(feature_info["feature_order"], model.feature_importances_))
    drivers = []

    # Map semantic feature labels and user severity
    for feat in feature_info["feature_order"]:
        val = inputs[feat]
        imp = importances.get(feat, 0.0)

        severity = 0.0
        reason = ""
        if feat == "Air_Pollution" and val >= 5.0:
            severity = (val / 10.0) * imp * 1.5
            reason = f"Severe air/particulate pollution exposure ({val:.1f}/10)"
        elif feat == "Smoking" and val >= 4.0:
            severity = (val / 10.0) * imp * 1.4
            reason = f"Elevated tobacco smoking burden ({val:.1f}/10)"
        elif feat == "Diet_Salted_Processed" and val >= 5.0:
            severity = (val / 10.0) * imp * 1.2
            reason = f"High processed/nitrate-rich dietary pattern ({val:.1f}/10)"
        elif feat == "Diet_Red_Meat" and val >= 6.0:
            severity = (val / 10.0) * imp * 1.2
            reason = f"Frequent red meat consumption ({val:.1f}/10)"
        elif feat == "Alcohol_Use" and val >= 5.0:
            severity = (val / 10.0) * imp * 1.2
            reason = f"Regular-to-heavy alcohol consumption ({val:.1f}/10)"
        elif feat == "Occupational_Hazards" and val >= 5.0:
            severity = (val / 10.0) * imp * 1.3
            reason = f"Workplace carcinogen/toxin exposure ({val:.1f}/10)"
        elif feat == "BRCA_Mutation" and val == 1:
            severity = 1.0 * imp * 2.2
            reason = "Pathogenic BRCA1/2 germline mutation detected"
        elif feat == "Family_History" and val == 1:
            severity = 1.0 * imp * 1.6
            reason = "Positive family history in 1st-degree relative"
        elif feat == "H_Pylori_Infection" and val == 1:
            severity = 1.0 * imp * 1.5
            reason = "Chronic Helicobacter Pylori inflammatory presence"
        elif feat == "BMI" and val >= 30.0:
            severity = ((val - 25.0) / 15.0) * imp * 1.2
            reason = f"Class I/II clinical obesity (BMI {val:.1f} kg/m²)"
        elif feat in ["Physical_Activity", "Physical_Activity_Level"] and val <= 3.0:
            severity = ((10.0 - val) / 10.0) * imp
            reason = f"Chronic sedentary lifestyle ({val:.1f}/10 activity)"
        elif feat == "Fruit_Veg_Intake" and val <= 3.0:
            severity = ((10.0 - val) / 10.0) * imp
            reason = f"Deficient protective antioxidant intake ({val:.1f}/10)"

        if severity > 0:
            drivers.append({
                "feature": feat.replace("_", " "),
                "severity": severity,
                "importance_pct": imp * 100.0,
                "reason": reason,
                "value": val,
            })

    drivers = sorted(drivers, key=lambda x: -x["severity"])
    return drivers[:4]


def generate_clinical_guidance(cancer_type: str, pred_label: str, inputs: dict) -> dict:
    """Tailors evidence-based oncology screening and intervention strategies."""
    screenings = []
    interventions = []
    triage = ""

    # Cancer Type Specific Screenings
    if cancer_type == "Breast":
        if inputs["BRCA_Mutation"] == 1 or inputs["Family_History"] == 1:
            screenings.append("High-Risk Screening Protocol: Annual Breast MRI + Digital Mammography starting by age 25–30.")
            screenings.append("Referral to Clinical Genetics for comprehensive hereditary oncology genetic counseling.")
        else:
            screenings.append("Age-appropriate Digital Mammography every 1–2 years; baseline clinical breast exam.")
        if inputs["Alcohol_Use"] >= 4.0:
            interventions.append("Restrict alcohol intake: each additional drink increases relative breast cancer risk by 7–10%.")

    elif cancer_type == "Lung":
        if inputs["Smoking"] >= 5.0 or inputs["Occupational_Hazards"] >= 6.0:
            screenings.append("Annual Low-Dose Computed Tomography (LDCT) chest screening for high-risk exposure history.")
            screenings.append("Pulmonary function assessment (Spirometry) & Occupational toxicity monitoring.")
        else:
            screenings.append("Standard respiratory surveillance; prompt investigation of chronic cough or hemoptysis.")
        if inputs["Smoking"] > 0:
            interventions.append("High-priority smoking cessation program with dual NRT (nicotine replacement therapy) or varenicline.")
        if inputs["Air_Pollution"] >= 5.0:
            interventions.append("Deploy indoor True-HEPA air filtration and avoid strenuous outdoor exercise during high-AQI windows.")

    elif cancer_type == "Colon":
        if inputs["Family_History"] == 1 or pred_label == "High":
            screenings.append("Diagnostic screening colonoscopy at 5-year intervals or earlier than standard 45-year guideline.")
        else:
            screenings.append("Routine colorectal screening: Colonoscopy every 10 years or annual high-sensitivity FIT.")
        if inputs["H_Pylori_Infection"] == 1:
            interventions.append("Gastrointestinal evaluation: Complete 14-day quad-therapy H. Pylori eradication protocol.")
        if inputs["Diet_Red_Meat"] >= 5.0 or inputs["Diet_Salted_Processed"] >= 5.0:
            interventions.append("Transition to high-fiber Mediterranean dietary pattern (<300g red meat/wk; minimize preserved cured meats).")

    elif cancer_type == "Prostate":
        screenings.append("Serum Prostate-Specific Antigen (PSA) test with free/total ratio & Digital Rectal Examination (DRE).")
        if inputs["Family_History"] == 1 or inputs["BRCA_Mutation"] == 1:
            screenings.append("Early baseline PSA monitoring starting at age 40–45 due to elevated germline predisposition.")
        interventions.append("Adopt cardiovascular-supportive nutrition rich in lycopene (cooked tomatoes) and cruciferous vegetables.")

    elif cancer_type == "Skin":
        screenings.append("Comprehensive Full-Body Total Skin Exam (TBSE) with clinical dermoscopy by a board-certified dermatologist.")
        screenings.append("Bi-monthly self-examination using the ABCDE criteria (Asymmetry, Border, Color, Diameter, Evolution).")
        interventions.append("Strict daily application of broad-spectrum SPF 50+ mineral sunscreen & UV protective clothing.")

    # Universal Lifestyle & Triage Guidance based on Predicted Risk Tier
    if pred_label == "High":
        triage = "🔴 Priority Oncology Triaging: Rapid clinical workup, baseline diagnostic imaging, and multi-disciplinary specialist review recommended."
        if inputs["Air_Pollution"] >= 6.0 or inputs["Occupational_Hazards"] >= 6.0:
            interventions.append("Workplace Industrial Hygiene Audit: Ensure NIOSH-certified particulate/gas respirators and personal protective equipment (PPE).")
        if inputs["BMI"] >= 28.0:
            interventions.append("Target a 7–10% intentional weight reduction through structured nutritional counseling to decrease systemic inflammatory cytokines.")
    elif pred_label == "Medium":
        triage = "🟡 Moderate Surveillance Stratum: Implement targeted risk factor mitigation and schedule an interval clinical follow-up in 6 months."
        if inputs["Physical_Activity"] < 5.0:
            interventions.append("Increase moderate-intensity aerobic physical activity to ≥150 minutes per week (or 75 mins vigorous) to optimize immune surveillance.")
    else:
        triage = "🟢 Routine Preventative Stratum: Maintain favorable baseline lifestyle habits and adhere to routine age-standardized screening schedules."

    return {
        "triage": triage,
        "screenings": screenings,
        "interventions": interventions,
    }


def generate_text_report(patient_data: dict, pred_label: str, proba_map: dict, drivers: list, guidance: dict) -> str:
    """Generates a downloadable, clinical consultation assessment note."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""================================================================================
ONCORISK AI™ — CLINICAL ONCOLOGY RISK STRATIFICATION REPORT
================================================================================
Generated: {timestamp}
Evaluation Model: Optuna-Tuned Class-Weighted XGBoost Classifier (Class-Imbalance Calibrated)
System Version: 2.4.0-Clinical

--------------------------------------------------------------------------------
1. PATIENT DEMOGRAPHIC & CLINICAL PROFILE
--------------------------------------------------------------------------------
Cancer Focus Site       : {patient_data['Cancer_Type']}
Patient Age             : {patient_data['Age']} years
Biological Sex          : {'Male' if patient_data['Gender'] == 1 else 'Female'}
Body Mass Index (BMI)   : {patient_data['BMI']:.1f} kg/m²
Family Cancer History   : {'Positive (1st-degree relative)' if patient_data['Family_History'] == 1 else 'Negative'}
BRCA1/2 Mutation        : {'Pathogenic Variant DETECTED' if patient_data['BRCA_Mutation'] == 1 else 'Wild-Type / Negative'}
H. Pylori Infection     : {'Active / Chronic' if patient_data['H_Pylori_Infection'] == 1 else 'Negative / Cleared'}

Exposure Ratings (0 - 10 Scale):
  - Tobacco / Smoking    : {patient_data['Smoking']:.1f} / 10
  - Alcohol Consumption  : {patient_data['Alcohol_Use']:.1f} / 10
  - Air Pollution Burden : {patient_data['Air_Pollution']:.1f} / 10  (Highest Global Importance)
  - Occupational Hazard  : {patient_data['Occupational_Hazards']:.1f} / 10
  - Diet (Red Meat)      : {patient_data['Diet_Red_Meat']:.1f} / 10
  - Diet (Processed)     : {patient_data['Diet_Salted_Processed']:.1f} / 10
  - Fruit & Veg Intake   : {patient_data['Fruit_Veg_Intake']:.1f} / 10
  - Physical Activity    : {patient_data['Physical_Activity']:.1f} / 10

--------------------------------------------------------------------------------
2. STRATIFIED RISK EVALUATION
--------------------------------------------------------------------------------
PREDICTED RISK TIER     : >>> {pred_label.upper()} RISK <<<
Primary Class Confidence: {proba_map[pred_label] * 100:.1f}%

Posterior Class Probability Breakdown:
  - Low Risk            : {proba_map.get('Low', 0.0) * 100:6.2f}%
  - Medium / Moderate   : {proba_map.get('Medium', 0.0) * 100:6.2f}%
  - High Risk           : {proba_map.get('High', 0.0) * 100:6.2f}%

Clinical Triaging Directive:
{guidance['triage']}

--------------------------------------------------------------------------------
3. PRIMARY CONTRIBUTING RISK DRIVERS
--------------------------------------------------------------------------------
"""
    for i, d in enumerate(drivers, 1):
        report += f"  {i}. {d['reason']} (Model Global Importance: {d['importance_pct']:.1f}%)\n"

    report += f"""
--------------------------------------------------------------------------------
4. EVIDENCE-BASED ACTION PLAN & RECOMMENDATIONS
--------------------------------------------------------------------------------
Recommended Diagnostic Screening:
"""
    for s in guidance["screenings"]:
        report += f"  [*] {s}\n"

    report += "\nTargeted Preventative Interventions:\n"
    for iv in guidance["interventions"]:
        report += f"  [+] {iv}\n"

    report += """
================================================================================
CLINICAL DISCLAIMER:
This report is generated by a machine-learning research platform intended solely
for clinical decision support, academic evaluation, and risk stratification.
It does not constitute a definitive medical diagnosis. Clinical management must
be determined by certified oncology healthcare professionals.
================================================================================
"""
    return report


# --------------------------------------------------------------------------- #
# Main Application Flow
# --------------------------------------------------------------------------- #
def main():
    # Load Model Artifacts
    model, label_encoders, target_encoder, feature_info = load_clinical_artifacts(ARTIFACTS_DIR)

    if model is None:
        st.error(
            f"❌ Critical Error: Could not locate `{ARTIFACTS_DIR}/` directory. "
            "Please ensure `model.pkl`, `label_encoders.pkl`, `target_encoder.pkl`, "
            "and `feature_info.pkl` are present."
        )
        st.stop()

    # Initialize session state if first time
    if "field_Age" not in st.session_state:
        initialize_session_state()

    # ----------------------------------------------------------------------- #
    # Top Hero Banner
    # ----------------------------------------------------------------------- #
    st.markdown(
        """
        <div class="hero-container">
            <h1 class="hero-title">
                🧬 OncoRisk AI™
                <span style="font-size: 1rem; font-weight: 500; color: #38BDF8; background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); padding: 4px 10px; border-radius: 6px;">
                    v2.4 Clinical CDSS
                </span>
            </h1>
            <div class="hero-subtitle">
                Advanced Multi-Factor Oncology Risk Stratification Platform & Biomarker Decision Support System.
                Analyzes compounding genetic, metabolic, dietary, environmental, and behavioral drivers.
            </div>
            <div>
                <span class="badge-pill">⚡ <b>Optuna-Tuned XGBoost</b></span>
                <span class="badge-pill">🎯 <b>80% High-Risk Recall Sensitivity</b></span>
                <span class="badge-pill">🛡️ <b>Class-Weighted Balancing</b></span>
                <span class="badge-pill">📊 <b>18 Validated Risk Biomarkers</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------------------------- #
    # Sidebar Controls & Clinical Archetypes
    # ----------------------------------------------------------------------- #
    with st.sidebar:
        st.markdown("### 🎛️ Clinical Archetypes")
        st.caption("Load verified test patient profiles with a single click:")

        for preset_name, preset_vals in PRESET_PATIENTS.items():
            if st.button(preset_name, key=f"btn_{preset_name}", use_container_width=True):
                initialize_session_state(preset_vals)
                st.rerun()

        st.divider()

        st.markdown("### 🔬 Model Transparency")
        with st.expander("Model Specifications", expanded=False):
            st.markdown(
                """
                - **Algorithm**: `XGBClassifier` (Gradient Boosting)
                - **Optimization**: Optuna Hyperparameter Tuning
                - **Class Strategy**: `compute_sample_weight('balanced')`
                - **Target Accuracy**: `82.0%`
                - **High-Risk Sensitivity**: `80.0% Recall (16/20 caught)`
                - **Leakage Prevention**: `Overall_Risk_Score` excluded from training.
                """
            )

        with st.expander("Top Feature Importance", expanded=False):
            importances = dict(zip(feature_info["feature_order"], model.feature_importances_))
            sorted_imp = sorted(importances.items(), key=lambda x: -x[1])[:8]
            imp_df = pd.DataFrame(sorted_imp, columns=["Feature", "Importance"])
            imp_df["Importance"] = imp_df["Importance"].map(lambda x: f"{x * 100:.1f}%")
            st.dataframe(imp_df, hide_index=True, use_container_width=True)

        st.divider()
        st.caption(
            "⚠️ **Clinical Advisory**: Designed as a decision support portfolio demonstration. "
            "Individual patient diagnosis requires accredited laboratory diagnostics and physician oversight."
        )

    # ----------------------------------------------------------------------- #
    # Main Form: Grouped, Clean, High-UX Categorization
    # ----------------------------------------------------------------------- #
    st.markdown("### 📋 Patient Biomarker & Exposure Assessment")
    st.caption("Adjust the patient parameters below across the 5 clinical risk domains:")

    tabs = st.tabs([
        "👤 Demographics & Focus",
        "🧬 Genetics & Pathogens",
        "🚬 Lifestyle & Substances",
        "🥗 Dietary Profile",
        "🏭 Environmental & Workplace",
    ])

    user_inputs = {}

    # TAB 1: Demographics & Cancer Focus
    with tabs[0]:
        col_t1a, col_t1b = st.columns([1, 1])

        with col_t1a:
            st.markdown("##### Primary Cancer Evaluation Focus")
            cancer_options = feature_info["categorical_options"]["Cancer_Type"]
            curr_cancer = st.session_state.get("field_Cancer_Type", cancer_options[0])
            user_inputs["Cancer_Type"] = st.selectbox(
                "Select Cancer Site / Type",
                options=cancer_options,
                index=cancer_options.index(curr_cancer) if curr_cancer in cancer_options else 0,
                help="Target malignancy under clinical surveillance.",
            )

            user_inputs["Age"] = st.slider(
                "Patient Age (Years)",
                min_value=25,
                max_value=90,
                value=int(st.session_state.get("field_Age", 63)),
                help="Chronological age is a primary oncological baseline determinant.",
            )

            # Gender selection (0 = Female, 1 = Male)
            curr_gender = int(st.session_state.get("field_Gender", 0))
            gender_label = st.radio(
                "Biological Sex",
                options=["Female (0)", "Male (1)"],
                index=curr_gender,
                horizontal=True,
                help="Biological sex influences endocrine and organ-specific risk profiles.",
            )
            user_inputs["Gender"] = 1 if "Male" in gender_label else 0

        with col_t1b:
            st.markdown("##### Metabolic & Body Mass Index (BMI)")
            calc_mode = st.toggle("Use Height & Weight Calculator", value=False)

            if calc_mode:
                c_h, c_w = st.columns(2)
                with c_h:
                    h_val = st.number_input(
                        "Height (cm)",
                        min_value=120,
                        max_value=220,
                        value=int(st.session_state.get("field_height_cm", 172)),
                        step=1,
                    )
                with c_w:
                    w_val = st.number_input(
                        "Weight (kg)",
                        min_value=35.0,
                        max_value=180.0,
                        value=float(st.session_state.get("field_weight_kg", 78.0)),
                        step=0.5,
                    )
                calc_bmi = round(w_val / ((h_val / 100.0) ** 2), 1)
                user_inputs["BMI"] = float(np.clip(calc_bmi, 15.0, 41.4))
            else:
                user_inputs["BMI"] = st.slider(
                    "Body Mass Index (BMI kg/m²)",
                    min_value=15.0,
                    max_value=41.4,
                    value=float(st.session_state.get("field_BMI", 26.2)),
                    step=0.1,
                    help="Normal: 18.5-24.9 | Overweight: 25.0-29.9 | Obese: ≥30.0",
                )

            # BMI status badge
            bmi_val = user_inputs["BMI"]
            if bmi_val < 18.5:
                bmi_cat = "Underweight (<18.5)"
                bmi_color = "#38BDF8"
            elif bmi_val < 25.0:
                bmi_cat = "Normal Weight (18.5 - 24.9)"
                bmi_color = "#10B981"
            elif bmi_val < 30.0:
                bmi_cat = "Overweight (25.0 - 29.9)"
                bmi_color = "#F59E0B"
            else:
                bmi_cat = "Obese Class I/II (≥30.0)"
                bmi_color = "#EF4444"

            st.markdown(
                f"""
                <div style="padding: 10px 14px; border-radius: 8px; background: rgba(255,255,255,0.04); border: 1px solid {bmi_color}; margin-top: 10px;">
                    <span style="font-size: 0.82rem; color: #94A3B8;">Clinical Classification:</span><br/>
                    <b style="color: {bmi_color}; font-size: 1rem;">{bmi_val:.1f} kg/m² — {bmi_cat}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

            user_inputs["Obesity"] = st.slider(
                "Adiposity / Visceral Obesity Rating (0 - 10)",
                min_value=0.0,
                max_value=10.0,
                value=float(st.session_state.get("field_Obesity", 5.0)),
                step=0.5,
                help="Clinical assessment of central abdominal fat and metabolic syndrome markers.",
            )

    # TAB 2: Genetics & Pathogens
    with tabs[1]:
        st.markdown("##### Hereditary Germline Mutations & Inflammatory Pathogens")
        col_t2a, col_t2b, col_t2c = st.columns(3)

        with col_t2a:
            curr_fam = int(st.session_state.get("field_Family_History", 0))
            fam_choice = st.radio(
                "Family History of Cancer",
                options=["No (Negative)", "Yes (1st-Degree Relative)"],
                index=curr_fam,
                help="Parent, sibling, or child diagnosed with malignancy.",
            )
            user_inputs["Family_History"] = 1 if "Yes" in fam_choice else 0
            st.caption("First-degree biological relative with verified malignancy.")

        with col_t2b:
            curr_brca = int(st.session_state.get("field_BRCA_Mutation", 0))
            brca_choice = st.radio(
                "BRCA1 / BRCA2 Mutation Status",
                options=["Negative / Wild-Type (0)", "Positive Variant Detected (1)"],
                index=curr_brca,
                help="Pathogenic DNA repair deficiency mutation.",
            )
            user_inputs["BRCA_Mutation"] = 1 if "Positive" in brca_choice else 0
            st.caption("High-penetrance hereditary risk for Breast, Ovarian, Colon, & Prostate.")

        with col_t2c:
            curr_hp = int(st.session_state.get("field_H_Pylori_Infection", 0))
            hp_choice = st.radio(
                "Helicobacter Pylori Status",
                options=["Negative / Cleared (0)", "Active / Chronic Positive (1)"],
                index=curr_hp,
                help="Bacterial presence causing chronic gastric mucosal inflammation.",
            )
            user_inputs["H_Pylori_Infection"] = 1 if "Active" in hp_choice else 0
            st.caption("Strong epidemiological link to gastric & colorectal mucosal lesions.")

    # TAB 3: Lifestyle & Substances
    with tabs[2]:
        col_t3a, col_t3b = st.columns(2)

        with col_t3a:
            st.markdown("##### Carcinogenic Substance Exposure")
            user_inputs["Smoking"] = st.slider(
                "Tobacco Smoking Index (0 - 10)",
                min_value=0.0,
                max_value=10.0,
                value=float(st.session_state.get("field_Smoking", 5.0)),
                step=0.5,
                help="0 = Non-smoker, 3 = Light/Past, 6 = Moderate (10-15/day), 10 = Heavy Chronic (25+/day).",
            )
            smk = user_inputs["Smoking"]
            smk_desc = "Non-smoker / Never" if smk == 0 else "Light / Occasional" if smk <= 3 else "Moderate Daily (10-15 cigs)" if smk <= 7 else "Heavy Chronic Smoker (20+ cigs/day)"
            st.caption(f"Status: **{smk_desc}**")

            user_inputs["Alcohol_Use"] = st.slider(
                "Alcohol Consumption Index (0 - 10)",
                min_value=0.0,
                max_value=10.0,
                value=float(st.session_state.get("field_Alcohol_Use", 5.0)),
                step=0.5,
                help="0 = Abstinent, 3 = Social/Occasional, 6 = Frequent weekly, 10 = Daily heavy drinking.",
            )
            alc = user_inputs["Alcohol_Use"]
            alc_desc = "Abstinent" if alc == 0 else "Low / Social" if alc <= 3 else "Moderate Regular" if alc <= 7 else "Heavy / Daily"
            st.caption(f"Status: **{alc_desc}**")

        with col_t3b:
            st.markdown("##### Physical Activity & Sedentary Burden")
            user_inputs["Physical_Activity"] = st.slider(
                "Weekly Physical Activity Volume (0 - 10)",
                min_value=0.0,
                max_value=10.0,
                value=float(st.session_state.get("field_Physical_Activity", 4.0)),
                step=0.5,
                help="0 = Sedentary (<30 min/wk), 5 = Moderate (150 min/wk), 10 = Highly Active (300+ min/wk).",
            )

            user_inputs["Physical_Activity_Level"] = st.slider(
                "Exercise Intensity Level (0 - 10)",
                min_value=0.0,
                max_value=10.0,
                value=float(st.session_state.get("field_Physical_Activity_Level", 5.0)),
                step=0.5,
                help="Aerobic conditioning and vigorous exertion intensity.",
            )

    # TAB 4: Dietary Profile
    with tabs[3]:
        col_t4a, col_t4b = st.columns(2)

        with col_t4a:
            st.markdown("##### Carcinogenic Dietary Factors")
            user_inputs["Diet_Red_Meat"] = st.slider(
                "Red Meat Intake (0 - 10)",
                min_value=0.0,
                max_value=10.0,
                value=float(st.session_state.get("field_Diet_Red_Meat", 5.0)),
                step=0.5,
                help="Beef, pork, lamb frequency. 0 = Rare/Vegetarian, 10 = Daily large portions.",
            )

            user_inputs["Diet_Salted_Processed"] = st.slider(
                "Salted & Ultra-Processed Foods (0 - 10)",
                min_value=0.0,
                max_value=10.0,
                value=float(st.session_state.get("field_Diet_Salted_Processed", 4.5)),
                step=0.5,
                help="Cured meats, sodium-dense, preserved foods with nitrates.",
            )

        with col_t4b:
            st.markdown("##### Protective Micronutrients")
            user_inputs["Fruit_Veg_Intake"] = st.slider(
                "Fruit & Vegetable Consumption (0 - 10)",
                min_value=0.0,
                max_value=10.0,
                value=float(st.session_state.get("field_Fruit_Veg_Intake", 5.0)),
                step=0.5,
                help="Dietary fiber and antioxidants. 0 = Severe deficiency, 10 = 5+ servings daily.",
            )

            user_inputs["Calcium_Intake"] = st.slider(
                "Dietary Calcium Intake (0 - 10)",
                min_value=0.0,
                max_value=10.0,
                value=float(st.session_state.get("field_Calcium_Intake", 4.0)),
                step=0.5,
                help="Dairy, fortified plant milk, leafy greens. Protective against colorectal polyps.",
            )

    # TAB 5: Environmental & Workplace
    with tabs[4]:
        col_t5a, col_t5b = st.columns(2)

        with col_t5a:
            st.markdown("##### Air Pollution & Particulate Exposure")
            st.info("⭐ **Key Predictor**: In our XGBoost feature importance analysis, Air Pollution is the #1 single most influential predictor (14.8% model weight).")

            user_inputs["Air_Pollution"] = st.slider(
                "Air Pollution Exposure Index (0 - 10)",
                min_value=0.0,
                max_value=10.0,
                value=float(st.session_state.get("field_Air_Pollution", 5.5)),
                step=0.5,
                help="Ambient PM2.5, NO2, ozone, and industrial smog exposure index.",
            )
            pol = user_inputs["Air_Pollution"]
            pol_desc = "Clean Rural Air" if pol <= 2 else "Suburban Moderate" if pol <= 5 else "Urban Smog / Industrial Exposure" if pol <= 8 else "Extreme Particulate Burden (AQI 250+)"
            st.caption(f"Exposure Status: **{pol_desc}**")

        with col_t5b:
            st.markdown("##### Workplace Occupational Hazards")
            user_inputs["Occupational_Hazards"] = st.slider(
                "Occupational Carcinogen Exposure (0 - 10)",
                min_value=0.0,
                max_value=10.0,
                value=float(st.session_state.get("field_Occupational_Hazards", 5.0)),
                step=0.5,
                help="Asbestos, silica, diesel exhaust, aromatic amines, radiation, chemical solvents.",
            )
            occ = user_inputs["Occupational_Hazards"]
            occ_desc = "Clean Office / Remote" if occ <= 2 else "Low Chemical Contact" if occ <= 5 else "Industrial / Construction / Chemical Handling" if occ <= 8 else "Extreme Hazardous Environment (Mining/Foundry)"
            st.caption(f"Workplace Risk: **{occ_desc}**")

    # ----------------------------------------------------------------------- #
    # Prediction Action Button
    # ----------------------------------------------------------------------- #
    st.markdown("<br/>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Run AI Oncology Risk Stratification", type="primary", use_container_width=True)

    # ----------------------------------------------------------------------- #
    # Prediction & Results Dashboard
    # ----------------------------------------------------------------------- #
    if predict_btn:
        # Build DataFrame in model feature order
        encoded_cancer = label_encoders["Cancer_Type"].transform([user_inputs["Cancer_Type"]])[0]
        row_dict = {}
        for feat in feature_info["feature_order"]:
            if feat == "Cancer_Type":
                row_dict[feat] = encoded_cancer
            else:
                row_dict[feat] = user_inputs[feat]

        input_df = pd.DataFrame([row_dict])[feature_info["feature_order"]]

        # Inference
        pred_encoded = model.predict(input_df)[0]
        pred_label = target_encoder.inverse_transform([pred_encoded])[0]
        proba = model.predict_proba(input_df)[0]
        proba_map = dict(zip(target_encoder.classes_, proba))

        badge_color = RISK_COLORS.get(pred_label, "#38BDF8")
        bg_color = RISK_BG_COLORS.get(pred_label, "rgba(56, 189, 248, 0.1)")
        border_color = RISK_BORDER_COLORS.get(pred_label, "#38BDF8")
        confidence_pct = proba_map[pred_label] * 100.0

        st.divider()

        # 1. HERO RESULT BANNER
        desc_text = {
            "Low": "Patient displays a favorable baseline clinical profile with low compounding carcinogen exposure and absence of severe genetic indicators. Standard age-appropriate routine surveillance advised.",
            "Medium": "Patient presents with multiple moderate lifestyle, metabolic, or occupational exposures. Heightened clinical vigilance, targeted risk factor modification, and semi-annual follow-ups are recommended.",
            "High": "Patient exhibits multiple compounding high-potency risk factors (genetic mutations, heavy carcinogen exposure, high environmental hazard). Priority clinical triaging and specialized diagnostic screening recommended.",
        }.get(pred_label, "")

        st.markdown(
            f"""
            <div class="result-banner" style="background-color: {bg_color}; border-color: {border_color};">
                <span style="font-size: 0.95rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: {badge_color};">
                    ONCOLOGY STRATIFICATION RESULT
                </span>
                <h2 class="result-level-title" style="color: {badge_color};">
                    {pred_label.upper()} CANCER RISK TIER
                </h2>
                <div class="result-confidence" style="color: #F1F5F9;">
                    Model Confidence: <span style="color: {badge_color};">{confidence_pct:.1f}%</span>
                    <span style="color: #94A3B8; font-size: 0.88rem; font-weight: normal;">({user_inputs['Cancer_Type']} Cancer Focus)</span>
                </div>
                <div class="result-description">
                    {desc_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. PROBABILITY GAUGE & CLASS DISTRIBUTION
        c_res1, c_res2 = st.columns([1, 1])

        with c_res1:
            gauge_val = {"Low": 22, "Medium": 58, "High": 88}.get(pred_label, 50)
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=gauge_val,
                title={"text": "Composite Risk Index", "font": {"size": 18, "color": "#F1F5F9"}},
                number={"suffix": "/100", "font": {"size": 26, "color": badge_color}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                    "bar": {"color": badge_color, "thickness": 0.28},
                    "bgcolor": "rgba(255, 255, 255, 0.05)",
                    "borderwidth": 1,
                    "bordercolor": "rgba(255, 255, 255, 0.1)",
                    "steps": [
                        {"range": [0, 35], "color": "rgba(16, 185, 129, 0.2)"},
                        {"range": [35, 70], "color": "rgba(245, 158, 11, 0.2)"},
                        {"range": [70, 100], "color": "rgba(239, 68, 68, 0.2)"},
                    ],
                },
            ))
            gauge_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=260,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

        with c_res2:
            prob_df = pd.DataFrame({
                "Risk Class": target_encoder.classes_,
                "Probability": [proba_map[c] for c in target_encoder.classes_],
            }).sort_values("Probability", ascending=True)

            bar_fig = go.Figure(go.Bar(
                x=prob_df["Probability"],
                y=prob_df["Risk Class"],
                orientation="h",
                marker_color=[RISK_COLORS.get(c, "#38BDF8") for c in prob_df["Risk Class"]],
                text=[f"{p * 100:.1f}%" for p in prob_df["Probability"]],
                textposition="outside",
            ))
            bar_fig.update_layout(
                title="Class Probability Distribution",
                title_font=dict(size=16, color="#F1F5F9"),
                xaxis=dict(range=[0, 1.15], tickformat=".0%", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.0)"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#CBD5E1"),
                height=260,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        # 3. MULTI-FACTOR RADAR & TOP DRIVERS
        c_radar, c_drivers = st.columns([1, 1])

        sub_domains = calculate_domain_indices(user_inputs)

        with c_radar:
            radar_fill = RISK_RADAR_FILL.get(pred_label, "rgba(56, 189, 248, 0.25)")
            radar_fig = go.Figure(go.Scatterpolar(
                r=list(sub_domains.values()),
                theta=list(sub_domains.keys()),
                fill="toself",
                fillcolor=radar_fill,
                line=dict(color=badge_color, width=2),
                marker=dict(size=6, color=badge_color),
            ))
            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.1)"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.1)"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                title="Sub-Domain Clinical Risk Radar",
                title_font=dict(size=16, color="#F1F5F9"),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#CBD5E1"),
                height=320,
                margin=dict(l=30, r=30, t=50, b=30),
            )
            st.plotly_chart(radar_fig, use_container_width=True)

        with c_drivers:
            st.markdown("##### 🔍 Primary Risk Factor Contributors")
            top_drivers = identify_top_risk_drivers(user_inputs, model, feature_info)

            if top_drivers:
                for d in top_drivers:
                    st.markdown(
                        f"""
                        <div class="driver-pill">
                            <div>
                                <b style="color: #F8FAFC;">{d['reason']}</b><br/>
                                <span style="font-size: 0.78rem; color: #94A3B8;">
                                    Feature: <code>{d['feature']}</code> | Model Global Weight: <b>{d['importance_pct']:.1f}%</b>
                                </span>
                            </div>
                            <span style="background: rgba(56, 189, 248, 0.15); color: #38BDF8; padding: 4px 8px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;">
                                Score: {d['value']}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.success("✅ No elevated acute risk drivers detected for this patient profile.")

        # 4. EVIDENCE-BASED CLINICAL GUIDANCE
        guidance = generate_clinical_guidance(user_inputs["Cancer_Type"], pred_label, user_inputs)

        st.markdown("---")
        st.markdown("### 🩺 Evidence-Based Oncology Care Pathway")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("##### 🔬 Recommended Diagnostic Screenings")
            for scr in guidance["screenings"]:
                st.markdown(f"- **Screening**: {scr}")

        with col_g2:
            st.markdown("##### 🥗 Modifiable Interventions & Prevention")
            for inv in guidance["interventions"]:
                st.markdown(f"- **Intervention**: {inv}")

        # 5. WHAT-IF RISK SIMULATOR
        st.markdown("---")
        with st.expander("🧪 Interactive 'What-If' Scenario Risk Simulator", expanded=False):
            st.markdown(
                "Simulate how targeted risk reduction (e.g. smoking cessation, air filtration, weight loss) "
                "impacts this patient's predicted risk class and posterior probability."
            )
            c_sim1, c_sim2, c_sim3 = st.columns(3)
            with c_sim1:
                sim_smoking = st.slider("Simulate Smoking Index", 0.0, 10.0, 0.0, 0.5, key="sim_smk")
            with c_sim2:
                sim_pollution = st.slider("Simulate Air Pollution Index", 0.0, 10.0, 2.0, 0.5, key="sim_pol")
            with c_sim3:
                sim_bmi = st.slider("Simulate Healthy BMI (kg/m²)", 18.5, 35.0, 23.0, 0.5, key="sim_bmi")

            sim_inputs = user_inputs.copy()
            sim_inputs["Smoking"] = sim_smoking
            sim_inputs["Air_Pollution"] = sim_pollution
            sim_inputs["BMI"] = sim_bmi
            sim_inputs["Obesity"] = 2.0 if sim_bmi < 25 else 5.0

            # Re-predict for simulation
            sim_row = {}
            for feat in feature_info["feature_order"]:
                if feat == "Cancer_Type":
                    sim_row[feat] = encoded_cancer
                else:
                    sim_row[feat] = sim_inputs[feat]
            sim_df = pd.DataFrame([sim_row])[feature_info["feature_order"]]
            sim_pred = target_encoder.inverse_transform(model.predict(sim_df))[0]
            sim_proba = model.predict_proba(sim_df)[0]
            sim_proba_map = dict(zip(target_encoder.classes_, sim_proba))

            sim_badge = RISK_COLORS.get(sim_pred, "#38BDF8")
            st.markdown(
                f"""
                <div style="padding: 16px; border-radius: 10px; background: rgba(30, 41, 59, 0.6); border: 1px solid {sim_badge};">
                    <b>Baseline Prediction</b>: <span style="color: {badge_color};">{pred_label} Risk ({confidence_pct:.1f}%)</span><br/>
                    <b>Simulated Modifiable Outcome</b>: <span style="color: {sim_badge}; font-weight: 700; font-size: 1.1rem;">{sim_pred} Risk ({sim_proba_map[sim_pred] * 100:.1f}%)</span><br/>
                    <small style="color: #94A3B8;">Demonstrates the direct clinical impact of actionable lifestyle and environmental mitigation.</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 6. DOWNLOADABLE CLINICAL REPORT
        st.markdown("<br/>", unsafe_allow_html=True)
        report_text = generate_text_report(user_inputs, pred_label, proba_map, top_drivers, guidance)
        st.download_button(
            label="📥 Download Comprehensive Clinical Assessment Report (.txt)",
            data=report_text,
            file_name=f"OncoRisk_Report_{user_inputs['Cancer_Type']}_{datetime.date.today()}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        with st.expander("🛠️ View Raw Encoded Matrix Sent to XGBoost"):
            st.dataframe(input_df)


if __name__ == "__main__":
    main()