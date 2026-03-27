import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stApp { max-width: 1100px; margin: 0 auto; }

    /* Section card */
    .section-card {
        background: white;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-left: 5px solid #e63946;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1d3557;
        margin-bottom: 4px;
    }
    .section-subtitle {
        font-size: 0.85rem;
        color: #6c757d;
        margin-bottom: 0;
    }

    /* Risk tier badges */
    .risk-low    { background:#d4edda; color:#155724; padding:6px 14px; border-radius:20px; font-weight:700; font-size:1rem; }
    .risk-mod    { background:#fff3cd; color:#856404; padding:6px 14px; border-radius:20px; font-weight:700; font-size:1rem; }
    .risk-high   { background:#f8d7da; color:#721c24; padding:6px 14px; border-radius:20px; font-weight:700; font-size:1rem; }
    .risk-vhigh  { background:#c82333; color:#fff;    padding:6px 14px; border-radius:20px; font-weight:700; font-size:1rem; }

    /* Result box */
    .result-box {
        background: white;
        border-radius: 14px;
        padding: 32px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    .prob-number {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .prob-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: 4px;
    }
    .disclaimer {
        background: #e9ecef;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.82rem;
        color: #495057;
        margin-top: 24px;
    }

    /* Progress bar override */
    .stProgress > div > div { border-radius: 10px; }

    div[data-testid="stHorizontalBlock"] { gap: 16px; }
    .stButton > button {
        background-color: #e63946;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 36px;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
    }
    .stButton > button:hover { background-color: #c1121f; }
</style>
""", unsafe_allow_html=True)

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    scalers = {}

    def _load_pkl(path):
        return joblib.load(path)

    # Gradient Boosting
    gb_path = os.path.join(MODEL_DIR, "gradient_boosting.pkl")
    if os.path.exists(gb_path):
        models["Gradient Boosting"] = _load_pkl(gb_path)

    # Random Forest
    rf_path = os.path.join(MODEL_DIR, "random_forest.pkl")
    if os.path.exists(rf_path):
        models["Random Forest"] = _load_pkl(rf_path)

    # SVM + scaler
    svm_path    = os.path.join(MODEL_DIR, "svm.pkl")
    svm_sc_path = os.path.join(MODEL_DIR, "svm_scaler.pkl")
    if os.path.exists(svm_path) and os.path.exists(svm_sc_path):
        models["SVM"]          = _load_pkl(svm_path)
        scalers["SVM"]         = _load_pkl(svm_sc_path)

    # Neural Network + scaler (optional – requires PyTorch)
    nn_path    = os.path.join(MODEL_DIR, "neural_network.pt")
    nn_sc_path = os.path.join(MODEL_DIR, "nn_scaler.pkl")
    if os.path.exists(nn_path) and os.path.exists(nn_sc_path):
        try:
            import torch
            import torch.nn as nn

            class HeartNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(15, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3),
                        nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.2),
                        nn.Linear(32, 1),
                    )
                def forward(self, x):
                    return self.net(x)

            net = HeartNet()
            net.load_state_dict(torch.load(nn_path, map_location="cpu"))
            net.eval()
            models["Neural Network"] = net
            scalers["Neural Network"] = _load_pkl(nn_sc_path)
        except Exception:
            pass  # PyTorch not installed or load failed

    return models, scalers

models, scalers = load_models()

FEATURE_ORDER = [
    "male", "age", "education", "currentSmoker", "cigsPerDay",
    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
    "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose",
]

# ── Risk tier helper ──────────────────────────────────────────────────────────
def get_risk_tier(prob: float):
    """Map a 0-1 probability to a named risk tier."""
    if prob < 0.10:
        return "Low Risk",       "risk-low",   "✅", "#28a745"
    elif prob < 0.20:
        return "Moderate Risk",  "risk-mod",   "⚠️", "#ffc107"
    elif prob < 0.35:
        return "High Risk",      "risk-high",  "🔶", "#dc3545"
    else:
        return "Very High Risk", "risk-vhigh", "🚨", "#c82333"

def predict_proba(model_name: str, features: dict) -> float:
    """Return CHD probability (0-1) for the given model and feature dict."""
    X = pd.DataFrame([features])[FEATURE_ORDER].astype(float)

    if model_name == "Neural Network":
        import torch
        scaler = scalers["Neural Network"]
        X_sc   = scaler.transform(X)
        tensor = torch.tensor(X_sc, dtype=torch.float32)
        with torch.no_grad():
            logit = models["Neural Network"](tensor)
        return torch.sigmoid(logit).item()

    model = models[model_name]
    if model_name in scalers:
        X = pd.DataFrame(scalers[model_name].transform(X), columns=FEATURE_ORDER)
    return model.predict_proba(X)[0][1]

def ensemble_proba(features: dict) -> float:
    """Average probability across all loaded models."""
    probs = [predict_proba(name, features) for name in models]
    return float(np.mean(probs))

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("## ❤️ 10-Year Heart Disease Risk Predictor")
st.markdown(
    "Answer the questions below as accurately as possible. "
    "This tool estimates your **10-year risk of coronary heart disease** "
    "based on the Framingham Heart Study. It takes about 2 minutes to complete."
)
st.markdown("---")

# ── SECTION 1: About You ──────────────────────────────────────────────────────
st.markdown('<div class="section-card">'
            '<div class="section-title">👤 About You</div>'
            '<div class="section-subtitle">Basic personal information</div>'
            '</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    sex_label = st.radio("Biological Sex", ["Female", "Male"], horizontal=True)
    male = 1 if sex_label == "Male" else 0

with col2:
    age = st.slider("Age (years)", min_value=20, max_value=90, value=45, step=1)

with col3:
    edu_labels = {
        "Some High School or Less": 1,
        "High School / GED":        2,
        "Some College":             3,
        "College Degree or Higher": 4,
    }
    edu_label  = st.selectbox("Highest Level of Education", list(edu_labels.keys()))
    education  = edu_labels[edu_label]

st.markdown("---")

# ── SECTION 2: Lifestyle ──────────────────────────────────────────────────────
st.markdown('<div class="section-card">'
            '<div class="section-title">🚬 Lifestyle</div>'
            '<div class="section-subtitle">Smoking habits</div>'
            '</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    smoker_label   = st.radio("Do you currently smoke cigarettes?", ["No", "Yes"], horizontal=True)
    currentSmoker  = 1 if smoker_label == "Yes" else 0

with col2:
    if currentSmoker:
        cigsPerDay = st.slider(
            "How many cigarettes do you smoke per day?",
            min_value=1, max_value=60, value=10, step=1
        )
    else:
        cigsPerDay = 0
        st.info("Cigarettes per day: **0** (non-smoker)")

st.markdown("---")

# ── SECTION 3: Medical History ────────────────────────────────────────────────
st.markdown('<div class="section-card">'
            '<div class="section-title">🏥 Medical History</div>'
            '<div class="section-subtitle">Pre-existing conditions and medications</div>'
            '</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    bp_med_label  = st.radio("Are you taking blood pressure medication?",    ["No", "Yes"], horizontal=True, key="bpmeds")
    stroke_label  = st.radio("Have you ever had a stroke?",                  ["No", "Yes"], horizontal=True, key="stroke")
    BPMeds        = 1 if bp_med_label == "Yes" else 0
    prevalentStroke = 1 if stroke_label == "Yes" else 0

with col2:
    hyp_label     = st.radio("Have you been diagnosed with high blood pressure (hypertension)?", ["No", "Yes"], horizontal=True, key="hyp")
    diab_label    = st.radio("Have you been diagnosed with diabetes?",       ["No", "Yes"], horizontal=True, key="diab")
    prevalentHyp  = 1 if hyp_label == "Yes" else 0
    diabetes      = 1 if diab_label == "Yes" else 0

st.markdown("---")

# ── SECTION 4: Vitals & Lab Results ──────────────────────────────────────────
st.markdown('<div class="section-card">'
            '<div class="section-title">🩺 Vitals & Lab Results</div>'
            '<div class="section-subtitle">From a recent physical exam or blood test — ask your doctor if unsure</div>'
            '</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Blood Pressure (mmHg)**")
    st.caption("Your systolic BP is the top number; diastolic is the bottom number.")
    sysBP  = st.number_input("Systolic BP  (top number)",    min_value=70,  max_value=250, value=120, step=1)
    diaBP  = st.number_input("Diastolic BP (bottom number)", min_value=40,  max_value=150, value=80,  step=1)

with col2:
    st.markdown("**Blood Test Results**")
    st.caption("These values come from a standard blood panel.")
    totChol = st.number_input("Total Cholesterol (mg/dL)",      min_value=100, max_value=700, value=200, step=1)
    glucose = st.number_input("Fasting Blood Glucose (mg/dL)",  min_value=40,  max_value=400, value=85,  step=1)

with col3:
    st.markdown("**Other Measurements**")
    heartRate = st.number_input("Resting Heart Rate (bpm)", min_value=30, max_value=200, value=75, step=1)

    st.markdown("**Body Mass Index (BMI)**")
    bmi_method = st.radio("How would you like to enter BMI?", ["Enter BMI directly", "Calculate from height & weight"], key="bmi_method")
    if bmi_method == "Enter BMI directly":
        BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1, format="%.1f")
    else:
        units = st.radio("Units", ["Imperial (lbs / inches)", "Metric (kg / cm)"], key="units")
        if "Imperial" in units:
            weight_lbs = st.number_input("Weight (lbs)",   min_value=50,   max_value=500, value=160, step=1)
            height_in  = st.number_input("Height (inches)", min_value=40,  max_value=100, value=67,  step=1)
            BMI = round((weight_lbs / (height_in ** 2)) * 703, 1) if height_in > 0 else 25.0
        else:
            weight_kg = st.number_input("Weight (kg)", min_value=20,  max_value=300, value=70, step=1)
            height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=170, step=1)
            BMI = round(weight_kg / ((height_cm / 100) ** 2), 1) if height_cm > 0 else 25.0
        st.info(f"Calculated BMI: **{BMI}**")

st.markdown("---")

# ── Model selector ────────────────────────────────────────────────────────────
available_models = list(models.keys())
model_options    = ["Ensemble (All Models Combined)"] + available_models

st.markdown("**Model Selection**")
st.caption("The Ensemble option averages all available models for the most robust result.")
model_choice = st.selectbox("Which prediction model would you like to use?", model_options)

st.markdown("---")

# ── Predict button ────────────────────────────────────────────────────────────
predict_clicked = st.button("🔍 Calculate My Heart Risk")

if predict_clicked:
    features = {
        "male":           male,
        "age":            age,
        "education":      education,
        "currentSmoker":  currentSmoker,
        "cigsPerDay":     cigsPerDay,
        "BPMeds":         BPMeds,
        "prevalentStroke":prevalentStroke,
        "prevalentHyp":   prevalentHyp,
        "diabetes":       diabetes,
        "totChol":        totChol,
        "sysBP":          sysBP,
        "diaBP":          diaBP,
        "BMI":            BMI,
        "heartRate":      heartRate,
        "glucose":        glucose,
    }

    with st.spinner("Analysing your risk factors…"):
        if model_choice == "Ensemble (All Models Combined)":
            prob = ensemble_proba(features)
            model_used = "Ensemble"
        else:
            prob = predict_proba(model_choice, features)
            model_used = model_choice

    tier, css_class, icon, color = get_risk_tier(prob)
    pct = round(prob * 100, 1)

    # ── Results layout ──────────────────────────────────────────────────
    st.markdown("## Your Results")
    res_col, detail_col = st.columns([1, 2])

    with res_col:
        st.markdown(
            f'<div class="result-box">'
            f'<div style="font-size:3rem; line-height:1;">{icon}</div>'
            f'<div class="prob-number" style="color:{color};">{pct}%</div>'
            f'<div class="prob-label">Estimated 10-year CHD probability</div>'
            f'<br><span class="{css_class}">{tier}</span>'
            f'<br><br><span style="font-size:0.78rem;color:#adb5bd;">Model: {model_used}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with detail_col:
        st.markdown("#### What does this mean?")

        # Probability bar
        bar_col, label_col = st.columns([4, 1])
        with bar_col:
            st.progress(min(prob, 1.0))
        with label_col:
            st.markdown(f"**{pct}%**")

        # Tier explanation
        tier_info = {
            "Low Risk": (
                "Your combination of risk factors suggests a **low likelihood** of developing "
                "coronary heart disease in the next 10 years. Continue your current healthy habits."
            ),
            "Moderate Risk": (
                "You have a **moderate chance** of developing coronary heart disease over the next "
                "10 years. Talk to your doctor about steps you can take to lower this risk."
            ),
            "High Risk": (
                "Your risk factors indicate a **higher-than-average risk** of heart disease. "
                "It's important to speak with a healthcare provider soon to discuss prevention strategies."
            ),
            "Very High Risk": (
                "Your profile suggests a **significantly elevated risk** of coronary heart disease. "
                "Please consult a doctor as soon as possible to discuss immediate preventive care."
            ),
        }
        st.info(tier_info[tier])

        # Key risk drivers (plain-language summary)
        st.markdown("#### Your Key Risk Factors")
        flags = []
        if age >= 55:           flags.append("🔴 Age 55 or older — risk increases with age")
        if sysBP >= 140:        flags.append("🔴 High systolic blood pressure (≥ 140 mmHg)")
        elif sysBP >= 130:      flags.append("🟡 Elevated systolic blood pressure (≥ 130 mmHg)")
        if totChol >= 240:      flags.append("🔴 High total cholesterol (≥ 240 mg/dL)")
        elif totChol >= 200:    flags.append("🟡 Borderline-high total cholesterol (≥ 200 mg/dL)")
        if BMI >= 30:           flags.append("🔴 Obesity (BMI ≥ 30)")
        elif BMI >= 25:         flags.append("🟡 Overweight (BMI 25–29.9)")
        if glucose >= 126:      flags.append("🔴 High fasting glucose (≥ 126 mg/dL — diabetes range)")
        elif glucose >= 100:    flags.append("🟡 Elevated fasting glucose (pre-diabetes range)")
        if currentSmoker:       flags.append("🔴 Current smoker")
        if diabetes:            flags.append("🔴 Diabetes diagnosis")
        if prevalentHyp:        flags.append("🔴 Diagnosed with hypertension")
        if prevalentStroke:     flags.append("🔴 Prior stroke history")
        if BPMeds:              flags.append("🟡 Currently on blood pressure medication")
        if male:                flags.append("🟡 Male sex (statistically higher baseline risk)")

        if flags:
            for f in flags:
                st.markdown(f"- {f}")
        else:
            st.markdown("✅ No major individual risk flags identified — your risk is driven by the combination of your values.")

    # Model breakdown (if ensemble)
    if model_choice == "Ensemble (All Models Combined)" and len(models) > 1:
        with st.expander("📊 Individual Model Breakdown"):
            breakdown_data = {}
            for name in models:
                p = predict_proba(name, features)
                tier_name, _, _, col = get_risk_tier(p)
                breakdown_data[name] = {"Probability": f"{p*100:.1f}%", "Risk Tier": tier_name}
            st.table(pd.DataFrame(breakdown_data).T)

    # Disclaimer
    st.markdown(
        '<div class="disclaimer">'
        '<strong>⚕️ Medical Disclaimer:</strong> This tool is for informational and educational purposes only. '
        'It is based on statistical patterns from the Framingham Heart Study and is <strong>not a medical diagnosis</strong>. '
        'Always consult a qualified healthcare provider for personalised medical advice, diagnosis, or treatment. '
        'Do not make medical decisions based solely on this tool.'
        '</div>',
        unsafe_allow_html=True,
    )

elif not predict_clicked:
    st.markdown(
        '<div style="background:#e8f4fd;border-radius:10px;padding:20px;text-align:center;color:#0d6efd;">'
        '👆 Fill in all fields above and click <strong>Calculate My Heart Risk</strong> to see your results.'
        '</div>',
        unsafe_allow_html=True,
    )
