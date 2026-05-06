import streamlit as st
import joblib
import numpy as np

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HEART DISEASE PREDICTION SYSTEM",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #e8e6f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(220,38,38,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(220,38,38,0.10) 0%, transparent 55%),
        #0a0a0f !important;
}

[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.block-container { max-width: 1100px !important; padding: 3rem 2rem 4rem !important; margin: auto; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3.5rem 0 2.5rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #ef4444;
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 100px;
    padding: 0.35rem 1rem;
    margin-bottom: 1.4rem;
    background: rgba(239,68,68,0.06);
}
.hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(2.6rem, 5vw, 4rem) !important;
    font-weight: 800 !important;
    line-height: 1.05 !important;
    letter-spacing: -0.03em !important;
    color: #f5f3ff !important;
    margin-bottom: 0.8rem !important;
}
.hero h1 span { color: #ef4444; }
.hero p {
    font-size: 1.05rem;
    font-weight: 300;
    color: rgba(232,230,240,0.55);
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.65;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(239,68,68,0.3), transparent);
    margin: 2rem 0;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(239,68,68,0.7);
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(239,68,68,0.15);
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.8rem 1.8rem 1.4rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(12px);
    transition: border-color 0.2s;
}
.card:hover { border-color: rgba(239,68,68,0.25); }

/* ── Streamlit widget overrides ── */
label, .stSelectbox label, .stNumberInput label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    color: rgba(232,230,240,0.6) !important;
    text-transform: uppercase !important;
    margin-bottom: 0.3rem !important;
}

/* Number inputs */
[data-testid="stNumberInput"] input,
input[type="number"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #f0eeff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    padding: 0.65rem 1rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: rgba(239,68,68,0.6) !important;
    box-shadow: 0 0 0 3px rgba(239,68,68,0.12) !important;
    outline: none !important;
}
[data-testid="stNumberInput"] button {
    background: rgba(255,255,255,0.06) !important;
    border: none !important;
    color: #e8e6f0 !important;
    border-radius: 8px !important;
}

/* Selectboxes */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #f0eeff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: rgba(239,68,68,0.6) !important;
    box-shadow: 0 0 0 3px rgba(239,68,68,0.12) !important;
}
[data-testid="stSelectbox"] svg { fill: rgba(239,68,68,0.7) !important; }

/* Dropdown menu */
[data-testid="stSelectboxVirtualDropdown"] {
    background: #1a1825 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
}
[data-testid="stSelectboxVirtualDropdown"] li {
    color: #e8e6f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSelectboxVirtualDropdown"] li:hover {
    background: rgba(239,68,68,0.15) !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #dc2626, #ef4444) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.85rem 2rem !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 24px rgba(220,38,38,0.35) !important;
    margin-top: 0.5rem !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(220,38,38,0.5) !important;
    background: linear-gradient(135deg, #b91c1c, #dc2626) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Result alerts ── */
[data-testid="stAlert"] {
    border-radius: 14px !important;
    border-left-width: 4px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    margin-top: 1rem !important;
    padding: 1.2rem 1.5rem !important;
}

/* ── Metrics row ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.metric-chip {
    flex: 1;
    min-width: 140px;
    background: rgba(239,68,68,0.07);
    border: 1px solid rgba(239,68,68,0.18);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-chip .label {
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(232,230,240,0.45);
    margin-bottom: 0.3rem;
}
.metric-chip .value {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: #ef4444;
}

/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 3rem;
    color: rgba(232,230,240,0.2);
    font-size: 0.75rem;
    letter-spacing: 0.06em;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: rgba(239,68,68,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🫀 AI-Powered Cardiac Risk Assessment</div>
    <h1>Cardio<span>Scan</span> AI</h1>
    <p>Enter your clinical parameters below for an instant, data-driven heart disease risk prediction.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─── Model load ────────────────────────────────────────────────────────────────
try:
    model = joblib.load("model.pkl")
    model_loaded = True
except Exception:
    model_loaded = False
    st.warning("⚠️ model.pkl not found. Predictions are disabled until the model file is present.")

# ─── Layout ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.05, 1], gap="large")

with col_left:
    # ── Vitals card ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Patient Vitals</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (yrs)", min_value=1, max_value=120, value=45)
    with c2:
        resting_bp = st.number_input("Resting BP (mmHg)", min_value=0, value=120)

    c3, c4 = st.columns(2)
    with c3:
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, value=200)
    with c4:
        max_hr = st.number_input("Max Heart Rate (bpm)", min_value=0, value=150)

    c5, c6 = st.columns(2)
    with c5:
        oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, value=1.0, step=0.1, format="%.1f")
    with c6:
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120?", ["No", "Yes"])

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    # ── Clinical findings card ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Clinical Findings</div>', unsafe_allow_html=True)

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ASY — Asymptomatic", "ATA — Atypical Angina", "NAP — Non-Anginal Pain", "TA — Typical Angina"],
    )
    rest_ecg = st.selectbox(
        "Resting ECG Result",
        ["Normal", "LVH — Left Ventricular Hypertrophy", "ST — ST-T Wave Abnormality"],
    )
    st_slope = st.selectbox(
        "ST Slope",
        ["Up — Upsloping", "Flat — Flat", "Down — Downsloping"],
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Quick summary chips ──
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-chip">
            <div class="label">Age</div>
            <div class="value">{age}</div>
        </div>
        <div class="metric-chip">
            <div class="label">BP</div>
            <div class="value">{int(resting_bp)}</div>
        </div>
        <div class="metric-chip">
            <div class="label">HR Max</div>
            <div class="value">{int(max_hr)}</div>
        </div>
        <div class="metric-chip">
            <div class="label">Oldpeak</div>
            <div class="value">{oldpeak:.1f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Predict button ──
    predict_clicked = st.button("⚡ Run Cardiac Risk Prediction")

# ─── Prediction logic ──────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

if predict_clicked:
    if not model_loaded:
        st.error("Cannot run prediction — model.pkl is missing.")
    else:
        fasting_bs_val = 1 if fasting_bs == "Yes" else 0

        cp_key = chest_pain.split(" — ")[0]
        cp_asy = 1 if cp_key == "ASY" else 0
        cp_ata = 1 if cp_key == "ATA" else 0
        cp_nap = 1 if cp_key == "NAP" else 0
        cp_ta  = 1 if cp_key == "TA"  else 0

        ecg_key    = rest_ecg.split(" — ")[0]
        ecg_lvh    = 1 if ecg_key == "LVH"    else 0
        ecg_normal = 1 if ecg_key == "Normal" else 0
        ecg_st     = 1 if ecg_key == "ST"     else 0

        slope_key   = st_slope.split(" — ")[0]
        st_slope_val = 1 if slope_key == "Up" else 0

        data = np.array([[
            age, resting_bp, cholesterol, fasting_bs_val, max_hr, oldpeak,
            st_slope_val,
            cp_asy, cp_ata, cp_nap, cp_ta,
            ecg_lvh, ecg_normal, ecg_st,
        ]])

        prediction = model.predict(data)

        if prediction[0] == 1:
            st.error(
                "🚨 **High Risk Detected** — The model indicates an elevated probability of heart disease. "
                "Please consult a cardiologist for a thorough evaluation."
            )
        else:
            st.success(
                "✅ **Low Risk** — Based on the provided parameters, the model indicates a low likelihood "
                "of heart disease. Continue maintaining a heart-healthy lifestyle."
            )

        st.markdown("""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
             border-radius:12px;padding:1rem 1.4rem;margin-top:1rem;
             font-size:0.78rem;color:rgba(232,230,240,0.4);line-height:1.7;">
        ⚠️ <strong style="color:rgba(232,230,240,0.55)">Disclaimer:</strong>
        This tool is for educational and research purposes only. It is not a substitute for
        professional medical advice, diagnosis, or treatment.
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="footer">CardioScan AI &nbsp;·&nbsp; Powered by Machine Learning &nbsp;·&nbsp; Not a medical device</div>', unsafe_allow_html=True)
