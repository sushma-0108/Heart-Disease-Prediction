import streamlit as st
import joblib
import numpy as np

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Nunito:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #f7f4f0 !important;
    color: #1e1a16 !important;
    font-family: 'Nunito', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 70% 40% at 95% 5%, rgba(192,57,43,0.08) 0%, transparent 55%),
        radial-gradient(ellipse 50% 35% at 5% 95%, rgba(192,57,43,0.06) 0%, transparent 55%),
        #f7f4f0 !important;
}

[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }

.block-container {
    max-width: 1080px !important;
    padding: 2.5rem 2rem 4rem !important;
    margin: auto;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #c0392b;
    background: rgba(192,57,43,0.09);
    border: 1px solid rgba(192,57,43,0.2);
    border-radius: 100px;
    padding: 0.38rem 1.1rem;
    margin-bottom: 1.3rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: clamp(2.2rem, 4.5vw, 3.6rem) !important;
    font-weight: 800 !important;
    line-height: 1.1 !important;
    letter-spacing: -0.02em !important;
    color: #1e1a16 !important;
    margin-bottom: 0.75rem !important;
}
.hero h1 em {
    font-style: normal;
    color: #c0392b;
}
.hero p {
    font-size: 1rem;
    font-weight: 400;
    color: #6b6358;
    max-width: 460px;
    margin: 0 auto;
    line-height: 1.7;
}

/* ── Rule ── */
.ruled {
    border: none;
    border-top: 1.5px solid rgba(30,26,22,0.08);
    margin: 2rem 0;
}

/* ── Section heading ── */
.sec-head {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-weight: 700;
    color: #1e1a16;
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.sec-head span {
    display: inline-block;
    width: 28px; height: 28px;
    border-radius: 8px;
    background: #c0392b;
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    text-align: center;
    line-height: 28px;
    font-family: 'Nunito', sans-serif;
}

/* ── Card ── */
.card {
    background: #ffffff;
    border: 1.5px solid rgba(30,26,22,0.07);
    border-radius: 18px;
    padding: 1.7rem 1.7rem 1.4rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 20px rgba(30,26,22,0.05);
    transition: box-shadow 0.2s, border-color 0.2s;
}
.card:hover {
    box-shadow: 0 6px 30px rgba(192,57,43,0.1);
    border-color: rgba(192,57,43,0.2);
}

/* ── Input labels ── */
label,
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label {
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #8c7f74 !important;
}

/* Number inputs */
[data-testid="stNumberInput"] input {
    background: #faf9f7 !important;
    border: 1.5px solid rgba(30,26,22,0.12) !important;
    border-radius: 10px !important;
    color: #1e1a16 !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    padding: 0.6rem 0.9rem !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #c0392b !important;
    box-shadow: 0 0 0 3px rgba(192,57,43,0.1) !important;
    outline: none !important;
    background: #fff !important;
}
[data-testid="stNumberInput"] button {
    background: #f0ede9 !important;
    border: none !important;
    border-radius: 8px !important;
    color: #1e1a16 !important;
}
[data-testid="stNumberInput"] button:hover {
    background: rgba(192,57,43,0.1) !important;
}

/* Selectboxes */
[data-testid="stSelectbox"] > div > div {
    background: #faf9f7 !important;
    border: 1.5px solid rgba(30,26,22,0.12) !important;
    border-radius: 10px !important;
    color: #1e1a16 !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #c0392b !important;
    box-shadow: 0 0 0 3px rgba(192,57,43,0.1) !important;
    background: #fff !important;
}
[data-testid="stSelectbox"] svg { fill: #c0392b !important; }

/* ── Summary chips ── */
.chip-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin: 0.5rem 0 1.2rem;
}
.chip {
    flex: 1;
    min-width: 110px;
    background: #fdf2f0;
    border: 1.5px solid rgba(192,57,43,0.18);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    text-align: center;
}
.chip .clabel {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #b08070;
    margin-bottom: 0.25rem;
}
.chip .cvalue {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 800;
    color: #c0392b;
}

/* ── Predict button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #c0392b, #e74c3c) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.85rem 2rem !important;
    box-shadow: 0 4px 20px rgba(192,57,43,0.3) !important;
    transition: all 0.2s !important;
    margin-top: 0.4rem !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(192,57,43,0.45) !important;
    background: linear-gradient(135deg, #a93226, #c0392b) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 14px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 1.2rem 1.5rem !important;
    margin-top: 1rem !important;
}

/* ── Disclaimer ── */
.disclaimer {
    background: #faf9f7;
    border: 1.5px solid rgba(30,26,22,0.08);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-top: 1rem;
    font-size: 0.78rem;
    color: #8c7f74;
    line-height: 1.7;
}
.disclaimer strong { color: #6b6358; }

/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 3.5rem;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    color: #b0a498;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f7f4f0; }
::-webkit-scrollbar-thumb { background: rgba(192,57,43,0.25); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    m = joblib.load("model.pkl")
    s = joblib.load("scaler.pkl")
    return m, s

try:
    model, scaler = load_artifacts()
    artifacts_ok = True
except Exception as e:
    artifacts_ok = False
    st.warning(f"⚠️ Could not load model/scaler: {e}")

# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">🫀 Clinical AI Decision Support</div>
    <h1>Heart Disease<br><em>Prediction System</em></h1>
    <p>Enter patient clinical parameters to receive an instant AI-powered cardiac risk assessment.</p>
</div>
<hr class="ruled">
""", unsafe_allow_html=True)

# ─── Layout ────────────────────────────────────────────────────────────────────
col_l, col_r = st.columns([1.05, 1], gap="large")

with col_l:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-head"><span>1</span> Patient Vitals</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
    with c2:
        resting_bp = st.number_input("Resting BP (mmHg)", min_value=0, max_value=300, value=130)

    c3, c4 = st.columns(2)
    with c3:
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=700, value=200)
    with c4:
        max_hr = st.number_input("Max Heart Rate (bpm)", min_value=0, max_value=250, value=150)

    c5, c6 = st.columns(2)
    with c5:
        oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0,
                                  value=1.0, step=0.1, format="%.1f")
    with c6:
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120?", ["No", "Yes"])

    st.markdown('</div>', unsafe_allow_html=True)

with col_r:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-head"><span>2</span> Clinical Findings</div>', unsafe_allow_html=True)

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ASY — Asymptomatic", "ATA — Atypical Angina",
         "NAP — Non-Anginal Pain", "TA — Typical Angina"],
    )
    rest_ecg = st.selectbox(
        "Resting ECG Result",
        ["Normal", "LVH — Left Ventricular Hypertrophy", "ST — ST-T Wave Abnormality"],
    )
    st_slope = st.selectbox(
        "ST Slope (Exercise)",
        ["Up — Upsloping", "Flat — Flat", "Down — Downsloping"],
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Live summary chips
    st.markdown(f"""
    <div class="chip-row">
        <div class="chip">
            <div class="clabel">Age</div>
            <div class="cvalue">{age}</div>
        </div>
        <div class="chip">
            <div class="clabel">BP</div>
            <div class="cvalue">{int(resting_bp)}</div>
        </div>
        <div class="chip">
            <div class="clabel">Max HR</div>
            <div class="cvalue">{int(max_hr)}</div>
        </div>
        <div class="chip">
            <div class="clabel">Oldpeak</div>
            <div class="cvalue">{oldpeak:.1f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    predict_clicked = st.button("🫀 Predict Heart Disease Risk")

# ─── Prediction ────────────────────────────────────────────────────────────────
st.markdown('<hr class="ruled">', unsafe_allow_html=True)

if predict_clicked:
    if not artifacts_ok:
        st.error("Cannot run prediction — model.pkl or scaler.pkl is missing.")
    else:
        # Encode binary
        fasting_bs_val = 1 if fasting_bs == "Yes" else 0

        # Chest pain one-hot
        cp_key = chest_pain.split(" — ")[0]
        cp_asy = 1 if cp_key == "ASY" else 0
        cp_ata = 1 if cp_key == "ATA" else 0
        cp_nap = 1 if cp_key == "NAP" else 0
        cp_ta  = 1 if cp_key == "TA"  else 0

        # ECG one-hot
        ecg_key    = rest_ecg.split(" — ")[0]
        ecg_lvh    = 1 if ecg_key == "LVH"    else 0
        ecg_normal = 1 if ecg_key == "Normal" else 0
        ecg_st     = 1 if ecg_key == "ST"     else 0

        # ST slope binary
        slope_key    = st_slope.split(" — ")[0]
        st_slope_val = 1 if slope_key == "Up" else 0

        # Scale 5 numeric features: Age, RestingBP, Cholesterol, MaxHR, Oldpeak
        numeric_raw    = np.array([[age, resting_bp, cholesterol, max_hr, oldpeak]])
        numeric_scaled = scaler.transform(numeric_raw)[0]

        # Build 14-feature vector matching training order:
        # Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak,
        # ST_Slope, CP_ASY, CP_ATA, CP_NAP, CP_TA,
        # ECG_LVH, ECG_Normal, ECG_ST
        data = np.array([[
            numeric_scaled[0],          # Age (scaled)
            numeric_scaled[1],          # RestingBP (scaled)
            numeric_scaled[2],          # Cholesterol (scaled)
            fasting_bs_val,             # FastingBS
            numeric_scaled[3],          # MaxHR (scaled)
            numeric_scaled[4],          # Oldpeak (scaled)
            st_slope_val,               # ST_Slope
            cp_asy, cp_ata, cp_nap, cp_ta,
            ecg_lvh, ecg_normal, ecg_st,
        ]])

        prediction = model.predict(data)
        prob = None
        try:
            prob = model.predict_proba(data)[0][1]
        except Exception:
            pass

        if prediction[0] == 1:
            prob_text = f" — Confidence: {prob*100:.1f}%" if prob is not None else ""
            st.error(
                f"🚨 **High Risk of Heart Disease Detected{prob_text}** — "
                "The model indicates an elevated probability of heart disease. "
                "Please consult a cardiologist for a comprehensive evaluation."
            )
        else:
            prob_text = f" — Confidence: {(1-prob)*100:.1f}%" if prob is not None else ""
            st.success(
                f"✅ **Low Risk of Heart Disease{prob_text}** — "
                "Based on the provided parameters, the model indicates a low likelihood "
                "of heart disease. Continue maintaining a heart-healthy lifestyle."
            )

        st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>Medical Disclaimer:</strong> This tool is intended for educational and
            research purposes only. It is not a substitute for professional medical advice,
            diagnosis, or treatment. Always consult a qualified healthcare provider.
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Heart Disease Prediction System &nbsp;·&nbsp; '
    'Powered by Machine Learning &nbsp;·&nbsp; Not a medical device</div>',
    unsafe_allow_html=True
)
