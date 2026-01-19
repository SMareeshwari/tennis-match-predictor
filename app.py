import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="🎾 ATP Tennis Match Predictor",
    layout="wide"
)

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
/* ===== GLOBAL BACKGROUND ===== */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #e8f1f5;
    font-family: 'Segoe UI', sans-serif;
}

/* ===== TITLES ===== */
h1 {
    color: #00ffd5;
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
}

h2 {
    color: #ffd166;
    text-align: center;
    font-weight: 700;
}

h3 {
    color: #ff9f1c;
    font-weight: 600;
}

/* ===== SUBHEADERS / LABELS ===== */
label, .css-1cpxqw2 {
    color: #c7f9ff !important;
    font-weight: 600;
}

/* ===== SLIDER VALUE COLOR ===== */
.css-1p05t8e {
    color: #00ffab !important;
    font-weight: bold;
}

/* ===== BUTTON ===== */
.stButton>button {
    background: linear-gradient(90deg, #00f5d4, #00bbf9);
    color: black;
    font-size: 18px;
    padding: 0.6em 2em;
    border-radius: 30px;
    font-weight: 700;
    border: none;
    box-shadow: 0px 0px 15px rgba(0,255,213,0.6);
    transition: all 0.3s ease;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #ffd166, #ef476f);
    color: black;
    transform: scale(1.05);
}

/* ===== METRICS / PROBABILITY TEXT ===== */
.metric-value {
    color: #06d6a0 !important;
    font-size: 36px !important;
    font-weight: 800;
}

/* ===== SUCCESS MESSAGE ===== */
.stAlert-success {
    background: linear-gradient(90deg, #06d6a0, #1b9aaa);
    color: black;
    font-weight: bold;
    border-radius: 12px;
}

/* ===== PROGRESS BAR ===== */
.stProgress > div > div {
    background: linear-gradient(90deg, #00f5d4, #ffd166);
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL (ROOT PATH)
# ===============================
MODEL_PATH = "random_forest.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Make sure random_forest.pkl is in the repo root.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("❌ Failed to load model.")
    st.code(str(e))
    st.stop()
# ===============================
# TITLE
# ===============================
st.markdown("<h1>🎾 ATP Tennis Match Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h2>AI-Powered Match Outcome Prediction</h2>", unsafe_allow_html=True)


# ===============================
# INPUTS
# ===============================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🎯 Match Inputs")

col1, col2 = st.columns(2)

with col1:
    winner_elo = st.slider("Player A ELO", 1200, 3000, 1600)
    winner_rank = st.slider("Player A Rank", 1, 500, 20)

with col2:
    loser_elo = st.slider("Player B ELO", 1200, 3000, 1500)
    loser_rank = st.slider("Player B Rank", 1, 500, 35)

best_of = st.selectbox("Match Format (Best of)", [3, 5])
st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# PREDICTION
# ===============================
if st.button("🔮 Predict Match Outcome"):

    X = pd.DataFrame([{
        "winner_elo": winner_elo,
        "loser_elo": loser_elo,
        "winner_rank": winner_rank,
        "loser_rank": loser_rank,
        "rank_diff": loser_rank - winner_rank,
        "elo_diff": winner_elo - loser_elo,
        "best_of": best_of
    }])

    probs = model.predict_proba(X)[0]
    player_a_prob = probs[1]
    player_b_prob = probs[0]

    st.markdown('<div class="card">', unsafe_allow_html=True)
   st.markdown(
    f"""
    <h2>🏆 Prediction Result</h2>
    <p style="color:#06d6a0;font-size:32px;font-weight:800;">
        Player A Win Probability: {prob_a:.2f}%
    </p>
    <p style="color:#ef476f;font-size:28px;font-weight:700;">
        Player B Win Probability: {prob_b:.2f}%
    </p>
    """,
    unsafe_allow_html=True
)


    if prob_a > 65:
    st.success("🔥 Player A is a STRONG FAVORITE!")
elif prob_b > 65:
    st.success("⚡ Player B is a STRONG FAVORITE!")
else:
    st.info("⚖️ This match looks evenly balanced")


    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=player_a_prob * 100,
        title={"text": "Player A Win Probability (%)"},
        gauge={"axis": {"range": [0, 100]}}
    ))

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# FOOTER
# ===============================
st.markdown("""
<div class="card">
<h3>📌 About</h3>
<ul>
<li>ELO ratings</li>
<li>ATP rankings</li>
<li>Random Forest ML model</li>
</ul>
</div>
""", unsafe_allow_html=True)

