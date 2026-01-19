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
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #e8f1f5;
    font-family: 'Segoe UI', sans-serif;
}

/* TITLES */
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

/* GLASS CARD */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(14px);
    border-radius: 18px;
    padding: 25px;
    margin-top: 25px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(90deg, #00f5d4, #00bbf9);
    color: black;
    font-size: 18px;
    padding: 0.6em 2em;
    border-radius: 30px;
    font-weight: 700;
    border: none;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #ffd166, #ef476f);
}

/* PROBABILITY BARS */
.bar-container {
    width: 100%;
    background: rgba(255,255,255,0.15);
    border-radius: 12px;
    overflow: hidden;
    height: 22px;
    margin-top: 8px;
}
.bar-a {
    height: 100%;
    background: linear-gradient(90deg, #06d6a0, #00f5d4);
}
.bar-b {
    height: 100%;
    background: linear-gradient(90deg, #ef476f, #ff758f);
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================
MODEL_PATH = "random_forest.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. random_forest.pkl must be in repo root.")
    st.stop()

model = joblib.load(MODEL_PATH)

# ===============================
# TITLE
# ===============================
st.markdown("<h1>🎾 ATP Tennis Match Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h2>AI-Powered Match Outcome Prediction</h2>", unsafe_allow_html=True)

# ===============================
# INPUTS
# ===============================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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
    prob_a = probs[1] * 100
    prob_b = probs[0] * 100

    # ===== RESULT CARD =====
    st.markdown(
        f"""
        <div class="glass-card">
            <h2>🏆 Prediction Result</h2>

            <p style="color:#06d6a0;font-size:28px;font-weight:800;">
                Player A Win Probability: {prob_a:.2f}%
            </p>
            <div class="bar-container">
                <div class="bar-a" style="width:{prob_a}%"></div>
            </div>

            <br>

            <p style="color:#ef476f;font-size:26px;font-weight:700;">
                Player B Win Probability: {prob_b:.2f}%
            </p>
            <div class="bar-container">
                <div class="bar-b" style="width:{prob_b}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ===== MESSAGE =====
    if prob_a > 65:
        st.success("🔥 Player A is a STRONG FAVORITE!")
    elif prob_b > 65:
        st.success("⚡ Player B is a STRONG FAVORITE!")
    else:
        st.info("⚖️ This match looks evenly balanced")

    # ===== GAUGE =====
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_a,
        title={"text": "Player A Win Probability (%)"},
        gauge={"axis": {"range": [0, 100]}}
    ))

    st.plotly_chart(fig, use_container_width=True)

# ===============================
# FOOTER
# ===============================
st.markdown("""
<div class="glass-card">
<h3>📌 About This Model</h3>
<ul>
<li>ATP ELO ratings</li>
<li>ATP rankings</li>
<li>Random Forest classifier</li>
</ul>
</div>
""", unsafe_allow_html=True)
