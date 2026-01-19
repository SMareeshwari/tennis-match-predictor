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
    color: white;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3 { color: #00ffcc; text-align: center; }
.card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    margin-bottom: 25px;
}
.stButton > button {
    background: linear-gradient(90deg, #00ffcc, #00ccff);
    color: black;
    border-radius: 30px;
    padding: 14px 34px;
    font-weight: bold;
}
footer, header {visibility: hidden;}
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
st.markdown("""
<h1>🎾 ATP Tennis Match Predictor</h1>
<p style="text-align:center; color:#ccc;">
AI-powered tennis match outcome prediction
</p>
""", unsafe_allow_html=True)

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
    st.subheader("🏆 Prediction Result")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🏆 Player A Win Probability", f"{player_a_prob*100:.2f}%")
        st.progress(player_a_prob)

    with col2:
        st.metric("⚡ Player B Win Probability", f"{player_b_prob*100:.2f}%")
        st.progress(player_b_prob)

    if player_a_prob >= 0.7:
        st.success("🔥 Player A is a strong favorite!")
        st.balloons()
    elif player_a_prob <= 0.3:
        st.warning("⚡ Player B could cause an upset!")
    else:
        st.info("⚖️ This looks like a close match!")

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
