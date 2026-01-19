update in this:import streamlit as st
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

label {
    color: #c7f9ff !important;
    font-weight: 600;
}

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
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================
MODEL_PATH = "random_forest.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found.")
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
st.subheader("🎯 Match Inputs")

col1, col2 = st.columns(2)

with col1:
    winner_elo = st.slider("Player A ELO", 1200, 3000, 1600)
    winner_rank = st.slider("Player A Rank", 1, 500, 20)

with col2:
    loser_elo = st.slider("Player B ELO", 1200, 3000, 1500)
    loser_rank = st.slider("Player B Rank", 1, 500, 35)

best_of = st.selectbox("Match Format (Best of)", [3, 5])

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
    player_a_prob = probs[1] * 100
    player_b_prob = probs[0] * 100

    st.markdown(
        f"""
        <h2>🏆 Prediction Result</h2>
        <p style="color:#06d6a0;font-size:32px;font-weight:800;">
            Player A Win Probability: {player_a_prob:.2f}%
        </p>
        <p style="color:#ef476f;font-size:28px;font-weight:700;">
            Player B Win Probability: {player_b_prob:.2f}%
        </p>
        """,
        unsafe_allow_html=True
    )

    if player_a_prob > 65:
        st.success("🔥 Player A is a STRONG FAVORITE!")
    elif player_b_prob > 65:
        st.success("⚡ Player B is a STRONG FAVORITE!")
    else:
        st.info("⚖️ This match looks evenly balanced")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=player_a_prob,
        title={"text": "Player A Win Probability (%)"},
        gauge={"axis": {"range": [0, 100]}}
    ))

    st.plotly_chart(fig, use_container_width=True)

# ===============================
# FOOTER
# ===============================
st.markdown("""
<h3>📌 About</h3>
<ul>
<li>ELO ratings</li>
<li>ATP rankings</li>
<li>Random Forest ML model</li>
</ul>
""", unsafe_allow_html=True)
