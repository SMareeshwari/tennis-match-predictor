import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="🎾 Tennis Match Predictor",
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

h1, h2, h3 {
    color: #00ffcc;
    text-align: center;
}

.card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    margin-bottom: 25px;
    animation: fadeIn 0.8s ease-in-out;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

.glow {
    animation: glow 2s infinite;
}

@keyframes glow {
    0% { box-shadow: 0 0 5px #00ffcc; }
    50% { box-shadow: 0 0 25px #00ffcc; }
    100% { box-shadow: 0 0 5px #00ffcc; }
}

.stButton > button {
    background: linear-gradient(90deg, #00ffcc, #00ccff);
    color: black;
    border-radius: 30px;
    padding: 14px 34px;
    font-weight: bold;
    border: none;
    transition: 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.08);
}

[data-testid="stMetricValue"] {
    font-size: 40px;
    color: #00ffcc;
}

footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL (SAFE PATH)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join("models", "random_forest.pkl")


if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found")
    st.stop()

model = joblib.load(MODEL_PATH)


# ===============================
# TITLE
# ===============================
st.markdown("""
<h1>🎾 ATP Tennis Match Predictor</h1>
<p style="text-align:center; color:#ccc; font-size:18px;">
AI-powered tennis match outcome prediction
</p>
""", unsafe_allow_html=True)

# ===============================
# INPUT SECTION
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

    prob = model.predict_proba(X)[0][1]

    # ===============================
    # RESULT CARD
    # ===============================
    st.markdown('<div class="card glow">', unsafe_allow_html=True)
    st.subheader("🏆 Prediction Result")
    st.metric("Win Probability for Player A", f"{prob*100:.2f}%")
    st.progress(prob)

    # 🎉 CONFETTI
    if prob >= 0.70:
        st.success("🔥 Strong favorite! Celebration time!")
        st.balloons()
    elif prob <= 0.30:
        st.warning("⚡ Potential upset incoming!")

    st.markdown('</div>', unsafe_allow_html=True)

    # ===============================
    # GAUGE CHART
    # ===============================
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={"text": "Win Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#00ffcc"},
            "steps": [
                {"range": [0, 50], "color": "#ff6b6b"},
                {"range": [50, 70], "color": "#feca57"},
                {"range": [70, 100], "color": "#1dd1a1"}
            ],
        }
    ))

    st.plotly_chart(fig, width="stretch")

    # ===============================
    # EXPLANATION
    # ===============================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 Why this prediction?")
    st.write(f"""
    • **ELO difference:** {winner_elo - loser_elo}  
    • **Ranking difference:** {loser_rank - winner_rank}  
    • **Match format:** Best of {best_of}
    """)
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
<li>Match format</li>
<li>Random Forest machine learning</li>
</ul>
</div>
""", unsafe_allow_html=True)
