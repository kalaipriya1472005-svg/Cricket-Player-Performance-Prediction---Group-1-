import streamlit as st
import pandas as pd

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="IPL Performance Predictor",
    page_icon="🏏",
    layout="wide"
)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 🧠 IPL ML Dashboard")
    st.markdown("**Cricket Player Performance Prediction**")
    st.markdown("---")
    st.markdown("👨‍💻 Developed by **Kalaipriya K M**")
    st.markdown("🎯 Role: Data Science / ML Engineer")
    st.markdown("📍 Project: End-to-End ML App")
    st.markdown("---")
    st.info("This dashboard predicts player performance using machine learning based on match statistics.")

# ---------- MAIN HEADER ----------
st.markdown(
    """
    <h1 style='text-align: center;'>🏏 IPL Player Performance Predictor</h1>
    <p style='text-align: center; font-size:18px;'>
    A Machine Learning powered system to evaluate player match performance
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------- INPUT SECTION ----------
st.markdown("## 📊 Match Statistics Input")

col1, col2, col3 = st.columns(3)

with col1:
    runs = st.number_input("🏃 Runs Scored", min_value=0, max_value=200, step=1)
    balls = st.number_input("🎯 Balls Faced", min_value=0, max_value=150, step=1)

with col2:
    fours = st.number_input("4️⃣ Number of Fours", min_value=0, max_value=50, step=1)
    sixes = st.number_input("6️⃣ Number of Sixes", min_value=0, max_value=30, step=1)

with col3:
    strike_rate = st.number_input("⚡ Strike Rate", min_value=0.0, max_value=300.0, step=0.1)

st.markdown("---")

# ---------- PREDICTION BUTTON ----------
predict_btn = st.button("🚀 Predict Player Performance", use_container_width=True)

if predict_btn:
    st.markdown("## 🔍 Prediction Result")

    # TEMP LOGIC (will be replaced by ML model)
    if runs >= 75 and strike_rate >= 140:
        performance = "🔥 High Performance"
        st.success(performance)
    elif runs >= 30:
        performance = "⚖️ Average Performance"
        st.warning(performance)
    else:
        performance = "❌ Low Performance"
        st.error(performance)

    # Visualization
    df = pd.DataFrame({
        "Metric": ["Runs", "Balls", "Fours", "Sixes", "Strike Rate"],
        "Value": [runs, balls, fours, sixes, strike_rate]
    })

    st.bar_chart(df.set_index("Metric"))

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    """
    <p style='text-align: center; font-size:14px; color:gray;'>
    © 2026 | IPL Player Performance Prediction System | Built with Streamlit & Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)
