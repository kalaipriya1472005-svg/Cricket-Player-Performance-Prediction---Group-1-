import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="IPL Player Performance", layout="wide")

# =============================
# LOAD MODEL ONLY (NO PIPELINE)
# =============================
@st.cache_resource
def load_model():
    return joblib.load("models/xgb_model.joblib")

model = load_model()

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    return pd.read_csv("ipl_cleaned_data.csv")

data = load_data()

# =============================
# TITLE
# =============================
st.markdown("<h1 style='text-align:center;'>🏏 IPL Player Performance Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Submission Ready • ML Based Prediction</p>", unsafe_allow_html=True)
st.divider()

# =============================
# SIDEBAR
# =============================
st.sidebar.header("🔍 Match Inputs")

player = st.sidebar.selectbox("Select Player", sorted(data["batter"].unique()))
venue = st.sidebar.selectbox("Select Venue", sorted(data["venue"].dropna().unique()))
predict_btn = st.sidebar.button("🎯 Predict Runs")

# =============================
# PLAYER DATA
# =============================
pdata = data[data["batter"] == player]

# =============================
# FEATURE VECTOR (EXACTLY 12)
# =============================
def build_feature_vector(player, venue):
    pdata = data[data["batter"] == player]

    features = np.array([[
        pdata["runs_batter"].tail(5).mean(),          # batting_form
        pdata[pdata["venue"] == venue]["runs_batter"].mean(),  # runs_at_venue
        pdata["runs_batter"].mean(),                  # runs_vs_opponent (proxy)
        pdata["runs_batter"].mean(),                  # runs_batter
        pdata["runs_batter"].sum(),                   # career_runs
        len(pdata),                                   # balls_faced
        0,                                            # wickets
        0,                                            # career_wickets
        0,                                            # wickets_at_venue
        0,                                            # wickets_vs_opponent
        0,                                            # bowling_form
        0                                             # runs_conceded
    ]])

    return features

# =============================
# PLAYER SUMMARY
# =============================
total_runs = int(pdata["runs_batter"].sum())
balls = len(pdata)
matches = pdata["match_number"].nunique()
avg = round(total_runs / max(matches, 1), 2)
sr = round((total_runs / max(balls, 1)) * 100, 2)

c1, c2, c3, c4 = st.columns(4)
c1.metric("🏏 Total Runs", total_runs)
c2.metric("⚾ Balls Faced", balls)
c3.metric("📊 Average", avg)
c4.metric("🚀 Strike Rate", sr)

st.divider()

# =============================
# MINI CHARTS
# =============================
g1, g2, g3 = st.columns(3)

with g1:
    st.subheader("📈 Last 10 Innings")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(pdata["runs_batter"].tail(10).values, marker="o")
    ax.set_xticks([])
    st.pyplot(fig)

with g2:
    st.subheader("🏟 Venue Avg")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["Venue"], [pdata[pdata["venue"] == venue]["runs_batter"].mean()])
    st.pyplot(fig)

with g3:
    st.subheader("📅 Season Trend")
    season_runs = pdata.groupby("season")["runs_batter"].sum()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(season_runs.index, season_runs.values, marker="o")
    st.pyplot(fig)

# =============================
# ML PREDICTION (SAFE)
# =============================
if predict_btn:
    X = build_feature_vector(player, venue)
    predicted_runs = int(model.predict(X)[0])

    st.divider()
    st.subheader("🎯 Predicted Performance")
    st.metric("Expected Runs", predicted_runs)

# =============================
# FOOTER
# =============================
st.markdown("<p style='text-align:center;color:gray;'>Streamlit • XGBoost • IPL Analytics</p>", unsafe_allow_html=True)
