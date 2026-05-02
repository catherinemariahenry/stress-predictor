import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Stress Predictor", layout="wide")

# ---------- UI ----------
st.markdown("""
<style>
.main {background-color: #f5efe6;}
.block-container {padding-top: 2rem;}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}

.center-card {
    text-align:center;
    padding:30px;
    border-radius:15px;
    background:white;
    font-size:28px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center;'>🎈 Stress Predictor</h1>
<p style='text-align:center; color:gray;'>AI-powered lifestyle stress analysis</p>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
model = joblib.load(os.path.join(os.path.dirname(__file__), "model.pkl"))

# ---------- NAV ----------
step = st.radio("Navigate", ["Personal", "Lifestyle", "Health", "Result"], horizontal=True)

if "inputs" not in st.session_state:
    st.session_state.inputs = {}

# ---------- STEP 1 ----------
if step == "Personal":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👤 Personal Info")

    st.session_state.inputs["age"] = st.slider("Age", 10, 70, 25)
    st.session_state.inputs["gender"] = st.selectbox("Gender", ["Male", "Female"])
    st.session_state.inputs["occupation"] = st.selectbox("Occupation", [
        "Doctor","Freelancer","Manager",
        "Researcher","Software Engineer",
        "Student","Teacher"
    ])
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- STEP 2 ----------
if step == "Lifestyle":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📱 Lifestyle")

    st.session_state.inputs["screen"] = st.slider("Screen Time (hrs)", 0.0, 15.0, 5.0)
    st.session_state.inputs["phone"] = st.slider("Phone Before Sleep (hrs)", 0.0, 3.0, 1.0)
    st.session_state.inputs["sleep"] = st.slider("Sleep Duration (hrs)", 0.0, 12.0, 7.0)
    st.session_state.inputs["sleep_quality"] = st.slider("Sleep Quality", 0, 100, 70)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- STEP 3 ----------
if step == "Health":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("⚡ Health")

    st.session_state.inputs["caffeine"] = st.slider("Caffeine (cups)", 0, 10, 2)
    st.session_state.inputs["activity"] = st.slider("Physical Activity (hrs)", 0.0, 3.0, 1.0)
    st.session_state.inputs["notifications"] = st.slider("Notifications/day", 0, 300, 50)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- STEP 4 ----------
if step == "Result":

    data = st.session_state.inputs

    if len(data) < 9:
        st.warning("⚠️ Complete all previous steps")
        st.stop()

    # ---------- PROCESS ----------
    gender_val = 0 if data["gender"] == "Male" else 1
    phone_minutes = data["phone"] * 60
    activity_minutes = data["activity"] * 60

    occ = {col:0 for col in [
        'occupation_Doctor','occupation_Freelancer','occupation_Manager',
        'occupation_Researcher','occupation_Software Engineer',
        'occupation_Student','occupation_Teacher'
    ]}
    occ[f'occupation_{data["occupation"]}'] = 1

    input_data = np.array([[
        data["age"], gender_val, data["screen"], phone_minutes,
        data["sleep"], data["sleep_quality"],
        data["caffeine"], activity_minutes,
        data["notifications"],
        occ['occupation_Doctor'], occ['occupation_Freelancer'],
        occ['occupation_Manager'], occ['occupation_Researcher'],
        occ['occupation_Software Engineer'],
        occ['occupation_Student'], occ['occupation_Teacher']
    ]])

    pred = model.predict(input_data)[0]

    # ---------- GAUGE ----------
    st.subheader("🎯 Stress Level")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        gauge={
            'axis': {'range': [0, 10]},
            'steps': [
                {'range': [0, 3], 'color': "#d8f3dc"},
                {'range': [3, 6], 'color': "#fff3bf"},
                {'range': [6, 10], 'color': "#ffccd5"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"<div class='center-card'>Stress Score: {pred:.2f}</div>", unsafe_allow_html=True)

    # ---------- METRICS ----------
    st.subheader("📊 Quick Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Screen", f"{data['screen']} hrs")
    m2.metric("Sleep", f"{data['sleep']} hrs")
    m3.metric("Activity", f"{data['activity']} hrs")

    # ---------- BIGGEST ISSUE ----------
    issues = {
        "Screen Overuse": data["screen"],
        "Sleep Deficit": 8 - data["sleep"],
        "Phone Usage": data["phone"]
    }
    main_issue = max(issues, key=issues.get)
    st.error(f"⚠️ Biggest Issue: {main_issue}")

    # ---------- WHAT-IF ----------
    st.subheader("🔄 What if you improve sleep?")
    improved_sleep = st.slider("Try better sleep (hrs)", 0.0, 12.0, data["sleep"])

    temp = input_data.copy()
    temp[0][4] = improved_sleep

    new_pred = model.predict(temp)[0]
    st.write(f"New Stress Score: {new_pred:.2f}")

    # ---------- CHART ----------
    st.subheader("📉 Risk Breakdown")
    risk = pd.DataFrame({
        "Factor": ["Screen", "Sleep Deficit", "Phone", "Activity", "Caffeine"],
        "Risk": [
            data["screen"]/10,
            max(0, 8-data["sleep"]),
            data["phone"],
            max(0, 1-data["activity"]),
            data["caffeine"]/5
        ]
    })

    fig2 = px.bar(risk, x="Factor", y="Risk", color="Risk",
                  color_continuous_scale="Reds")
    st.plotly_chart(fig2, use_container_width=True)

    # ---------- AI EXPLANATION ----------
    st.subheader("🧠 AI Explanation")
    st.write(f"""
Your stress level is influenced by:
- Screen time: {data['screen']} hrs  
- Sleep: {data['sleep']} hrs  
- Activity: {data['activity']} hrs  

Improving sleep and reducing screen exposure will significantly reduce stress.
""")

    # ---------- ROUTINE ----------
    st.subheader("📅 Suggested Routine")
    if pred > 6:
        st.write("• Sleep before 11 PM")
        st.write("• Reduce screen after 8 PM")
        st.write("• Exercise 30 mins daily")
    else:
        st.write("• Maintain current habits")
        st.write("• Keep sleep consistent")
        st.write("• Stay active")

    # ---------- RECOMMENDATIONS ----------
    st.subheader("💡 Recommendations")

    if data["screen"] > 8:
        st.info("📱 Reduce screen time")
    if data["sleep"] < 6:
        st.info("😴 Improve sleep duration")
    if data["phone"] > 1:
        st.info("📵 Avoid phone before sleep")
    if data["activity"] < 0.5:
        st.info("🏃 Increase activity")
    if data["caffeine"] > 4:
        st.info("☕ Reduce caffeine")

    if data["screen"] <= 8 and data["sleep"] >= 6:
        st.success("Great habits! Keep it up 🎉")
