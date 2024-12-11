import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler
import random
import os

file_path = os.path.join("C:/Users/Aravind/Downloads/SIHHACK/bp", "scaler_metadata.pkl")
# Load the scaler and model metadata
metadata = joblib.load('C:/Users/Aravind/Downloads/SIHHACK/bp/scaler_metadata.pkl')
scaler = metadata['scaler']
scaler_feature_names = metadata['features']

# Load trained models
best_model_systolic = joblib.load('C:/Users/Aravind/Downloads/SIHHACK/bp/systolic_bp_model.pkl')
best_model_diastolic = joblib.load('C:/Users/Aravind/Downloads/SIHHACK/bp/diastolic_bp_model.pkl')

# Load test data for SHAP explainability
test_data = joblib.load('C:/Users/Aravind/Downloads/SIHHACK/bp/test_data.pkl')
X_test_scaled = test_data["X_test_scaled"]
X_test = test_data["X_test"]
y_test_systolic = test_data["y_test_systolic"]
y_test_diastolic = test_data["y_test_diastolic"]
# Streamlit app title and intro with styling
st.markdown("""
    <style>
        .title {
            text-align: center;
            color: #4CAF50;
            font-size: 40px;
            font-weight: bold;
            font-family: 'Arial', sans-serif;
        }
        .subtitle {
            font-size: 20px;
            text-align: center;
            color: #555;
        }
        .result-text {
            font-size: 25px;
            color: #333;
            text-align: center;
        }
        .alert-box {
            background-color: #FFEBEE;
            color: #D32F2F;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #D32F2F;
        }
        .success-box {
            background-color: #E8F5E9;
            color: #388E3C;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #388E3C;
        }
        .recommendation {
            font-size: 18px;
            color: #555;
            margin: 10px;
        }
        .gamification {
            font-size: 22px;
            text-align: center;
            font-weight: bold;
            margin-top: 20px;
        }
        .habit-section {
            font-size: 18px;
            color: #555;
            margin: 20px 0;
        }
        .habit-heading {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .habit-list {
            font-size: 16px;
            color: #555;
        }
        .habit-list li {
            margin-bottom: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

# Streamlit app header
st.markdown('<div class="title">🏋️‍♂️ Health & Blood Pressure Dashboard 🏃‍♀️</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict and Analyze Your Blood Pressure with Interactive Insights!</div>', unsafe_allow_html=True)


# User input form
st.header("Input Your Health Data")
col1, col2 = st.columns(2)  # Create two columns
with col1:
    st.metric(label="Normal Stress Level", value="5")
with col2:
    st.metric(label="Normal Cholesterol", value="180")
with st.form("user_input_form"):
    col1, col2, col3 = st.columns(3)
    

    with col1:
        age = st.slider("Age", 18, 60, 25)
        weight = st.slider("Weight (kg)", 40, 120, 70)
        height = st.slider("Height (cm)", 140, 220, 175)
        cholesterol = st.slider("Cholesterol Level", 100, 300, 180)
        exercise = st.slider("Exercise (hours/week)", 0, 20, 10)

    with col2:
        smoking = st.radio("Smoking", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        stress_score = st.slider("Stress Level (0-10)", 0, 10, 5)
        alcohol_units = st.slider("Alcohol Units/Week", 0, 50, 5)
        education = st.slider("Education Level (1-5)", 1, 5, 3)
        current_smoker = st.radio("Current Smoker", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    with col3:
        prevalent_hyp = st.radio("Prevalent Hypertension", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        heart_rate = st.slider("Heart Rate (bpm)", 50, 200, 85)
        hypertension_score = st.slider("Hypertension Score", 0, 50, 20)
        hypertension_bmi_interaction = st.slider("Hypertension-BMI Interaction", 0, 50, 15)

    # Submit button
    submit_button = st.form_submit_button("Predict BP")

# Calculated fields
bmi = weight / (height / 100) ** 2
bmi_age_interaction = bmi * age
stress_smoking_interaction = stress_score * smoking
alcohol_exercise_interaction = alcohol_units * exercise
if(exercise>0&exercise<8):
    lifestyle='Sedentary'
elif(exercise>8&stress_score<16):
    lifestyle='Moderate'
elif(stress_score>16&stress_score<=20):
    lifestyle='Active'

# Prepare input data for prediction
input_data = pd.DataFrame([{
    'Age': age, 'Weight': weight, 'Height': height, 'Cholesterol': cholesterol, 'Exercise': exercise,
    'Smoking': smoking, 'Stress_Score': stress_score, 'Alcohol_Units_Per_Week': alcohol_units,
    'BMI_Age_Interaction': bmi_age_interaction, 'Stress_Smoking_Interaction': stress_smoking_interaction,
    'Alcohol_Exercise_Interaction': alcohol_exercise_interaction, 'education': education,
    'currentSmoker': current_smoker, 'prevalentHyp': prevalent_hyp, 'heartRate': heart_rate,
    'Hypertension_Score': hypertension_score, 'Hypertension_BMI_Interaction': hypertension_bmi_interaction
}])

input_data_scaled = scaler.transform(input_data[scaler_feature_names])

# Predict blood pressure
systolic_prediction = best_model_systolic.predict(input_data_scaled)[0]
diastolic_prediction = best_model_diastolic.predict(input_data_scaled)[0]

# Determine blood pressure category
def determine_bp_category(systolic, diastolic):
    if systolic > 180 or diastolic > 120:
        return "Hypertensive Crisis", "red"
    elif systolic >= 140 or diastolic >= 90:
        return "Hypertension Stage 2", "orange"
    elif 130 <= systolic <= 139 or 80 <= diastolic <= 89:
        return "Hypertension Stage 1", "yellow"
    elif 121 <= systolic <= 129 and diastolic < 80:
        return "Elevated", "blue"
    else:
        return "Normal", "green"

bp_category, category_color = determine_bp_category(systolic_prediction, diastolic_prediction)

# Display BP results
st.markdown(f'<div class="result-text">Systolic BP: {systolic_prediction:.2f} mmHg</div>', unsafe_allow_html=True)
st.markdown(f'<div class="result-text">Diastolic BP: {diastolic_prediction:.2f} mmHg</div>', unsafe_allow_html=True)
st.markdown(f'<div style="color:{category_color}; font-size:30px; text-align:center;">● {bp_category}</div>', unsafe_allow_html=True)

# Visualization: Bar Chart for Systolic and Diastolic BP
st.subheader("Blood Pressure Visualization")
fig_bar, ax = plt.subplots(figsize=(6, 4))

# Bar chart data
categories = ['Systolic BP', 'Diastolic BP']
values = [systolic_prediction, diastolic_prediction]

# Create bar chart
ax.bar(categories, values, color=['blue', 'orange'], alpha=0.8)
ax.set_ylim(0, 200)  # Set Y-axis range based on expected BP levels
ax.set_ylabel('Blood Pressure (mmHg)')
ax.set_title('Blood Pressure Predictions')
ax.axhline(y=120, color='green', linestyle='--', label='Normal SBP')
ax.axhline(y=80, color='purple', linestyle='--', label='Normal DBP')
ax.legend()

# Display chart in Streamlit
st.pyplot(fig_bar)

# SHAP Explanation for Model Interpretability
st.header("Model Explainability with SHAP")
explainer = shap.Explainer(best_model_systolic, X_test_scaled)
shap_values = explainer(input_data_scaled)

# Display SHAP summary plot
st.subheader("Feature Importance - Systolic BP")
fig = plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, input_data[scaler_feature_names], show=False)
st.pyplot(fig)

# Gamification: Tracking Progress and Achievements
st.header("Gamification for Motivation")

# Define user levels and progression
current_level = "Beginner"
fitness_progress = systolic_prediction - 120  # Simple way to track improvement

# Level based on age and systolic BP
if age > 40 and systolic_prediction > 130:
    current_level = "Intermediate"
elif age > 50 and systolic_prediction > 140:
    current_level = "Advanced"
else:
    current_level = "Beginner"

# Display current fitness level
st.markdown(f'<div class="gamification">Your Current Fitness Level: {current_level}</div>', unsafe_allow_html=True)
st.write(f"Keep improving to unlock new achievements!")
# Progress Bar to visualize improvement (e.g., towards normal BP range)
progress = max(min(fitness_progress / 60, 1), 0)  # Ensure progress is between 0 and 1
st.markdown(f"<div style='text-align:center; font-size: 20px;'>Progress Towards Optimal BP</div>", unsafe_allow_html=True)
st.progress(progress)


# Achievement Badges for Milestones
if systolic_prediction <= 120:
    badge = "🏅 BP Champion Badge"
    st.markdown(f'<div style="text-align:center; font-size: 24px; color:#388E3C;">{badge}</div>', unsafe_allow_html=True)
    st.write("Congratulations! You've reached an optimal BP level. Keep it up!")
elif systolic_prediction <= 130:
    badge = "🎯 BP Reducer Badge"
    st.markdown(f'<div style="text-align:center; font-size: 24px; color:#FFEB3B;">{badge}</div>', unsafe_allow_html=True)
    st.write("Great progress! Keep pushing to maintain a healthy BP!")
else:
    badge = "⚡ BP Challenger Badge"
    st.markdown(f'<div style="text-align:center; font-size: 24px; color:#FF7043;">{badge}</div>', unsafe_allow_html=True)
    st.write("You're working hard! Aim for further improvement to reach optimal BP levels.")

# Personalized Milestones & Motivational Messages
st.markdown("<div style='text-align:center; font-size: 20px; font-weight: bold;'>Your Health Milestones</div>", unsafe_allow_html=True)
st.write(f"🏃‍♂️ Goal: Lower systolic BP by 10 mmHg within the next 2 weeks. This will improve your heart health!")
st.write(f"💪 Challenge: Increase your weekly exercise to 12 hours to boost cardiovascular health and lower BP!")

# Progress Bar for Habit Change (e.g., exercise goal)
exercise_goal = 12  # Target weekly exercise hours for improvement
exercise_progress = min(exercise / exercise_goal, 1)
st.markdown(f"<div style='text-align:center; font-size: 20px;'>Exercise Goal Progress</div>", unsafe_allow_html=True)
st.progress(exercise_progress)

# Display motivational quote
motivational_quotes = [
    "The only bad workout is the one that didn’t happen!",
    "Push yourself, because no one else is going to do it for you.",
    "Your body can stand almost anything. It’s your mind that you have to convince.",
    "Strength does not come from physical capacity. It comes from an indomitable will."
]
st.markdown(f"<div style='text-align:center; font-size: 18px; font-style: italic;'>💬 {random.choice(motivational_quotes)}</div>", unsafe_allow_html=True)
# Habit Change Recommendations with Icons and Structured Layout
col1, col2 = st.columns(2)

# Exercise Habit Recommendation
with col1:
    st.markdown('<div class="habit-heading" style="color: #388E3C;">🚶‍♂️ Exercise</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="habit-section">
            Increasing weekly exercise can significantly improve cardiovascular health and lower blood pressure.
            <ul class="habit-list">
                <li>Aim for at least 150 minutes of moderate exercise per week.</li>
                <li>Incorporate activities like walking, cycling, and swimming.</li>
                <li>Exercise strengthens the heart, improving overall blood flow.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Smoking Habit Recommendation
with col2:
    st.markdown('<div class="habit-heading" style="color: #D32F2F;">🚭 Smoking</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="habit-section">
            Reducing smoking can lower your systolic and diastolic levels significantly.
            <ul class="habit-list">
                <li>Consider quitting or reducing the number of cigarettes.</li>
                <li>Smoking increases heart rate and blood pressure.</li>
                <li>Seek support from health professionals or cessation programs.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Alcohol Habit Recommendation
col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="habit-heading" style="color: #1976D2;">🍷 Alcohol Consumption</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="habit-section">
            Limiting alcohol intake can help lower your blood pressure and improve overall health.
            <ul class="habit-list">
                <li>Limit alcohol consumption to no more than 1 drink per day for women and 2 for men.</li>
                <li>Excessive alcohol intake can increase both systolic and diastolic BP.</li>
                <li>Consider reducing or eliminating alcohol to improve heart health.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

model = joblib.load('C:/Users/Aravind/Downloads/SIHHACK/bp/recommendation_model.pkl')
recommendation_data = joblib.load('C:/Users/Aravind/Downloads/SIHHACK/bp/recommendation_data.pkl')

def get_personalized_recommendations_with_emoji(age, bp_category, lifestyle="Moderate"):
    # Encode inputs
    age_group = 0 if age < 30 else 1 if age <= 50 else 2
    bp_cat = 0 if bp_category == "Normal_BP" else 1
    life_style = {"Sedentary": 0, "Moderate": 1, "Active": 2}[lifestyle]

    # Predict recommendation index
    input_features = [[age_group, bp_cat, life_style]]
    rec_index = model.predict(input_features)[0]

    # Fetch recommendation and emoji
    recommendation = recommendation_data.iloc[rec_index]["Recommendation"]
    emoji = recommendation_data.iloc[rec_index]["Emoji"]

    # Return recommendation with emoji
    return f"{recommendation} {emoji}"
if st.button("Get Recommendations"):
    recommendation_with_emoji = get_personalized_recommendations_with_emoji(age, bp_category, lifestyle)
    st.subheader("Personalized Recommendation:")
    st.write(recommendation_with_emoji)
