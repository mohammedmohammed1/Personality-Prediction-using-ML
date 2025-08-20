import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open('random_forest_model.pkl', 'rb') as file:       # read binary(rb)  and write binary(wb)
    model = pickle.load(file)

# Prediction descriptions (customize as needed)
personality_descriptions = {
    0: {
        "type": "Introvert",
        "description": "You are reflective, observant, and prefer solitude or quiet environments. You find energy in your inner world and enjoy meaningful one-on-one conversations. While not shy, you are introspective and thoughtful in your interactions.",
        "compliment": "You're deep, thoughtful, and wonderfully calm. ✨"
    },
    1: {
        "type": "Extrovert",
        "description": "You are outgoing, energetic, and enjoy being around people. Social interaction fuels your energy, and you are typically enthusiastic, talkative, and action-oriented. You're the life of the party!",
        "compliment": "You're full of life and light up every room! 🌟"
    },
    2: {
        "type": "Ambivert",
        "description": "You have a balanced personality with traits of both introverts and extroverts. You adapt to situations easily — enjoying social time as well as time alone. You are flexible, intuitive, and emotionally intelligent.",
        "compliment": "You’re versatile, balanced, and beautifully unique. 🌈"
    },
    3: {
        "type": "Analyst",
        "description": "You are logical, analytical, and love solving problems. You rely on reason more than emotion and often think strategically. You seek depth, patterns, and clarity in everything you do.",
        "compliment": "You're smart, sharp, and an insightful thinker. 🧠"
    },
    4: {
        "type": "Diplomat",
        "description": "You are empathetic and cooperative. You value harmony, relationships, and emotional connection. You are intuitive and often the peacemaker in social groups, creating warmth and understanding wherever you go.",
        "compliment": "You're kind-hearted, wise, and deeply appreciated. 💖"
    }
    # Add more if needed based on your model's output
}

# App layout
st.title("🧠 ML Based Personality Profiling")
st.markdown("Fill in your traits to get your predicted personality type and a personalized message!")

# Inputs
gender_code = st.selectbox("Gender Code", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
age = st.number_input("Enter your Age", min_value=10, max_value=100, value=25)

openness = st.number_input("Openness (0-10)", min_value=0.0, max_value=10.0, value=5.0)
conscientiousness = st.number_input("Conscientiousness (0-10)", min_value=0.0, max_value=10.0, value=5.0)
extraversion = st.number_input("Extraversion (0-10)", min_value=0.0, max_value=10.0, value=5.0)
agreeableness = st.number_input("Agreeableness (0-10)", min_value=0.0, max_value=10.0, value=5.0)
neuroticism = st.number_input("Neuroticism (0-10)", min_value=0.0, max_value=10.0, value=5.0)

# Derived features (scaled for 0–10)
openness_conscientiousness_ratio = openness / conscientiousness if conscientiousness != 0 else 0
risk_index = (neuroticism + (10 - agreeableness) + (10 - conscientiousness)) / 3
total_trait_score = openness + conscientiousness + extraversion + agreeableness + neuroticism
trait_balance = abs(openness - neuroticism)

# Build DataFrame for prediction
input_df = pd.DataFrame([{
    "Age": age,
    "Gender_Code": gender_code,
    "Openness": openness,
    "Conscientiousness": conscientiousness,
    "Extraversion": extraversion,
    "Agreeableness": agreeableness,
    "Neuroticism": neuroticism,
    "Openness_Conscientiousness_Ratio": openness_conscientiousness_ratio,
    "Risk_Index": risk_index,
    "Total_Trait_Score": total_trait_score,
    "Trait_Balance": trait_balance,
}])

# Align with model features
try:
    model_features = model.feature_names_in_
except AttributeError:
    model_features = input_df.columns.tolist()

for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[model_features]


# Predict button
if st.button("🔮 Predict My Personality"):
    try:
        prediction = model.predict(input_df)[0]
        result = personality_descriptions.get(prediction, {
            "type": f"Unknown ({prediction})",
            "description": "No description available for this type.",
            "compliment": "You are unique in your own way! ✨"
        })

        # Build result table
        result_df = pd.DataFrame({
            "Your Age": [age],
            "Predicted Personality": [result["type"]],
            "Character Description": [result["description"]],
            "Final Note": [result["compliment"]]
        })

        st.success("🎉 Prediction Complete!")
        st.markdown("### 🧾 Prediction Summary")
        st.table(result_df)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
