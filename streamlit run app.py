import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Load models safely
base_path = os.path.dirname(__file__)

diabetes_model = pickle.load(open(os.path.join(base_path, 'diabetes_model.sav'), 'rb'))
heart_model = pickle.load(open(os.path.join(base_path, 'heart_disease_model.sav'), 'rb'))
parkinsons_model = pickle.load(open(os.path.join(base_path, 'parkinsons_model.sav'), 'rb'))

st.title("🩺 Multiple Disease Prediction System")
option = st.sidebar.selectbox("Select Disease to Predict", ["Diabetes", "Heart Disease", "Parkinson's Disease"])

# ---------------------- Diabetes ----------------------
if option == "Diabetes":
    st.header("Diabetes Prediction")

    preg = st.number_input("Pregnancies", 0)
    glu = st.number_input("Glucose", 0)
    bp = st.number_input("Blood Pressure", 0)
    skin = st.number_input("Skin Thickness", 0)
    insulin = st.number_input("Insulin", 0)
    bmi = st.number_input("BMI", 0.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0)
    age = st.number_input("Age", 0)

    if st.button("Predict Diabetes"):
        input_df = pd.DataFrame([[preg, glu, bp, skin, insulin, bmi, dpf, age]],
                                columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        result = diabetes_model.predict(input_df)
        st.success("✅ Diabetic" if result[0] == 1 else "❌ Not Diabetic")

# ---------------------- Heart Disease ----------------------
elif option == "Heart Disease":
    st.header("Heart Disease Prediction")

    age = st.number_input("Age", 0)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.number_input("Chest Pain Type (0-3)", 0)
    trestbps = st.number_input("Resting Blood Pressure", 0)
    chol = st.number_input("Cholesterol", 0)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True)", [0, 1])
    restecg = st.number_input("Resting ECG results (0-2)", 0)
    thalach = st.number_input("Max Heart Rate Achieved", 0)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes)", [0, 1])
    oldpeak = st.number_input("ST depression", 0.0)
    slope = st.number_input("Slope of ST segment", 0)
    ca = st.number_input("Major Vessels (0-3)", 0)
    thal = st.number_input("Thal (0 = normal; 1 = fixed; 2 = reversible)", 0)

    if st.button("Predict Heart Disease"):
        input_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                  thalach, exang, oldpeak, slope, ca, thal]],
                                columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                         'restecg', 'thalach', 'exang', 'oldpeak',
                                         'slope', 'ca', 'thal'])
        result = heart_model.predict(input_df)
        st.success("❤️ Has Heart Disease" if result[0] == 1 else "💖 No Heart Disease")

# ---------------------- Parkinson's ----------------------
elif option == "Parkinson's Disease":
    st.header("Parkinson's Disease Prediction")

    fo = st.number_input("MDVP:Fo(Hz)")
    fhi = st.number_input("MDVP:Fhi(Hz)")
    flo = st.number_input("MDVP:Flo(Hz)")
    jitter_percent = st.number_input("MDVP:Jitter(%)")
    shimmer = st.number_input("MDVP:Shimmer")
    rap = st.number_input("MDVP:RAP")
    ppq = st.number_input("MDVP:PPQ")
    ddp = st.number_input("Jitter:DDP")
    shimmer_dB = st.number_input("MDVP:Shimmer(dB)")
    apq3 = st.number_input("Shimmer:APQ3")
    apq5 = st.number_input("Shimmer:APQ5")
    apq = st.number_input("Shimmer:APQ")
    dda = st.number_input("Shimmer:DDA")
    nhr = st.number_input("NHR")
    hnr = st.number_input("HNR")
    rpde = st.number_input("RPDE")
    dfa = st.number_input("DFA")
    spread1 = st.number_input("Spread1")
    spread2 = st.number_input("Spread2")
    d2 = st.number_input("D2")
    ppe = st.number_input("PPE")

    if st.button("Predict Parkinson's"):
        input_df = pd.DataFrame([[fo, fhi, flo, jitter_percent, shimmer, rap, ppq, ddp,
                                  shimmer_dB, apq3, apq5, apq, dda, nhr, hnr,
                                  rpde, dfa, spread1, spread2, d2, ppe]],
                                columns=['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                                         'MDVP:Shimmer', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                                         'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ',
                                         'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'Spread1',
                                         'Spread2', 'D2', 'PPE'])
        result = parkinsons_model.predict(input_df)
        st.success("🧠 Has Parkinson's" if result[0] == 1 else "🧠 No Parkinson's Disease")
