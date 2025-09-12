# -*- coding: utf-8 -*-
"""
Created on Sun May 22 11:53:51 2022

@author: Prerna
"""

import pickle
import numpy as np
import streamlit as st
import os
from streamlit_option_menu import option_menu

# Load models
base_path = os.path.dirname(__file__)
diabetes_model = pickle.load(open(os.path.join(base_path, 'diabetes_model.sav'), 'rb'))
heart_model = pickle.load(open(os.path.join(base_path, 'heart_disease_model.sav'), 'rb'))
parkinsons_model = pickle.load(open(os.path.join(base_path, 'parkinsons_model.sav'), 'rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')

    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        try:
            diab_prediction = diabetes_model.predict([[
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ]])
            diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
        except:
            diab_diagnosis = '⚠️ Please enter valid numerical values.'

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex (1 = Male, 0 = Female)')

    with col3:
        cp = st.text_input('Chest Pain types (0-3)')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholesterol in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)')

    with col1:
        restecg = st.text_input('Resting ECG results (0-2)')

    with col2:
        thalach = st.text_input('Max Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina (1 = Yes, 0 = No)')

    with col1:
        oldpeak = st.text_input('ST depression')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by fluoroscopy (0-3)')

    with col1:
        thal = st.text_input('Thal (0 = normal, 1 = fixed defect, 2 = reversible)')

    heart_diagnosis = ''

    if st.button('Heart Disease Test Result'):
        try:
            heart_prediction = heart_model.predict([[
                float(age), float(sex), float(cp), float(trestbps), float(chol),
                float(fbs), float(restecg), float(thalach), float(exang),
                float(oldpeak), float(slope), float(ca), float(thal)
            ]])
            heart_diagnosis = 'The person has heart disease' if heart_prediction[0] == 1 else 'The person does not have heart disease'
        except:
            heart_diagnosis = '⚠️ Please enter valid numerical values.'

    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == 'Parkinsons Prediction':
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    parkinsons_diagnosis = ''

    if st.button("Parkinson's Test Result"):
        try:
            parkinsons_prediction = parkinsons_model.predict([[
                float(fo), float(fhi), float(flo), float(Jitter_percent), float(Jitter_Abs),
                float(RAP), float(PPQ), float(DDP), float(Shimmer), float(Shimmer_dB),
                float(APQ3), float(APQ5), float(APQ), float(DDA), float(NHR), float(HNR),
                float(RPDE), float(DFA), float(spread1), float(spread2), float(D2), float(PPE)
            ]])
            parkinsons_diagnosis = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease"
        except:
            parkinsons_diagnosis = '⚠️ Please enter valid numerical values.'

    st.success(parkinsons_diagnosis)
