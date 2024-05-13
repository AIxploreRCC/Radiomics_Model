import streamlit as st
import pandas as pd
import dill
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# Decorator to cache model loading
@st.cache(allow_output_mutation=True)


def load_model_direct():
    with open('model_cox.pkl', 'rb') as f:
        model = dill.load(f)
    return model

# Call the function directly
model_cox = load_model_direct()

# Title of the Streamlit app
st.title("Survival Prediction with the Cox Model")

# Inputs for the model's variables
hb = st.slider("Hemoglobin Level", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
kn564 = st.selectbox("KEYNOTE-564 inclusion criteria", options=[0, 1], format_func=lambda x: "High risk" if x == 1 else "Intermediate-high")
rad_score = st.slider("Radiomics Signature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

# DataFrame for model input
input_df = pd.DataFrame({
    'Hb': [hb],
    'KN564': [kn564],
    'RAD_Score': [rad_score]
})

# Button to predict survival
if st.button('Predict Survival'):
    with st.spinner('Calculating... Please wait.'):
        try:
            survival_function = model_cox.predict_survival_function(input_df)
            st.subheader('Estimated Survival Probability:')
            st.line_chart(survival_function)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
