import streamlit as st
import pandas as pd
import dill
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib



@st.cache(allow_output_mutation=True)
def load_model():
    with open('model_cox.pkl', 'rb') as f:
        model = dill.load(f)
    return model


# Chargement du modèle
model_cox = load_model()

# Titre de l'application Streamlit
st.title("Survival Prediction with the Cox Model")

# Entrées pour les variables du modèle
hb = st.slider("Hemoglobin Level", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
kn564 = st.selectbox("KEYNOTE-564 inclusion criteria", options=[0, 1], format_func=lambda x: "high risk" if x == 1 else "intermediate-high")
rad_score = st.slider("Radiomics Signature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

input_df = pd.DataFrame({
    'Hb': [hb],
    'KN564': [kn564],
    'RAD_Score': [rad_score]
})

# Bouton pour prédire la survie
if st.button('Predict Survival'):
    try:
        survival_function = model_cox.predict_survival_function(input_df)
        st.subheader('Estimated Survival Probability:')
        st.line_chart(survival_function)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
