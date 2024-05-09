import streamlit as st
import pandas as pd
import dill
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# Fonction pour charger le modèle sauvegardé
@st.cache(allow_output_mutation=True)
def load_model():
    with open('model_cox.pkl', 'rb') as f:
        model = dill.load(f)
    return model

# Fonction pour charger des fichiers NIfTI
@st.cache(show_spinner=True)
def load_nifti_file(file):
    try:
        nifti = nib.load(file)
        data = nifti.get_fdata()
        return data
    except Exception as e:
        st.error(f"Failed to load NIfTI file: {e}")
        return None

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

# Section pour le calcul du RAD
st.title('RAD Calculation with CT and Segmentation')

# Chargement des fichiers TDM et de segmentation via l'interface utilisateur
uploaded_tdm = st.file_uploader("Choose a CT file (.nii)", type=["nii", "nii.gz"])
uploaded_seg = st.file_uploader("Choose a Segmentation file (.nii)", type=["nii", "nii.gz"])

# Bouton pour calculer le RAD après le chargement des fichiers
if st.button('Calculate RAD'):
    if uploaded_tdm is not None and uploaded_seg is not None:
        # Charger les données NIfTI
        tdm_data = load_nifti_file(uploaded_tdm)
        seg_data = load_nifti_file(uploaded_seg)

        if tdm_data is not None and seg_data is not None:
            # Ici, vous pouvez ajouter votre logique de calcul du RAD
            # Par exemple :
            # rad_score = calculate_rad(tdm_data, seg_data)
            # st.write(f'The RAD score is: {rad_score}')

            # Placeholder pour la fonction de calcul du RAD
            st.write("CT and Segmentation data loaded. Ready to calculate RAD.")
    else:
        st.error("Please upload both CT and Segmentation files.")
