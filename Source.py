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

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import dill
import SimpleITK as sitk
from radiomics import featureextractor
import joblib

# Constants
FEATURE_KEYS = [
    'original_glcm_JointEnergy', 'original_gldm_LargeDependenceEmphasis', 'original_gldm_SmallDependenceLowGrayLevelEmphasis',
    'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LongRunEmphasis', 'original_glrlm_LongRunLowGrayLevelEmphasis',
    'original_glrlm_RunVariance', 'original_glrlm_ShortRunEmphasis', 'original_glrlm_ShortRunHighGrayLevelEmphasis',
    'original_glrlm_ShortRunLowGrayLevelEmphasis', 'original_glszm_GrayLevelVariance', 'original_glszm_HighGrayLevelZoneEmphasis',
    'original_glszm_LargeAreaLowGrayLevelEmphasis', 'original_glszm_SmallAreaHighGrayLevelEmphasis', 'original_glszm_ZoneVariance',
    'wavelet-LLH_firstorder_Entropy', 'wavelet-LLH_firstorder_InterquartileRange', 'wavelet-LLH_firstorder_Kurtosis',
    'wavelet-LLH_glcm_Contrast', 'wavelet-LLH_glcm_DifferenceVariance', 'wavelet-LLH_glcm_Idm', 'wavelet-LLH_glcm_Idn',
    'wavelet-LLH_glcm_Imc1', 'wavelet-LLH_gldm_HighGrayLevelEmphasis', 'wavelet-LLH_gldm_LargeDependenceEmphasis',
    'wavelet-LLH_gldm_SmallDependenceHighGrayLevelEmphasis', 'wavelet-LLH_glrlm_GrayLevelNonUniformityNormalized',
    'wavelet-LLH_glrlm_HighGrayLevelRunEmphasis', 'wavelet-LLH_glrlm_LongRunLowGrayLevelEmphasis', 'wavelet-LLH_glrlm_RunLengthNonUniformity',
    'wavelet-LLH_glrlm_RunPercentage', 'wavelet-LLH_ngtdm_Busyness', 'wavelet-LHL_firstorder_RobustMeanAbsoluteDeviation',
    'wavelet-LHL_glcm_ClusterTendency', 'wavelet-LHL_glcm_Correlation', 'wavelet-LHL_glcm_DifferenceEntropy',
    'wavelet-LHL_glcm_Idmn', 'wavelet-LHL_glcm_JointEntropy', 'wavelet-LHL_glcm_SumAverage', 'wavelet-LHL_gldm_DependenceNonUniformityNormalized',
    'wavelet-LHL_glrlm_LongRunEmphasis', 'wavelet-LHL_glszm_SizeZoneNonUniformityNormalized', 'wavelet-LHL_ngtdm_Complexity',
    'wavelet-LHH_firstorder_RootMeanSquared'
]

# Function to load the survival model
@st.cache(allow_output_mutation=True)
def load_model():
    with open('model_cox.pkl', 'rb') as f:
        return dill.load(f)

# Function to load the RSF model
@st.cache(allow_output_mutation=True)
def load_rsf_model():
    return joblib.load("random_survival_forest_model.joblib")

# Function to load NIfTI files
@st.cache(show_spinner=True)
def load_nifti_file(file):
    try:
        nifti = nib.load(file)
        return nifti.get_fdata()
    except Exception as e:
        st.error(f"Failed to load NIfTI file: {e}")
        return None

# Load models
model_cox = load_model()
model_rsf = load_rsf_model()

# Configuring feature extractor
def setup_extractor():
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllImageTypes()
    return extractor

extractor = setup_extractor()

# Streamlit UI
st.title("Radiomics Feature Extraction and RSF Prediction")
uploaded_ct = st.file_uploader("Upload CT Image", type=["nii", "nii.gz"])
uploaded_seg = st.file_uploader("Upload Segmentation Mask", type=["nii", "nii.gz"])
hb = st.slider("Hemoglobin Level", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
kn564 = st.selectbox("KEYNOTE-564 inclusion criteria", options=[0, 1], format_func=lambda x: "high risk" if x == 1 else "intermediate-high")
rad_score = st.slider("Radiomics Signature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
input_df = pd.DataFrame({'Hb': [hb], 'KN564': [kn564], 'RAD_Score': [rad_score]})

if st.button('Predict Survival'):
    survival_function = model_cox.predict_survival_function(input_df)
    st.subheader('Estimated Survival Probability:')
    st.line_chart(survival_function)

if uploaded_ct and uploaded_seg and st.button("Predict"):
    ct_image = sitk.ReadImage(uploaded_ct.name)
    seg_image = sitk.ReadImage(uploaded_seg.name)
    features = extractor.execute(ct_image, seg_image)
    selected_features = {k: features[k] for k in features.keys() if k in FEATURE_KEYS}
    features_array = np.array(list(selected_features.values())).reshape(1, -1)
    risk_score = model_rsf.predict(features_array)[0]
    st.write(f"Predicted Risk Score: {risk_score}")
    rad_score = np.dot(features_array.flatten(), np.random.rand(features_array.size))  # Example calculation
    st.write(f"RAD Score: {rad_score}")
