import streamlit as st
import pandas as pd
import numpy as np
import dill
import nibabel as nib
import SimpleITK as sitk
import pyradiomics as pr
from radiomics import featureextractor
import joblib


# Fonction pour charger des fichiers NIfTI
@st.cache(show_spinner=True)
def load_nifti_file(file):
    try:
        nifti = nib.load(file)
        return nifti.get_fdata()
    except Exception as e:
        st.error(f"Failed to load NIfTI file: {e}")
        return None

# Configurer l'extracteur de caractéristiques
@st.cache(allow_output_mutation=True)
def setup_extractor():
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllImageTypes()
    return extractor


extractor = setup_extractor()


# Téléchargement des images
uploaded_ct = st.file_uploader("Upload CT Image", type=["nii", "nii.gz"])
uploaded_seg = st.file_uploader("Upload Segmentation Mask", type=["nii", "nii.gz"])


# Bouton pour prédire avec RSF
if uploaded_ct and uploaded_seg and st.button("Predict with RSF"):
    ct_image = sitk.ReadImage(uploaded_ct.name)
    seg_image = sitk.ReadImage(uploaded_seg.name)
    features = extractor.execute(ct_image, seg_image)
    selected_features = {k: features[k] for k in features if k in FEATURE_KEYS}
    features_array = np.array(list(selected_features.values())).reshape(1, -1)
   

FEATURE_KEYS = FEATURE_KEYS = [
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

