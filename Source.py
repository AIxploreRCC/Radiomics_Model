import streamlit as st
import pandas as pd
import dill
import numpy as np
import matplotlib.pyplot as plt

# Fonction pour charger le modèle sauvegardé
@st.cache(allow_output_mutation=True)
def load_model():
    with open('model_cox.pkl', 'rb') as f:
        model = dill.load(f)
    return model

# Chargement du modèle de Cox
model_cox = load_model()

# Titre de l'application Streamlit
st.title("Prédiction de Survie avec le Modèle de Cox")

# Slider pour RAD Score
rad_score = st.slider("RAD Score", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

# Autres entrées nécessaires pour le modèle (exemple simple avec une variable supplémentaire)
hb = st.slider("Hemoglobin Level", min_value=0.0, max_value=20.0, value=10.0, step=0.1)

# Bouton de prédiction
if st.button('Prédire la survie'):
    input_df = pd.DataFrame({
        'Hb': [hb],
        'RAD_Score': [rad_score]
    })

    # Prédiction de la fonction de survie
    survival_function = model_cox.predict_survival_function(input_df)
    st.subheader('Probabilité de survie estimée:')
    st.line_chart(survival_function)

# Instruction pour exécuter l'application (à utiliser dans le terminal, non dans le code)
# st.write("Pour exécuter cette application, utilisez: streamlit run <nom_de_fichier>.py")
