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

model_cox = load_model()

st.title("Prédiction de Survie avec le Modèle de Cox")

# Entrées pour les variables du modèle
hb = st.slider("Hemoglobin Level", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
kn564 = st.selectbox("KN564 Presence", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
rad_score = st.slider("RAD Score", min_value=0.0, max_value=10.0, value=5.0, step=0.1)

input_df = pd.DataFrame({
    'Hb': [hb],
    'KN564': [kn564],
    'RAD_Score': [rad_score]
})

# Bouton pour prédire la survie
if st.button('Prédire la survie'):
    # Prédiction de survie
    survival_function = model_cox.predict_survival_function(input_df)
    st.subheader('Probabilité de survie estimée:')
    st.write(survival_function.iloc[:, 0])  # Affiche la première colonne de la fonction de survie

# Bouton pour afficher la fonction de survie
if st.button("Afficher la fonction de survie"):
    pred_surv = model_cox.predict_survival_function(input_df, return_array=True)
    fig, ax = plt.subplots()
    for i, s in enumerate(pred_surv):
        plt.step(model_cox.event_times_, s, where="post", label=f'ID {i}')
    plt.ylabel("Probabilité de survie")
    plt.xlabel("Temps en jours")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)



