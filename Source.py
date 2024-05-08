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


# Bouton de prédiction
if st.button('Prédire la survie'):
    input_df = pd.DataFrame({
        'Hb': [hb],
        'KN564': [kn564],
        'RAD_Score': [rad_score]
    })

    # Prédiction de survie
    survival_prob = model_cox.predict(input_df)
    st.subheader('Probabilité de survie estimée:')
    st.write(survival_prob[0])


# Prédire la fonction de survie
if st.button("Afficher la fonction de survie"):
    pred_surv = model_cox.predict_survival_function(input_df, return_array=True)
    fig, ax = plt.subplots()
    for i, s in enumerate(pred_surv):
        plt.step(model_cox.event_times_, s, where="post", label=str(i))
    plt.ylabel("Probabilité de survie")
    plt.xlabel("Temps en jours")
    st.pyplot(fig)

