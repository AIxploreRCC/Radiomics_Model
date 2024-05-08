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


# Title of the Streamlit application
st.title("Survival Prediction with the Cox Model")

# Slider for 'Duration' limited to 60 months
duration = st.slider("Duration in Months", min_value=0, max_value=60, value=30, step=1)


# Entrées pour les variables du modèle
hb = st.slider("Hemoglobin Level", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
kn564 = st.selectbox("KN564 Presence", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
rad_score = st.slider("RAD Score", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

input_df = pd.DataFrame({
    'Hb': [hb],
    'KN564': [kn564],
    'RAD_Score': [rad_score]
})

# Bouton pour prédire la survie
if st.button('Prédire la survie'):
    survival_function = model_cox.predict_survival_function(input_df)
    st.subheader('Probabilité de survie estimée:')
    st.line_chart(survival_function)

# Bouton pour afficher la fonction de survie
if st.button("Afficher la fonction de survie"):
    pred_surv = model_cox.predict_survival_function(input_df)
    fig, ax = plt.subplots()
    pred_surv.plot(ax=ax)  # Utilisez DataFrame.plot pour un affichage correct
    ax.set_ylabel("Probabilité de survie")
    ax.set_xlabel("Temps en jours")
    st.pyplot(fig)




