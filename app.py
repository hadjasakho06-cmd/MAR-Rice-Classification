import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import urllib.request
import pandas as pd

# 1. Configuration de la page
st.set_page_config(
    page_title="ServAgri - MAC",
    page_icon="🌾",
    layout="centered"
)

# 2. Design personnalisé (Bleu Ciel)
st.markdown("""
    <style>
    .main {
        background-color: #f0f8ff;
    }
    h1 {
        color: #0077b6;
        text-align: center;
    }
    .stButton>button {
        background-color: #0077b6;
        color: white;
        border-radius: 10px;
        width: 100%;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Gestion du modèle lourd (VGG16)
MODEL_URL = "https://huggingface.co/Hadjita/vgg16-rice-model/resolve/main/Modele_riz_final.keras"
MODEL_PATH = "Modele_riz_final.keras"

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        try:
            with st.spinner("Initialisation du système... Téléchargement du modèle VGG16 (ServAgri) en cours..."):
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
                st.success("Modèle chargé avec succès !")
        except Exception as e:
            st.error(f"Erreur lors du téléchargement du modèle : {e}")
            return None
    
    # Chargement de l'architecture VGG16 personnalisée
    return tf.keras.models.load_model(MODEL_PATH)

# Chargement du modèle
model = load_my_model()

# 4. Interface Utilisateur
st.title("🌾 Mon Assistant Cultural (MAC)")
st.subheader("Système intelligent de classification des grains de riz")
st.write("Ce module utilise l'architecture Deep Learning VGG16 pour identifier les variétés de riz avec précision.")

# Zone de téléchargement
uploaded_file = st.file_uploader("Déposez une image de grain de riz (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Affichage de l'image
    image = Image.open(uploaded_file)
    st.image(image, caption='Image à analyser', use_container_width=True)
    
    # --- PRÉTRAITEMENT SYNC AVEC TON ENTRAÎNEMENT ---
    img = image.resize((224, 224))
    img_array = np.array(img)
    
    # Normalisation : Assure-toi que c'est bien ce qui a été fait dans ton notebook (rescale=1./255)
    img_array = img_array / 255.0 
    
    img_array = np.expand_dims(img_array, axis=0)

    # Bouton de prédiction
    if st.button("Lancer l'analyse"):
        if model is not None:
            with st.spinner('Analyse en cours par VGG16...'):
                prediction = model.predict(img_array)
                
                # Liste des variétés (Vérifie l'ordre alphabétique de tes dossiers d'entraînement)
                classes = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
                
                # Préparation des données pour le graphique (Correction image_87e73c.png)
                df_results = pd.DataFrame({
                    'Variété': classes,
                    'Probabilité': prediction[0]
                }).set_index('Variété')
                
                # Identification de la meilleure prédiction
                indice_max = np.argmax(prediction)
                nom_variete = classes[indice_max]
                score_confiance = prediction[0][indice_max] * 100

                st.write("---")
                st.header("Résultat de l'analyse")
                
                # Résultat principal
                st.success(f"✅ Variété détectée : **{nom_variete}**")
                st.info(f"📊 Indice de confiance : **{score_confiance:.2f}%**")
                
                # Graphique avec les noms de variétés
                st.write("Détails des probabilités par variété :")
                st.bar_chart(df_results)
                
        else:
            st.error("Le modèle n'est pas disponible. Vérifiez le terminal.")

# Pied de page
st.write("---")
st.caption("Projet ServAgri - Développé par Racky Sakho Sow - Master 1 Data Science")
