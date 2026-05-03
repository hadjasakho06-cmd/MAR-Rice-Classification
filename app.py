import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import urllib.request
import pandas as pd
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="ServAgri - MAC",
    page_icon="🌾",
    layout="centered"
)

# --- PARAMÈTRES DU MODÈLE ---
# Utilisation de l'URL permanente Hugging Face pour contourner la limite de taille de GitHub
MODEL_URL = "https://huggingface.co/Hadjita/classification-resnet-final/resolve/main/ResNet50_Final_Classification_riz.keras"
MODEL_PATH = "ResNet50_Final_Classification_riz.keras"

@st.cache_resource
def load_my_model():
    """Télécharge le modèle si nécessaire et le charge en mémoire."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Téléchargement du modèle ServAgri (217 Mo)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

# Chargement du modèle
try:
    model = load_my_model()
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")

# --- INTERFACE UTILISATEUR ---
st.title("🌾 Mon Assistant Cultural (MAC)")
st.markdown("""
    **Système de diagnostic automatisé des variétés de riz.**
    *Projet ServAgri - Expertise en Data Science Agricole.*
""")

st.divider()

# Zone de téléchargement
uploaded_file = st.file_uploader("Veuillez charger une photo d'un grain de riz (JPG ou JPEG)", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Affichage de l'image source
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Échantillon prêt pour analyse", use_container_width=True)
    
    # --- PRÉTRAITEMENT TECHNIQUE ---
    # Redimensionnement aux dimensions d'entrée de ResNet50 (224x224)
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Application du prétraitement spécifique au modèle ResNet50
    img_array = preprocess_input(img_array) 

    # --- PRÉDICTION ---
    if st.button("Lancer le diagnostic 🚀"):
        with st.spinner("Analyse des caractéristiques morphologiques..."):
            predictions = model.predict(img_array)
            classes = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
            
            # Application de Softmax pour obtenir des probabilités claires
            probabilites = tf.nn.softmax(predictions[0]).numpy()
            idx_max = np.argmax(probabilites)
            
            # Préparation des données pour le graphique
            df_results = pd.DataFrame({
                'Variété': classes, 
                'Confiance': probabilites
            }).set_index('Variété')

            # --- AFFICHAGE DES RÉSULTATS ---
            st.write("---")
            st.success(f"✅ Variété identifiée : **{classes[idx_max]}**")
            st.info(f"📊 Indice de fiabilité : **{probabilites[idx_max]*100:.2f}%**")
            
            # Graphique de répartition
            st.markdown("### Détails des probabilités")
            st.bar_chart(df_results)

            # Note de bas de page technique
            st.caption("Modèle basé sur l'architecture ResNet50 | Précision globale : 96.88%")

st.divider()
st.caption("Développé par Rassoul'S Job | UFR Sciences et Technologies | Sénégal")
