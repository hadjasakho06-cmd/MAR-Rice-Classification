import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import urllib.request
import pandas as pd

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="ServAgri - MAC",
    page_icon="🌾",
    layout="centered"
)

# --- PARAMÈTRES DU MODÈLE ---
# URL permanente sans accents pour éviter l'erreur ASCII
MODEL_URL = "https://huggingface.co/Hadjita/classification-resnet-final/resolve/main/ResNet50_Final_Classification_riz.keras"
MODEL_PATH = "ResNet50_Final_Classification_riz.keras"

@st.cache_resource
def load_my_model():
    """Télécharge le modèle depuis Hugging Face si absent et le charge."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Téléchargement du modèle ServAgri (217 Mo)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

# Chargement sécurisé du modèle
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

# Zone de téléchargement d'image
uploaded_file = st.file_uploader("Veuillez charger une photo d'un grain de riz (JPG ou JPEG)", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Lecture et affichage de l'image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Échantillon à analyser", use_container_width=True)
    
    # --- PRÉTRAITEMENT TECHNIQUE ---
    # 1. Redimensionnement standard ResNet
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 2. NORMALISATION (Crucial : division par 255 pour correspondre à l'entraînement standard)
    img_array = img_array / 255.0 

    # --- PRÉDICTION ---
    if st.button("Lancer le diagnostic 🚀"):
        with st.spinner("Analyse des caractéristiques morphologiques..."):
            # Exécution de la prédiction
            predictions = model.predict(img_array)
            
            # Ordre alphabétique des dossiers d'entraînement
            classes = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
            
            # Récupération du score le plus élevé
            probabilites = predictions[0]
            idx_max = np.argmax(probabilites)
            
            # --- AFFICHAGE DES RÉSULTATS ---
            st.write("---")
            
            # Ligne de Debug pour vérifier les sorties brutes (à supprimer après tes tests)
            st.write("🔍 **Debug (Scores bruts) :**", probabilites)
            
            st.success(f"✅ Variété identifiée : **{classes[idx_max]}**")
            st.info(f"📊 Indice de fiabilité : **{probabilites[idx_max]*100:.2f}%**")
            
            # Graphique de confiance
            df_results = pd.DataFrame({
                'Variété': classes, 
                'Confiance': probabilites
            }).set_index('Variété')
            
            st.markdown("### Détails des probabilités")
            st.bar_chart(df_results)

            st.caption("Modèle : ResNet50 | Précision d'entraînement : 96.88%")

st.divider()
st.caption("Développé par Rassoul'S Job | UFR Sciences et Technologies | Sénégal")
