import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
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
    .main { background-color: #f0f8ff; }
    h1 { color: #0077b6; text-align: center; }
    .stButton>button {
        background-color: #0077b6;
        color: white;
        border-radius: 10px;
        width: 100%;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Chargement du modèle VGG16 (ServAgri)
MODEL_URL = "https://huggingface.co/Hadjita/vgg16-rice-model/resolve/main/Modele_riz_final.keras"
MODEL_PATH = "Modele_riz_final.keras"

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as e:
            st.error(f"Erreur de téléchargement : {e}")
            return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_my_model()

# 4. Interface Utilisateur
st.title("🌾 Mon Assistant Cultural (MAC)")
st.subheader("Système intelligent de classification des grains de riz")

uploaded_file = st.file_uploader("Déposez une image (JPG)", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Lecture de l'image
    image = Image.open(uploaded_file)
    st.image(image, caption='Image originale', use_container_width=True)
    
    # --- PRÉTRAITEMENT TECHNIQUE RIGOUREUX ---
    # 1. Conversion forcée en RGB (pour éviter les erreurs de matrice)
    img_rgb = image.convert('RGB')
    
    # 2. Redimensionnement avec haute qualité (LANCZOS) pour garder les détails du grain
    img_resized = img_rgb.resize((224, 224), Image.Resampling.LANCZOS)
    
    # 3. Conversion en Array et Normalisation (doit matcher ton entraînement)
    img_array = np.array(img_resized).astype('float32') / 255.0
    
    # 4. Ajout de la dimension batch (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Bouton de prédiction
    if st.button('Lancer l\'analyse'):
        if model is not None:
            with st.spinner('Analyse en cours...'):
                prediction = model.predict(img_array)
                
                # Liste des classes (Assure-toi que c'est l'ordre alphabétique de ton dataset)
                classes = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
                
                # Création du DataFrame pour le graphique (Axe X = Noms)
                df_results = pd.DataFrame({
                    'Variété': classes,
                    'Confiance': prediction[0]
                }).set_index('Variété')

                # Identification du gagnant
                idx_max = np.argmax(prediction)
                nom_final = classes[idx_max]
                score_final = prediction[0][idx_max] * 100

                st.write("---")
                st.header("Résultat de l'analyse")
                
                # Affichage des résultats
                st.success(f"✅ Variété détectée : **{nom_final}**")
                st.info(f"📊 Indice de confiance : **{score_final:.2f}%**")
                
                # Affichage du graphique avec les noms de variétés
                st.bar_chart(df_results)
        else:
            st.error("Le modèle n'est pas chargé.")

# Pied de page
st.write("---")
st.caption("Projet ServAgri - Développé par Racky Sakho Sow - Master 1 Data Science")
