import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np
import pandas as pd
import os
import urllib.request

# --- CONFIGURATION ---
st.set_page_config(
    page_title="RIZIERE - MAR",
    page_icon="🌾",
    layout="centered"
)

# --- CSS PERSONNALISÉ ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Fond général */
.stApp {
    background-color: #F7F4EF;
    font-family: 'DM Sans', sans-serif;
}

/* Header principal */
.hero {
    background: linear-gradient(135deg, #2D4A1E 0%, #4A7A2E 60%, #6BA03E 100%);
    border-radius: 20px;
    padding: 40px 32px 32px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "🌾";
    font-size: 120px;
    position: absolute;
    right: -10px;
    top: -10px;
    opacity: 0.15;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #FFFFFF;
    margin: 0 0 6px 0;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.75);
    margin: 0;
    font-weight: 300;
    letter-spacing: 0.02em;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    color: #fff;
    font-size: 0.75rem;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 14px;
    letter-spacing: 0.08em;
    font-weight: 500;
    text-transform: uppercase;
}

/* Zone de résultat */
.result-success {
    background: #EDFAF0;
    border: 1.5px solid #4A7A2E;
    border-radius: 14px;
    padding: 20px 24px;
    margin: 16px 0;
}
.result-variety {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #2D4A1E;
    margin: 0 0 4px 0;
}
.result-confidence {
    font-size: 0.9rem;
    color: #4A7A2E;
    font-weight: 500;
    margin: 0;
}

/* Alerte rejet */
.result-alert {
    background: #FFF8EC;
    border: 1.5px solid #D4830A;
    border-radius: 14px;
    padding: 20px 24px;
    margin: 16px 0;
}
.alert-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #8A5200;
    margin: 0 0 6px 0;
}
.alert-text {
    font-size: 0.88rem;
    color: #8A5200;
    margin: 0;
    line-height: 1.5;
}

/* Carte info variété */
.variety-info {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 16px 20px;
    margin-top: 12px;
    border: 1px solid #E8E2D9;
}
.variety-info h4 {
    font-family: 'DM Serif Display', serif;
    color: #2D4A1E;
    margin: 0 0 6px 0;
    font-size: 1rem;
}
.variety-info p {
    font-size: 0.85rem;
    color: #666;
    margin: 0;
    line-height: 1.6;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 0.78rem;
    color: #AAA;
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid #E8E2D9;
}

/* Bouton Streamlit */
.stButton > button {
    background: linear-gradient(135deg, #2D4A1E, #4A7A2E) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
}
</style>
""", unsafe_allow_html=True)

# --- DONNÉES VARIÉTÉS ---
VARIETY_INFO = {
    'Arborio': {
        'emoji': '🇮🇹',
        'desc': 'Riz italien à grains courts et ronds, riche en amidon. Idéal pour les risottos grâce à sa texture crémeuse.'
    },
    'Basmati': {
        'emoji': '🇮🇳',
        'desc': 'Riz indien à grains longs et fins, au parfum floral caractéristique. Incontournable dans la cuisine d\'Asie du Sud.'
    },
    'Ipsala': {
        'emoji': '🇹🇷',
        'desc': 'Riz turc à grains moyens et aplatis, cultivé dans la région d\'Ipsala. Polyvalent en cuisine.'
    },
    'Jasmine': {
        'emoji': '🇹🇭',
        'desc': 'Riz thaïlandais légèrement parfumé, à grains longs. Texture moelleuse après cuisson, emblématique de l\'Asie du Sud-Est.'
    },
    'Karacadag': {
        'emoji': '🇹🇷',
        'desc': 'Riz turc à indication géographique protégée, cultivé sur le plateau de Karacadag. Grains courts et saveur distinctive.'
    }
}

# --- MODÈLE ---
MODEL_URL = "https://huggingface.co/Hadjita/classification-resnet-final/resolve/main/ResNet50_BON.keras"
MODEL_PATH = "ResNet50_BON.keras"
CLASSES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Seuils de rejet
SEUIL_CONFIANCE = 0.65      # En dessous → image rejetée
SEUIL_ENTROPIE = 1.2        # Au dessus → distribution trop uniforme → rejetée

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Chargement du modèle depuis HuggingFace..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

def compute_entropy(preds):
    """Entropie de Shannon — mesure d'incertitude."""
    preds = np.clip(preds, 1e-10, 1.0)
    return -np.sum(preds * np.log(preds))

def is_valid_rice_image(preds):
    """Retourne True si l'image ressemble à du riz connu."""
    confiance_max = np.max(preds)
    entropie = compute_entropy(preds)
    if confiance_max < SEUIL_CONFIANCE:
        return False, f"Confiance trop faible ({confiance_max*100:.1f}%)"
    if entropie > SEUIL_ENTROPIE:
        return False, f"Distribution trop incertaine (entropie={entropie:.2f})"
    return True, ""

# --- CHARGEMENT ---
model = load_my_model()

# --- INTERFACE ---
st.markdown("""
<div class="hero">
    <div class="hero-badge">ServAgri · IA agricole</div>
    <h1 class="hero-title">Mon Assistant Rizicole<br>Cultural</h1>
    <p class="hero-subtitle">Classification de variétés de riz par intelligence artificielle · ResNet50 · 98.8% d'accuracy</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Chargez une image d'un grain de riz",
    type=["jpg", "jpeg", "png"],
    help="L'image doit montrer un grain de riz sur fond neutre pour de meilleurs résultats."
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.image(image, caption="Image chargée", use_container_width=True)
    with col2:
        st.markdown("**Variétés reconnues :**")
        for v, info in VARIETY_INFO.items():
            st.markdown(f"{info['emoji']} {v}")

    st.markdown("---")

    if st.button("🔍 Lancer le diagnostic"):
        with st.spinner("Analyse en cours..."):
            # Prétraitement
            img = image.resize((224, 224))
            img_array = np.array(img).astype('float32')
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            # Prédiction
            preds = model.predict(img_array)[0]
            idx = np.argmax(preds)
            confiance = preds[idx] * 100
            variete = CLASSES[idx]

            # Vérification
            valide, raison = is_valid_rice_image(preds)

        if not valide:
            st.markdown(f"""
            <div class="result-alert">
                <p class="alert-title">⚠️ Image non reconnue</p>
                <p class="alert-text">
                    Le modèle ne peut pas identifier cette image comme un grain de riz connu.<br>
                    <strong>Raison :</strong> {raison}<br><br>
                    Veuillez charger une image claire d'un grain de riz appartenant aux variétés
                    Arborio, Basmati, Ipsala, Jasmine ou Karacadag.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            info = VARIETY_INFO[variete]
            st.markdown(f"""
            <div class="result-success">
                <p class="result-variety">{info['emoji']} {variete}</p>
                <p class="result-confidence">Confiance : {confiance:.1f}%</p>
            </div>
            <div class="variety-info">
                <h4>À propos de cette variété</h4>
                <p>{info['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Distribution des probabilités")
            df = pd.DataFrame({'Probabilité (%)': preds * 100}, index=CLASSES)
            st.bar_chart(df, color="#4A7A2E")

st.markdown("""
<div class="footer">
    RIZIERE · Mon Assistant Rizicole · Modèle ResNet50 entraîné sur 5 variétés de riz<br>
    Projet Deep Learning · 2025
</div>
""", unsafe_allow_html=True)
