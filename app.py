import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np
import os
import urllib.request

# --- CONFIGURATION ---
st.set_page_config(
    page_title="RIZIERE – MAR",
    page_icon="🌾",
    layout="wide"
)

# --- CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

html, body, .stApp {
    background-color: #F2EFE9;
    font-family: 'DM Sans', sans-serif;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E3512 0%, #2D4A1E 60%, #3A5F26 100%) !important;
}
[data-testid="stSidebar"] * { color: #E8F5D8 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    font-family: 'DM Serif Display', serif !important;
}
.sidebar-logo { font-size: 3rem; text-align: center; margin-bottom: 4px; }
.sidebar-brand { font-family: 'DM Serif Display', serif; font-size: 1.5rem; color: #fff !important; text-align: center; margin-bottom: 2px; }
.sidebar-tagline { font-size: 0.72rem; color: rgba(232,245,216,0.6) !important; text-align: center; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 20px; }
.sidebar-divider { border: none; border-top: 1px solid rgba(255,255,255,0.12); margin: 14px 0; }
.sidebar-section { font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase; color: rgba(232,245,216,0.45) !important; margin-bottom: 8px; font-weight: 600; }
.variety-pill { background: rgba(255,255,255,0.08); border-radius: 8px; padding: 7px 12px; margin-bottom: 5px; font-size: 0.83rem; }
.stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.07); font-size: 0.81rem; }
.stat-val { font-weight: 600; color: #A8D97F !important; }
.tip-text { font-size: 0.79rem; line-height: 1.75; color: rgba(232,245,216,0.72) !important; }

/* HERO */
.hero {
    background: linear-gradient(135deg, #2D4A1E 0%, #4A7A2E 70%, #6BA03E 100%);
    border-radius: 18px; padding: 36px 40px; margin-bottom: 24px;
    position: relative; overflow: hidden;
}
.hero::after {
    content: "🌾"; font-size: 140px; position: absolute;
    right: 20px; top: -15px; opacity: 0.1; pointer-events: none;
}
.hero-eyebrow { font-size: 0.7rem; letter-spacing: 0.14em; text-transform: uppercase; color: rgba(255,255,255,0.55); margin-bottom: 10px; }
.hero-title { font-family: 'DM Serif Display', serif; font-size: 2.5rem; color: #fff; margin: 0 0 8px; line-height: 1.15; }
.hero-sub { font-size: 0.88rem; color: rgba(255,255,255,0.68); margin: 0; font-weight: 300; max-width: 460px; }
.hero-chips { margin-top: 18px; display: flex; gap: 8px; flex-wrap: wrap; }
.chip { background: rgba(255,255,255,0.14); color: #fff; font-size: 0.71rem; padding: 4px 12px; border-radius: 20px; font-weight: 500; letter-spacing: 0.04em; }

/* UPLOAD */
.upload-label { font-size: 0.78rem; font-weight: 600; color: #3A5226; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px; }

/* RÉSULTAT */
.result-card { border-radius: 16px; padding: 22px 26px; margin: 12px 0 10px; }
.result-success { background: #EAF7E4; border: 1.5px solid #4A7A2E; }
.result-alert { background: #FFF6E8; border: 1.5px solid #C07B10; }
.result-variety { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #1E3512; margin: 0 0 4px; }
.result-conf { font-size: 0.88rem; color: #4A7A2E; font-weight: 600; margin: 0; }
.alert-title { font-family: 'DM Serif Display', serif; font-size: 1.25rem; color: #7A4B00; margin: 0 0 8px; }
.alert-body { font-size: 0.84rem; color: #7A4B00; line-height: 1.65; margin: 0; }
.info-card { background: #fff; border-radius: 14px; padding: 16px 20px; border: 1px solid #DDD8CF; margin-top: 10px; }
.info-card h4 { font-family: 'DM Serif Display', serif; font-size: 1rem; color: #1E3512; margin: 0 0 6px; }
.info-card p { font-size: 0.83rem; color: #666; margin: 0; line-height: 1.65; }

/* BAR CHART CUSTOM */
.chart-section-title { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase; color: #3A5226; margin: 18px 0 10px; }
.bar-container { background: #fff; border-radius: 14px; padding: 18px 22px; border: 1px solid #DDD8CF; }
.bar-row { display: flex; align-items: center; margin-bottom: 9px; gap: 10px; }
.bar-row:last-child { margin-bottom: 0; }
.bar-label { width: 95px; font-size: 0.81rem; font-weight: 500; color: #333; flex-shrink: 0; }
.bar-track { flex: 1; background: #EEEBE6; border-radius: 6px; height: 20px; overflow: hidden; }
.bar-fill-best { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #2D4A1E, #6BA03E); }
.bar-fill-other { height: 100%; border-radius: 6px; background: #C5BBAD; }
.bar-pct { width: 44px; font-size: 0.81rem; font-weight: 700; color: #2D4A1E; text-align: right; flex-shrink: 0; }
.bar-pct-other { color: #888; font-weight: 400; }

/* PLACEHOLDER */
.placeholder { background: #fff; border-radius: 14px; padding: 48px 24px; border: 1.5px dashed #C8C0B4; text-align: center; color: #B0A898; }
.placeholder-icon { font-size: 2.8rem; margin-bottom: 12px; }
.placeholder-text { font-size: 0.88rem; line-height: 1.6; }

/* FOOTER */
.footer { text-align: center; font-size: 0.75rem; color: #B0A898; margin-top: 32px; padding-top: 14px; border-top: 1px solid #DDD8CF; }

/* BOUTON */
.stButton > button {
    background: linear-gradient(135deg, #2D4A1E, #4A7A2E) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 12px 28px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
    font-size: 0.95rem !important; width: 100% !important;
}
.stButton > button:hover { opacity: 0.86 !important; }
</style>
""", unsafe_allow_html=True)

# --- DONNÉES ---
VARIETY_INFO = {
    'Arborio':   {'emoji': '🫧', 'desc': 'Riz italien à grains courts, riche en amidon. Idéal pour les risottos grâce à sa texture crémeuse.'},
    'Basmati':   {'emoji': '🌬️', 'desc': 'Riz indien à grains longs et fins, au parfum floral. Incontournable dans la cuisine d\'Asie du Sud.'},
    'Ipsala':    {'emoji': '🪨', 'desc': 'Riz turc à grains moyens et aplatis, cultivé dans la région d\'Ipsala. Très polyvalent en cuisine.'},
    'Jasmine':   {'emoji': '🕊️', 'desc': 'Riz thaïlandais légèrement parfumé, à grains longs. Texture moelleuse, emblématique de l\'Asie du Sud-Est.'},
    'Karacadag': {'emoji': '🌋', 'desc': 'Riz turc IGP cultivé sur le plateau de Karacadag. Grains courts et saveur très distinctive.'},
}
CLASSES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# --- MODÈLE ---
MODEL_URL = "https://huggingface.co/Hadjita/classification-resnet-final/resolve/main/ResNet50_BON.keras"
MODEL_PATH = "ResNet50_BON.keras"

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

# --- LOGIQUE DE REJET ---
def analyse_visuelle(image: Image.Image) -> tuple:
    """
    Vérifie si l'image ressemble à un grain de riz sur fond sombre.
    Le dataset d'entraînement utilise exclusivement ce type d'image.
    """
    img_small = np.array(image.resize((224, 224)).convert('L'))

    pixels_sombres = np.sum(img_small < 60) / img_small.size   # fond noir
    pixels_clairs  = np.sum(img_small > 160) / img_small.size  # grain blanc
    contraste      = img_small.std()

    if pixels_sombres < 0.30:
        return False, "Le fond n'est pas assez sombre — image hors domaine"
    if pixels_clairs < 0.03 or pixels_clairs > 0.60:
        return False, "Aucun grain isolé détectable dans l'image"
    if contraste < 35:
        return False, "Image trop uniforme (fond blanc ou image vide)"
    return True, ""

def valider_prediction(preds) -> tuple:
    preds_clip = np.clip(preds, 1e-10, 1.0)
    confiance  = np.max(preds)
    entropie   = -np.sum(preds_clip * np.log(preds_clip))
    if confiance < 0.60:
        return False, f"Confiance insuffisante ({confiance*100:.1f}%) — grain atypique ou hors variété"
    if entropie > 1.3:
        return False, "Prédiction trop incertaine — image ambiguë"
    return True, ""

def render_bars(preds):
    best_idx = np.argmax(preds)
    html = '<div class="bar-container">'
    for i, (cls, prob) in enumerate(zip(CLASSES, preds)):
        pct = prob * 100
        is_best = (i == best_idx)
        fill_cls = "bar-fill-best" if is_best else "bar-fill-other"
        pct_cls  = "bar-pct" if is_best else "bar-pct bar-pct-other"
        html += f"""
        <div class="bar-row">
            <span class="bar-label">{VARIETY_INFO[cls]['emoji']} {cls}</span>
            <div class="bar-track"><div class="{fill_cls}" style="width:{pct:.1f}%"></div></div>
            <span class="{pct_cls}">{pct:.1f}%</span>
        </div>"""
    html += '</div>'
    return html

model = load_my_model()

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🌾</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-brand">RIZIERE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Mon Assistant Rizicole</div>', unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Variétés reconnues</div>', unsafe_allow_html=True)
    for cls, info in VARIETY_INFO.items():
        st.markdown(f'<div class="variety-pill">{info["emoji"]} <strong>{cls}</strong></div>', unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Performances</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stat-row"><span>Modèle</span><span class="stat-val">ResNet50</span></div>
    <div class="stat-row"><span>Accuracy</span><span class="stat-val">98.82%</span></div>
    <div class="stat-row"><span>Classes</span><span class="stat-val">5 variétés</span></div>
    <div class="stat-row"><span>Train</span><span class="stat-val">500 images</span></div>
    <div class="stat-row"><span>Test</span><span class="stat-val">169 images</span></div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Conseils</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tip-text">
    ✅ Grain isolé sur fond sombre<br>
    ✅ Image nette et bien éclairée<br>
    ✅ Format JPG ou PNG<br>
    ❌ Pas de fond blanc ou coloré<br>
    ❌ Pas d'image sans rapport avec le riz
    </div>
    """, unsafe_allow_html=True)

# ===== MAIN =====
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">RIZIERE · Intelligence Artificielle Agricole</div>
    <h1 class="hero-title">Mon Assistant<br>Rizicole (MAR)</h1>
    <p class="hero-sub">Identifiez instantanément la variété de votre grain de riz par intelligence artificielle.</p>
    <div class="hero-chips">
        <span class="chip">Modèle IA</span>
        <span class="chip">98.82% de Précision</span>
        <span class="chip">5 variétés</span>
        <span class="chip">Deep Learning</span>
    </div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown('<div class="upload-label">📁 Charger une image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)
        st.markdown("---")

        if st.button("🔍 Lancer le diagnostic"):
            with st.spinner("Analyse en cours..."):
                # Étape 1 : analyse visuelle
                ok_visuel, raison_visuel = analyse_visuelle(image)
                if not ok_visuel:
                    st.session_state['result'] = ('alert', raison_visuel)
                else:
                    # Étape 2 : prédiction
                    img_arr = np.expand_dims(
                        preprocess_input(np.array(image.resize((224, 224))).astype('float32')), axis=0
                    )
                    preds = model.predict(img_arr)[0]

                    # Étape 3 : validation
                    ok_pred, raison_pred = valider_prediction(preds)
                    if not ok_pred:
                        st.session_state['result'] = ('alert', raison_pred)
                    else:
                        idx = np.argmax(preds)
                        st.session_state['result'] = ('success', CLASSES[idx], float(preds[idx]*100), preds)

with col_right:
    st.markdown('<div class="upload-label">📊 Résultat du diagnostic</div>', unsafe_allow_html=True)

    if 'result' not in st.session_state:
        st.markdown("""
        <div class="placeholder">
            <div class="placeholder-icon">🔬</div>
            <div class="placeholder-text">Chargez une image et lancez<br>le diagnostic pour voir le résultat</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        status = st.session_state['result'][0]

        if status == 'alert':
            raison = st.session_state['result'][1]
            st.markdown(f"""
            <div class="result-card result-alert">
                <p class="alert-title">⚠️ Image non reconnue</p>
                <p class="alert-body">
                    Ce modèle est spécialisé dans la classification de grains de riz sur fond sombre.<br><br>
                    <strong>Raison :</strong> {raison}<br><br>
                    Veuillez charger une image d'un grain isolé (Arborio, Basmati, Ipsala, Jasmine ou Karacadag).
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            _, variete, confiance, preds = st.session_state['result']
            info = VARIETY_INFO[variete]
            st.markdown(f"""
            <div class="result-card result-success">
                <p class="result-variety">{info['emoji']} {variete}</p>
                <p class="result-conf">Confiance : {confiance:.1f}%</p>
            </div>
            <div class="info-card">
                <h4>À propos de cette variété</h4>
                <p>{info['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="chart-section-title">Distribution des probabilités</div>', unsafe_allow_html=True)
            st.markdown(render_bars(preds), unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    RIZIERE · Mon Assistant Rizicole · IA · Projet Deep Learning 2025
</div>
""", unsafe_allow_html=True)
