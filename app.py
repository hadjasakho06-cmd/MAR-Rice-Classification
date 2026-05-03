import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import urllib.request
import pandas as pd

# --- CONFIGURATION ---
st.set_page_config(page_title="ServAgri - Diagnostic", page_icon="🌾")

MODEL_URL = "https://huggingface.co/Hadjita/classification-resnet-final/resolve/main/ResNet50_Final_Classification_riz.keras"
MODEL_PATH = "ResNet50_Final_Classification_riz.keras"

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_my_model()

# --- INTERFACE ---
st.title("🌾 Mon Assistant Cultural (MAC)")
uploaded_file = st.file_uploader("Chargez un grain de riz", type=["jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, width=300)
    
    # Sélecteur pour tester l'ordre des classes en direct
    ordre_choisi = st.radio("Si le résultat est faux, changez l'ordre ici :", 
                            ["Ordre A (Standard)", "Ordre B (Inversé)"])
    
    if ordre_choisi == "Ordre A (Standard)":
        classes = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
    else:
        # On inverse juste les deux premiers pour tester ton cas Arborio/Basmati
        classes = ['Basmati', 'Arborio', 'Ipsala', 'Jasmine', 'Karacadag']

    # --- PRÉTRAITEMENT ---
    img = image.resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0  # Normalisation standard
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Lancer le diagnostic 🚀"):
        preds = model.predict(img_array)[0]
        idx = np.argmax(preds)
        
        st.subheader(f"Résultat : {classes[idx]}")
        st.write(f"Confiance : {preds[idx]*100:.2f}%")
        
        # Affichage pour comprendre le décalage
        st.bar_chart(pd.DataFrame({'Probabilité': preds}, index=classes))
