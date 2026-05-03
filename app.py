import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np
import pandas as pd
import os
import urllib.request

st.set_page_config(page_title="ServAgri - Diagnostic", page_icon="🌾")

MODEL_URL = "https://huggingface.co/Hadjita/classification-resnet-final/resolve/main/ResNet50_BON.keras"
MODEL_PATH = "ResNet50_BON.keras"

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_my_model()

CLASSES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

st.title("🌾 Mon Assistant Cultural (MAC)")
st.write("Chargez une image d'un grain de riz pour identifier sa variété.")

uploaded_file = st.file_uploader("Chargez un grain de riz", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, width=300)

    if st.button("Lancer le diagnostic 🚀"):
        img = image.resize((224, 224))
        img_array = np.array(img).astype('float32')
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)[0]
        idx = np.argmax(preds)
        confiance = preds[idx] * 100

        st.success(f"Variété détectée : **{CLASSES[idx]}**")
        st.write(f"Confiance : **{confiance:.2f}%**")

        if confiance < 70:
            st.warning("⚠️ Confiance faible — image peut-être de mauvaise qualité.")

        st.write("Distribution des probabilités :")
        st.bar_chart(pd.DataFrame({'Probabilité (%)': preds * 100}, index=CLASSES))
