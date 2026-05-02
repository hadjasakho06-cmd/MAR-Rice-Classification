import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import urllib.request
import pandas as pd
from tensorflow.keras.applications.vgg16 import preprocess_input

# Configuration
st.set_page_config(page_title="ServAgri - MAC", page_icon="🌾")

MODEL_URL = "https://huggingface.co/Hadjita/vgg16-rice-model/resolve/main/Modele_riz_final.keras"
MODEL_PATH = "Modele_riz_final.keras"

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_my_model()

st.title("🌾 Mon Assistant Cultural (MAC)")
uploaded_file = st.file_uploader("Image du grain de riz", type=["jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=True)
    
    # Prétraitement VGG16 standard
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # --- LA CORRECTION EST ICI ---
    img_array = preprocess_input(img_array) 

    if st.button("Lancer l'analyse"):
        predictions = model.predict(img_array)
        classes = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
        
        df_results = pd.DataFrame({'Variété': classes, 'Confiance': predictions[0]}).set_index('Variété')
        idx_max = np.argmax(predictions)
        
        st.write("---")
        st.success(f"✅ Variété détectée : **{classes[idx_max]}**")
        st.info(f"📊 Confiance : **{predictions[0][idx_max]*100:.2f}%**")
        st.bar_chart(df_results)
