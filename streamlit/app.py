# coding: utf-8
"""
Streamlit App — Detección de Neumonía con Generación de Imágenes Sintéticas
-------------------------------------------------------------------------------

• Sube una radiografía de tórax y recibe una predicción (Neumonía / Normal).
• Usa un autoencoder preentrenado (`autoencoder.h5`) como extractor y generador.
• Clasificación con modelo VGG16 entrenado por el usuario (`modelo_vgg16.h5`).
• Permite generar imágenes nuevas usando el decodificador del autoencoder.

**Requisitos mínimos**:
```bash
pip install streamlit tensorflow pillow numpy transformers
```
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from PIL import Image
import base64

# --------------------------------------------------
# Configuración general
# --------------------------------------------------
st.set_page_config(page_title="Detección de Neumonía", page_icon="🫁", layout="centered")

st.markdown(
    """
    <style>
        .block-container {max-width: 900px;}
        footer {visibility: hidden;}
        .stImage > img {max-width: 300px; height: auto;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Carga de modelos
# --------------------------------------------------
@st.cache_resource

def load_autoencoder(path="/Users/martaalvarez/Pneumonia-Detection/model/autoencoder.h5"):
    return load_model(path, compile=False)

autoencoder = load_autoencoder()

@st.cache_resource
def load_classifier(path="/Users/martaalvarez/Pneumonia-Detection/model/modelo_vgg16.h5"):
    return load_model(path)

classifier = load_classifier()

# --------------------------------------------------
# Utilidades
# --------------------------------------------------
@st.cache_data
def preprocess_image(uploaded_file):
    image_rgb = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    image_gray = Image.open(uploaded_file).convert("L").resize((150, 150))

    img_rgb = np.array(image_rgb, dtype="float32") / 255.0
    img_gray = np.array(image_gray, dtype="float32") / 255.0

    return img_rgb, img_gray[..., np.newaxis]  # (224, 224, 3), (150, 150, 1)

def classify_image_with_vgg(img: np.ndarray) -> str:
    pred = classifier.predict(np.expand_dims(img, 0))[0][0]
    return "Neumonía" if pred > 0.5 else "Normal"

# --------------------------------------------------
# App principal
# -------------------------------------------------
st.title("🫁 Detección de Neumonía en Radiografías")

uploaded_file = st.file_uploader(" ", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    def get_image_base64(uploaded_file):
        img_bytes = uploaded_file.read()
        encoded = base64.b64encode(img_bytes).decode()
        return f"data:image/png;base64,{encoded}"

    img_url = get_image_base64(uploaded_file)
    st.markdown(
        f"<div style='text-align: center'><img src='{img_url}' width='360'></div>",
        unsafe_allow_html=True
    )

    # Procesar y predecir
    img_rgb, img_gray = preprocess_image(uploaded_file)

    # Clasificación real con modelo VGG16
    pred_label = classify_image_with_vgg(img_rgb)

    if pred_label == "Neumonía":
        st.error("La radiografía muestra signos de **Neumonía**")
    else:
        st.success("La radiografía **no** muestra signos de **Neumonía**")

    # --------------------------------------------------
    # Generación de imágenes sintéticas
    # --------------------------------------------------
    generate_more = st.radio(
        "¿Quieres generar más imágenes similares para prácticas?",
        ("No", "Sí"),
        index=0
    )

    if generate_more == "Sí":
        st.markdown("### 🧬 Imágenes generadas con el decoder del autoencoder")

        num_images = st.number_input(
            "Introduce cuántas imágenes quieres generar (máximo 10):",
            min_value=1,
            max_value=10,
            value=4,
            step=1
        )

        if st.button("Generar imágenes sintéticas"):
            decoder_input = autoencoder.get_layer(index=6).output
            decoder = Model(inputs=decoder_input, outputs=autoencoder.output)

            # Extraemos la representación latente desde el encoder
            encoder_model = Model(inputs=autoencoder.input, outputs=decoder_input)
            latent = encoder_model.predict(np.expand_dims(img_gray, 0))
            latent_shape = decoder_input.shape[1:]

            for row in range((num_images + 1) // 2):
                row_images = []
                for col in range(2):
                    idx = row * 2 + col
                    if idx >= num_images:
                        break
                    noise = np.random.normal(0, 0.05, size=latent.shape).astype("float32")
                    synthetic_latent = latent + noise
                    generated = decoder.predict(synthetic_latent)[0, ..., 0]
                    generated = np.clip(generated * 255, 0, 255).astype("uint8")
                    pil_img = Image.fromarray(generated, mode="L")
                    row_images.append(pil_img)
                cols = st.columns(len(row_images))
                for i, img in enumerate(row_images):
                    cols[i].image(img, caption=f"Sintética #{row * 2 + i + 1}")

# --------------------------------------------------
# Estilo: ocultar menú y footer
# --------------------------------------------------
st.markdown("<style>#MainMenu {visibility:hidden;} header {visibility:hidden;}</style>", unsafe_allow_html=True)
