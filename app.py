import streamlit as st
from PIL import Image
import numpy as np
import os
import tensorflow as tf

st.set_page_config(page_title="UTS Big Data – Dashboard", layout="wide")

st.title("Dashboard UTS – Klasifikasi Cacat Kain")
st.write("Unggah gambar untuk melihat hasil klasifikasi menggunakan model.")

# =====================================================
# LOAD MODEL KLASIFIKASI (.h5)
# =====================================================

model_path = "model/model_classification.h5"

if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Model klasifikasi berhasil dimuat.")
    except:
        model = None
        st.error("❌ Gagal memuat model klasifikasi.")
else:
    model = None
    st.error("❌ File model tidak ditemukan di folder /model.")


# =====================================================
# UPLOAD GAMBAR
# =====================================================

st.subheader("Upload Gambar")

uploaded_file = st.file_uploader("Pilih gambar (.jpg/.png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)
else:
    image = None


# =====================================================
# PROSES KLASIFIKASI
# =====================================================

if image is not None and model is not None:
    st.subheader("Hasil Klasifikasi")

    # Preprocessing gambar
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array)

    class_labels = [
        "broken_stitch", "needle_mark", "pinched_fabric",
        "vertical", "defect_free", "hole",
        "horizontal", "lines", "stain"
    ]

    predicted_label = class_labels[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    st.success(f"**Prediksi: {predicted_label}** (akurasi: {confidence:.2f})")
