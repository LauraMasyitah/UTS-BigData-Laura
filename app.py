import streamlit as st
from PIL import Image
import numpy as np
import os
import gdown

st.set_page_config(page_title="UTS Big Data – Dashboard", layout="wide")

st.title("Dashboard UTS – Klasifikasi & Deteksi Objek")
st.write("Unggah gambar untuk melihat hasil klasifikasi dan deteksi menggunakan model.")

import tensorflow as tf
from ultralytics import YOLO

# =====================================================
# DOWNLOAD MODEL DARI GOOGLE DRIVE (agar Streamlit bisa load)
# =====================================================

os.makedirs("model", exist_ok=True)

# Link Google Drive (ganti ID saja)
classification_id = "1uuibbXLgeKR9IurVz2bOrn6LPOYKFWHX"
detection_id = "1ltzOqydL9Om99kZt_731SVhzeNG_bC4T"

classification_model_path = "model/model_classification.h5"
detection_model_path = "model/model_detection.pt"

# Download file jika belum ada
if not os.path.exists(classification_model_path):
    st.write("Mengunduh model klasifikasi...")
    gdown.download(f"https://drive.google.com/uc?id={classification_id}", classification_model_path, quiet=False)

if not os.path.exists(detection_model_path):
    st.write("Mengunduh model deteksi...")
    gdown.download(f"https://drive.google.com/uc?id={detection_id}", detection_model_path, quiet=False)

# =====================================================
# LOAD MODEL
# =====================================================

# Load model klasifikasi
try:
    classification_model = tf.keras.models.load_model(classification_model_path)
except:
    classification_model = None
    st.error("❌ Gagal load model klasifikasi (.h5)")

# Load model deteksi
try:
    detection_model = YOLO(detection_model_path)
except:
    detection_model = None
    st.error("❌ Gagal load model deteksi (.pt)")


# =====================================================
# UPLOAD GAMBAR
# =====================================================

st.subheader("Upload Gambar")

uploaded_file = st.file_uploader("Pilih gambar (.jpg, .png)", type=["jpg", "jpeg", "png"])

image_to_predict = None
if uploaded_file:
    image_to_predict = Image.open(uploaded_file)
    st.image(image_to_predict, caption="Gambar yang diunggah", use_column_width=True)


# =====================================================
# KLASIFIKASI (.h5)
# =====================================================

if image_to_predict is not None and classification_model is not None:
    st.subheader("Hasil Klasifikasi")

    img = image_to_predict.convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = classification_model.predict(img_array)

    class_labels = [
        "broken_stitch", "needle_mark", "pinched_fabric",
        "vertical", "defect_free", "hole",
        "horizontal", "lines", "stain"
    ]

    predicted_label = class_labels[np.argmax(prediction)]

    st.success(f"**Prediksi Klasifikasi: {predicted_label}**")


# =====================================================
# DETEKSI OBJEK (.pt)
# =====================================================

if image_to_predict is not None and detection_model is not None:
    st.subheader("Hasil Deteksi Objek")

    results = detection_model(image_to_predict)
    result_image = results[0].plot()

    st.image(result_image, caption="Hasil Deteksi Objek", use_column_width=True)

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        st.write("Objek terdeteksi:")
        for b in boxes:
            class_name = results[0].names[int(b.cls[0])]
            conf = float(b.conf[0])
            st.write(f"- **{class_name}** ({conf:.2f})")
    else:
        st.write("Tidak ada objek terdeteksi.")
