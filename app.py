import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="UTS Big Data – Dashboard", layout="wide")

st.title("Dashboard UTS – Klasifikasi & Deteksi Objek")
st.write("Unggah gambar untuk melihat hasil klasifikasi dan deteksi menggunakan model.")


import tensorflow as tf
from ultralytics import YOLO
import os

# ---- Load model KLASIFIKASI (.h5) ----
import gdown

# Pastikan folder model ada
os.makedirs("model", exist_ok=True)

# === Download model .h5 dari Google Drive ===
h5_url = "https://drive.google.com/uc?id=AAAABBBB"   # ganti ID file kamu
h5_output = "model/model_classification.h5"

if not os.path.exists(h5_output):
    with st.spinner("Mengunduh model klasifikasi..."):
        gdown.download(h5_url, h5_output, quiet=False)


# === Download model .pt dari Google Drive ===
pt_url = "https://drive.google.com/uc?id=CCCCDDDD"   # ganti ID file kamu
pt_output = "model/model_detection.pt"

if not os.path.exists(pt_output):
    with st.spinner("Mengunduh model deteksi..."):
        gdown.download(pt_url, pt_output, quiet=False)

classification_model_path = "model/model_classification.h5"

if os.path.exists(classification_model_path):
    classification_model = tf.keras.models.load_model(classification_model_path)
else:
    classification_model = None
    st.error("Model klasifikasi (.h5) tidak ditemukan di folder model/")

# ---- Load model DETEKSI (.pt) ----
detection_model_path = "model/model_detection.pt"

if os.path.exists(detection_model_path):
    detection_model = YOLO(detection_model_path)
else:
    detection_model = None
    st.error("Model deteksi (.pt) tidak ditemukan di folder model/")


st.subheader("Upload Gambar")

uploaded_file = st.file_uploader("Pilih gambar (.jpg, .png)", type=["jpg", "jpeg", "png"])

image_to_predict = None

if uploaded_file is not None:
    # Baca file sebagai gambar PIL
    image_to_predict = Image.open(uploaded_file)
    st.image(image_to_predict, caption="Gambar yang diunggah", use_column_width=True)


# =======================
#     KLASIFIKASI (.h5)
# =======================

if image_to_predict is not None and classification_model is not None:
    st.subheader("Hasil Klasifikasi")

    # Preprocessing gambar
    img = image_to_predict.convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)   # tambah dimensi batch

    # Prediksi
    prediction = classification_model.predict(img_array)

    # Tentukan label (ganti sesuai label kamu)
    class_labels = ["broken_stitch", "needle_mark", "pinched_fabric", "vertical",
                    "defect_free", "hole", "horizontal", "lines", "stain"]

    predicted_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_index]

    st.write(f"**Prediksi:** {predicted_label}")


# =======================
#     DETEKSI OBJEK (.pt)
# =======================

if image_to_predict is not None and detection_model is not None:
    st.subheader("Hasil Deteksi Objek")

    # Jalankan YOLO
    results = detection_model(image_to_predict)

    # YOLO menghasilkan gambar dengan bounding box
    result_image = results[0].plot()   # numpy array

    st.image(result_image, caption="Hasil Deteksi Objek", use_column_width=True)

    # List objek yang terdeteksi
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        st.write("Objek terdeteksi:")
        for b in boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            class_name = results[0].names[cls_id]
            st.write(f"- **{class_name}** (confidence: {conf:.2f})")
    else:
        st.write("Tidak ada objek yang terdeteksi.")

