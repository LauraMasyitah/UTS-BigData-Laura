import streamlit as st
from PIL import Image
import numpy as np
import os
from ultralytics import YOLO
import cv2

st.set_page_config(page_title="UTS Big Data – Dashboard", layout="wide")

st.title("Dashboard UTS – Klasifikasi & Deteksi Objek")
st.write("Unggah gambar untuk melihat hasil klasifikasi dan deteksi menggunakan model.")

# =====================================================
# LOAD MODEL
# =====================================================

# --- Load TFLite Classification Model ---
classification_path = "model/model_classification.tflite"

if os.path.exists(classification_path):
    interpreter = None
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=classification_path)
        interpreter.allocate_tensors()
    except Exception:
        st.error("Gagal load model TFLite.")
else:
    st.error("Model klasifikasi (.tflite) tidak ditemukan.")


# --- Load YOLO Detection Model ---
detection_path = "model/model_detection.pt"

if os.path.exists(detection_path):
    try:
        detection_model = YOLO(detection_path)
    except:
        detection_model = None
        st.error("Gagal load model deteksi YOLO.")
else:
    st.error("Model deteksi (.pt) tidak ditemukan.")


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
# KLASIFIKASI (TFLITE)
# =====================================================

if image_to_predict is not None and 'interpreter' in locals() and interpreter is not None:
    st.subheader("Hasil Klasifikasi")

    # Preprocess gambar
    img = image_to_predict.convert("RGB").resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # TFLite input & output
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    interpreter.set_tensor(input_index, img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_index)

    # Label (9 kelas)
    class_labels = [
        "broken_stitch", "needle_mark", "pinched_fabric",
        "vertical", "defect_free", "hole",
        "horizontal", "lines", "stain"
    ]

    predicted_label = class_labels[np.argmax(prediction)]

    st.success(f"**Prediksi Klasifikasi: {predicted_label}**")


# =====================================================
# DETEKSI (YOLO)
# =====================================================

if image_to_predict is not None and 'detection_model' in locals() and detection_model is not None:
    st.subheader("Hasil Deteksi Objek")

    results = detection_model(image_to_predict)
    result_image = results[0].plot()

    st.image(result_image, caption="Hasil Deteksi Objek", use_column_width=True)

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        st.write("Objek terdeteksi:")
        for b in boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            class_name = results[0].names[cls_id]
            st.write(f"- **{class_name}** ({conf:.2f})")
    else:
        st.write("Tidak ada objek terdeteksi.")
