
import cv2
import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Tampilkan versi OpenCV untuk memastikan berhasil
st.write("✅ OpenCV version:", cv2.__version__)

# Path model YOLO
model_path = 'best.pt'

if not os.path.exists(model_path):
    st.error(f"❌ Error: Model file not found at {model_path}")
else:
    model = YOLO(model_path)
    st.title("🧠 Deteksi Objek dengan YOLOv8")

    # Upload gambar
    uploaded_file = st.file_uploader("📤 Unggah gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Gambar Asli", use_container_width=True)

        # Jalankan deteksi
        results = model(image)
        for r in results:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])  # ubah BGR → RGB
            st.image(im, caption="🔍 Hasil Deteksi", use_container_width=True)
