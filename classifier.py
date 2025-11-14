import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import io

try:
    from tensorflow.keras.models import load_model
except ImportError:
    try:
        from keras.models import load_model
    except ImportError:
        st.error("TensorFlow/Keras not found. Please install: pip install tensorflow")
        st.stop()

MODEL_PATH = "model_tensorflow/keras_model.h5"
LABELS_PATH = "model_tensorflow/labels.txt"
INPUT_SIZE = (224, 224)


@st.cache_resource
def get_model():
    try:
        model = load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None


@st.cache_data
def load_labels():
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        # If labels are like '0 Star Fish', remove the leading index
        parsed = [l.partition(' ')[2] if ' ' in l else l for l in lines]
        return parsed
    except Exception as e:
        st.error(f"Gagal memuat labels: {e}")
        return []


def preprocess_image(image: Image.Image):
    # Ensure RGB
    image = image.convert('RGB')
    # Resize and crop to square then to target size
    image = ImageOps.fit(image, INPUT_SIZE, Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32)
    normalized = (image_array / 127.5) - 1.0
    return np.expand_dims(normalized, axis=0)


def predict_image(model, labels, image: Image.Image, top_k=3):
    if model is None:
        return None
    input_arr = preprocess_image(image)
    preds = model.predict(input_arr)
    probs = preds[0]
    # get top k
    top_idx = probs.argsort()[-top_k:][::-1]
    results = []
    for idx in top_idx:
        label = labels[idx] if idx < len(labels) else f"Class {idx}"
        results.append({"label": label, "confidence": float(probs[idx])})
    return results


def main():
    st.title("🧭 Marine Image Classifier")
    st.write("Ambil gambar dari kamera atau unggah file untuk mengklasifikasikan jenis hewan laut.")

    model = get_model()
    labels = load_labels()

    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.image("model_tensorflow/labels.txt", caption="Label file (for reference)", width=1)

    # Camera input (Streamlit provides camera capture in the browser)
    st.subheader("📷 Ambil foto (kamera) atau unggah gambar")
    cam_file = st.camera_input("Ambil foto menggunakan kamera")
    upload_file = st.file_uploader("Atau unggah gambar", type=['png', 'jpg', 'jpeg'])

    image = None
    if cam_file is not None:
        image = Image.open(cam_file)
    elif upload_file is not None:
        image = Image.open(upload_file)

    if image is not None:
        st.image(image, caption='Input Image', use_column_width=True)

        if st.button("🔍 Prediksi"):
            with st.spinner('Memproses...'):
                results = predict_image(model, labels, image, top_k=5)
            if results:
                st.success("Hasil prediksi:")
                for r in results:
                    st.write(f"- {r['label']} — {r['confidence']*100:.2f}%")
            else:
                st.error("Model belum dimuat atau terjadi kesalahan saat prediksi.")

    st.markdown("---")
    st.markdown("**Catatan:** Model dilatih menggunakan Teachable Machine. Jika akurasi rendah, coba gunakan gambar yang jelas dan terpusat pada objek.")
