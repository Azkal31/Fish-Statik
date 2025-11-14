# Setup and Usage Guide - Marine Image Classifier

## Integrasi Model Teachable Machine ke Sistem Analisis Perikanan

Sistem ini telah berhasil diintegrasikan dengan model klasifikasi gambar hewan laut yang Anda latih menggunakan Teachable Machine.

### Fitur yang Ditambahkan:

1. **Marine Image Classifier** - Menu baru di Streamlit app
2. **Camera Capture** - Ambil foto langsung dari kamera browser
3. **File Upload** - Upload file gambar (PNG, JPG, JPEG)
4. **Prediksi Real-time** - Klasifikasi dengan confidence score
5. **21 Kelas Hewan Laut** - Sesuai dengan model Teachable Machine Anda

### Kelas yang Dapat Dideteksi:

- Star Fish, Squid, Shrimp, Sea Urchins, Puffers
- Octopus, Nudibranchs, Lobster, Jelly Fish, Fish
- Eel, Crabs, Coral, Whale, Shark
- Sea Turtle, Rays, Sea Horse, Dolphin, Clams
- Not a marine animal

### Cara Menjalankan:

#### Opsi 1: Menggunakan Virtual Environment yang Sudah Ada
```powershell
# Aktivasi environment yang sudah memiliki TensorFlow
# (Pastikan TensorFlow 2.15.0, Pillow, dan Streamlit terinstall)
streamlit run main.py
```

#### Opsi 2: Setup Fresh Environment (Direkomendasikan)
```powershell
# 1. Buat virtual environment baru
python -m venv marine_env

# 2. Aktivasi environment
marine_env\Scripts\Activate.ps1

# 3. Install packages yang diperlukan
pip install tensorflow==2.15.0 Pillow==10.0.1 streamlit==1.38.0 numpy==1.24.4

# 4. Install dependencies tambahan
pip install pandas matplotlib plotly

# 5. Jalankan aplikasi
streamlit run main.py
```

### Cara Menggunakan Marine Image Classifier:

1. **Buka aplikasi** - http://localhost:8502
2. **Pilih menu** - "Marine Image Classifier" 
3. **Ambil/Upload gambar**:
   - Klik "Ambil foto menggunakan kamera" untuk camera capture
   - Atau drag & drop file gambar ke "Atau unggah gambar"
4. **Klik "🔍 Prediksi"** untuk mendapatkan hasil klasifikasi
5. **Lihat hasil** - Top 5 prediksi dengan confidence score

### File yang Ditambahkan/Dimodifikasi:

- `classifier.py` - Module klasifikasi gambar marine
- `main.py` - Ditambah menu "Marine Image Classifier" 
- `requirements.txt` - Ditambah Pillow untuk image processing
- `model_tensorflow/keras_model.h5` - Model Teachable Machine Anda
- `model_tensorflow/labels.txt` - 21 label kelas hewan laut

### Troubleshooting:

**Jika TensorFlow tidak terdeteksi:**
```powershell
# Check instalasi TensorFlow
pip list | findstr tensorflow

# Reinstall jika perlu
pip uninstall tensorflow tensorflow-intel
pip install tensorflow==2.15.0
```

**Jika ada error NumPy compatibility:**
```powershell
pip install --force-reinstall numpy==1.24.4
```

**Jika webcam tidak bekerja:**
- Pastikan browser memberikan izin akses kamera
- Gunakan HTTPS atau localhost (bukan IP lain)
- Coba browser lain (Chrome/Firefox)

### Catatan Teknis:

- Model: Keras/TensorFlow (.h5 format)
- Input size: 224x224 RGB
- Preprocessing: Normalisasi (-1 to 1)
- Output: 21 classes dengan softmax
- Epochs: 50, Batch size: 12 (sesuai training Anda)

### Next Steps:

Jika ingin menambahkan fitur lain:
- REST API endpoint untuk mobile apps
- Batch processing untuk multiple images  
- Integration dengan database results
- Performance monitoring dan logging

Model siap digunakan untuk produksi! 🐟🦑🦐