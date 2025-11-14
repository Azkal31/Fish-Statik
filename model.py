import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

def main():
    st.title("🎣 Model Prediksi Sistem Stok Ikan")
    st.write("Prediksi CPUE dan Produksi Total berdasarkan data penangkapan ikan di Karangantu.")

    # -----------------------------
    # 1️⃣ Load Data
    # -----------------------------
    try:
        df = pd.read_excel(
            r"D:\BERKULIAH DI UPI SERANG\SEMESTER 5\ASIK 2025\analisis-perikanan\DATA\Data penangkapan ikan karangantu.xlsx",
            engine="openpyxl"
        )
    except Exception as e:
        st.error(f"Gagal memuat file Excel: {e}")
        st.stop()  # lebih baik daripada 'return' di luar fungsi


    # -----------------------------    streamlit run main.py
    # 2️⃣ Pembersihan & Pra-pemrosesan Data
    # -----------------------------
    for col in ['Produksi_total (kg)', 'Effort (trip)']:
        df[col] = df[col].fillna(df[col].mean())

    df_cleaned = df.dropna(subset=['CPUE']).copy()

    # Tangani outlier CPUE (IQR method)
    Q1 = df_cleaned['CPUE'].quantile(0.25)
    Q3 = df_cleaned['CPUE'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_cleaned['CPUE'] = np.where(
        df_cleaned['CPUE'] > upper_bound, upper_bound,
        np.where(df_cleaned['CPUE'] < lower_bound, lower_bound, df_cleaned['CPUE'])
    )

    # One-hot encoding kolom "Alat tangkap"
    df_encoded = pd.get_dummies(df_cleaned, columns=['Alat tangkap'])
    features_encoded = ['Effort (trip)'] + [col for col in df_encoded.columns if 'Alat tangkap' in col]
    X = df_encoded[features_encoded]
    y = df_encoded['CPUE']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    training_columns = X_train.columns

    # -----------------------------
    # 3️⃣ Pelatihan Model Ridge Regression
    # -----------------------------
    best_params_ridge = {"alpha": 1.0}  # default, ubah sesuai hasil tuning jika ada
    model = Ridge(**best_params_ridge)
    model.fit(X_train, y_train)

    # -----------------------------
    # 4️⃣ Input Form Streamlit
    # -----------------------------
    st.subheader("🧮 Masukkan Parameter Prediksi")

    col1, col2 = st.columns(2)
    with col1:
        tahun = st.number_input("Tahun", min_value=2000, max_value=2100, value=2024)
        effort = st.number_input("Effort (trip)", min_value=0.0, value=100.0, step=10.0)
    with col2:
        alat_tangkap = st.selectbox("Alat tangkap", df['Alat tangkap'].unique())

    # -----------------------------
    # 5️⃣ Tombol Prediksi
    # -----------------------------
    if st.button("🔍 Prediksi CPUE & Produksi"):
        input_df = pd.DataFrame({
            'Tahun': [tahun],
            'Alat tangkap': [alat_tangkap],
            'Effort (trip)': [effort]
        })

        input_encoded = pd.get_dummies(input_df, columns=['Alat tangkap'])
        processed_input = input_encoded.reindex(columns=training_columns, fill_value=0)
        processed_input = processed_input.drop(columns=['Tahun'], errors='ignore')

        predicted_cpue = model.predict(processed_input)[0]
        predicted_production = predicted_cpue * effort

        st.success(f"**Prediksi CPUE:** {predicted_cpue:.2f}")
        st.info(f"**Prediksi Produksi Total (kg):** {predicted_production:.2f}")

    # -----------------------------
    # 6️⃣ Info tambahan
    # -----------------------------
    st.caption("Model Ridge Regression dilatih ulang dari data penangkapan ikan Karangantu.")

