"""
market_foresight.py

Lightweight Streamlit page that exposes a `main()` function and uses
the `MarketForesight` class implemented in `market_model.py`.

This file was corrupted and has been replaced with a minimal, safe UI
that supports sample-data generation, CSV/Excel upload, selecting a
fish type, running ARIMA predictions (via MarketForesight), showing
results, plotting, and exporting CSV results.
"""

import io
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from market_model import MarketForesight


def _ensure_forecaster():
    if "market_forecaster" not in st.session_state:
        st.session_state["market_forecaster"] = MarketForesight()
    return st.session_state["market_forecaster"]


def _plot_predictions(df, hist_data=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot historical data if available
    if hist_data is not None and not hist_data.empty:
        hist_dates = hist_data.index
        hist_values = hist_data['Harga_Per_Kg']
        ax.plot(hist_dates, hist_values, 'b-', label='Data Historis', alpha=0.7)
    
    # Plot predictions with confidence intervals if available
    pred_dates = pd.to_datetime(df["Tanggal"])
    ax.plot(pred_dates, df["Prediksi"], 'r-', marker='o', label='Prediksi ARIMA')
    
    if "Batas_Bawah" in df.columns and "Batas_Atas" in df.columns:
        ax.fill_between(pred_dates, 
                       df["Batas_Bawah"], 
                       df["Batas_Atas"], 
                       color='red', 
                       alpha=0.1, 
                       label='Interval Kepercayaan')
    
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga (Rp)")
    ax.set_title("Prediksi Harga ARIMA")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Market Foresight", layout="wide")
    st.title("Prediksi Harga Ikan — Market Foresight (ARIMA)")
    
    # Informasi model
    with st.expander("ℹ️ Informasi Model ARIMA"):
        st.markdown("""
        **Model ARIMA (Autoregressive Integrated Moving Average)**
        - Metode statistik untuk analisis dan prediksi data time series
        - Mempertimbangkan tren, musiman, dan pola historis
        - Komponen:
            - AR (Autoregressive): Menggunakan nilai historis
            - I (Integrated): Transformasi diferensiasi untuk stasionaritas
            - MA (Moving Average): Memperhitungkan error prediksi sebelumnya
        """)

    forecaster = _ensure_forecaster()
    
    st.sidebar.header("Data & Pengaturan")
    uploaded = st.sidebar.file_uploader("Upload CSV atau Excel (opsional)", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            forecaster.data = df
            st.sidebar.success(f"Data dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
        except Exception as e:
            st.sidebar.error(f"Gagal membaca file: {e}")

    if st.sidebar.button("Buat Data Sampel (36 bulan)"):
        sample = forecaster.create_sample_data(n_months=36)
        forecaster.data = sample
        st.sidebar.success("Data sampel dibuat dan dimuat ke session.")

    if forecaster.data is None:
        st.info("Tidak ada data. Upload file atau buat data sampel dari sidebar.")
        return

    # Select fish types available in the data
    if "Jenis_Ikan" in forecaster.data.columns:
        fish_options = sorted(forecaster.data["Jenis_Ikan"].dropna().unique())
    else:
        st.error("Kolom 'Jenis_Ikan' tidak ditemukan di data. Pastikan format file sesuai.")
        return

    selected_fish = st.selectbox("Pilih Jenis Ikan", fish_options)
    periods = st.number_input("Periode prediksi (bulan)", min_value=1, max_value=60, value=6)
    threshold = st.number_input("Ambang alert (%) untuk perubahan", min_value=1, max_value=100, value=10)

    if st.button("Jalankan Prediksi ARIMA"):
        with st.spinner("Memproses prediksi..."):
            try:
                # Prepare data and generate predictions
                forecaster.preprocess_data(fish_type=selected_fish)
                _ = forecaster.train_arima()
                # Generate predictions
                preds = forecaster.predict_future(periods=periods, method="arima")

                st.subheader("Hasil Prediksi")
                preds_display = preds.copy()
                preds_display["Tanggal"] = pd.to_datetime(preds_display["Tanggal"]).dt.strftime("%Y-%m-%d")
                st.dataframe(preds_display)

                # Plot both historical and predicted data
                fig = _plot_predictions(preds, forecaster.processed_data)
                st.pyplot(fig)

                alerts = forecaster.generate_alerts(preds, threshold_percent=threshold, method='arima')
                if not alerts.empty:
                    st.warning("Perhatian: Terdapat alert berikut berdasarkan ambang yang ditetapkan:")
                    st.dataframe(alerts)
                else:
                    st.success("Tidak ada alert signifikan pada prediksi.")

                csv_buf = io.StringIO()
                preds.to_csv(csv_buf, index=False)
                st.download_button("Download Hasil Prediksi (CSV)", data=csv_buf.getvalue(), file_name="prediksi_harga.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")


if __name__ == "__main__":
    main()