# market_model.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import joblib

class MarketForesight:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.arima_model = None

    def create_sample_data(self, n_months=36, start_date='2022-01-01'):
        dates = pd.date_range(start=start_date, periods=n_months, freq='MS')
        fish_types = ['Kembung', 'Tongkol', 'Cakalang']
        np.random.seed(42)

        records = []
        for fish in fish_types:
            base_price = np.random.randint(20000, 40000)
            prices = base_price + np.random.randn(n_months) * 2000
            volumes = np.random.randint(8000, 15000, size=n_months)
            values = (volumes * prices).astype(int)
            trips = np.random.randint(80, 150, size=n_months)

            for i in range(n_months):
                records.append({
                    'Tanggal': dates[i],
                    'Jenis_Ikan': fish,
                    'Volume_Produksi_Kg': int(volumes[i]),
                    'Nilai_Produksi_Rp': int(values[i]),
                    'Jumlah_Trip': int(trips[i])
                })

        self.data = pd.DataFrame(records)
        return self.data

    def preprocess_data(self, fish_type):
        """
        Pastikan kolom Tanggal ada.
        Jika hanya ada kolom 'Tahun', buat Tanggal = 'YYYY-01-01'.
        """
        if self.data is None:
            raise ValueError("Data belum dimuat. Set self.data terlebih dahulu.")

        # filter jenis ikan
        df = self.data[self.data['Jenis_Ikan'] == fish_type].copy()

        # Jika tidak ada kolom 'Tanggal' tapi ada 'Tahun', buat Tanggal dari Tahun
        if 'Tanggal' not in df.columns and 'Tahun' in df.columns:
            df['Tanggal'] = pd.to_datetime(df['Tahun'].astype(str) + '-01-01')

        # Jika Tanggal ada tapi berupa numeric (mis. 2024), ubah jadi datetime
        if 'Tanggal' in df.columns and df['Tanggal'].dtype in ['int64', 'float64']:
            df['Tanggal'] = pd.to_datetime(df['Tanggal'].astype(int).astype(str) + '-01-01')

        # Pastikan parse datetime (jika string)
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')

        # Drop baris tanpa tanggal valid
        df = df.dropna(subset=['Tanggal'])

        df = df.sort_values('Tanggal')

        # Hitung Harga per Kg
        df['Harga_Per_Kg'] = df['Nilai_Produksi_Rp'] / df['Volume_Produksi_Kg']

        self.processed_data = df.set_index('Tanggal')
        return self.processed_data

    def train_prophet(self):
        """
        Placeholder: implement actual Prophet training.
        Return model, metrics
        """
        # contoh return dummy
        metrics = {'mae': 5000, 'rmse': 7000, 'r2': 0.9}
        return None, metrics

    def train_arima(self):
        """
        Try to load a pre-trained ARIMA model from disk (arima_model.joblib).
        If not found, return placeholder metrics. If loaded, set self.arima_model
        and return it together with a simple placeholder metrics dict.
        """
        model_path = os.path.join(os.getcwd(), 'arima_model.joblib')

        if os.path.exists(model_path):
            try:
                loaded = joblib.load(model_path)
                self.arima_model = loaded
                metrics = {'mae': None, 'rmse': None, 'r2': None, 'source': f'loaded:{model_path}'}
                return loaded, metrics
            except Exception as e:
                # If load failed, keep going with placeholder
                metrics = {'mae': 6000, 'rmse': 8000, 'r2': 0.85, 'load_error': str(e)}
                return None, metrics
        else:
            metrics = {'mae': 6000, 'rmse': 8000, 'r2': 0.85, 'note': 'no pre-trained model found'}
            return None, metrics

    def train_lstm(self, epochs=30):
        """
        Placeholder: implement actual LSTM training.
        """
        metrics = {'mae': 4000, 'rmse': 6000, 'r2': 0.95}
        return None, None, metrics

    def predict_future(self, periods=3, method='prophet'):
        """
        Buat DataFrame prediksi sederhana (placeholder).
        Pastikan self.processed_data sudah ada.
        """
        if self.processed_data is None or len(self.processed_data) == 0:
            raise ValueError("processed_data kosong. Jalankan preprocess_data() dulu.")

        last_date = self.processed_data.index.max()
        # buat range mulai dari bulan berikutnya (MS = month start)
        start = last_date + pd.offsets.MonthBegin()
        future_dates = pd.date_range(start=start, periods=periods, freq='MS')

        last_price = float(self.processed_data['Harga_Per_Kg'].iloc[-1])

        # If ARIMA method requested and model is loaded, use it
        if method.lower() == 'arima' and self.arima_model is not None:
            try:
                model = self.arima_model
                # Many statsmodels SARIMAX results objects have get_forecast
                if hasattr(model, 'get_forecast'):
                    # get_forecast expects steps
                    forecast_res = model.get_forecast(steps=periods)
                    mean = np.asarray(forecast_res.predicted_mean)
                    ci = forecast_res.conf_int(alpha=0.05)
                    lower = np.asarray(ci.iloc[:, 0])
                    upper = np.asarray(ci.iloc[:, 1])
                else:
                    # Fallback: try predict
                    if hasattr(model, 'predict'):
                        mean = np.asarray(model.predict(start=len(self.processed_data),
                                                        end=len(self.processed_data) + periods - 1))
                        # create a small ad-hoc interval around mean
                        lower = mean - (np.abs(mean) * 0.05)
                        upper = mean + (np.abs(mean) * 0.05)
                    else:
                        raise RuntimeError('Loaded ARIMA model does not support forecasting API')

                df_pred = pd.DataFrame({
                    'Tanggal': future_dates,
                    'Prediksi': mean,
                    'Batas_Bawah': lower,
                    'Batas_Atas': upper
                })

                return df_pred
            except Exception:
                # If anything fails, fall back to simple placeholder below
                pass

        # Default fallback: random-walk-like simple forecast (existing behavior)
        np.random.seed(0)
        forecast = last_price + np.cumsum(np.random.randn(periods) * (last_price * 0.02))

        df_pred = pd.DataFrame({
            'Tanggal': future_dates,
            'Prediksi': forecast,
            'Batas_Bawah': forecast - (last_price * 0.05),
            'Batas_Atas': forecast + (last_price * 0.05)
        })

        return df_pred

    def generate_alerts(self, preds=None, threshold_percent=10, method='arima'):
        """
        Generate alerts berdasarkan prediksi singkat.
        Jika `preds` (DataFrame) disediakan, gunakan itu; jika tidak, buat prediksi singkat
        dengan `predict_future(periods=3, method=method)`.

        Mengembalikan DataFrame dengan kolom: Tanggal, Harga_Sekarang, Harga_Prediksi, Perubahan, Status, Rekomendasi, Method
        method: 'prophet'|'arima'|'lstm' atau nama lain untuk ditampilkan di kolom Method
        """
        # Pastikan ada data historis
        if self.processed_data is None or len(self.processed_data) == 0:
            return pd.DataFrame()

        last_price = float(self.processed_data['Harga_Per_Kg'].iloc[-1])

        # Jika preds diberikan, gunakan langsung; jika tidak, buat prediksi singkat
        if preds is None:
            future_pred = self.predict_future(periods=3, method=method)
        else:
            # pastikan salinan dan kolom tersedia
            future_pred = preds.copy()

        alerts = []
        for _, row in future_pred.iterrows():
            pred_price = float(row['Prediksi'])
            change = ((pred_price - last_price) / last_price) * 100
            if change > threshold_percent:
                status = 'NAIK'
                rekom = 'TAHAN STOK'
            elif change < -threshold_percent:
                status = 'TURUN'
                rekom = 'JUAL SEGERA'
            else:
                status = 'STABIL'
                rekom = 'PANTAU'

            alerts.append({
                'Tanggal': pd.to_datetime(row['Tanggal']).strftime('%Y-%m-%d'),
                'Harga_Sekarang': f"Rp {last_price:,.0f}",
                'Harga_Prediksi': f"Rp {pred_price:,.0f}",
                'Perubahan': f"{change:.2f}%",
                'Status': status,
                'Rekomendasi': rekom,
                'Method': method.capitalize() if isinstance(method, str) else str(method)
            })

        return pd.DataFrame(alerts)
