# -----------------------------------------------------------------
# IMPOR GABUNGAN (dari semua file)
# -----------------------------------------------------------------
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import io
import matplotlib.pyplot as plt
import os
import sys
import joblib
import warnings
from scipy import stats

# Impor Model & Gambar
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from PIL import Image, ImageOps

# Import dari file lokal Anda (PASTIKAN market_model.py ADA)
try:
    from market_model import MarketForesight
except ImportError:
    st.error("FATAL ERROR: File 'market_model.py' tidak ditemukan. Pastikan file tersebut ada di folder yang sama dengan 'dashboard.py'.")
    st.stop()

# Impor untuk Klasifikasi Gambar
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------
# 1. KONFIGURASI HALAMAN & FUNGSI BANTU UTAMA
# -----------------------------------------------------------------

st.set_page_config(
    page_title="Dashboard Analisis Perikanan",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi untuk memuat file CSS eksternal
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"File CSS '{file_name}' tidak ditemukan. Pastikan 'pages.css' ada di folder yang sama.")

# Fungsi untuk memuat file HTML
def get_html_file(file_name):
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"File HTML 'game.html' tidak ditemukan.")
        return None

# Panggil fungsi untuk memuat CSS eksternal Anda
local_css("pages.css")

st.markdown('<h1 class="main-header">🐟 Dashboard Analisis & Simulasi Perikanan</h1>', unsafe_allow_html=True)


# -----------------------------------------------------------------
# 2. FUNGSI UNTUK TAB 1: KLASIFIKASI GAMBAR (dari classifier.py)
# -----------------------------------------------------------------

@st.cache_resource
def get_classifier_model():
    MODEL_PATH = "model_tensorflow/keras_model.h5"
    if not TENSORFLOW_AVAILABLE:
        st.error("TensorFlow/Keras tidak terinstal. Tab ini tidak akan berfungsi.")
        return None
    try:
        model = load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model klasifikasi: {e}")
        st.info("Pastikan file 'model_tensorflow/keras_model.h5' ada.")
        return None

@st.cache_data
def load_classifier_labels():
    LABELS_PATH = "model_tensorflow/labels.txt"
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        parsed = [l.partition(' ')[2] if ' ' in l else l for l in lines]
        return parsed
    except Exception as e:
        st.error(f"Gagal memuat labels: {e}")
        st.info("Pastikan file 'model_tensorflow/labels.txt' ada.")
        return []

def preprocess_image(image: Image.Image):
    INPUT_SIZE = (224, 224)
    image = image.convert('RGB')
    image = ImageOps.fit(image, INPUT_SIZE, Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32)
    normalized = (image_array / 127.5) - 1.0
    return np.expand_dims(normalized, axis=0)

def predict_image(model, labels, image: Image.Image, top_k=3):
    if model is None or not labels:
        return None
    input_arr = preprocess_image(image)
    preds = model.predict(input_arr)
    probs = preds[0]
    top_idx = probs.argsort()[-top_k:][::-1]
    results = []
    for idx in top_idx:
        label = labels[idx] if idx < len(labels) else f"Class {idx}"
        results.append({"label": label, "confidence": float(probs[idx])})
    return results

def render_tab_classifier():
    st.header("🧭 Marine Image Classifier")
    st.write("Ambil gambar dari kamera atau unggah file untuk mengklasifikasikan jenis hewan laut.")

    model = get_classifier_model()
    labels = load_classifier_labels()

    if model is None or not labels:
        st.warning("Model atau file label tidak berhasil dimuat. Fungsionalitas tab ini terbatas.")
        return

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
        if st.button("🔍 Prediksi Gambar"):
            with st.spinner('Memproses...'):
                results = predict_image(model, labels, image, top_k=5)
            if results:
                st.success("Hasil prediksi:")
                for r in results:
                    st.write(f"- **{r['label']}** — {r['confidence']*100:.2f}%")
            else:
                st.error("Model belum dimuat atau terjadi kesalahan saat prediksi.")

# -----------------------------------------------------------------
# 3. FUNGSI UNTUK TAB 2: ANALISIS MSY (dari analysis.py)
# -----------------------------------------------------------------
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')
def generate_years(start_year, num_years):
    return [start_year + i for i in range(num_years)]
def hitung_cpue(produksi_df, upaya_df, gears):
    cpue_data = []
    years = produksi_df['Tahun'].values
    for year in years:
        year_data = {'Tahun': year}
        for gear in gears:
            try:
                prod = produksi_df.loc[produksi_df['Tahun'] == year, gear].values[0]
            except Exception: prod = 0
            try:
                eff = upaya_df.loc[upaya_df['Tahun'] == year, gear].values[0]
            except Exception: eff = 0
            year_data[gear] = (prod / eff) if eff > 0 else 0
        year_data['Jumlah'] = sum([year_data[gear] for gear in gears])
        cpue_data.append(year_data)
    return pd.DataFrame(cpue_data)
def hitung_fpi_per_tahun(cpue_df, gears, standard_gear):
    fpi_data = []
    years = cpue_df['Tahun'].values
    for year in years:
        year_data = {'Tahun': year}
        try:
            cpue_standard = cpue_df.loc[cpue_df['Tahun'] == year, standard_gear].values[0]
        except Exception: cpue_standard = 0
        for gear in gears:
            try:
                cpue_gear = cpue_df.loc[cpue_df['Tahun'] == year, gear].values[0]
            except Exception: cpue_gear = 0
            year_data[gear] = (cpue_gear / cpue_standard) if cpue_standard > 0 else 0
        year_data['Jumlah'] = sum([year_data[gear] for gear in gears])
        fpi_data.append(year_data)
    return pd.DataFrame(fpi_data)
def hitung_upaya_standar(upaya_df, fpi_df, gears):
    standard_effort_data = []
    years = upaya_df['Tahun'].values
    for year in years:
        year_data = {'Tahun': year}
        total_standard_effort = 0
        for gear in gears:
            try:
                eff = upaya_df.loc[upaya_df['Tahun'] == year, gear].values[0]
            except Exception: eff = 0
            try:
                fpi = fpi_df.loc[fpi_df['Tahun'] == year, gear].values[0]
            except Exception: fpi = 0
            standard_effort = eff * fpi
            year_data[gear] = standard_effort
            total_standard_effort += standard_effort
        year_data['Jumlah'] = total_standard_effort
        standard_effort_data.append(year_data)
    return pd.DataFrame(standard_effort_data)
def hitung_cpue_standar(produksi_df, standard_effort_df):
    standard_cpue_data = []
    years = produksi_df['Tahun'].values
    for year in years:
        year_data = {'Tahun': year}
        try:
            total_production = produksi_df.loc[produksi_df['Tahun'] == year, 'Jumlah'].values[0]
        except Exception: total_production = 0
        try:
            total_standard_effort = standard_effort_df.loc[standard_effort_df['Tahun'] == year, 'Jumlah'].values[0]
        except Exception: total_standard_effort = 0
        cpue_standar_total = (total_production / total_standard_effort) if total_standard_effort > 0 else 0
        year_data['CPUE_Standar_Total'] = cpue_standar_total
        year_data['Ln_CPUE'] = np.log(cpue_standar_total) if cpue_standar_total > 0 else 0
        standard_cpue_data.append(year_data)
    return pd.DataFrame(standard_cpue_data)
def analisis_msy_schaefer(standard_effort_total, cpue_standard_total):
    if len(standard_effort_total) < 2: return None
    slope, intercept, r_value, p_value, std_err = stats.linregress(standard_effort_total, cpue_standard_total)
    a = intercept; b = slope
    if b >= 0: return {'success': False, 'error': 'Slope (b) harus negatif'}
    F_MSY = -a / (2 * b)
    C_MSY = -(a ** 2) / (4 * b)
    return {'a': a, 'b': b, 'r_squared': r_value**2, 'p_value': p_value, 'std_err': std_err, 'F_MSY': F_MSY, 'C_MSY': C_MSY, 'success': True}
def buat_grafik_lengkap(results, effort_values, cpue_values, production_values, years, df_cpue, df_fpi, df_standard_effort, gears, display_names, standard_gear):
    plt.style.use('dark_background') # Tema gelap
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('none')
    
    ax1 = plt.subplot(2, 3, 1); ax1.set_facecolor('none')
    ax1.bar(years, production_values, alpha=0.7); ax1_twin = ax1.twinx()
    ax1_twin.plot(years, effort_values, 'ro-'); ax1.set_title("Produksi vs Upaya")
    
    ax2 = plt.subplot(2, 3, 2); ax2.set_facecolor('none')
    ax2.scatter(effort_values, cpue_values, s=80); ax2.set_title("CPUE vs Upaya (Schaefer)")
    if results and results.get('success'):
        effort_range = np.linspace(min(effort_values), max(effort_values) * 1.2, 100)
        cpue_pred = results['a'] + results['b'] * effort_range
        ax2.plot(effort_range, cpue_pred, 'red', linewidth=2)
        ax2.axvline(results['F_MSY'], color='green', linestyle='--')
        
    ax3 = plt.subplot(2, 3, 3); ax3.set_facecolor('none')
    ax3.set_title("Kurva Produksi (MSY)")
    if results and results.get('success'):
        effort_range_prod = np.linspace(0, max(effort_values) * 1.5, 100)
        catch_pred = results['a'] * effort_range_prod + results['b'] * (effort_range_prod ** 2)
        ax3.plot(effort_range_prod, catch_pred, 'purple', linewidth=3)
        ax3.axvline(results['F_MSY'], color='green', linestyle='--')
        ax3.axhline(results['C_MSY'], color='orange', linestyle='--')

    ax4 = plt.subplot(2, 3, 4); ax4.set_facecolor('none')
    ax4.set_title("CPUE per Alat Tangkap")
    for i, gear in enumerate(gears):
        if gear in df_cpue.columns:
            ax4.plot(years, df_cpue[gear].values, 'o-', label=display_names[i] if i < len(display_names) else gear)
    ax4.legend()

    ax5 = plt.subplot(2, 3, 5); ax5.set_facecolor('none')
    ax5.set_title("FPI (vs Standar)")
    fpi_gears = [g for g in gears if g != standard_gear]
    for i, gear in enumerate(fpi_gears):
        if gear in df_fpi.columns:
            idx = gears.index(gear)
            label = display_names[idx] if idx < len(display_names) else gear
            ax5.plot(years, df_fpi[gear].values, 's-', label=label)
    ax5.legend()
    
    ax6 = plt.subplot(2, 3, 6); ax6.set_facecolor('none')
    ax6.set_title("Rata-rata Komposisi Upaya Standar")
    avg_effort = []
    for g in gears:
        if g in df_standard_effort.columns: avg_effort.append(df_standard_effort[g].mean())
        else: avg_effort.append(0)
    if all(v == 0 for v in avg_effort): avg_effort = [1 for _ in avg_effort]
    ax6.pie(avg_effort, labels=[(display_names[i] if i < len(display_names) else gears[i]) for i in range(len(gears))])
    
    plt.tight_layout()
    return fig
def parse_uploaded_file(uploaded_file):
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        df = pd.read_csv(io.StringIO(content))
        required_columns = ['Tahun', 'Alat_Tangkap', 'Produksi', 'Upaya']
        if not all(col in df.columns for col in required_columns):
            return None, "File harus memiliki kolom: Tahun, Alat_Tangkap, Produksi, Upaya"
        gears = sorted(df['Alat_Tangkap'].unique().tolist())
        years = sorted(df['Tahun'].unique().tolist())
        index = pd.MultiIndex.from_product([years, gears], names=['Tahun', 'Alat_Tangkap'])
        template = pd.DataFrame(index=index).reset_index()
        df_prod = pd.merge(template, df[['Tahun', 'Alat_Tangkap', 'Produksi']], on=['Tahun', 'Alat_Tangkap'], how='left')
        df_effort = pd.merge(template, df[['Tahun', 'Alat_Tangkap', 'Upaya']], on=['Tahun', 'Alat_Tangkap'], how='left')
        df_prod['Produksi'] = df_prod['Produksi'].fillna(0)
        df_effort['Upaya'] = df_effort['Upaya'].fillna(0)
        df_production = df_prod.pivot(index='Tahun', columns='Alat_Tangkap', values='Produksi').reset_index()
        df_effort = df_effort.pivot(index='Tahun', columns='Alat_Tangkap', values='Upaya').reset_index()
        df_production['Jumlah'] = df_production.sum(axis=1, numeric_only=True)
        df_effort['Jumlah'] = df_effort.sum(axis=1, numeric_only=True)
        return {'production': df_production, 'effort': df_effort, 'gears': gears, 'years': years, 'display_names': [g.replace('_', ' ') for g in gears]}, "Success"
    except Exception as e:
        return None, f"Error membaca file: {str(e)}"
def validate_fishing_data(effort_values, production_values):
    if len(effort_values) != len(production_values): return False, "Data effort dan produksi tidak sama"
    if len(effort_values) < 3: return False, "Minimal diperlukan 3 tahun data"
    return True, "Data valid" # Validasi tren dihapus karena terlalu ketat

# Fungsi utama untuk Tab 2
def render_tab_analysis_msy():
    st.header("📈 Analisis CPUE & MSY (Schaefer)")
    
    # Inisialisasi session state
    if 'gear_config' not in st.session_state:
        st.session_state.gear_config = {'gears': ['Jaring_Insang', 'Pancing'], 'display_names': ['Jaring Insang', 'Pancing'], 'standard_gear': 'Jaring_Insang', 'years': generate_years(2020, 3), 'num_years': 3}
    if 'data_tables' not in st.session_state:
        st.session_state.data_tables = {'production': [], 'effort': []}

    # --- Konfigurasi dipindahkan dari sidebar ke expander ---
    with st.expander("⚙️ Konfigurasi Analisis (Wajib diisi jika Input Manual)"):
        start_year = st.number_input("Tahun Mulai", min_value=2000, max_value=2030, value=2020, key="start_year_a")
        num_years = st.number_input("Jumlah Tahun", min_value=2, max_value=20, value=5, key="num_years_a")
        num_gears = st.number_input("Jumlah Alat Tangkap", min_value=1, max_value=8, value=2, key="num_gears_a")

        config = st.session_state.gear_config
        gear_names = []
        display_names = []
        for i in range(num_gears):
            default_internal = config['gears'][i] if i < len(config['gears']) else f"Alat_{i+1}"
            default_display = config['display_names'][i] if i < len(config['display_names']) else f"Alat Tangkap {i+1}"
            col1, col2 = st.columns(2)
            with col1:
                internal_name = st.text_input(f"Kode Unik {i+1} (tanpa spasi)", value=default_internal, key=f"internal_a_{i}")
            with col2:
                display_name = st.text_input(f"Nama Tampilan {i+1}", value=default_display, key=f"display_a_{i}")
            gear_names.append(internal_name); display_names.append(display_name)

        standard_gear = st.selectbox("Pilih Alat Standar (untuk FPI)", gear_names, index=0)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Simpan Konfigurasi"):
                years_generated = generate_years(start_year, num_years)
                st.session_state.gear_config = {'gears': gear_names, 'display_names': display_names, 'standard_gear': standard_gear, 'years': years_generated, 'num_years': num_years}
                st.success("Konfigurasi disimpan.")
        with col2:
            if st.button("🔄 Reset Data"):
                st.session_state.data_tables = {'production': [], 'effort': []}
                st.success("Data input manual direset.")
    
    # Variabel lokal dari state
    config = st.session_state.gear_config
    gears = config.get('gears', [])
    display_names = config.get('display_names', [g.replace('_',' ') for g in gears])
    years = config.get('years', [])
    standard_gear = config.get('standard_gear', gears[0] if gears else None)

    st.header("📊 Input Data")
    input_method = st.radio("Metode Input", ["Upload File", "Input Manual"], horizontal=True, key="msy_input_method")

    # Template download dipindahkan ke sini
    example_data = "Tahun,Alat_Tangkap,Produksi,Upaya\n2020,Jaring_Insang,5000,100\n2020,Pancing,3000,80\n2021,Jaring_Insang,4500,120\n2021,Pancing,2800,100"
    st.download_button("📥 Download Template CSV", data=example_data, file_name="template_data.csv", mime="text/csv")

    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload file CSV/TXT (Format: Tahun,Alat_Tangkap,Produksi,Upaya)", type=['csv', 'txt'], key="msy_uploader")
        if uploaded_file:
            data, message = parse_uploaded_file(uploaded_file)
            if data:
                st.success("File berhasil di-parse!")
                st.session_state.gear_config = {'gears': data['gears'], 'display_names': data['display_names'], 'standard_gear': data['gears'][0] if data['gears'] else None, 'years': data['years'], 'num_years': len(data['years'])}
                st.session_state.data_tables = {'production': data['production'].to_dict('records'), 'effort': data['effort'].to_dict('records')}
                st.dataframe(data['production'])
                st.dataframe(data['effort'])
                # Refresh local vars
                config = st.session_state.gear_config; gears = config['gears']; display_names = config['display_names']; years = config['years']; standard_gear = config['standard_gear']
            else:
                st.error(f"Error: {message}")
    
    else: # Input Manual
        st.subheader("1. Data Produksi (Kg)")
        if not gears:
            st.warning("Konfigurasikan alat tangkap di expander di atas.")
            return

        headers = ["Tahun"] + display_names + ["Jumlah"]
        prod_cols = st.columns(len(headers))
        for i, header in enumerate(headers):
            with prod_cols[i]: st.markdown(f"**{header}**")
        
        production_inputs = []
        prod_records = st.session_state.data_tables.get('production', [])
        for i, year in enumerate(years):
            cols = st.columns(len(headers))
            total_prod = 0; row = {'Tahun': year}
            with cols[0]: st.markdown(f"**{year}**")
            for j, gear in enumerate(gears):
                default_val = prod_records[i].get(gear, 0) if i < len(prod_records) else 0.0
                with cols[j+1]:
                    v = st.number_input(f"Prod {display_names[j]} {year}", min_value=0.0, value=float(default_val), key=f"prod_a_{gear}_{year}")
                row[gear] = v; total_prod += v
            with cols[-1]: st.markdown(f"**{total_prod:,.0f}**"); row['Jumlah'] = total_prod
            production_inputs.append(row)
        st.session_state.data_tables['production'] = production_inputs

        st.subheader("2. Data Upaya (Trip)")
        effort_inputs = []
        eff_records = st.session_state.data_tables.get('effort', [])
        for i, year in enumerate(years):
            cols = st.columns(len(headers))
            total_eff = 0; row = {'Tahun': year}
            with cols[0]: st.markdown(f"**{year}**")
            for j, gear in enumerate(gears):
                default_val = eff_records[i].get(gear, 0) if i < len(eff_records) else 0
                with cols[j+1]:
                    v = st.number_input(f"Effort {display_names[j]} {year}", min_value=0, value=int(default_val), key=f"eff_a_{gear}_{year}")
                row[gear] = v; total_eff += v
            with cols[-1]: st.markdown(f"**{total_eff:,}**"); row['Jumlah'] = total_eff
            effort_inputs.append(row)
        st.session_state.data_tables['effort'] = effort_inputs

    st.markdown("---")
    if st.button("🚀 Lakukan Analisis CPUE dan MSY", type="primary"):
        if not st.session_state.data_tables.get('production') or not st.session_state.data_tables.get('effort'):
            st.error("Data produksi atau upaya kosong!")
            return
            
        df_production = pd.DataFrame(st.session_state.data_tables['production'])
        df_effort = pd.DataFrame(st.session_state.data_tables['effort'])

        if df_production.empty or df_effort.empty:
            st.error("Dataframe kosong. Harap isi data.")
            return

        effort_values = df_effort['Jumlah'].values
        production_values = df_production['Jumlah'].values
        valid, message = validate_fishing_data(effort_values, production_values)
        if not valid:
            st.error(f"Data tidak valid: {message}")
            return
            
        st.header("📈 Hasil Perhitungan")
        for g in gears:
            if g not in df_production.columns: df_production[g] = 0
            if g not in df_effort.columns: df_effort[g] = 0
        
        df_cpue = hitung_cpue(df_production, df_effort, gears)
        if not standard_gear or standard_gear not in gears:
            standard_gear = gears[0] if gears else None
            st.session_state.gear_config['standard_gear'] = standard_gear
        
        if standard_gear is None:
             st.error("Tidak ada alat tangkap standar yang valid.")
             return

        df_fpi = hitung_fpi_per_tahun(df_cpue, gears, standard_gear)
        df_standard_effort = hitung_upaya_standar(df_effort, df_fpi, gears)
        df_standard_cpue = hitung_cpue_standar(df_production, df_standard_effort)

        st.subheader("Data CPUE per Alat Tangkap")
        st.dataframe(df_cpue)
        st.download_button("📥 Download CPUE (CSV)", data=convert_df_to_csv(df_cpue), file_name="data_cpue.csv", mime="text/csv")
        
        st.subheader("Data FPI (Fishing Power Index)")
        st.dataframe(df_fpi)
        
        st.subheader("Data Upaya Standar")
        st.dataframe(df_standard_effort)
        
        st.subheader("Data CPUE Standar (Total)")
        st.dataframe(df_standard_cpue)

        effort_values_std = df_standard_effort['Jumlah'].values
        cpue_values_std = df_standard_cpue['CPUE_Standar_Total'].values
        results = analisis_msy_schaefer(effort_values_std, cpue_values_std)

        if results and results.get('success'):
            st.subheader("Hasil Regresi Linear (Schaefer)")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Intercept (a)", f"{results['a']:.4f}")
            col2.metric("Slope (b)", f"{results['b']:.4f}")
            col3.metric("R²", f"{results['r_squared']:.4f}")
            col4.metric("p-value", f"{results['p_value']:.4f}")

            st.subheader("Estimasi MSY")
            col1, col2 = st.columns(2)
            col1.metric("F_MSY (Upaya Optimal)", f"{results['F_MSY']:,.2f} trip standar")
            col2.metric("C_MSY (Hasil Tangkapan Maks)", f"{results['C_MSY']:,.2f} kg")

            fig = buat_grafik_lengkap(results, effort_values_std, cpue_values_std, production_values, years, df_cpue, df_fpi, df_standard_effort, gears, display_names, standard_gear)
            st.pyplot(fig)

            latest_effort = effort_values_std[-1]
            utilization_effort = (latest_effort / results['F_MSY']) * 100
            st.subheader(f"Tingkat Pemanfaatan (Tahun {years[-1]}): {utilization_effort:.2f}%")
            if utilization_effort < 80: st.success("🟢 UNDER EXPLOITED — Rekomendasi: Tingkatkan pemanfaatan.")
            elif utilization_effort <= 100: st.info("🟡 FULLY EXPLOITED — Rekomendasi: Pertahankan dan monitoring.")
            else: st.warning("🔴 OVER EXPLOITED — Rekomendasi: Kurangi upaya segera.")
        else:
            st.error(f"❌ Analisis MSY gagal. Error: {results.get('error', 'Data tidak cukup atau slope tidak negatif.')}")

# -----------------------------------------------------------------
# 4. FUNGSI UNTUK TAB 3: PREDIKSI HARGA (dari market_foresight.py)
# -----------------------------------------------------------------
@st.cache_resource
def _ensure_forecaster():
    if "market_forecaster" not in st.session_state:
        st.session_state["market_forecaster"] = MarketForesight()
    return st.session_state["market_forecaster"]

def _plot_predictions(df, hist_data=None):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    if hist_data is not None and not hist_data.empty:
        ax.plot(hist_data.index, hist_data['Harga_Per_Kg'], 'b-', label='Data Historis', alpha=0.7)
    
    pred_dates = pd.to_datetime(df["Tanggal"])
    ax.plot(pred_dates, df["Prediksi"], 'r-', marker='o', label='Prediksi ARIMA')
    
    if "Batas_Bawah" in df.columns and "Batas_Atas" in df.columns:
        ax.fill_between(pred_dates, df["Batas_Bawah"], df["Batas_Atas"], color='red', alpha=0.1, label='Interval Kepercayaan')
    
    ax.set_xlabel("Tanggal"); ax.set_ylabel("Harga (Rp)"); ax.set_title("Prediksi Harga ARIMA")
    ax.legend(); ax.grid(True, alpha=0.3); plt.xticks(rotation=45); plt.tight_layout()
    return fig

def render_tab_arima():
    st.header("💹 Prediksi Harga Ikan — Market Foresight (ARIMA)")
    
    with st.expander("ℹ️ Informasi Model ARIMA"):
        st.markdown("Model statistik untuk prediksi data time series (tren, musiman, pola historis).")

    forecaster = _ensure_forecaster()
    
    # --- UI dipindahkan dari sidebar ke expander ---
    with st.expander("Data & Pengaturan"):
        uploaded = st.file_uploader("Upload CSV atau Excel (opsional)", type=["csv", "xlsx", "xls"], key="arima_uploader")
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
                forecaster.data = df
                st.success(f"Data dimuat: {df.shape[0]} baris.")
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

        if st.button("Buat Data Sampel (36 bulan)"):
            forecaster.data = forecaster.create_sample_data(n_months=36)
            st.success("Data sampel dibuat dan dimuat.")

    if forecaster.data is None:
        st.info("Tidak ada data. Upload file atau buat data sampel di atas.")
        return

    if "Jenis_Ikan" not in forecaster.data.columns:
        st.error("Kolom 'Jenis_Ikan' tidak ditemukan di data.")
        return

    fish_options = sorted(forecaster.data["Jenis_Ikan"].dropna().unique())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_fish = st.selectbox("Pilih Jenis Ikan", fish_options)
    with col2:
        periods = st.number_input("Periode prediksi (bulan)", min_value=1, max_value=60, value=6)
    with col3:
        threshold = st.number_input("Ambang alert (%)", min_value=1, max_value=100, value=10)

    if st.button("Jalankan Prediksi ARIMA"):
        with st.spinner("Memproses prediksi..."):
            try:
                forecaster.preprocess_data(fish_type=selected_fish)
                _ = forecaster.train_arima()
                preds = forecaster.predict_future(periods=periods, method="arima")

                st.subheader("Hasil Prediksi")
                preds_display = preds.copy()
                preds_display["Tanggal"] = pd.to_datetime(preds_display["Tanggal"]).dt.strftime("%Y-%m-%d")
                st.dataframe(preds_display)

                fig = _plot_predictions(preds, forecaster.processed_data)
                st.pyplot(fig)

                alerts = forecaster.generate_alerts(preds, threshold_percent=threshold, method='arima')
                if not alerts.empty:
                    st.warning("Perhatian: Terdapat alert berikut:")
                    st.dataframe(alerts)
                else:
                    st.success("Tidak ada alert signifikan pada prediksi.")

                csv_buf = io.StringIO()
                preds.to_csv(csv_buf, index=False)
                st.download_button("Download Hasil Prediksi (CSV)", data=csv_buf.getvalue(), file_name="prediksi_harga.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")

# -----------------------------------------------------------------
# 5. FUNGSI UNTUK TAB 4: PREDIKSI CPUE (dari api.py)
# -----------------------------------------------------------------
@st.cache_resource
def load_cpue_model():
    try:
        # Menggunakan path lengkap sesuai yang Anda berikan
        model_path = r"D:\BERKULIAH DI UPI SERANG\SEMESTER 5\ASIK 2025\analisis-perikanan\analisis-perikanan\tuned_ridge_model_retrain.joblib"
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model CPUE: {e}")
        return None

# Fungsi preprocess dari api.py
def preprocess_cpue_input(alat_tangkap, effort, fishing_gear_options):
    data = {'Effort (trip)': [effort]}
    for gear in fishing_gear_options:
        column_name = f'Alat tangkap_{gear}'
        data[column_name] = [1 if alat_tangkap == gear else 0]
    
    processed_input = pd.DataFrame(data)
    expected_columns = ['Effort (trip)'] + [f'Alat tangkap_{gear}' for gear in fishing_gear_options]
    processed_input = processed_input[expected_columns]
    return processed_input

def render_tab_cpue_api():
    st.header("⚙️ Prediksi Stok (CPUE)")
    st.write("Prediksi CPUE dan Produksi Total menggunakan model Ridge Regression yang sudah dilatih (dari `api.py`).")
    
    model = load_cpue_model()
    if model is None:
        st.error("Model prediksi CPUE tidak dapat dimuat. Tab ini tidak akan berfungsi.")
        return
        
    # Opsi dari api.py
    fishing_gear_options = [
        'Bagan Tancap', 'Bubu', 'Jaring Hela Dasar', 'Jaring Insang Hanyut',
        'Jaring Insang Tetap', 'Jaring Payang', 'Lain-lain', 'Pancing'
    ]
    
    col1, col2 = st.columns(2)
    with col1:
        alat_tangkap = st.selectbox("Pilih Alat Tangkap", fishing_gear_options)
    with col2:
        effort = st.number_input("Masukkan Effort (trip)", min_value=1.0, value=100.0, step=10.0)
        
    if st.button("🔍 Prediksi CPUE (Ridge)"):
        try:
            processed_input = preprocess_cpue_input(alat_tangkap, effort, fishing_gear_options)
            predicted_cpue = model.predict(processed_input)[0]
            predicted_production = predicted_cpue * effort
            
            st.success("Prediksi Berhasil!")
            kpi1, kpi2 = st.columns(2)
            kpi1.metric("Prediksi CPUE", f"{predicted_cpue:.2f}")
            kpi2.metric("Prediksi Produksi (kg)", f"{predicted_production:,.2f}")
            
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")

# -----------------------------------------------------------------
# 6. FUNGSI UNTUK TAB 5: GAME SIMULASI
# -----------------------------------------------------------------
def render_tab_game():
    st.header("🎣 Game Simulasi Penangkapan Ikan")
    st.info("Mainkan simulasi ini untuk memahami dampak overfishing secara visual.")
    html_code = get_html_file("game.html")
    if html_code:
        components.html(html_code, height=650, scrolling=True)
    else:
        st.error("Gagal memuat 'game.html'.")

# -----------------------------------------------------------------
# APLIKASI UTAMA (MAIN)
# -----------------------------------------------------------------
def main():
    # Definisi TABS
    tab_list = [
        "🧭 Klasifikasi Gambar", 
        "📈 Analisis MSY (Schaefer)",
        "💹 Prediksi Harga (ARIMA)",
        "⚙️ Prediksi CPUE (Ridge)",
        "🎮 Simulasi Game"
    ]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_list)

    # --- ISI TAB 1: Klasifikasi Gambar ---
    with tab1:
        render_tab_classifier()

    # --- ISI TAB 2: Analisis MSY ---
    with tab2:
        render_tab_analysis_msy()

    # --- ISI TAB 3: Prediksi Harga ARIMA ---
    with tab3:
        render_tab_arima()

    # --- ISI TAB 4: Prediksi CPUE (dari API) ---
    with tab4:
        render_tab_cpue_api()
        
    # --- ISI TAB 5: Game ---
    with tab5:
        render_tab_game()

if __name__ == "__main__":
    main()