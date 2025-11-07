import warnings
import streamlit as st
from streamlit.components.v1 import html

warnings.filterwarnings('ignore')

# PANGGIL HANYA SEKALI dan PERTAMA kali Streamlit API
st.set_page_config(
    page_title="Analisis CPUE & MSY - Model Schaefer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
with open('pages.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Menu navigasi horizontal di atas
menu_items = [
    "Analisis CPUE & MSY",
    "Model Prediksi Sistem Stok Ikan",
    "Prediksi Harga Ikan (Market Foresight)", 
    "Dashboard Produksi Ikan",
    "Game Simulasi Penangkapan Ikan"
]

# Buat tombol horizontal untuk setiap menu
cols = st.columns(len(menu_items))
for idx, col in enumerate(cols):
    if col.button(menu_items[idx], use_container_width=True):
        st.session_state['page'] = menu_items[idx]

# Inisialisasi state jika belum ada
if 'page' not in st.session_state:
    st.session_state['page'] = menu_items[0]

st.title("🐟 Analisis CPUE dan MSY dengan Standarisasi Alat Tangkap")

# Gunakan session state untuk menyimpan pilihan halaman
page = st.session_state['page']


if page == "Analisis CPUE & MSY":
    # import modul analisis hanya saat dipilih (lazy import)
    from analysis import main as analysis_main
    analysis_main()

elif page == "Model Prediksi Sistem Stok Ikan":
    # Import model prediksi stok ikan
    from model import main as prediction_model_main
    prediction_model_main()
    
elif page == "Prediksi Harga Ikan (Market Foresight)":
    # Launch Market Foresight UI (price prediction)
    from market_foresight import main as market_foresight_main
    market_foresight_main()

elif page == "Dashboard Produksi Ikan":
    # lazy import agar dashboard.py tidak dieksekusi pada import time
    from dashboard import main as dashboard_main
    dashboard_main()

elif page == "Game Simulasi Penangkapan Ikan":
    # Tampilkan file game.html
    with open("game.html", "r", encoding="utf-8") as f:
        game_html = f.read()
    html(game_html, height=1000, scrolling=True)

# FOOTER
st.markdown("---")
st.markdown("*Dikembangkan untuk ASIK*")