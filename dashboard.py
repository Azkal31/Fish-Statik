import streamlit as st
from plotly.subplots import make_subplots
from io import StringIO
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .kpi-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_clean_data(uploaded_file=None):
    """Membaca dan membersihkan data produksi ikan"""
    
    if uploaded_file is not None:
        try:
            # Baca file yang diupload
            content = uploaded_file.getvalue().decode('utf-8')
            df = pd.read_csv(StringIO(content), sep='\t')
        except:
            # Coba dengan separator koma jika tab gagal
            uploaded_file.seek(0)
            content = uploaded_file.getvalue().decode('utf-8')
            df = pd.read_csv(StringIO(content))
    else:
        return pd.DataFrame()
    
    # Bersihkan nama kolom
    df.columns = df.columns.str.strip()
    
    # Konversi volume produksi
    df['Volume Produksi (kg)'] = pd.to_numeric(
        df['Volume Produksi (kg)'].astype(str).replace(['-', '', ' ', 'nan'], '0'), 
        errors='coerce'
    ).fillna(0)
    
    # Standarisasi nama jenis ikan
    df['Jenis Ikan'] = df['Jenis Ikan'].astype(str).str.strip().str.replace('"', '')
    
    # Hapus duplikat
    df = df.drop_duplicates(subset=['Tahun', 'Bulan', 'Jenis Ikan'])
    
    # Buat kolom periode untuk sorting
    bulan_order = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
                   'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
    df['Bulan_Num'] = df['Bulan'].map({b: i+1 for i, b in enumerate(bulan_order)})
    
    # Handle bulan yang tidak ada di mapping
    df['Bulan_Num'] = df['Bulan_Num'].fillna(1)
    
    # Buat kolom tanggal
    try:
        df['Tanggal'] = pd.to_datetime(
            df['Tahun'].astype(str) + '-' + df['Bulan_Num'].astype(int).astype(str) + '-01',
            format='%Y-%m-%d',
            errors='coerce'
        )
    except:
        df['Tanggal'] = pd.to_datetime('2020-01-01')
    
    return df

def create_kpi_cards(df, col1, col2, col3, col4):
    """Membuat KPI cards dengan error handling"""
    
    # Cek apakah data kosong
    if df.empty:
        st.info("Unggah data untuk melihat KPI.")
        return
    
    total_produksi = df['Volume Produksi (kg)'].sum()
    
    # Hitung rata-rata bulanan dengan safety check
    monthly_sum = df.groupby(['Tahun', 'Bulan'])['Volume Produksi (kg)'].sum()
    rata_bulanan = monthly_sum.mean() if len(monthly_sum) > 0 else 0
    
    jumlah_jenis = df['Jenis Ikan'].nunique()
    
    # Hitung tahun terbaik dengan safety check
    yearly_sum = df.groupby('Tahun')['Volume Produksi (kg)'].sum()
    if len(yearly_sum) > 0 and yearly_sum.max() > 0:
        tahun_terbaik = yearly_sum.idxmax()
    else:
        tahun_terbaik = "N/A"
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_produksi/1_000_000:.2f}M kg</div>
            <div class="kpi-label">Total Produksi</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{rata_bulanan/1000:.1f}K kg</div>
            <div class="kpi-label">Rata-rata/Bulan</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{jumlah_jenis}</div>
            <div class="kpi-label">Jenis Ikan</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{tahun_terbaik}</div>
            <div class="kpi-label">Tahun Terbaik</div>
        </div>
        """, unsafe_allow_html=True)

def plot_trend_tahunan(df):
    """Grafik tren produksi tahunan"""
    if df.empty:
        st.info("Data kosong untuk plot trend.")
        return
    
    df_bulanan = df.groupby('Tanggal')['Volume Produksi (kg)'].sum().reset_index()
    
    fig = px.line(
        df_bulanan, 
        x='Tanggal', 
        y='Volume Produksi (kg)',
        title='📈 Tren Produksi Ikan Bulanan (2020-2024)',
        labels={'Volume Produksi (kg)': 'Volume Produksi (kg)', 'Tanggal': 'Periode'}
    )
    
    fig.update_traces(line_color='#1E88E5', line_width=3)
    fig.update_layout(
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_top_species(df, n=10):
    """Grafik top N jenis ikan"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Tidak ada data untuk ditampilkan",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    top_species = df.groupby('Jenis Ikan')['Volume Produksi (kg)'].sum().nlargest(n).reset_index()
    
    if top_species.empty:
        fig = go.Figure()
        fig.add_annotation(text="Tidak ada data untuk ditampilkan",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = px.bar(
        top_species,
        x='Volume Produksi (kg)',
        y='Jenis Ikan',
        orientation='h',
        title=f'🏆 Top {n} Jenis Ikan dengan Produksi Tertinggi',
        color='Volume Produksi (kg)',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_heatmap_bulanan(df):
    """Heatmap produksi bulanan per tahun"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Tidak ada data untuk ditampilkan",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    pivot_data = df.groupby(['Tahun', 'Bulan'])['Volume Produksi (kg)'].sum().reset_index()
    pivot_table = pivot_data.pivot(index='Bulan', columns='Tahun', values='Volume Produksi (kg)')
    
    # Urutkan bulan
    bulan_order = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
                   'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
    pivot_table = pivot_table.reindex(bulan_order)
    
    fig = px.imshow(
        pivot_table.T,
        labels=dict(x="Bulan", y="Tahun", color="Volume (kg)"),
        title='🔥 Heatmap Produksi Bulanan per Tahun',
        color_continuous_scale='YlOrRd',
        aspect='auto'
    )
    
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)

def plot_comparison_yearly(df):
    """Perbandingan produksi tahunan"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Tidak ada data untuk ditampilkan",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    yearly_prod = df.groupby('Tahun')['Volume Produksi (kg)'].sum().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=yearly_prod['Tahun'],
        y=yearly_prod['Volume Produksi (kg)'],
        text=yearly_prod['Volume Produksi (kg)'].apply(lambda x: f'{x/1000:.0f}K'),
        textposition='auto',
        marker_color='#764ba2'
    ))
    
    fig.update_layout(
        title='📊 Perbandingan Produksi Tahunan',
        xaxis_title='Tahun',
        yaxis_title='Volume Produksi (kg)',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🐟 Dashboard Produksi Ikan</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Data")
        uploaded_file = st.file_uploader(
            "Upload file CSV/TXT (format: Tahun, Bulan, Jenis Ikan, Volume)",
            type=['csv', 'txt']
        )
        
        st.markdown("---")
        st.header("🔧 Filter Data")
    
    # Load data
    if uploaded_file is not None:
        with st.spinner('Memproses data...'):
            df = load_and_clean_data(uploaded_file)
        
        if df is not None and not df.empty:
            st.success(f'✅ Data berhasil dimuat! Total {len(df)} baris data.')
            
            # Filter di sidebar
            with st.sidebar:
                tahun_options = sorted(df['Tahun'].unique())
                selected_years = st.multiselect(
                    "Pilih Tahun",
                    options=tahun_options,
                    default=tahun_options
                )
                
                bulan_options = df['Bulan'].unique().tolist()
                selected_months = st.multiselect(
                    "Pilih Bulan",
                    options=bulan_options,
                    default=bulan_options
                )
                
                jenis_options = sorted(df['Jenis Ikan'].unique())
                selected_species = st.multiselect(
                    "Pilih Jenis Ikan",
                    options=jenis_options,
                    default=jenis_options[:20] if len(jenis_options) > 20 else jenis_options
                )
            
            # Apply filters
            df_filtered = df[
                (df['Tahun'].isin(selected_years)) &
                (df['Bulan'].isin(selected_months)) &
                (df['Jenis Ikan'].isin(selected_species))
            ]
            
            if df_filtered.empty:
                st.warning("⚠️ Tidak ada data yang sesuai dengan filter. Silakan ubah filter.")
            else:
                # KPI Cards
                col1, col2, col3, col4 = st.columns(4)
                create_kpi_cards(df_filtered, col1, col2, col3, col4)
                
                st.markdown("---")
                
                # Grafik utama
                col1, col2 = st.columns(2)
                
                with col1:
                    plot_trend_tahunan(df_filtered)
                
                with col2:
                    plot_comparison_yearly(df_filtered)
                
                # Grafik sekunder
                col1, col2 = st.columns(2)
                
                with col1:
                    plot_top_species(df_filtered, 10)
                
                with col2:
                    plot_heatmap_bulanan(df_filtered)
                
                st.markdown("---")
                
                # Tabel detail
                st.subheader("📋 Data Detail")
                
                # Agregasi data untuk tampilan
                df_display = df_filtered.groupby(['Tahun', 'Bulan', 'Jenis Ikan'])['Volume Produksi (kg)'].sum().reset_index()
                df_display = df_display.sort_values(['Tahun', 'Volume Produksi (kg)'], 
                                                    ascending=[False, False])
                
                st.dataframe(
                    df_display[['Tahun', 'Bulan', 'Jenis Ikan', 'Volume Produksi (kg)']],
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = df_display.to_csv(index=False)
                st.download_button(
                    label="📥 Download Data (CSV)",
                    data=csv,
                    file_name="produksi_ikan_filtered.csv",
                    mime="text/csv"
                )
            
        else:
            st.error("❌ Gagal memproses file. Pastikan format file sesuai dengan contoh di bawah.")
            st.code("""
Tahun    Bulan      Jenis Ikan             Volume Produksi (kg)
2020     Januari    Teri (Anchovy)         63163
2020     Januari    Kembung (Indian...)    5232
            """)
    else:
        st.info("👈 Silakan upload file data di sidebar untuk memulai analisis.")
        
        # Tampilkan instruksi
        st.markdown("""
        ### 📖 Cara Menggunakan Dashboard:
        
        1. **Upload Data**: Klik tombol "Browse files" di sidebar dan pilih file data
        2. **Filter Data**: Gunakan filter tahun, bulan, dan jenis ikan sesuai kebutuhan
        3. **Eksplorasi Visualisasi**: Lihat berbagai grafik dan chart interaktif
        4. **Analisis Detail**: Scroll ke bawah untuk melihat tabel data lengkap
        5. **Download Hasil**: Klik tombol download untuk menyimpan data hasil filter
        
        ### 📊 Fitur Dashboard:
        - ✅ KPI Cards: Ringkasan metrik utama
        - ✅ Tren Temporal: Grafik time series produksi
        - ✅ Ranking Jenis Ikan: Top performers
        - ✅ Heatmap: Pola musiman produksi
        - ✅ Filter Interaktif: Multi-dimensi filtering
        - ✅ Export Data: Download hasil analisis
        
        ### 📁 Format Data yang Dibutuhkan:
        File harus berisi kolom berikut (tab atau comma separated):
        - **Tahun** (numeric, contoh: 2020)
        - **Bulan** (text, contoh: Januari)
        - **Jenis Ikan** (text, contoh: Teri (Anchovy))
        - **Volume Produksi (kg)** (numeric, contoh: 63163)
        
        ### 💡 Tips:
        - File bisa format .txt (tab-separated) atau .csv (comma-separated)
        - Pastikan nama kolom persis seperti di atas
        - Nilai kosong atau "-" akan otomatis diubah menjadi 0
        """)

# Jangan jalankan main() di import time
if __name__ == "__main__":
    main()