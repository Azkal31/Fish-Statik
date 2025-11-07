import io
import csv
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import streamlit as st

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

warnings.filterwarnings('ignore')

# -------------------------
# Helper functions (pure/logic)
# -------------------------
def generate_years(start_year, num_years):
    return [start_year + i for i in range(num_years)]

def hitung_cpue(produksi_df, upaya_df, gears):
    cpue_data = []
    years = produksi_df['Tahun'].values
    for year in years:
        year_data = {'Tahun': year}
        for gear in gears:
            # safe access: gunakan .get jika kolom tidak ada
            try:
                prod = produksi_df.loc[produksi_df['Tahun'] == year, gear].values[0]
            except Exception:
                prod = 0
            try:
                eff = upaya_df.loc[upaya_df['Tahun'] == year, gear].values[0]
            except Exception:
                eff = 0
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
        except Exception:
            cpue_standard = 0
        for gear in gears:
            try:
                cpue_gear = cpue_df.loc[cpue_df['Tahun'] == year, gear].values[0]
            except Exception:
                cpue_gear = 0
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
            except Exception:
                eff = 0
            try:
                fpi = fpi_df.loc[fpi_df['Tahun'] == year, gear].values[0]
            except Exception:
                fpi = 0
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
        except Exception:
            total_production = 0
        try:
            total_standard_effort = standard_effort_df.loc[standard_effort_df['Tahun'] == year, 'Jumlah'].values[0]
        except Exception:
            total_standard_effort = 0
        cpue_standar_total = (total_production / total_standard_effort) if total_standard_effort > 0 else 0
        year_data['CPUE_Standar_Total'] = cpue_standar_total
        year_data['Ln_CPUE'] = np.log(cpue_standar_total) if cpue_standar_total > 0 else 0
        standard_cpue_data.append(year_data)
    return pd.DataFrame(standard_cpue_data)

def analisis_msy_schaefer(standard_effort_total, cpue_standard_total):
    if len(standard_effort_total) < 2:
        return None
    slope, intercept, r_value, p_value, std_err = stats.linregress(standard_effort_total, cpue_standard_total)
    a = intercept; b = slope
    if b >= 0:
        return {'success': False, 'error': 'Slope (b) harus negatif untuk model Schaefer yang valid'}
    if b != 0:
        F_MSY = -a / (2 * b)
        C_MSY = -(a ** 2) / (4 * b)
    else:
        F_MSY = 0; C_MSY = 0
    return {'a': a, 'b': b, 'r_squared': r_value**2, 'p_value': p_value, 'std_err': std_err,
            'F_MSY': F_MSY, 'C_MSY': C_MSY, 'success': True}

def buat_grafik_lengkap(results, effort_values, cpue_values, production_values, years,
                       df_cpue, df_fpi, df_standard_effort, gears, display_names, standard_gear):
    fig = plt.figure(figsize=(20, 12))
    ax1 = plt.subplot(2, 3, 1)
    ax1.bar(years, production_values, alpha=0.7)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(years, effort_values, 'ro-')
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(effort_values, cpue_values, s=80)
    # protect if results is None
    if results and results.get('success'):
        effort_range = np.linspace(min(effort_values), max(effort_values) * 1.2, 100)
        cpue_pred = results['a'] + results['b'] * effort_range
        ax2.plot(effort_range, cpue_pred, 'red', linewidth=2)
        ax2.axvline(results['F_MSY'], color='green', linestyle='--')
    ax3 = plt.subplot(2, 3, 3)
    if results and results.get('success'):
        effort_range_prod = np.linspace(0, max(effort_values) * 1.5, 100)
        catch_pred = results['a'] * effort_range_prod + results['b'] * (effort_range_prod ** 2)
        ax3.plot(effort_range_prod, catch_pred, 'purple', linewidth=3)
        ax3.axvline(results['F_MSY'], color='green', linestyle='--')
        ax3.axhline(results['C_MSY'], color='orange', linestyle='--')
    ax4 = plt.subplot(2, 3, 4)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    for i, gear in enumerate(gears):
        if gear in df_cpue.columns:
            ax4.plot(years, df_cpue[gear].values, 'o-', label=display_names[i] if i < len(display_names) else gear)
    ax5 = plt.subplot(2, 3, 5)
    fpi_gears = [g for g in gears if g != standard_gear]
    for i, gear in enumerate(fpi_gears):
        if gear in df_fpi.columns:
            idx = gears.index(gear)
            label = display_names[idx] if idx < len(display_names) else gear
            ax5.plot(years, df_fpi[gear].values, 's-', label=label)
    ax6 = plt.subplot(2, 3, 6)
    # Avoid division by zero / empty
    avg_effort = []
    for g in gears:
        if g in df_standard_effort.columns:
            avg_effort.append(df_standard_effort[g].mean())
        else:
            avg_effort.append(0)
    # if all zeros, add tiny positive to avoid pie crash
    if all(v == 0 for v in avg_effort):
        avg_effort = [1 for _ in avg_effort]
    ax6.pie(avg_effort, labels=[(display_names[i] if i < len(display_names) else gears[i]) for i in range(len(gears))])
    plt.tight_layout()
    return fig

def hitung_analisis_statistik(df_production, df_effort, df_standard_effort, df_standard_cpue, gears):
    stat_produksi = df_production[gears + ['Jumlah']].describe()
    stat_upaya = df_effort[gears + ['Jumlah']].describe()
    years_idx = np.arange(len(df_production))
    trend_produksi, _, _, _, _ = stats.linregress(years_idx, df_production['Jumlah'].values)
    trend_upaya, _, _, _, _ = stats.linregress(years_idx, df_standard_effort['Jumlah'].values)
    trend_cpue, _, _, _, _ = stats.linregress(years_idx, df_standard_cpue['CPUE_Standar_Total'].values)
    return {'stat_produksi': stat_produksi, 'stat_upaya': stat_upaya,
            'trend_produksi': trend_produksi, 'trend_upaya': trend_upaya, 'trend_cpue': trend_cpue}

# Tambahkan fungsi helper untuk membaca file
def parse_uploaded_file(uploaded_file):
    """Parse uploaded CSV/TXT file into production and effort data"""
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        df = pd.read_csv(io.StringIO(content))
        
        # Validate structure
        required_columns = ['Tahun', 'Alat_Tangkap', 'Produksi', 'Upaya']
        if not all(col in df.columns for col in required_columns):
            return None, "File harus memiliki kolom: Tahun, Alat_Tangkap, Produksi, Upaya"
        
        # Get unique gears and years
        gears = sorted(df['Alat_Tangkap'].unique().tolist())
        years = sorted(df['Tahun'].unique().tolist())
        
        # Create empty DataFrames with all combinations
        index = pd.MultiIndex.from_product([years, gears], names=['Tahun', 'Alat_Tangkap'])
        template = pd.DataFrame(index=index).reset_index()
        
        # Merge with actual data
        df_prod = pd.merge(template, df[['Tahun', 'Alat_Tangkap', 'Produksi']], 
                          on=['Tahun', 'Alat_Tangkap'], how='left')
        df_effort = pd.merge(template, df[['Tahun', 'Alat_Tangkap', 'Upaya']], 
                            on=['Tahun', 'Alat_Tangkap'], how='left')
        
        # Fill missing values with 0
        df_prod['Produksi'] = df_prod['Produksi'].fillna(0)
        df_effort['Upaya'] = df_effort['Upaya'].fillna(0)
        
        # Pivot tables
        df_production = df_prod.pivot(index='Tahun', columns='Alat_Tangkap', 
                                    values='Produksi').reset_index()
        df_effort = df_effort.pivot(index='Tahun', columns='Alat_Tangkap', 
                                   values='Upaya').reset_index()
        
        # Add total columns
        df_production['Jumlah'] = df_production.sum(axis=1, numeric_only=True)
        df_effort['Jumlah'] = df_effort.sum(axis=1, numeric_only=True)
        
        return {
            'production': df_production,
            'effort': df_effort,
            'gears': gears,
            'years': years,
            'display_names': [g.replace('_', ' ') for g in gears]
        }, "Success"
        
    except Exception as e:
        return None, f"Error membaca file: {str(e)}"

def validate_fishing_data(effort_values, production_values):
    """Validasi data sebelum analisis MSY"""
    if len(effort_values) != len(production_values):
        return False, "Jumlah data effort dan produksi tidak sama"
    
    if len(effort_values) < 3:
        return False, "Minimal diperlukan 3 tahun data"
    
    # Check effort trend (should increase)
    if not all(effort_values[i] <= effort_values[i+1] for i in range(len(effort_values)-1)):
        return False, "Data effort sebaiknya meningkat setiap tahun untuk analisis yang valid"
    
    # Calculate and check CPUE trend
    cpue_values = [p/e if e > 0 else 0 for p,e in zip(production_values, effort_values)]
    if not all(cpue_values[i] >= cpue_values[i+1] for i in range(len(cpue_values)-1)):
        return False, "CPUE seharusnya menurun seiring meningkatnya effort"
    
    return True, "Data valid untuk analisis MSY"

# -------------------------
# Main UI function (call from main.py)
# -------------------------
def main():
    # session-state defaults
    if 'gear_config' not in st.session_state:
        st.session_state.gear_config = {
            'gears': ['Jaring_Insang_Tetap', 'Jaring_Hela_Dasar', 'Bagan_Berperahu', 'Pancing'],
            'display_names': ['Jaring Insang Tetap', 'Jaring Hela Dasar', 'Bagan Berperahu', 'Pancing'],
            'standard_gear': 'Jaring_Hela_Dasar',
            'years': generate_years(2020, 5),
            'num_years': 5
        }
    if 'data_tables' not in st.session_state:
        # store as list-of-dict (records)
        default_years = st.session_state.gear_config['years']
        default_gears = st.session_state.gear_config['gears']
        # create default sample records to avoid indexing issues
        production_template = []
        effort_template = []
        for y in default_years:
            row_p = {'Tahun': y}
            row_e = {'Tahun': y}
            for g in default_gears:
                row_p[g] = 1000
                row_e[g] = 100
            row_p['Jumlah'] = sum([row_p[g] for g in default_gears])
            row_e['Jumlah'] = sum([row_e[g] for g in default_gears])
            production_template.append(row_p)
            effort_template.append(row_e)
        st.session_state.data_tables = {'production': production_template, 'effort': effort_template}

    st.header("Analisis CPUE & MSY")
    # --- sidebar configuration (local to this page) ---
    st.sidebar.header("⚙️ Konfigurasi Analisis")
    start_year = st.sidebar.number_input("Tahun Mulai", min_value=2000, max_value=2030, value=2020, key="start_year_a")
    num_years = st.sidebar.number_input("Jumlah Tahun", min_value=2, max_value=20, value=5, key="num_years_a")
    num_gears = st.sidebar.number_input("Jumlah Alat Tangkap", min_value=2, max_value=8, value=4, key="num_gears_a")

    # build config inputs
    config = st.session_state.gear_config
    gear_names = []
    display_names = []
    for i in range(num_gears):
        default_internal = config['gears'][i] if i < len(config['gears']) else f"Alat_{i+1}"
        default_display = config['display_names'][i] if i < len(config['display_names']) else f"Alat Tangkap {i+1}"
        internal_name = st.sidebar.text_input(f"Kode {i+1}", value=default_internal, key=f"internal_a_{i}")
        display_name = st.sidebar.text_input(f"Nama Tampilan {i+1}", value=default_display, key=f"display_a_{i}")
        gear_names.append(internal_name); display_names.append(display_name)

    standard_gear = st.sidebar.selectbox("Pilih Alat Standar (untuk FPI)", gear_names, index=1 if len(gear_names)>1 else 0)
    if st.sidebar.button("💾 Simpan Konfigurasi Analisis"):
        years_generated = generate_years(start_year, num_years)
        st.session_state.gear_config = {'gears': gear_names, 'display_names': display_names,
                                       'standard_gear': standard_gear, 'years': years_generated, 'num_years': num_years}
        st.success("Konfigurasi disimpan.")
    if st.sidebar.button("🔄 Reset Data Analisis"):
        # reset to default templates
        config_default = st.session_state.gear_config
        years_def = config_default.get('years', generate_years(2020,5))
        gears_def = config_default.get('gears', [])
        production_template = []
        effort_template = []
        for y in years_def:
            row_p = {'Tahun': y}
            row_e = {'Tahun': y}
            for g in gears_def:
                row_p[g] = 1000
                row_e[g] = 100
            row_p['Jumlah'] = sum([row_p[g] for g in gears_def]) if gears_def else 0
            row_e['Jumlah'] = sum([row_e[g] for g in gears_def]) if gears_def else 0
            production_template.append(row_p)
            effort_template.append(row_e)
        st.session_state.data_tables = {'production': production_template, 'effort': effort_template}
        st.success("Data direset.")

    # Input Method Selection
    st.header("📊 Input Data")
    input_method = st.radio(
        "Pilih Metode Input Data",
        ["Input Manual", "Upload File"],
        horizontal=True
    )

    # Local variables that must always exist (pull from session_state)
    config = st.session_state.gear_config
    gears = config.get('gears', [])
    display_names = config.get('display_names', [g.replace('_',' ') for g in gears])
    years = config.get('years', generate_years(2020, 5))
    standard_gear = config.get('standard_gear', gears[0] if gears else None)

    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload file CSV/TXT", type=['csv', 'txt'])
        
        if uploaded_file:
            data, message = parse_uploaded_file(uploaded_file)
            if data:
                st.success("File berhasil diupload!")
                
                # Update gear config in session_state
                st.session_state.gear_config = {
                    'gears': data['gears'],
                    'display_names': data['display_names'],
                    'standard_gear': data['gears'][0] if len(data['gears'])>0 else None,
                    'years': data['years'],
                    'num_years': len(data['years'])
                }
                
                # Update data tables
                st.session_state.data_tables = {
                    'production': data['production'].to_dict('records'),
                    'effort': data['effort'].to_dict('records')
                }
                
                # Refresh local copies
                config = st.session_state.gear_config
                gears = config['gears']
                display_names = config['display_names']
                years = config['years']
                standard_gear = config['standard_gear']
                
                # Show preview
                st.write("Preview Data:")
                st.dataframe(data['production'])
                st.dataframe(data['effort'])
            else:
                st.error(f"Error: {message}")
    
    else:  # Input Manual
        st.subheader("1. Data Produksi (Kg)")
        config = st.session_state.gear_config
        gears = config['gears']
        display_names = config['display_names']
        years = config['years']

        st.info(f"**Alat Tangkap:** {', '.join(display_names)}")
        if years:
            st.info(f"**Periode:** {years[0]} - {years[-1]} ({len(years)} tahun)")
        else:
            st.info("**Periode:** belum dikonfigurasi")

        # Manual input tables
        headers = ["Tahun"] + display_names + ["Jumlah"]
        prod_cols = st.columns(len(headers))
        for i, header in enumerate(headers):
            with prod_cols[i]:
                st.markdown(f"**{header}**")

        production_inputs = []
        # ensure session_state.data_tables has sensible structure
        prod_records = st.session_state.data_tables.get('production', [])
        eff_records = st.session_state.data_tables.get('effort', [])

        for i, year in enumerate(years):
            cols = st.columns(len(headers))
            total_prod = 0
            row = {'Tahun': year}
            with cols[0]:
                st.markdown(f"**{year}**")
            for j, gear in enumerate(gears):
                default_val = 0
                if prod_records and i < len(prod_records):
                    default_val = prod_records[i].get(gear, 1000 * (i+1) * (j+1))
                else:
                    default_val = 1000 * (i+1) * (j+1)
                with cols[j+1]:
                    v = st.number_input(f"{display_names[j]} {year}", min_value=0.0, value=float(default_val), key=f"prod_a_{gear}_{year}")
                row[gear] = v; total_prod += v
            with cols[-1]:
                st.markdown(f"**{total_prod:,.0f}**")
                row['Jumlah'] = total_prod
            production_inputs.append(row)
        st.session_state.data_tables['production'] = production_inputs

        # Effort inputs
        effort_inputs = []
        st.subheader("2. Data Upaya (Trip)")
        for i, year in enumerate(years):
            cols = st.columns(len(headers))
            total_eff = 0
            row = {'Tahun': year}
            with cols[0]:
                st.markdown(f"**{year}**")
            for j, gear in enumerate(gears):
                default_val = 0
                if eff_records and i < len(eff_records):
                    default_val = eff_records[i].get(gear, 100 * (i+1) * (j+1))
                else:
                    default_val = 100 * (i+1) * (j+1)
                with cols[j+1]:
                    v = st.number_input(f"Upaya {display_names[j]} {year}", min_value=0, value=int(default_val), key=f"eff_a_{gear}_{year}")
                row[gear] = v; total_eff += v
            with cols[-1]:
                st.markdown(f"**{total_eff:,}**")
                row['Jumlah'] = total_eff
            effort_inputs.append(row)
        st.session_state.data_tables['effort'] = effort_inputs

    # Separator before analysis
    st.markdown("---")

    # Analysis button
    if st.button("🚀 Lakukan Analisis CPUE dan MSY", type="primary"):
        if not st.session_state.data_tables.get('production') or not st.session_state.data_tables.get('effort'):
            st.error("Mohon input data produksi dan upaya terlebih dahulu!")
            return

        # ensure local config variables reflect latest session state
        config = st.session_state.gear_config
        gears = config.get('gears', gears)
        display_names = config.get('display_names', display_names)
        years = config.get('years', years)
        standard_gear = config.get('standard_gear', standard_gear)

        df_production = pd.DataFrame(st.session_state.data_tables['production'])
        df_effort = pd.DataFrame(st.session_state.data_tables['effort'])

        # Validate data
        if 'Jumlah' not in df_effort.columns or 'Jumlah' not in df_production.columns:
            st.error("Data produksi atau upaya tidak memiliki kolom 'Jumlah'. Periksa input.")
            return

        effort_values = df_effort['Jumlah'].values
        production_values = df_production['Jumlah'].values
        
        valid, message = validate_fishing_data(effort_values, production_values)
        if not valid:
            st.error(f"Data tidak valid: {message}")
            return

        # Continue with analysis
        st.header("📈 Perhitungan CPUE & MSY")
        # ensure gears exist in dataframe columns; if not, create zero columns
        for g in gears:
            if g not in df_production.columns:
                df_production[g] = 0
            if g not in df_effort.columns:
                df_effort[g] = 0

        df_cpue = hitung_cpue(df_production, df_effort, gears)
        # safe default standard_gear if missing
        if not standard_gear or standard_gear not in gears:
            standard_gear = gears[0] if gears else None

        df_fpi = hitung_fpi_per_tahun(df_cpue, gears, standard_gear)
        df_standard_effort = hitung_upaya_standar(df_effort, df_fpi, gears)
        df_standard_cpue = hitung_cpue_standar(df_production, df_standard_effort)

        # tampilkan tabel / download
        st.subheader("CPUE")
        # Align display names length with gears
        display_names_aligned = [display_names[i] if i < len(display_names) else g.replace('_',' ') for i,g in enumerate(gears)]
        cpue_display = df_cpue.copy()
        # Rename columns for display: Tahun, gear display names..., Jumlah
        rename_map = {}
        for i, g in enumerate(gears):
            rename_map[g] = display_names_aligned[i]
        cpue_display = cpue_display.rename(columns=rename_map)
        # show
        st.dataframe(cpue_display.style.format({name: '{:.4f}' for name in rename_map.values()}))

        st.download_button("📥 Download Data CPUE (CSV)", data=convert_df_to_csv(cpue_display), file_name="data_cpue.csv", mime="text/csv")

        # MSY
        effort_values = df_standard_effort['Jumlah'].values
        cpue_values = df_standard_cpue['CPUE_Standar_Total'].values
        production_values = df_production['Jumlah'].values
        results = analisis_msy_schaefer(effort_values, cpue_values)

        if results and results.get('success'):
            st.subheader("Hasil Regresi Linear")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Intercept (a)", f"{results['a']:,.1f}")
            col2.metric("Slope (b)", f"{results['b']:,.1f}")
            col3.metric("R²", f"{results['r_squared']:.6f}")
            col4.metric("p-value", f"{results['p_value']:.6f}")

            st.subheader("Estimasi MSY")
            col1, col2 = st.columns(2)
            col1.metric("F_MSY (Upaya Optimal)", f"{results['F_MSY']:.9f}")
            col2.metric("C_MSY (MSY)", f"{results['C_MSY']:,.2f} kg")

            # pass years as list for plotting
            fig = buat_grafik_lengkap(results, effort_values, cpue_values, production_values, years,
                                      df_cpue, df_fpi, df_standard_effort, gears, display_names_aligned, standard_gear)
            st.pyplot(fig)

            # rekomendasi singkat
            latest_effort = effort_values[-1]
            utilization_effort = (latest_effort / results['F_MSY']) * 100 if results['F_MSY'] > 0 else 0
            if utilization_effort < 80:
                st.success("🟢 UNDER EXPLOITED — rekomendasi: tingkatkan pemanfaatan")
            elif utilization_effort <= 100:
                st.info("🟡 FULLY EXPLOITED — rekomendasi: pertahankan dan monitoring")
            else:
                st.warning("🔴 OVER EXPLOITED — rekomendasi: kurangi upaya segera")
        else:
            if results and 'error' in results:
                st.error(f"❌ Error: {results['error']}")
            else:
                st.error("Data tidak mencukupi untuk analisis MSY.")
    
    # Tambahkan upload file setelah konfigurasi (sidebar)
    st.sidebar.markdown("---")
    st.sidebar.header("📤 Upload Data")
    uploaded_file_sidebar = st.sidebar.file_uploader(
        "Upload file CSV/TXT", 
        type=['csv', 'txt'],
        help="""Format file:
        Tahun,Alat_Tangkap,Produksi,Upaya
        2020,Jaring_Insang,5000,100
        2020,Pancing,3000,80
        ..."""
    )
    
    if uploaded_file_sidebar:
        data, message = parse_uploaded_file(uploaded_file_sidebar)
        if data:
            st.success("File berhasil diupload!")
            
            # Update session state dengan data dari file
            st.session_state.gear_config = {
                'gears': data['gears'],
                'display_names': [g.replace('_', ' ') for g in data['gears']],
                'standard_gear': data['gears'][0] if len(data['gears'])>0 else None,
                'years': data['years'],
                'num_years': len(data['years'])
            }
            
            # Konversi DataFrame ke format yang dibutuhkan
            production_data = data['production'].to_dict('records')
            effort_data = data['effort'].to_dict('records')
            
            st.session_state.data_tables = {
                'production': production_data,
                'effort': effort_data
            }
            
            # Tampilkan preview data
            st.subheader("Preview Data Uploaded")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Data Produksi:")
                st.dataframe(data['production'])
            with col2:
                st.write("Data Upaya:")
                st.dataframe(data['effort'])
        else:
            st.error(f"Error: {message}")
    
    # Tambahkan contoh template
    st.sidebar.markdown("---")
    st.sidebar.markdown("📝 **Template File**")
    
    # Generate contoh data
    example_data = """Tahun,Alat_Tangkap,Produksi,Upaya
2020,Jaring_Insang,5000,100
2020,Pancing,3000,80
2021,Jaring_Insang,4500,120
2021,Pancing,2800,100
2022,Jaring_Insang,4000,140
2022,Pancing,2500,120"""
    
    st.sidebar.download_button(
        label="📥 Download Template CSV",
        data=example_data,
        file_name="template_data.csv",
        mime="text/csv"
    )

# If this file is run directly for debugging:
if __name__ == "__main__":
    main()
