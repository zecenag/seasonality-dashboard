import streamlit as st
import pandas as pd
import numpy as np
import os

# Configuración de la página
st.set_page_config(
    page_title="Seasonality Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para cargar datos (versión simplificada)
@st.cache_data
def load_data_simple():
    """Carga datos del archivo Excel"""
    try:
        # Buscar archivo Excel en el directorio actual
        excel_files = [f for f in os.listdir('.') if f.endswith('.xlsx')]
        if excel_files:
            data_file = excel_files[0]  # Usar el primer archivo Excel encontrado
            df = pd.read_excel(data_file)
            
            # Verificar y limpiar columnas básicas
            if 'WEEK' in df.columns:
                df['WEEK'] = df['WEEK'].astype(str).str.strip()
            
            # Crear métricas básicas si no existen
            if 'BIAS_PCT' not in df.columns and 'FCST' in df.columns and 'SALES' in df.columns:
                df['BIAS_PCT'] = ((df['FCST'] - df['SALES']) / df['SALES'].replace(0, np.nan)) * 100
            
            if 'ACC_PCT' not in df.columns and 'FCST' in df.columns and 'SALES' in df.columns:
                df['ACC_PCT'] = 100 - (np.abs(df['FCST'] - df['SALES']) / df['SALES'].replace(0, np.nan)) * 100
            
            return df
        else:
            st.error("No se encontró archivo Excel en el directorio")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame()

# Título principal
st.title("📊 Seasonality Dashboard")
st.markdown("---")

# Cargar datos
df = load_data_simple()

if df.empty:
    st.warning("No hay datos para mostrar")
    st.stop()

# Mostrar información básica
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Total Registros", len(df))
with col2:
    if 'STORE_ID' in df.columns:
        st.metric("🏪 Tiendas", df['STORE_ID'].nunique())
with col3:
    if 'COUNTRY' in df.columns:
        st.metric("🌍 Países", df['COUNTRY'].nunique())
with col4:
    if 'WEEK' in df.columns:
        st.metric("📅 Semanas", df['WEEK'].nunique())

st.markdown("---")

# Sidebar para filtros
st.sidebar.header("🔧 Filtros")

# Filtros
filtered_df = df.copy()

if 'COUNTRY' in df.columns:
    countries = ["Todos"] + sorted(df["COUNTRY"].unique().tolist())
    selected_country = st.sidebar.selectbox("País", countries)
    if selected_country != "Todos":
        filtered_df = filtered_df[filtered_df["COUNTRY"] == selected_country]

if 'STORE_ID' in df.columns:
    stores = ["Todos"] + sorted(df["STORE_ID"].unique().tolist())
    selected_store = st.sidebar.selectbox("Tienda", stores)
    if selected_store != "Todos":
        filtered_df = filtered_df[filtered_df["STORE_ID"] == selected_store]

if 'WEEK' in df.columns:
    weeks = ["Todas"] + sorted(df["WEEK"].unique().tolist())
    selected_week = st.sidebar.selectbox("Semana", weeks)
    if selected_week != "Todas":
        filtered_df = filtered_df[filtered_df["WEEK"] == selected_week]

st.sidebar.markdown(f"**Registros filtrados:** {len(filtered_df)}")

# Tabs principales
tab1, tab2, tab3 = st.tabs(["📊 Datos", "📈 Métricas", "📋 Tabla"])

with tab1:
    st.header("📊 Vista General de Datos")
    
    # Mostrar primeras filas
    st.subheader("Primeras filas de datos")
    st.dataframe(filtered_df.head(10), use_container_width=True)
    
    # Estadísticas básicas
    st.subheader("Estadísticas Básicas")
    if not filtered_df.empty:
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(filtered_df[numeric_cols].describe(), use_container_width=True)

with tab2:
    st.header("📈 Métricas")
    
    if 'SALES' in filtered_df.columns and 'FCST' in filtered_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ventas vs Forecast")
            chart_data = filtered_df[['SALES', 'FCST']].head(20)
            st.line_chart(chart_data)
        
        with col2:
            st.subheader("Distribución de Ventas")
            st.bar_chart(filtered_df['SALES'].head(20))
        
        if 'BIAS_PCT' in filtered_df.columns:
            st.subheader("Distribución de BIAS (%)")
            st.histogram = st.bar_chart(filtered_df['BIAS_PCT'].value_counts().head(10))
    else:
        st.info("Columnas SALES y FCST requeridas para mostrar métricas")

with tab3:
    st.header("📋 Tabla Completa")
    
    # Filtro adicional por columnas
    if not filtered_df.empty:
        available_columns = filtered_df.columns.tolist()
        selected_columns = st.multiselect(
            "Seleccionar columnas a mostrar:",
            available_columns,
            default=available_columns[:8] if len(available_columns) > 8 else available_columns
        )
        
        if selected_columns:
            display_df = filtered_df[selected_columns]
            st.dataframe(display_df, use_container_width=True)
            
            # Opción de descarga
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Descargar CSV",
                data=csv,
                file_name="seasonality_data.csv",
                mime="text/csv"
            )
        else:
            st.warning("Selecciona al menos una columna para mostrar")

# Footer
st.markdown("---")
st.markdown("**Seasonality Dashboard** - Versión Streamlit Cloud Compatible")

# Debug info (solo en desarrollo)
if st.sidebar.checkbox("🔍 Debug Info"):
    st.sidebar.write("**Archivos en directorio:**")
    st.sidebar.write(os.listdir('.'))
    st.sidebar.write("**Columnas disponibles:**")
    st.sidebar.write(df.columns.tolist() if not df.empty else "No data")
