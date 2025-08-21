import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re

# Configuración de la página
st.set_page_config(
    page_title="Seasonality Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importar funciones del dashboard original
from FORECAST_SALES import (
    load_data, 
    create_forecast_vs_sales_chart,
    create_bias_sales_scatter,
    create_bias_histogram,
    create_pareto_chart,
    create_bias_matrix_heatmap,
    create_bias_country_table,
    create_acc_bias_scatter,
    create_enhanced_table
)

# Título principal
st.title("📊 Seasonality Dashboard")
st.markdown("---")

# Cargar datos
@st.cache_data
def get_data():
    return load_data()

try:
    df = get_data()
    st.success(f"✅ Datos cargados: {len(df)} registros")
except Exception as e:
    st.error(f"❌ Error cargando datos: {e}")
    st.stop()

# Sidebar para filtros
st.sidebar.header("🔧 Filtros")

# Filtros
countries = ["ALL"] + sorted(df["COUNTRY"].unique().tolist())
selected_country = st.sidebar.selectbox("País", countries)

stores = ["ALL"] + sorted(df["STORE_ID"].unique().tolist()) 
selected_store = st.sidebar.selectbox("Tienda", stores)

weeks = ["ALL"] + sorted(df["WEEK"].unique().tolist())
selected_week = st.sidebar.selectbox("Semana", weeks)

# Aplicar filtros
filtered_df = df.copy()

if selected_country != "ALL":
    filtered_df = filtered_df[filtered_df["COUNTRY"] == selected_country]

if selected_store != "ALL":
    filtered_df = filtered_df[filtered_df["STORE_ID"] == selected_store]
    
if selected_week != "ALL":
    filtered_df = filtered_df[filtered_df["WEEK"] == selected_week]

st.sidebar.markdown(f"**Registros filtrados:** {len(filtered_df)}")

# Tabs principales
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🎯 Accuracy", "📊 ACC vs BIAS", "📋 Tabla"])

with tab1:
    st.header("📈 Overview General")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Forecast vs Sales")
        fig_main = create_forecast_vs_sales_chart(filtered_df, "BIAS_PCT", "BIAS")
        st.plotly_chart(fig_main, use_container_width=True)
        
        st.subheader("Histograma de BIAS")
        fig_hist = create_bias_histogram(filtered_df)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("BIAS vs Sales Scatter")
        fig_scatter = create_bias_sales_scatter(filtered_df)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.subheader("Pareto Chart")
        fig_pareto = create_pareto_chart(filtered_df)
        st.plotly_chart(fig_pareto, use_container_width=True)
    
    st.subheader("🔥 Heatmap Matrix")
    fig_matrix = create_bias_matrix_heatmap(filtered_df, aggfunc="mean")
    st.plotly_chart(fig_matrix, use_container_width=True)
    
    st.subheader("🌍 Resumen por País")
    table_country = create_bias_country_table(filtered_df)
    st.components.v1.html(table_country.to_html(), height=400, scrolling=True)

with tab2:
    st.header("🎯 Accuracy Analysis")
    fig_acc = create_forecast_vs_sales_chart(filtered_df, "ACC_PCT", "ACC")
    st.plotly_chart(fig_acc, use_container_width=True)

with tab3:
    st.header("📊 Accuracy vs BIAS Scatter")
    fig_acc_bias = create_acc_bias_scatter(filtered_df)
    st.plotly_chart(fig_acc_bias, use_container_width=True)

with tab4:
    st.header("📋 Tabla de Datos Críticos")
    critical_data = filtered_df.loc[
        filtered_df["SEVERITY"] >= 4, 
        ["STORE_ID","COUNTRY","WEEK","SALES","FCST","BIAS_PCT","ACC_PCT","SEVERITY"]
    ].copy()
    
    if len(critical_data) > 0:
        st.dataframe(critical_data, use_container_width=True)
        
        # Opción de descarga
        csv = critical_data.to_csv(index=False)
        st.download_button(
            label="📥 Descargar CSV",
            data=csv,
            file_name="critical_data.csv",
            mime="text/csv"
        )
    else:
        st.info("✅ No hay datos críticos con SEVERITY >= 4")

# Footer
st.markdown("---")
st.markdown("**Dashboard de Seasonality** - Desarrollado con Streamlit")
