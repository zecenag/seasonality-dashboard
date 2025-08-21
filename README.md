# Seasonality Dashboard

Dashboard interactivo para análisis de seasonality y forecasting de ventas.

## 🚀 Demo Online

- **Streamlit Cloud**: [Ver Dashboard](https://seasonality-dashboard.streamlit.app)
- **Local**: Ejecuta `streamlit run streamlit_app.py`

## 📊 Características

- ✅ Análisis de BIAS y Accuracy
- ✅ Visualizaciones interactivas con Plotly
- ✅ Filtros por País, Tienda y Semana
- ✅ Matriz de calor y gráficos de dispersión
- ✅ Tabla de datos críticos
- ✅ Exportación de datos

## 🔧 Instalación Local

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 📁 Estructura

```
├── streamlit_app.py          # App principal Streamlit
├── FORECAST_SALES.py         # Dashboard Dash original
├── Graphs.py                 # Funciones de gráficos
├── requirements.txt          # Dependencias
└── data/                     # Archivos de datos
```

## 🚀 Deploy

### Streamlit Cloud
1. Fork este repositorio
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repo
4. ¡Listo!

### Docker (Railway/Render)
```bash
docker build -t seasonality-dashboard .
docker run -p 8050:8050 seasonality-dashboard
```

## 📊 Datos

El dashboard espera un archivo Excel con las siguientes columnas:
- `STORE_ID`: ID de la tienda
- `COUNTRY`: País
- `WEEK`: Semana (formato YYYY-WW)
- `SALES`: Ventas reales
- `FCST`: Forecast
- `BIAS`: Sesgo del forecast
- `ACC`: Accuracy del forecast

## 🔧 Desarrollo

Desarrollado con:
- **Frontend**: Streamlit / Dash
- **Visualizaciones**: Plotly
- **Datos**: Pandas, NumPy
- **Deploy**: Docker, Streamlit Cloud
