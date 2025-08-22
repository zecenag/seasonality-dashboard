# 🚀 INSTRUCCIONES PARA SUBIR A GITHUB Y STREAMLIT CLOUD

## 📁 Archivos listos para subir:

✅ streamlit_app.py (dashboard principal)
✅ FORECAST_SALES.py (funciones originales)
✅ Graphs.py (gráficos)
✅ requirements.txt (dependencias)
✅ README.md (documentación)
✅ .gitignore (configuración Git)
✅ STORE_SUMMARY_long_form.xlsx (datos de ejemplo)

## 🎯 PASOS PARA GITHUB:

### 1. Crear Repositorio
- Ve a: https://github.com/new
- Nombre: `seasonality-dashboard`
- Descripción: `Interactive dashboard for seasonality and forecasting analysis`
- ✅ Público (para Streamlit Cloud gratis)
- ❌ NO marques "Add README" (ya lo tienes)
- Click "Create repository"

### 2. Subir Archivos
- En tu nuevo repo, click "uploading an existing file"
- Arrastra TODOS los archivos de esta carpeta:
  * streamlit_app.py
  * FORECAST_SALES.py
  * Graphs.py
  * requirements.txt
  * README.md
  * .gitignore
  * STORE_SUMMARY_long_form.xlsx
  * print_store_summary_long_form.py
  * SEASONALITY_TREND/ (si tiene contenido)

### 3. Commit
- Message: "Initial dashboard upload"
- Click "Commit changes"

## 🎯 PASOS PARA STREAMLIT CLOUD:

### 1. Ir a Streamlit Cloud
- Ve a: https://share.streamlit.io/
- Sign in with GitHub

### 2. Crear Nueva App
- Click "New app"
- Repository: tu-usuario/seasonality-dashboard
- Branch: main
- Main file path: streamlit_app.py
- Click "Deploy!"

### 3. Esperar Deploy
- Toma 2-5 minutos
- Se abrirá automáticamente cuando esté listo
- URL será algo como: https://seasonality-dashboard-xxx.streamlit.app

## ⚠️ IMPORTANTE:

1. **Datos de ejemplo**: El archivo Excel actual tiene datos sintéticos
   - Para usar tus datos reales, reemplaza STORE_SUMMARY_long_form.xlsx
   - Puedes hacerlo desde GitHub después del deploy inicial

2. **Actualizar datos**: 
   - Sube nuevo Excel a GitHub
   - Streamlit Cloud se actualiza automáticamente

3. **URL compartible**:
   - Una vez deployed, comparte la URL con tu equipo
   - Es público pero solo quien tenga el link puede acceder

## 🆘 SI HAY PROBLEMAS:

### Error en Streamlit Cloud:
- Check "App logs" en el dashboard de Streamlit
- Común: falta dependencia en requirements.txt

### Error de datos:
- Verifica que STORE_SUMMARY_long_form.xlsx esté en el repo
- Asegúrate que tenga las columnas correctas

## ✨ RESULTADO FINAL:

Dashboard online accesible desde cualquier lugar:
- ✅ Sin instalar nada en otros PCs
- ✅ Actualizable subiendo nuevos archivos
- ✅ Compartible con URL
- ✅ Gratis en Streamlit Cloud

¡Tu dashboard estará disponible 24/7 en la nube! 🌤️
