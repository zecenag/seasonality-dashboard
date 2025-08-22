#!/usr/bin/env python3
"""
Test script para verificar que openpyxl funciona
"""
import pandas as pd
import os

print("🔍 Verificando archivos Excel...")
excel_files = [f for f in os.listdir('.') if f.endswith('.xlsx')]
print(f"Archivos Excel encontrados: {excel_files}")

if excel_files:
    try:
        print(f"\n📊 Cargando {excel_files[0]}...")
        df = pd.read_excel(excel_files[0])
        print(f"✅ Archivo cargado exitosamente!")
        print(f"📏 Dimensiones: {df.shape}")
        print(f"📋 Columnas: {list(df.columns)}")
        print(f"\n📄 Primeras 3 filas:")
        print(df.head(3))
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ No se encontraron archivos Excel")
