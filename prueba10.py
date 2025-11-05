import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import os
from datetime import datetime, timedelta

# === FUNCION PARA CALCULAR RSI MANUALMENTE (sin librerías de análisis técnico) ===
def calcular_rsi(series, periodo=14):
    """
    Calcula el RSI (Relative Strength Index) manualmente.
    No usa librerías de análisis técnico, solo operaciones básicas de pandas.
    """
    delta = series.diff()
    ganancia = delta.where(delta > 0, 0)
    perdida = -delta.where(delta < 0, 0)
    
    # Calcular medias móviles simples de ganancias y pérdidas
    media_gan = ganancia.rolling(window=periodo).mean()
    media_per = perdida.rolling(window=periodo).mean()
    
    # Evitar división por cero
    rs = media_gan / media_per.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Crear carpeta para guardar gráficos
os.makedirs("graficos", exist_ok=True)

# Leer símbolos desde el archivo
try:
    with open("simbolos.txt", "r") as f:
        simbolos = [line.strip().upper() for line in f if line.strip()]
except FileNotFoundError:
    print("Error: No se encontró el archivo 'simbolos.txt'")
    exit(1)

if not simbolos:
    print("Error: El archivo 'simbolos.txt' está vacío")
    exit(1)

print(f"Se encontraron {len(simbolos)} símbolos para procesar\n")

# Crear listas para guardar resultados
resultados_desvio = []
resultados_rsi = []

# Calcular fecha de inicio (2 años atrás)
fecha_inicio = datetime.now() - timedelta(days=730)

for simbolo in simbolos:
    print(f"Procesando {simbolo}...")
    
    try:
        # Descargar datos de los últimos 2 años
        data = yf.download(simbolo, period="2y", interval="1d", progress=False)
        
        if data.empty:
            print(f"  ⚠ No se encontraron datos para {simbolo}")
            continue
        
        # Obtener solo la columna de cierre
        precios = data["Close"]
        
        # Calcular RSI manualmente
        rsi_values = calcular_rsi(precios)
        ultimo_rsi = rsi_values.iloc[-1]
        
        # Guardar resultado RSI
        if not np.isnan(ultimo_rsi):
            resultados_rsi.append({
                "Simbolo": simbolo,
                "RSI": round(float(ultimo_rsi), 2)
            })
        else:
            resultados_rsi.append({
                "Simbolo": simbolo,
                "RSI": "N/A"
            })
        
        # Calcular máximos y mínimos parciales (locales)
        # Usar order=5 para detectar máximos y mínimos más significativos
        maxima_idx = argrelextrema(precios.values, np.greater_equal, order=5)[0]
        minima_idx = argrelextrema(precios.values, np.less_equal, order=5)[0]
        
        # Obtener valores de máximos y mínimos globales de toda la serie
        max_global = float(precios.max())
        min_global = float(precios.min())
        ultimo_precio = float(precios.iloc[-1])
        
        # Calcular desvíos porcentuales respecto al máximo y mínimo
        desvio_max = ((ultimo_precio - max_global) / max_global) * 100
        desvio_min = ((ultimo_precio - min_global) / min_global) * 100
        
        # Guardar resultado de desvío
        resultados_desvio.append({
            "Simbolo": simbolo,
            "Desvio_Max(%)": round(desvio_max, 2),
            "Desvio_Min(%)": round(desvio_min, 2)
        })
        
        # === CREAR GRÁFICO ===
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Graficar precio de cierre
        ax.plot(precios.index, precios.values, label="Precio de Cierre", 
                color="blue", linewidth=1.5)
        
        # Marcar máximos parciales
        if len(maxima_idx) > 0:
            ax.scatter(precios.index[maxima_idx], precios.values[maxima_idx],
                      color="red", marker="^", s=100, label="Máximos parciales",
                      zorder=5)
        
        # Marcar mínimos parciales
        if len(minima_idx) > 0:
            ax.scatter(precios.index[minima_idx], precios.values[minima_idx],
                      color="green", marker="v", s=100, label="Mínimos parciales",
                      zorder=5)
        
        # Marcar último precio
        ax.axhline(y=ultimo_precio, color="gray", linestyle="--", 
                  linewidth=1, alpha=0.7, label=f"Último precio: ${ultimo_precio:.2f}")
        
        # Marcar máximo y mínimo global
        ax.axhline(y=max_global, color="red", linestyle=":", 
                  linewidth=1, alpha=0.5, label=f"Máximo global: ${max_global:.2f}")
        ax.axhline(y=min_global, color="green", linestyle=":", 
                  linewidth=1, alpha=0.5, label=f"Mínimo global: ${min_global:.2f}")
        
        # Configurar gráfico
        ax.set_title(f"{simbolo} - Cotizaciones últimos 2 años\n"
                    f"Máximos y Mínimos Parciales", fontsize=14, fontweight='bold')
        ax.set_xlabel("Fecha", fontsize=11)
        ax.set_ylabel("Precio de Cierre (USD)", fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Agregar información adicional
        info_text = (
            f"RSI (14): {ultimo_rsi:.2f}\n"
            f"Desvío respecto al máximo: {desvio_max:.2f}%\n"
            f"Desvío respecto al mínimo: {desvio_min:.2f}%"
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Guardar gráfico
        ruta_img = os.path.join("graficos", f"{simbolo}_prueba10.png")
        plt.savefig(ruta_img, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Gráfico guardado: {ruta_img}")
        
    except Exception as e:
        print(f"  ✗ Error procesando {simbolo}: {str(e)}")
        continue

# === GUARDAR ARCHIVOS CSV ===
print("\n" + "="*50)
print("Generando archivos de salida...")

# Guardar desvio.CSV
if resultados_desvio:
    df_desvio = pd.DataFrame(resultados_desvio)
    df_desvio.to_csv("desvio.CSV", index=False, sep=";", encoding='utf-8-sig')
    print(f"✓ Archivo 'desvio.CSV' generado correctamente ({len(resultados_desvio)} registros)")
else:
    print("⚠ No se generaron datos de desvío")

# Guardar RSI.CSV
if resultados_rsi:
    df_rsi = pd.DataFrame(resultados_rsi)
    df_rsi.to_csv("RSI.CSV", index=False, sep=";", encoding='utf-8-sig')
    print(f"✓ Archivo 'RSI.CSV' generado correctamente ({len(resultados_rsi)} registros)")
else:
    print("⚠ No se generaron datos de RSI")

print("\n" + "="*50)
print("Proceso completado!")
print(f"Gráficos guardados en la carpeta 'graficos/'")

