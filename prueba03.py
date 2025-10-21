import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import numpy as np

# --- Función para calcular RSI manualmente ---
def calcular_rsi(precios, periodos=14):
    delta = precios.diff()
    ganancia = delta.clip(lower=0)
    perdida = -delta.clip(upper=0)

    # Promedio exponencial (mejor que simple)
    media_ganancia = ganancia.ewm(alpha=1/periodos, min_periods=periodos, adjust=False).mean()
    media_perdida = perdida.ewm(alpha=1/periodos, min_periods=periodos, adjust=False).mean()

    rs = media_ganancia / media_perdida
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # Si faltan datos iniciales, asigna valor neutro
    return rsi

# --- Leer símbolos desde el archivo ---
with open("simbolos.txt", "r") as f:
    simbolos = [line.strip() for line in f.readlines() if line.strip()]

# --- Crear carpeta para gráficos ---
os.makedirs("graficos_RSI", exist_ok=True)

# --- Fechas ---
hoy = datetime.now()
inicio = hoy - timedelta(days=40)  # Se amplía un poco para evitar bordes sin datos

# --- Lista para resultados ---
resultados = []

# --- Procesar cada símbolo ---
for simbolo in simbolos:
    try:
        data = yf.download(simbolo, start=inicio, end=hoy, progress=False)
        if data.empty or len(data) < 15:
            print(f"⚠️ No hay suficientes datos para {simbolo}")
            continue

        data["RSI"] = calcular_rsi(data["Close"])
        ultimo_rsi = float(data["RSI"].iloc[-1])
        ultimo_rsi = round(ultimo_rsi, 2)
        resultados.append({"Simbolo": simbolo, "RSI": ultimo_rsi})

        print(f"{simbolo}: RSI = {ultimo_rsi}")

        # --- Graficar RSI ---
        plt.figure(figsize=(10, 4))
        plt.plot(data.index, data["RSI"], label=f'RSI {simbolo}', color='blue')
        plt.axhline(70, color='red', linestyle='--', linewidth=1)
        plt.axhline(30, color='green', linestyle='--', linewidth=1)
        plt.title(f"RSI de {simbolo} (últimos 30 días)")
        plt.xlabel("Fecha")
        plt.ylabel("RSI")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        # Guardar gráfico
        ruta = os.path.join("graficos_RSI", f"{simbolo}_RSI.png")
        plt.savefig(ruta)
        plt.close()

    except Exception as e:
        print(f"❌ Error con {simbolo}: {e}")

# --- Guardar resultados en CSV ---
if resultados:
    df_rsi = pd.DataFrame(resultados)
    df_rsi.to_csv("RSI.csv", index=False, sep=";")
    print("\n✅ Archivo RSI.csv generado correctamente.")
else:
    print("\n⚠️ No se generó RSI.csv porque no se obtuvieron resultados válidos.")
