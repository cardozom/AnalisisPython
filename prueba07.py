import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import os

# === FUNCION PARA CALCULAR RSI MANUALMENTE ===
def calcular_rsi(series, periodo=14):
    delta = series.diff()
    ganancia = delta.where(delta > 0, 0)
    perdida = -delta.where(delta < 0, 0)
    
    media_gan = ganancia.rolling(window=periodo).mean()
    media_per = perdida.rolling(window=periodo).mean()
    
    rs = media_gan / media_per
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Crear carpeta para guardar gráficos
os.makedirs("graficos", exist_ok=True)

# Leer símbolos desde el archivo
with open("simbolos.txt", "r") as f:
    simbolos = [line.strip().upper() for line in f if line.strip()]

# Crear listas para guardar resultados
resultados_desvio = []
resultados_rsi = []

for simbolo in simbolos:
    print(f"Procesando {simbolo}...")
    data = yf.download(simbolo, period="2y", interval="1d", progress=False)
    
    if data.empty:
        print(f"No se encontraron datos para {simbolo}")
        continue

    # Calcular medias móviles
    data["MA50"] = data["Close"].rolling(window=50).mean()
    data["MA200"] = data["Close"].rolling(window=200).mean()

    # Calcular RSI
    data["RSI"] = calcular_rsi(data["Close"])
    ultimo_rsi = round(float(data["RSI"].iloc[-1]), 2) if not np.isnan(data["RSI"].iloc[-1]) else np.nan
    resultados_rsi.append({"Simbolo": simbolo, "RSI": ultimo_rsi})

    # Calcular máximos y mínimos locales
    data["max_local"] = np.nan
    data["min_local"] = np.nan
    maxima_idx = argrelextrema(data["Close"].values, np.greater_equal, order=5)[0]
    minima_idx = argrelextrema(data["Close"].values, np.less_equal, order=5)[0]
    data.loc[data.index[maxima_idx], "max_local"] = data["Close"].iloc[maxima_idx]
    data.loc[data.index[minima_idx], "min_local"] = data["Close"].iloc[minima_idx]

    # Obtener máximos y mínimos globales
    max_global = float(data["Close"].max())
    min_global = float(data["Close"].min())
    ultima = float(data["Close"].iloc[-1])  # ← aseguramos que sea un número real

    # Calcular desvíos porcentuales
    desvio_max = ((ultima - max_global) / max_global) * 100
    desvio_min = ((ultima - min_global) / min_global) * 100

    resultados_desvio.append({
        "Simbolo": simbolo,
        "Ultimo_Cierre": round(ultima, 2),
        "Maximo_Serie": round(max_global, 2),
        "Minimo_Serie": round(min_global, 2),
        "Desvio_Max(%)": round(desvio_max, 2),
        "Desvio_Min(%)": round(desvio_min, 2)
    })

    # === GRAFICAR ===
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data["Close"], label="Cierre", color="blue", linewidth=1)
    plt.plot(data.index, data["MA50"], label="Media 50 ruedas", color="orange", linewidth=1.2)
    plt.plot(data.index, data["MA200"], label="Media 200 ruedas", color="purple", linewidth=1.2)
    plt.scatter(data.index, data["max_local"], color="red", label="Máximos parciales", marker="^")
    plt.scatter(data.index, data["min_local"], color="green", label="Mínimos parciales", marker="v")
    plt.axhline(y=ultima, color="gray", linestyle="--", linewidth=1, label=f"Último precio ({ultima:.2f})")

    plt.title(f"{simbolo} - Cotización últimos 2 años")
    plt.xlabel("Fecha")
    plt.ylabel("Precio de Cierre (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Guardar gráfico como imagen PNG
    ruta_img = os.path.join("graficos", f"{simbolo}.png")
    plt.savefig(ruta_img)
    print(f"Gráfico guardado: {ruta_img}")

    # Mostrar en pantalla
    plt.show()

# === GUARDAR RESULTADOS ===
df_desvio = pd.DataFrame(resultados_desvio)
df_desvio.to_csv("desvio.csv", index=False)
print("\nArchivo 'desvio.csv' generado correctamente.")

df_rsi = pd.DataFrame(resultados_rsi)
df_rsi.to_csv("rsi.csv", index=False)
print("Archivo 'rsi.csv' generado correctamente.")

print("Gráficos guardados en la carpeta 'graficos'.")
