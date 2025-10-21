import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# Leer símbolos desde el archivo
with open("simbolos.txt", "r") as f:
    simbolos = [line.strip().upper() for line in f if line.strip()]

# Crear lista para guardar resultados
resultados = []

for simbolo in simbolos:
    print(f"Procesando {simbolo}...")
    data = yf.download(simbolo, period="2y", interval="1d", progress=False)
    
    if data.empty:
        print(f"No se encontraron datos para {simbolo}")
        continue

    # Calcular máximos y mínimos locales
    data["max_local"] = data["Close"].iloc[argrelextrema(data["Close"].values, np.greater_equal, order=5)[0]]
    data["min_local"] = data["Close"].iloc[argrelextrema(data["Close"].values, np.less_equal, order=5)[0]]

    # Obtener máximos y mínimos globales
    max_global = data["Close"].max()
    min_global = data["Close"].min()
    ultima = data["Close"].iloc[-1]

    # Calcular desvíos porcentuales
    desvio_max = ((ultima - max_global) / max_global) * 100
    desvio_min = ((ultima - min_global) / min_global) * 100

    resultados.append({
        "Simbolo": simbolo,
        "Desvio_Max(%)": round(desvio_max, 2),
        "Desvio_Min(%)": round(desvio_min, 2)
    })

    # Graficar cotización y extremos locales
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data["Close"], label="Cierre", color="blue")
    plt.scatter(data.index, data["max_local"], color="red", label="Máximos parciales", marker="^")
    plt.scatter(data.index, data["min_local"], color="green", label="Mínimos parciales", marker="v")
    plt.title(f"{simbolo} - Cotización últimos 2 años")
    plt.xlabel("Fecha")
    plt.ylabel("Precio de Cierre (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Guardar archivo CSV con desvíos
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("desvio.csv", index=False)
print("\nArchivo 'desvio.csv' generado correctamente.")
