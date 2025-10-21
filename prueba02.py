import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# --- Función para calcular RSI manualmente ---
def calcular_rsi(precios, periodos=14):
    delta = precios.diff()
    ganancia = delta.where(delta > 0, 0)
    perdida = -delta.where(delta < 0, 0)

    avg_gain = ganancia.rolling(window=periodos, min_periods=periodos).mean()
    avg_loss = perdida.rolling(window=periodos, min_periods=periodos).mean()

    # Suavizado exponencial (más preciso que promedio simple)
    avg_gain = avg_gain.ewm(alpha=1/periodos, min_periods=periodos, adjust=False).mean()
    avg_loss = avg_loss.ewm(alpha=1/periodos, min_periods=periodos, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Leer símbolos desde el archivo ---
with open("simbolos.txt", "r") as f:
    simbolos = [line.strip() for line in f.readlines() if line.strip()]

# --- Crear carpeta para guardar gráficos ---
os.makedirs("graficos_RSI", exist_ok=True)

# --- Fechas ---
hoy = datetime.now()
inicio = hoy - timedelta(days=30)

# --- Lista para resultados ---
resultados = []

# --- Procesar cada símbolo ---
for simbolo in simbolos:
    try:
        data = yf.download(simbolo, start=inicio, end=hoy)
        if data.empty:
            print(f"No se encontraron datos para {simbolo}")
            continue

        data['RSI'] = calcular_rsi(data['Close'])
        ultimo_rsi = round(data['RSI'].iloc[-1], 2)
        resultados.append({"Simbolo": simbolo, "RSI": ultimo_rsi})
        print(f"{simbolo}: RSI = {ultimo_rsi}")

        # --- Graficar RSI ---
        plt.figure(figsize=(10, 4))
        plt.plot(data.index, data['RSI'], label=f'RSI {simbolo}', color='blue')
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
        print(f"Error con {simbolo}: {e}")

# --- Guardar resultados en CSV ---
df_rsi = pd.DataFrame(resultados)
df_rsi.to_csv("RSI.csv", index=False, sep=";")

print("\n✅ Archivo RSI.csv generado correctamente.")
print("✅ Gráficos guardados en la carpeta 'graficos_RSI'.")
