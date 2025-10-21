import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# --- Función para calcular RSI manualmente ---
def calcular_rsi(precios, periodos=14):
    delta = precios.diff()
    ganancia = delta.where(delta > 0, 0)
    perdida = -delta.where(delta < 0, 0)

    avg_gain = ganancia.rolling(window=periodos, min_periods=periodos).mean()
    avg_loss = perdida.rolling(window=periodos, min_periods=periodos).mean()

    # Suavizado exponencial (opcional pero más preciso)
    avg_gain = avg_gain.ewm(alpha=1/periodos, min_periods=periodos, adjust=False).mean()
    avg_loss = avg_loss.ewm(alpha=1/periodos, min_periods=periodos, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Leer símbolos desde el archivo ---
with open("simbolos.txt", "r") as f:
    simbolos = [line.strip() for line in f.readlines() if line.strip()]

# --- Fechas ---
hoy = datetime.now()
inicio = hoy - timedelta(days=30)

# --- Crear lista para resultados ---
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

    except Exception as e:
        print(f"Error con {simbolo}: {e}")

# --- Guardar resultados en CSV ---
df_rsi = pd.DataFrame(resultados)
df_rsi.to_csv("RSI.csv", index=False, sep=";")

print("\nArchivo RSI.csv generado correctamente.")
