import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
import requests
import os
from tvDatafeed import TvDatafeed, Interval

# --- 1. CREDENCIALES Y PARÁMETROS ---
# GitHub Actions le va a inyectar estos valores de forma segura
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

VENTANA_BETA = 120
VENTANA_Z = 60
UMBRAL_ENTRADA = 1.5
TAKE_PROFIT = 1.0
UMBRAL_PVAL_ADF = 0.10

def enviar_telegram(mensaje):
    """Función para mandar la alerta a tu celular"""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("Faltan credenciales de Telegram. No se pudo enviar el mensaje.")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": mensaje,
        "parse_mode": "Markdown"
    }
    requests.post(url, json=payload)

def ejecutar_bot_diario():
    fecha_hoy = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"[{fecha_hoy}] Despertando bot de Pairs Trading...")

    # --- 2. OBTENER DATOS FRESCOS ---
    try:
        print("Conectando a TradingView...")
        tv = TvDatafeed() # Inicialización anónima gratuita
        
        # Bajamos un poco más de los 120 días que necesitamos por las dudas
        al30d_raw = tv.get_hist(symbol='AL30D', exchange='BCBA', interval=Interval.in_daily, n_bars=VENTANA_BETA + 10)
        gd30d_raw = tv.get_hist(symbol='GD30D', exchange='BCBA', interval=Interval.in_daily, n_bars=VENTANA_BETA + 10)

        # Filtramos solo la columna de cierre y emulamos tu viejo CSV
        al30d = al30d_raw[['close']].rename(columns={'close': 'al30d'})
        gd30d = gd30d_raw[['close']].rename(columns={'close': 'gd30d'})
        
        # Unimos ambas series por fecha para asegurar que estén alineadas
        df = pd.merge(al30d, gd30d, left_index=True, right_index=True, how='inner').reset_index()
        df.rename(columns={'datetime': 'fecha'}, inplace=True)
        
        # Recortamos exactamente a los últimos 120 días para el modelo
        df = df.tail(VENTANA_BETA).copy()
        
    except Exception as e:
        enviar_telegram(f"🚨 ERROR CRÍTICO: Falló la conexión a TradingView.\nDetalles: `{e}`")
        return

    df['lnal'] = np.log(df['al30d'])
    df['lngd'] = np.log(df['gd30d'])

    # --- 3. CÁLCULO ESTADÍSTICO ---
    X = sm.add_constant(df['lngd'])
    modelo = sm.OLS(df['lnal'], X).fit()
    beta_hoy = modelo.params['lngd']
    constante_hoy = modelo.params['const']

    try:
        _, pval, _, _, _, _ = adfuller(modelo.resid, maxlag=1)
    except:
        pval = 1.0

    df['spread'] = df['lnal'] - beta_hoy * df['lngd'] - constante_hoy
    
    tail_60 = df.tail(VENTANA_Z)
    media_z = tail_60['spread'].mean()
    std_z = tail_60['spread'].std()
    
    spread_hoy = df['spread'].iloc[-1]
    z_score_hoy = (spread_hoy - media_z) / std_z

    # --- 4. LÓGICA DE DECISIÓN Y ALERTAS ---
    # Asumimos estado flat (0) para la demostración. 
    estado_actual = 0 
    accion = "ESPERAR"

    if estado_actual == 0:
        if z_score_hoy > UMBRAL_ENTRADA and pval < UMBRAL_PVAL_ADF:
            accion = "🔴 ABRIR SHORT (Vender AL30, Comprar GD30)"
        elif z_score_hoy < -UMBRAL_ENTRADA and pval < UMBRAL_PVAL_ADF:
            accion = "🟢 ABRIR LONG (Comprar AL30, Vender GD30)"
    
    # Armamos el reporte visual para Telegram
    reporte = (
        f"🤖 *Reporte Quant Diario* ({fecha_hoy})\n\n"
        f"📊 *Métricas:*\n"
        f"• Z-Score: `{z_score_hoy:.2f}`\n"
        f"• ADF P-Valor: `{pval:.4f}`\n"
        f"• Beta Actual: `{beta_hoy:.4f}`\n\n"
        f"🎯 *Decisión:* \n*{accion}*"
    )
    
    print(reporte)
    # Disparamos el mensaje a tu celular
    enviar_telegram(reporte)

if __name__ == "__main__":
    ejecutar_bot_diario()
