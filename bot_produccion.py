import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
import requests
import os
import json
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

    precio_al_hoy = df['al30d'].iloc[-1]
    precio_gd_hoy = df['gd30d'].iloc[-1]

    # --- 4. LÓGICA DE DECISIÓN Y ALERTAS ---
    # Asumimos estado flat (0) para la demostración. 
    COSTO_TRANSACCION = 0.002 # 0.2%
    
    # 4.1. Abrir la memoria
    try:
        with open('cartera.json', 'r') as f:
            cartera = json.load(f)
    except Exception:
        enviar_telegram("🚨 Error: No se pudo leer cartera.json")
        return

    estado_actual = cartera['estado_actual']
    capital = cartera['capital_cash']
    accion = "ESPERAR (Mantener posición actual)"
    rendimiento_trade = 0.0

    # 4.2. Toma de Decisiones y Contabilidad
    if estado_actual == 0:
        if z_score_hoy > UMBRAL_ENTRADA and pval < UMBRAL_PVAL_ADF:
            accion = "🔴 ABRIR SHORT (Vender AL30, Comprar GD30)"
            cartera['estado_actual'] = -1
            cartera['precio_al30_entrada'] = precio_al_hoy
            cartera['precio_gd30_entrada'] = precio_gd_hoy
            cartera['beta_entrada'] = beta_hoy
            cartera['capital_cash'] = capital * (1 - COSTO_TRANSACCION) # Pago comisión entrada
            
        elif z_score_hoy < -UMBRAL_ENTRADA and pval < UMBRAL_PVAL_ADF:
            accion = "🟢 ABRIR LONG (Comprar AL30, Vender GD30)"
            cartera['estado_actual'] = 1
            cartera['precio_al30_entrada'] = precio_al_hoy
            cartera['precio_gd30_entrada'] = precio_gd_hoy
            cartera['beta_entrada'] = beta_hoy
            cartera['capital_cash'] = capital * (1 - COSTO_TRANSACCION)

    elif estado_actual == 1: # Estábamos comprados
        if z_score_hoy > -TAKE_PROFIT:
            accion = "✅ CERRAR LONG (Take Profit)"
            retorno = (precio_al_hoy / cartera['precio_al30_entrada'] - 1) - cartera['beta_entrada'] * (precio_gd_hoy / cartera['precio_gd30_entrada'] - 1)
            capital_bruto = capital * (1 + retorno)
            cartera['capital_cash'] = capital_bruto * (1 - COSTO_TRANSACCION) # Comisión salida
            rendimiento_trade = retorno * 100
            cartera['estado_actual'] = 0

    elif estado_actual == -1: # Estábamos vendidos
        if z_score_hoy < TAKE_PROFIT:
            accion = "✅ CERRAR SHORT (Take Profit)"
            # El retorno se invierte porque estamos shorteados en el spread
            retorno = -1 * ((precio_al_hoy / cartera['precio_al30_entrada'] - 1) - cartera['beta_entrada'] * (precio_gd_hoy / cartera['precio_gd30_entrada'] - 1))
            capital_bruto = capital * (1 + retorno)
            cartera['capital_cash'] = capital_bruto * (1 - COSTO_TRANSACCION)
            rendimiento_trade = retorno * 100
            cartera['estado_actual'] = 0

    # 4.3. Guardar la memoria para mañana
    with open('cartera.json', 'w') as f:
        json.dump(cartera, f, indent=4)

    # --- 5. REPORTE TELEGRAM ---
    # Traducimos el estado para el reporte
    txt_estado = "FLAT (Fuera del mercado)" if cartera['estado_actual'] == 0 else ("LONG" if cartera['estado_actual'] == 1 else "SHORT")
    
    reporte = (
        f"🤖 *Reporte Quant Diario* ({fecha_hoy})\n\n"
        f"📊 *Métricas del Mercado:*\n"
        f"• Z-Score: `{z_score_hoy:.2f}`\n"
        f"• ADF P-Valor: `{pval:.4f}`\n"
        f"• AL30D: `${precio_al_hoy:.2f}` | GD30D: `${precio_gd_hoy:.2f}`\n\n"
        f"💼 *Tu Billetera Virtual:*\n"
        f"• Capital Total: `USD ${cartera['capital_cash']:.2f}`\n"
        f"• Posición Actual: `{txt_estado}`\n\n"
        f"🎯 *Decisión de Hoy:* \n*{accion}*"
    )
    
    if rendimiento_trade != 0.0:
        reporte += f"\n\n📈 Rendimiento de este trade: `{rendimiento_trade:.2f}%`"

    print(reporte)
    enviar_telegram(reporte)
if __name__ == "__main__":
    ejecutar_bot_diario()
