import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

# Sayfa Ayarları
st.set_page_config(layout="wide", page_title="Quant Pro | Ultimate Signal Alpha")

# Borsa Bağlantısı (Bulut için en yüksek stabilite)
exchange = ccxt.bybit({'enableRateLimit': True, 'timeout': 60000})

st.markdown("# 🏛️ QUANT PRO - ULTIMATE TREND TRACKER")
st.write("---")

symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT', 'AVAX/USDT', 'SUI/USDT', 'PEPE/USDT']

def get_pro_signals():
    rows = []
    progress = st.progress(0)
    
    for idx, symbol in enumerate(symbols):
        try:
            # Daha derin analiz için 200 mumluk veri (H1)
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=200)
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # 1. Trend: EMA 200 & SuperTrend
            df['EMA200'] = ta.ema(df['c'], length=200)
            st_data = ta.supertrend(df['h'], df['l'], df['c'], length=10, multiplier=3)
            df = pd.concat([df, st_data], axis=1)
            
            # 2. Momentum: RSI & MACD
            df['RSI'] = ta.rsi(df['c'], length=14)
            macd = ta.macd(df['c'])
            df = pd.concat([df, macd], axis=1)
            
            last = df.iloc[-1]
            c, rsi, ema = last['c'], last['RSI'], last['EMA200']
            st_dir = last['SUPERTd_10_3.0'] # 1=Boğa, -1=Ayı
            
            # Gelişmiş Sinyal Skoru
            score = 0
            if c > ema: score += 25  # Ana Trend Pozitif
            if st_dir == 1: score += 25 # SuperTrend Pozitif
            if rsi < 40: score += 25 # Aşırı Satım (Toplama Alanı)
            if last['MACD_12_26_9'] > last['MACDs_12_26_9']: score += 25 # MACD Kesişimi
            
            # Short için tam tersi
            short_score = 0
            if c < ema: short_score += 25
            if st_dir == -1: short_score += 25
            if rsi > 60: short_score += 25
            if last['MACD_12_26_9'] < last['MACDs_12_26_9']: short_score += 25

            # Eylem Belirleme
            if score >= 75: eylem = "🚀 GÜÇLÜ LONG"
            elif score == 50: eylem = "🟢 LONG"
            elif short_score >= 75: eylem = "💥 GÜÇLÜ SHORT"
            elif short_score == 50: eylem = "🔴 SHORT"
            else: eylem = "⚪ BEKLE"

            rows.append({
                "VARLIK": symbol,
                "FİYAT": f"{c:.4f}",
                "TREND": "BOĞA" if c > ema else "AYI",
                "SİNYAL GÜCÜ": f"%{max(score, short_score)}",
                "İŞLEM EYLEMİ": eylem,
                "TEKNİK": f"RSI:{int(rsi)} | ST:{'BOĞA' if st_dir==1 else 'AYI'}"
            })
            time.sleep(0.2)
        except: continue
        progress.progress((idx + 1) / len(symbols))
    
    progress.empty()
    return pd.DataFrame(rows)

# Görsel Stil
def style_signal(val):
    if "LONG" in str(val): return 'background-color: #0c3e1e; color: #52ff8f; font-weight: bold'
    if "SHORT" in str(val): return 'background-color: #4b0a0a; color: #ff6e6e; font-weight: bold'
    return ''

# Uygulama Başlatma
data = get_pro_signals()
if not data.empty:
    st.dataframe(data.style.applymap(style_signal, subset=['İŞLEM EYLEMİ']), use_container_width=True, height=600)
else:
    st.warning("Veriler işleniyor, lütfen sayfayı yenilemeyin...")

if st.sidebar.button('🔄 Derin Analiz Yap'):
    st.rerun()
