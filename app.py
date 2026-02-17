import streamlit as st
import pandas as pd
import ccxt
import numpy as np
from datetime import datetime
import pytz

st.set_page_config(page_title="Sniper Pro v3", layout="wide")
st.title("🎯 Kucoin Optimum Sinyal Yakalayıcı (Stabil)")

# --- HIZLI İNDİKATÖR HESAPLAYICILAR ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger(series, window=20, std_dev=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def analiz_motoru():
    exchange = ccxt.kucoin({'enableRateLimit': True})
    sonuclar = []
    
    st.write("🔍 Piyasa taranıyor... (Python 3.13 Uyumlu Mod)")
    markets = exchange.load_markets()
    symbols = [s for s in markets.keys() if '/USDT' in s and markets[s]['active']][:50]
    
    progress_bar = st.progress(0)
    
    for i, symbol in enumerate(symbols):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=50)
            df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # İndikatörleri Hesapla (Manuel - Sürüm Hatası Vermez)
            df['rsi'] = calculate_rsi(df['c'])
            df['upper'], df['sma'], df['lower'] = calculate_bollinger(df['c'])
            
            son_fiyat = df['c'].iloc[-1]
            rsi_val = df['rsi'].iloc[-1]
            sma_val = df['sma'].iloc[-1]
            alt_bant = df['lower'].iloc[-1]
            ust_bant = df['upper'].iloc[-1]
            
            # --- PROFESYONEL SKORLAMA ---
            skor = 50
            if son_fiyat > sma_val: skor += 15 # Trend Pozitif
            if rsi_val < 35: skor += 25        # Aşırı Satım (Fırsat)
            if rsi_val > 65: skor -= 25        # Aşırı Alım (Risk)
            if son_fiyat <= alt_bant: skor += 20 # Bollinger Dibi
            
            durum = "İZLEMEDE"
            if skor >= 90: durum = "🔥 STRONG LONG"
            elif skor <= 20: durum = "💀 STRONG SHORT"
            
            if not np.isnan(rsi_val): # Veri bozuk değilse ekle
                sonuclar.append({
                    "Coin": symbol,
                    "Fiyat": round(son_fiyat, 4),
                    "RSI": round(rsi_val, 2),
                    "Skor": skor,
                    "Sinyal": durum
                })
        except:
            continue
        progress_bar.progress((i + 1) / len(symbols))
        
    return pd.DataFrame(sonuclar)

if st.button("🚀 OPTİMUM ANALİZİ BAŞLAT"):
    data = analiz_motoru()
    if not data.empty:
        firsatlar = data[data['Skor'] >= 90]
        if not firsatlar.empty:
            st.subheader("✅ BULUNAN FIRSATLAR")
            st.success(f"{len(firsatlar)} tane 90+ skorlu fırsat yakalandı!")
            st.table(firsatlar.sort_values(by="Skor", ascending=False))
        
        st.subheader("📋 Genel Piyasa Listesi")
        st.dataframe(data)
