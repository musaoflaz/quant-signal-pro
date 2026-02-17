import streamlit as st
import pandas as pd
import ccxt
import requests
from datetime import datetime
import pytz

# --- AYARLAR (Bilgilerini Buraya Gir) ---
TELEGRAM_TOKEN = "BURAYA_TOKEN_YAZ"
TELEGRAM_CHAT_ID = "BURAYA_ID_YAZ"
COINLER = ['BTC/USDT', 'ETH/USDT', 'NEAR/USDT', 'SOL/USDT', 'AVAX/USDT']

st.set_page_config(page_title="Sniper Bot Pro", layout="wide")
st.title("🎯 Long/Short Skor Sistemi")

# --- ANALİZ FONKSİYONU ---
def analiz_yap():
    sonuclar = []
    exchange = ccxt.binance()
    for coin in COINLER:
        try:
            ohlcv = exchange.fetch_ohlcv(coin, timeframe='1h', limit=50)
            df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            fiyat = df['c'].iloc[-1]
            degisim = ((df['c'].iloc[-1] - df['c'].iloc[-2]) / df['c'].iloc[-2]) * 100
            
            # Başarılı Skorlama Mantığın
            if degisim > 0:
                skor = f"{int(70 + degisim*10)} (LONG) ✅"
            else:
                skor = f"{int(30 + degisim*10)} (SHORT) ❌"
            
            sonuclar.append({"Coin": coin, "Fiyat": fiyat, "Skor": skor})
        except:
            continue
    return pd.DataFrame(sonuclar)

# --- SİSTEMİ BAŞLAT BUTONU ---
if st.button("🚀 SİSTEMİ BAŞLAT"):
    st.write(f"Tarama yapıldı: {datetime.now(pytz.timezone('Europe/Istanbul')).strftime('%H:%M:%S')}")
    df_sonuc = analiz_yap()
    
    # 1. EKRANA TABLO BAS
    st.table(df_sonuc)
    
    # 2. TELEGRAM'A MESAJ AT
    mesaj = "🚀 **GÜNCEL SKORLAR** 🚀\n\n"
    for i, row in df_sonuc.iterrows():
        mesaj += f"🔹 {row['Coin']}: {row['Fiyat']} | **{row['Skor']}**\n"
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": mesaj, "parse_mode": "Markdown"})
    st.success("Sinyaller Telegram'a başarıyla gönderildi!")
