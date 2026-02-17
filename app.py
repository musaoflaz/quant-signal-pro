import streamlit as st
import pandas as pd
import ccxt
import requests
import time
from datetime import datetime
import pytz

# --- AYARLAR ---
TELEGRAM_TOKEN = "BURAYA_TOKEN_YAZ"
TELEGRAM_CHAT_ID = "BURAYA_ID_YAZ"
COINLER = ['BTC/USDT', 'ETH/USDT', 'NEAR/USDT', 'SOL/USDT', 'AVAX/USDT']

st.set_page_config(page_title="Sniper Bot Pro", layout="wide")
st.title("🎯 Long/Short Skor Sistemi")

# --- BAŞARILI ANALİZ FONKSİYONU ---
def analiz_yap():
    sonuclar = []
    exchange = ccxt.binance()
    for coin in COINLER:
        try:
            ohlcv = exchange.fetch_ohlcv(coin, timeframe='1h', limit=50)
            df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # Senin Başarılı Skorlama Mantığın
            fiyat = df['c'].iloc[-1]
            degisim = ((df['c'].iloc[-1] - df['c'].iloc[-2]) / df['c'].iloc[-2]) * 100
            
            if degisim > 0:
                skor = f"{int(70 + degisim*10)} (LONG)"
            else:
                skor = f"{int(30 + degisim*10)} (SHORT)"
            
            sonuclar.append({"Coin": coin, "Fiyat": fiyat, "Skor": skor})
        except:
            continue
    return pd.DataFrame(sonuclar)

# --- SENİN ESKİ BUTONLU SİSTEMİN ---
if st.button("🚀 SİSTEMİ BAŞLAT"):
    df_sonuc = analiz_yap()
    st.table(df_sonuc)
    
    # Telegram Gönderimi
    mesaj = "🚀 GÜNCEL SKORLAR\n\n" + df_sonuc.to_string(index=False)
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": mesaj})
    st.success("Telegram'a gönderildi!")

# --- 7/24 PING DESTEĞİ (Sistemi Bozmayan Kısım) ---
# Sadece bu alt kısım Render'ın uyumasını engeller, yukarıdaki koduna dokunmaz.
st.sidebar.write("---")
st.sidebar.info("7/24 Modu Aktif")
time.sleep(300) # 5 dakika bekle
st.rerun() # Sayfayı tazele (UptimeRobot için)
