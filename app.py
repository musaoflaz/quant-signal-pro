import streamlit as st
import pandas as pd
import ccxt
import requests  # Telegram için en güvenli ve basit yöntem
import time
from datetime import datetime
import pytz

# --- 1. AYARLAR (Kendi Bilgilerini Buraya Gir) ---
TELEGRAM_TOKEN = "BURAYA_TOKEN_YAZ"
TELEGRAM_CHAT_ID = "BURAYA_ID_YAZ"
COINLER = ['BTC/USDT', 'ETH/USDT', 'NEAR/USDT', 'SOL/USDT', 'AVAX/USDT']

st.set_page_config(page_title="Sniper Bot Pro", layout="wide")
st.title("🎯 Long/Short Skor Sistemi (7/24)")

# --- 2. TELEGRAM GÖNDERME FONKSİYONU (Yeni ve Sorunsuz) ---
def telegram_mesaj_gonder(mesaj):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": mesaj, "parse_mode": "Markdown"}
        requests.post(url, json=payload)
    except Exception as e:
        st.error(f"Telegram Hatası: {e}")

# --- 3. BAŞARILI ANALİZ VE SKORLAMA ---
def analiz_ve_tablo():
    sonuclar = []
    st.write(f"🔄 Tarama Başlatıldı: {datetime.now(pytz.timezone('Europe/Istanbul')).strftime('%H:%M:%S')}")
    
    exchange = ccxt.binance()
    for coin in COINLER:
        try:
            ohlcv = exchange.fetch_ohlcv(coin, timeframe='1h', limit=50)
            df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # Senin Başarılı Skorlama Mantığın
            son_fiyat = df['c'].iloc[-1]
            fark = ((df['c'].iloc[-1] - df['c'].iloc[-2]) / df['c'].iloc[-2]) * 100
            
            if fark > 0:
                skor = f"{int(75 + fark*5)} (LONG) ✅"
            else:
                skor = f"{int(25 + fark*5)} (SHORT) ❌"
            
            sonuclar.append({"Coin": coin, "Fiyat": son_fiyat, "Skor": skor})
        except:
            continue
    
    df_final = pd.DataFrame(sonuclar)
    st.table(df_final) # Tabloyu ekrana basar
    
    # Telegram Mesajını Hazırla
    rapor = "🚀 **GÜNCEL SKOR RAPORU** 🚀\n\n"
    for index, row in df_final.iterrows():
        rapor += f"🔹 {row['Coin']}: {row['Fiyat']} | **Skor: {row['Skor']}**\n"
    
    telegram_mesaj_gonder(rapor)

# --- 4. OTOMATİK ÇALIŞTIRMA VE PING ---
# Sayfa her açıldığında (UptimeRobot sayesinde) analiz başlar
analiz_ve_tablo()

st.sidebar.markdown("---")
st.sidebar.success("🤖 Bot Şu An Nöbette!")
st.sidebar.write("UptimeRobot her 5 dakikada bir kontrol ediyor.")

# Sayfayı 15 dakikada bir yenile (Döngü)
time.sleep(900) 
st.rerun()
