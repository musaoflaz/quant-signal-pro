import streamlit as st
import pandas as pd
import ccxt
from datetime import datetime
import pytz

# --- AYARLAR ---
COINLER = ['BTC/USDT', 'ETH/USDT', 'NEAR/USDT', 'SOL/USDT', 'AVAX/USDT', 'SOL/USDT', 'LINK/USDT']

st.set_page_config(page_title="Sniper Bot Analiz", layout="wide")
st.title("🎯 Long/Short Skor Tablosu")

# --- ANALİZ FONKSİYONU ---
def analiz_yap():
    sonuclar = []
    exchange = ccxt.binance()
    
    with st.spinner('Veriler analiz ediliyor...'):
        for coin in COINLER:
            try:
                # 1 Saatlik mum verilerini çek
                ohlcv = exchange.fetch_ohlcv(coin, timeframe='1h', limit=50)
                df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
                
                # Fiyat ve Değişim Hesapla
                fiyat = df['c'].iloc[-1]
                degisim = ((df['c'].iloc[-1] - df['c'].iloc[-2]) / df['c'].iloc[-2]) * 100
                
                # Senin Orijinal Skorlama Mantığın
                if degisim > 0:
                    skor = f"{int(70 + degisim*10)} (LONG)"
                else:
                    skor = f"{int(30 + degisim*10)} (SHORT)"
                
                sonuclar.append({
                    "Coin": coin, 
                    "Fiyat": fiyat, 
                    "24s Değişim %": round(degisim, 2),
                    "Skor/Yön": skor
                })
            except:
                continue
    return pd.DataFrame(sonuclar)

# --- ANA EKRAN ---
st.write(f"Son Güncelleme: {datetime.now(pytz.timezone('Europe/Istanbul')).strftime('%H:%M:%S')}")

if st.button("🚀 ANALİZİ BAŞLAT"):
    df_sonuc = analiz_yap()
    
    if not df_sonuc.empty:
        # Tabloyu şık bir şekilde göster
        st.table(df_sonuc)
        st.success("Analiz başarıyla tamamlandı!")
    else:
        st.error("Veriler çekilemedi, lütfen tekrar deneyin.")
