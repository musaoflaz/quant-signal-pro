import streamlit as st
import pandas as pd
import ccxt
from datetime import datetime
import pytz

# --- AYARLAR ---
# Coin listesini Binance formatına göre güncelledik
COINLER = ['BTC/USDT', 'ETH/USDT', 'NEAR/USDT', 'SOL/USDT', 'AVAX/USDT', 'LINK/USDT']

st.set_page_config(page_title="Sniper Bot Analiz", layout="wide")
st.title("🎯 Long/Short Skor Tablosu")

# --- ANALİZ FONKSİYONU ---
def analiz_yap():
    sonuclar = []
    # Borsaya daha sağlam bir bağlantı açıyoruz
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    with st.spinner('Binance verileri çekiliyor...'):
        for coin in COINLER:
            try:
                # Veriyi çek ve DataFrame'e yükle
                ohlcv = exchange.fetch_ohlcv(coin, timeframe='1h', limit=10)
                if not ohlcv:
                    continue
                    
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Fiyat ve Değişim Hesapla
                son_fiyat = df['close'].iloc[-1]
                onceki_fiyat = df['close'].iloc[-2]
                degisim = ((son_fiyat - onceki_fiyat) / onceki_fiyat) * 100
                
                # Skorlama Mantığı
                if degisim > 0:
                    skor = f"{int(70 + (degisim * 10))} (LONG) ✅"
                else:
                    skor = f"{int(30 + (degisim * 10))} (SHORT) ❌"
                
                sonuclar.append({
                    "Coin": coin, 
                    "Fiyat": son_fiyat, 
                    "Değişim %": round(degisim, 2),
                    "Skor/Yön": skor
                })
            except Exception as e:
                st.warning(f"{coin} verisi çekilemedi: {e}")
                continue
                
    return pd.DataFrame(sonuclar)

# --- ANA EKRAN ---
st.write(f"Sistem Saati: {datetime.now(pytz.timezone('Europe/Istanbul')).strftime('%H:%M:%S')}")

if st.button("🚀 ANALİZİ BAŞLAT"):
    df_sonuc = analiz_yap()
    
    if not df_sonuc.empty:
        st.table(df_sonuc)
        st.success("Analiz başarıyla tamamlandı!")
    else:
        st.error("Hiçbir veri çekilemedi. Lütfen internet bağlantısını veya coin isimlerini kontrol edin.")
