import streamlit as st
import pandas as pd
import ccxt
from datetime import datetime
import pytz

# --- AYARLAR ---
# Kucoin formatında coin listesi
COINLER = ['BTC/USDT', 'ETH/USDT', 'NEAR/USDT', 'SOL/USDT', 'AVAX/USDT', 'LINK/USDT']

st.set_page_config(page_title="Sniper Bot Kucoin", layout="wide")
st.title("🎯 Kucoin Long/Short Skor Tablosu")

# --- ANALİZ FONKSİYONU (Kucoin Özel) ---
def analiz_yap():
    sonuclar = []
    # Borsayı KUCOIN olarak ayarlıyoruz
    exchange = ccxt.kucoin({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    with st.spinner('Kucoin verileri çekiliyor...'):
        for coin in COINLER:
            try:
                # Kucoin'den 1 saatlik verileri çek
                ohlcv = exchange.fetch_ohlcv(coin, timeframe='1h', limit=10)
                if not ohlcv:
                    continue
                    
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Fiyat ve Skorlama Hesapları
                son_fiyat = df['close'].iloc[-1]
                onceki_fiyat = df['close'].iloc[-2]
                degisim = ((son_fiyat - onceki_fiyat) / onceki_fiyat) * 100
                
                # Senin o başarılı skorlama mantığın
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
                st.warning(f"{coin} çekilemedi (Kucoin): {e}")
                continue
                
    return pd.DataFrame(sonuclar)

# --- ANA EKRAN ---
st.sidebar.info("Borsa: Kucoin")
st.write(f"Sistem Saati: {datetime.now(pytz.timezone('Europe/Istanbul')).strftime('%H:%M:%S')}")

if st.button("🚀 ANALİZİ BAŞLAT"):
    df_sonuc = analiz_yap()
    
    if not df_sonuc.empty:
        # Senin sevdiğin o temiz tablo
        st.table(df_sonuc)
        st.success("Kucoin skorları başarıyla güncellendi!")
    else:
        st.error("Veri çekme hatası! Lütfen Kucoin bağlantısını kontrol edin.")
