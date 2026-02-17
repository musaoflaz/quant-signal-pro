import streamlit as st
import pandas as pd
import ccxt
from datetime import datetime
import pytz

st.set_page_config(page_title="Kucoin Sniper Pro", layout="wide")
st.title("🎯 Kucoin Tüm Piyasalar Skor Analizi")

# --- ANALİZ MERKEZİ ---
def tum_piyasayi_analiz_et():
    exchange = ccxt.kucoin({'enableRateLimit': True})
    sonuclar = []
    
    try:
        # 1. Kucoin'deki tüm marketleri çek
        st.write("🔍 Tüm marketler listeleniyor...")
        markets = exchange.load_markets()
        # Sadece USDT çiftlerini ve aktif olanları filtrele
        usdt_pairs = [symbol for symbol, market in markets.items() if '/USDT' in symbol and market['active']]
        
        # İşlem yükünü azaltmak için hacimli olanlardan başla (İsteğe bağlı sınırlama: ilk 50 coin)
        tarama_listesi = usdt_pairs[:60] 
        
        progress_bar = st.progress(0)
        st.write(f"📊 {len(tarama_listesi)} coin analiz ediliyor, lütfen bekleyin...")

        for i, coin in enumerate(tarama_listesi):
            try:
                ohlcv = exchange.fetch_ohlcv(coin, timeframe='1h', limit=5)
                if not ohlcv: continue
                
                df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
                son_fiyat = df['c'].iloc[-1]
                onceki_fiyat = df['c'].iloc[-2]
                degisim = ((son_fiyat - onceki_fiyat) / onceki_fiyat) * 100
                
                # Skorlama Mantığı
                skor_degeri = int(50 + (degisim * 20))
                if skor_degeri > 95: skor_degeri = 95
                if skor_degeri < 5: skor_degeri = 5
                
                yon = "LONG ✅" if degisim > 0 else "SHORT ❌"
                
                sonuclar.append({
                    "Coin": coin,
                    "Fiyat": son_fiyat,
                    "Değişim %": round(degisim, 2),
                    "Skor": skor_degeri,
                    "Yön": yon
                })
            except:
                continue
            progress_bar.progress((i + 1) / len(tarama_listesi))
            
    except Exception as e:
        st.error(f"Piyasa verisi alınamadı: {e}")

    return pd.DataFrame(sonuclar)

# --- ANA EKRAN ---
if st.button("🚀 TÜM PİYASAYI TARA VE KIYASLA"):
    df_sonuc = tum_piyasayi_analiz_et()
    
    if not df_sonuc.empty:
        # Kıyaslama Paneli
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔥 En Güçlü Long Adayları")
            st.table(df_sonuc.sort_values(by="Skor", ascending=False).head(10))
            
        with col2:
            st.subheader("❄️ En Güçlü Short Adayları")
            st.table(df_sonuc.sort_values(by="Skor", ascending=True).head(10))
            
        st.subheader("📋 Tüm Liste")
        st.dataframe(df_sonuc) # Büyük liste için interaktif tablo
    else:
        st.error("Veri çekilemedi.")

st.sidebar.info("Kucoin üzerinden tüm USDT pariteleri taranmaktadır.")
