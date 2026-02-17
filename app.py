import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta  # Teknik analiz kütüphanesi
from datetime import datetime
import pytz

st.set_page_config(page_title="Sniper Pro v2", layout="wide")
st.title("🎯 Kucoin Optimum Sinyal Yakalayıcı")

def analiz_motoru():
    exchange = ccxt.kucoin({'enableRateLimit': True})
    sonuclar = []
    
    st.write("🔍 Piyasa taranıyor ve indikatörler hesaplanıyor...")
    markets = exchange.load_markets()
    symbols = [s for s in markets.keys() if '/USDT' in s and markets[s]['active']][:60] # İlk 60 hacimli coin
    
    progress_bar = st.progress(0)
    
    for i, symbol in enumerate(symbols):
        try:
            # Analiz için gerekli olan son 100 mumu çek
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
            df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # --- TEKNİK ANALİZ (Optimum İndikatörler) ---
            # 1. RSI (14)
            df['rsi'] = ta.rsi(df['c'], length=14)
            
            # 2. Bollinger Bantları
            bb = ta.bbands(df['c'], length=20, std=2)
            df = pd.concat([df, bb], axis=1)
            
            # 3. SMA (20) - Trend Yönü
            df['sma20'] = ta.sma(df['c'], length=20)
            
            # Son değerleri al
            son_fiyat = df['c'].iloc[-1]
            rsi_son = df['rsi'].iloc[-1]
            sma_son = df['sma20'].iloc[-1]
            alt_bant = df['BBL_20_2.0'].iloc[-1]
            ust_bant = df['BBU_20_2.0'].iloc[-1]
            
            # --- SKORLAMA MANTIĞI (Terste Kalmamak İçin) ---
            skor = 50 # Nötr başla
            
            # Trend Kontrolü
            if son_fiyat > sma_son: skor += 15 # Fiyat SMA üzerindeyse trend yukarı
            else: skor -= 15
            
            # RSI Kontrolü
            if 40 < rsi_son < 60: skor += 10 # RSI sağlıklı bölgedeyse
            elif rsi_son > 70: skor -= 20 # Aşırı şişmiş, girmek riskli!
            elif rsi_son < 30: skor += 20 # Aşırı düşmüş, tepki gelebilir.
            
            # Bollinger Kontrolü
            if son_fiyat <= alt_bant: skor += 20 # Alt banta dokunmuş (Alım fırsatı)
            if son_fiyat >= ust_bant: skor -= 20 # Üst banta dokunmuş (Direnç)

            # --- SİNYAL KARARI ---
            durum = "İZLEMEDE"
            if skor >= 90: durum = "🔥 GERÇEK SİNYAL (STRONG LONG)"
            elif skor <= 20: durum = "💀 GERÇEK SİNYAL (STRONG SHORT)"
            
            sonuclar.append({
                "Coin": symbol,
                "Fiyat": round(son_fiyat, 4),
                "RSI": round(rsi_son, 2),
                "Skor": skor,
                "Sinyal": durum
            })
            
        except:
            continue
        progress_bar.progress((i + 1) / len(symbols))
        
    return pd.DataFrame(sonuclar)

# --- ARAYÜZ ---
if st.button("🚀 OPTİMUM ANALİZİ BAŞLAT"):
    data = analiz_motoru()
    
    if not data.empty:
        # Sadece Gerçek Sinyalleri Öne Çıkar
        gercek_sinyaller = data[data['Skor'] >= 90]
        
        if not gercek_sinyaller.empty:
            st.subheader("✅ BULUNAN FIRSATLAR")
            st.success(f"{len(gercek_sinyaller)} tane 90+ skorlu coin bulundu!")
            st.table(gercek_sinyaller.sort_values(by="Skor", ascending=False))
        else:
            st.info("Şu an 90 skoruna ulaşan kusursuz bir fırsat yok. Beklemede kal.")

        st.subheader("📋 Genel Piyasa Durumu")
        st.dataframe(data)
    else:
        st.error("Veri çekilemedi.")
