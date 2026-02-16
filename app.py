import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta

# 1. Sayfa Ayarları
st.set_page_config(layout="wide", page_title="Quant Signal Pro | Terminal")

# 2. Borsa Bağlantısı
exchange = ccxt.binance({'enableRateLimit': True})

# 3. Başlık
st.markdown("# 🏛️ TRADE TERMINAL (Hız Modu)")
st.write("---")

# Takip edilecek varlıklar
symbols = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT',
    'PEPE/USDT', 'BNB/USDT', 'SUI/USDT', 'AVAX/USDT', 'LINK/USDT'
]

def veri_topla():
    rows = []
    for symbol in symbols:
        try:
            # 4 Saatlik veriler
            bars = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=50)
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # RSI Hesaplama
            rsi = ta.rsi(df['c'], length=14).iloc[-1]
            fiyat = df['c'].iloc[-1]
            
            # Sinyal Mantığı (Sadece Metin)
            if rsi < 35:
                eylem = "AL (LONG)"
                rejim = "YUKARI TREND"
            elif rsi > 65:
                eylem = "SAT (SHORT)"
                rejim = "AŞAĞI TREND"
            else:
                eylem = "BEKLE"
                rejim = "YATAY PİYASA"

            rows.append({
                "VARLIK": symbol,
                "FİYAT": f"{fiyat:.4f}",
                "PİYASA REJİMİ": rejim,
                "İŞLEM EYLEMİ": eylem,
                "GÜVEN %": f"%{int(abs(50-rsi)*2)}",
                "RSI": int(rsi)
            })
        except:
            continue
    return pd.DataFrame(rows)

# Veriyi çek ve göster
data = veri_topla()

if not data.empty:
    # Boyama/Style olmadan doğrudan tabloyu basıyoruz
    st.dataframe(data, use_container_width=True, height=600)
else:
    st.error("Veri alınamadı, Binance bağlantısı kontrol ediliyor...")

# Manuel Yenileme
if st.sidebar.button('Sinyalleri Güncelle'):
    st.rerun()
