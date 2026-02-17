import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time
import requests

# --- AYARLAR ---
TOKEN = "8330775219:AAHMGpdCdCEStj-B4Y3_WHD7xPEbjeaHWFM"
CHAT_ID = "1358384022"

def telegram_yolla(mesaj):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": mesaj}, timeout=10)
    except:
        pass

exchange = ccxt.kucoin({'enableRateLimit': True})

st.set_page_config(page_title="Alpha Sniper PRO", layout="wide")
st.title("🛡️ ALPHA SNIPER PRO: ÇOKLU TARAMA")

# GENİŞLETİLMİŞ COİN LİSTESİ
symbols = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'LTC/USDT', 'AVAX/USDT', 'FET/USDT', 'SUI/USDT', 'NEAR/USDT',
    'RNDR/USDT', 'ARB/USDT', 'OP/USDT', 'LINK/USDT', 'DOT/USDT', 'MATIC/USDT', 'TIA/USDT', 'APT/USDT',
    'STX/USDT', 'INJ/USDT', 'FIL/USDT', 'ATOM/USDT', 'ALGO/USDT', 'XRP/USDT', 'ADA/USDT', 'DOGE/USDT'
]

if 'status' not in st.session_state:
    st.session_state.status = False

c1, c2 = st.columns(2)
with c1:
    if st.button("🟢 TARAMAYI BAŞLAT"):
        st.session_state.status = True
        telegram_yolla("🚀 Pro Tarayıcı Aktif! Geniş liste taranıyor...")
with c2:
    if st.button("🔴 DURDUR"):
        st.session_state.status = False

# Tablo için boş alan
table_placeholder = st.empty()

if st.session_state.status:
    while st.session_state.status:
        results = []
        for s in symbols:
            try:
                bars = exchange.fetch_ohlcv(s, timeframe='1h', limit=100)
                df = pd.DataFrame(bars, columns=['t','o','h','l','c','v'])
                
                # Göstergeler
                df['EMA200'] = ta.ema(df['c'], length=200) or df['c'].rolling(50).mean()
                df['RSI'] = ta.rsi(df['c'], length=14)
                stoch = ta.stochrsi(df['c'])
                df = pd.concat([df, stoch], axis=1)
                
                l, p = df.iloc[-1], df.iloc[-2]
                sk = [c for c in df.columns if 'STOCHRSIk' in c][0]
                sd = [c for c in df.columns if 'STOCHRSId' in c][0]
                
                # Puanlama
                puan = 0
                if l['c'] > l['EMA200']: puan += 40
                if p[sk] < p[sd] and l[sk] > l[sd]: puan += 40
                if 40 <= l['RSI'] <= 70: puan += 20
                
                results.append({
                    "Coin": s,
                    "Fiyat": round(l['c'], 4),
                    "RSI": round(l['RSI'], 2),
                    "Puan": puan,
                    "Trend": "📈 YUKARI" if l['c'] > l['EMA200'] else "📉 AŞAĞI"
                })
                
                if puan >= 100:
                    telegram_yolla(f"🎯 100 PUAN SİNYALİ!\nCoin: {s}\nFiyat: {round(l['c'], 4)}\nTrend: EMA Üstü\nStoch: Kesişim Var!")
                
            except:
                continue
        
        # Tabloyu Güncelle
        df_res = pd.DataFrame(results)
        with table_placeholder.container():
            st.write(f"⏱️ Son Güncelleme: {time.strftime('%H:%M:%S')}")
            st.table(df_res.sort_values(by="Puan", ascending=False))
        
        time.sleep(60) # Geniş tarama olduğu için 1 dakikada bir yeniler
