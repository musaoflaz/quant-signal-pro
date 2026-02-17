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

# Borsa Bağlantısı
exchange = ccxt.kucoin({'enableRateLimit': True})

st.set_page_config(page_title="Alpha Sniper Pro V42", layout="wide")
st.title("🛡️ ALPHA SNIPER PRO (V42)")
st.info("Log hataları giderildi. Canlı takip tablosu ve ultra keskin sinyal filtresi aktif.")

# --- COİN LİSTESİNİ OTOMATİK AL ---
@st.cache_data
def get_all_symbols():
    try:
        markets = exchange.load_markets()
        # Sadece USDT çiftlerini ve aktif olanları al (İlk 50 hacimli gibi filtreleyebiliriz)
        all_symbols = [symbol for symbol in markets if '/USDT' in symbol and markets[symbol]['active']]
        return all_symbols[:40] # Performans için en popüler 40 tanesini tarar
    except:
        return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'FET/USDT', 'SUI/USDT', 'NEAR/USDT', 'TIA/USDT', 'PEPE/USDT', 'DOGE/USDT']

symbols = get_all_symbols()

if 'run' not in st.session_state:
    st.session_state.run = False

# Analiz Başlat/Durdur Butonları
c1, c2 = st.columns([1, 4])
with c1:
    if st.button("🎯 ANALİZİ BAŞLAT"):
        st.session_state.run = True
        telegram_yolla("🚀 Pro Tarayıcı Yayında! Tablo güncelleniyor...")
with c2:
    if st.button("🛑 DURDUR"):
        st.session_state.run = False

# Tablo Alanı
placeholder = st.empty()

if st.session_state.run:
    while st.session_state.run:
        data_list = []
        for s in symbols:
            try:
                bars = exchange.fetch_ohlcv(s, timeframe='1h', limit=100)
                df = pd.DataFrame(bars, columns=['t','o','h','l','c','v'])
                
                # İndikatörler
                df['EMA200'] = ta.ema(df['c'], length=200) or df['c'].rolling(100).mean()
                df['RSI'] = ta.rsi(df['c'], length=14)
                stoch = ta.stochrsi(df['c'])
                df = pd.concat([df, stoch], axis=1)
                
                l, p = df.iloc[-1], df.iloc[-2]
                sk = [col for col in df.columns if 'STOCHRSIk' in col][0]
                sd = [col for col in df.columns if 'STOCHRSId' in col][0]
                
                # Skor Hesaplama
                skor = 0
                komut = "🔍 TAKİP ET"
                
                if l['c'] > l['EMA200']: skor += 40
                if p[sk] < p[sd] and l[sk] > l[sd]: skor += 40
                if 30 <= l['RSI'] <= 60: skor += 20
                
                if skor >= 90: komut = "🚀 KESİN LONG"
                elif skor >= 70: komut = "⚡ SİNYAL YAKIN"
                
                data_list.append({
                    "COIN": s,
                    "SKOR": skor,
                    "KOMUT": komut,
                    "RSI": int(l['RSI']),
                    "FİYAT": l['c']
                })
                
                # 100 Puan Sinyali Telegram'a
                if skor >= 100:
                    telegram_yolla(f"🎯 100 PUAN! {s}\nFiyat: {l['c']}\nKomut: {komut}")
                
            except:
                continue
        
        # Tabloyu Ekrana Bas
        final_df = pd.DataFrame(data_list).sort_values(by="SKOR", ascending=False)
        
        with placeholder.container():
            # Renklendirme Fonksiyonu
            def color_skor(val):
                color = 'white'
                if val >= 90: color = '#FFD700' # Altın sarısı
                elif val >= 70: color = '#90EE90' # Yeşilimsi
                return f'background-color: {color}; color: black; font-weight: bold'

            st.write(f"🔄 Son Tarama: {time.strftime('%H:%M:%S')}")
            st.table(final_df.style.applymap(color_skor, subset=['SKOR']))
        
        time.sleep(60) # Her dakika tabloyu tazeler
