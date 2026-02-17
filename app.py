import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time
import requests

# --- KİMLİK BİLGİLERİ ---
TOKEN = "8330775219:AAHMGpdCdCEStj-B4Y3_WHD7xPEbjeaHWFM"
CHAT_ID = "1358384022"

def telegram_yolla(mesaj):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try: requests.post(url, json={"chat_id": CHAT_ID, "text": mesaj}, timeout=10)
    except: pass

exchange = ccxt.kucoin({'enableRateLimit': True})

st.set_page_config(page_title="Alpha Sniper Ultra-Elite", layout="wide")
st.title("🛡️ ALPHA SNIPER V42: ULTRA-ELITE SCANNER")
st.sidebar.success("MOD: Haftalık 3-5 Garantici İşlem")
st.sidebar.info("Bu modda kriterler çok ağırdır. Tablo genelde 0-30 puan arası kalacaktır.")

@st.cache_data
def get_symbols():
    try:
        m = exchange.load_markets()
        return [s for s in m if '/USDT' in s and m[s]['active']][:60]
    except:
        return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'FET/USDT', 'SUI/USDT', 'NEAR/USDT']

symbols = get_symbols()

if 'run' not in st.session_state: st.session_state.run = False

c1, c2 = st.columns([1, 4])
with c1:
    if st.button("🎯 ULTRA TARAMAYI BAŞLAT"):
        st.session_state.run = True
        telegram_yolla("💎 Ultra-Elite Sniper Pusuya Yattı. Sadece 'Kusursuz' sinyaller bekleniyor.")
with c2:
    if st.button("🛑 SİSTEMİ KAPAT"): st.session_state.run = False

placeholder = st.empty()

if st.session_state.run:
    while st.session_state.run:
        data_list = []
        for s in symbols:
            try:
                bars = exchange.fetch_ohlcv(s, timeframe='1h', limit=200)
                df = pd.DataFrame(bars, columns=['t','o','h','l','c','v'])
                
                # --- AĞIRLAŞTIRILMIŞ ANALİZ ---
                df['EMA200'] = ta.ema(df['c'], length=200)
                df['RSI'] = ta.rsi(df['c'], length=14)
                stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
                df = pd.concat([df, stoch], axis=1)
                
                l, p = df.iloc[-1], df.iloc[-2]
                sk = [col for col in df.columns if 'STOCHRSIk' in col][0]
                sd = [col for col in df.columns if 'STOCHRSId' in col][0]
                
                skor = 0
                notlar = []

                # 1. Trend Sert Filtre (30 Puan)
                if l['c'] > l['EMA200'] and l['EMA200'] > df['EMA200'].iloc[-5]:
                    skor += 30
                    notlar.append("Trend+")

                # 2. Dip Kesişimi Sert Filtre (40 Puan) - Mutlaka 25'in altında kesişmeli
                if p[sk] < p[sd] and l[sk] > l[sd] and l[sk] < 25:
                    skor += 40
                    notlar.append("Dip-OK")

                # 3. Hacim Onayı (20 Puan) - Hacim ortalamanın %50 üzerinde olmalı
                vol_avg = df['v'].tail(15).mean()
                if l['v'] > (vol_avg * 1.5):
                    skor += 20
                    notlar.append("Hacim-OK")

                # 4. RSI Stratejik Bölge (10 Puan)
                if 45 <= l['RSI'] <= 65:
                    skor += 10
                    notlar.append("Güç-OK")

                durum = "🔍 Beklemede"
                if skor >= 100: durum = "💎 ELMAS SİNYAL"
                elif skor >= 70: durum = "🔥 RADARDA"

                data_list.append({
                    "COIN": s, "SKOR": skor, "ANALİZ": durum,
                    "FİYAT": f"{l['c']:.4f}", "RSI": int(l['RSI']), "ONAYLAR": " | ".join(notlar)
                })

                if skor >= 100:
                    telegram_yolla(f"💎 **ELMAS SİNYAL YAKALANDI!**\n\nCoin: {s}\nFiyat: {l['c']}\nSkor: 100/100\nOnaylar: {', '.join(notlar)}\n\n⚠️ Bu sinyal nadir gelir, grafiği kontrol et!")
                
            except: continue
        
        final_df = pd.DataFrame(data_list).sort_values(by="SKOR", ascending=False)
        with placeholder.container():
            st.write(f"⏱️ Son Tarama: {time.strftime('%H:%M:%S')}")
            # Tabloyu görselleştir
            st.dataframe(final_df, use_container_width=True)
        
        time.sleep(60)
