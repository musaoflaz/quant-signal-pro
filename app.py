import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time
import requests
from datetime import datetime

# --- KİMLİK BİLGİLERİ ---
TOKEN = "8330775219:AAHMGpdCdCEStj-B4Y3_WHD7xPEbjeaHWFM"
CHAT_ID = "1358384022"

def telegram_yolla(mesaj):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": mesaj, "parse_mode": "Markdown"}, timeout=10)
    except:
        pass

exchange = ccxt.kucoin({'enableRateLimit': True})

st.set_page_config(page_title="Alpha Sniper Dual", layout="wide")
st.title("🛡️ ALPHA SNIPER V42: LONG & SHORT")

if 'bot_active' not in st.session_state:
    st.session_state.bot_active = False

c1, c2 = st.columns(2)
with c1:
    if st.button("🚀 SİSTEMİ BAŞLAT"):
        st.session_state.bot_active = True
        telegram_yolla("✅ *Sniper Çift Yönlü Modda Başladı!* \n🟢 Long ve 🔴 Short takibi aktif.")

with c2:
    if st.button("🛑 SİSTEMİ DURDUR"):
        st.session_state.bot_active = False
        telegram_yolla("⚠️ *Sistem Kapatıldı!*")

symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'FET/USDT', 'SUI/USDT', 'NEAR/USDT', 'RNDR/USDT', 'TIA/USDT', 'ARB/USDT', 'OP/USDT', 'LINK/USDT', 'DOT/USDT']

placeholder = st.empty()

if st.session_state.bot_active:
    last_report_time = 0 
    
    while st.session_state.bot_active:
        current_time = time.time()
        results = []
        
        for s in symbols:
            try:
                bars = exchange.fetch_ohlcv(s, timeframe='1h', limit=200)
                df = pd.DataFrame(bars, columns=['t','o','h','l','c','v'])
                
                # --- TEKNİK ANALİZ ---
                df['EMA200'] = ta.ema(df['c'], length=200)
                df['RSI'] = ta.rsi(df['c'], length=14)
                stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
                df = pd.concat([df, stoch], axis=1)
                
                l, p = df.iloc[-1], df.iloc[-2]
                sk = [col for col in df.columns if 'STOCHRSIk' in col][0]
                sd = [col for col in df.columns if 'STOCHRSId' in col][0]
                
                skor = 0
                yon = "NÖTR"

                # 🟢 LONG ŞARTLARI (Fiyat EMA200 Üstünde)
                if l['c'] > l['EMA200']:
                    skor += 30
                    if p[sk] < p[sd] and l[sk] > l[sd] and l[sk] < 40: # Dip dönüşü
                        skor += 50
                    if l['v'] > df['v'].tail(15).mean(): # Hacim
                        skor += 20
                    yon = "🟢 LONG"

                # 🔴 SHORT ŞARTLARI (Fiyat EMA200 Altında)
                elif l['c'] < l['EMA200']:
                    skor += 30
                    if p[sk] > p[sd] and l[sk] < l[sd] and l[sk] > 60: # Tepe dönüşü
                        skor += 50
                    if l['v'] > df['v'].tail(15).mean(): # Hacim
                        skor += 20
                    yon = "🔴 SHORT"

                results.append({"COIN": s.replace('/USDT',''), "YÖN": yon, "SKOR": skor, "FİYAT": l['c']})
                
                if skor >= 100:
                    telegram_yolla(f"🚨 *SİNYAL ÇAKTI!* 🚨\n\n*YÖN:* {yon}\n*Coin:* {s}\n*Fiyat:* {l['c']}\n*Skor:* 100/100")
            except:
                continue
        
        final_df = pd.DataFrame(results).sort_values(by="SKOR", ascending=False)
        
        with placeholder.container():
            st.write(f"⏱️ Son Tarama: {datetime.now().strftime('%H:%M:%S')}")
            st.table(final_df)
        
        # 20 Dakikalık Rapor
        if (current_time - last_report_time) >= 1200:
            rapor = "📋 *Gözcü Raporu (L/S)*\n"
            for _, row in final_df.head(10).iterrows():
                emoji = "🟩" if "LONG" in row['YÖN'] else "🟥"
                rapor += f"{emoji} *{row['COIN']}*: {row['SKOR']} Puan | {row['YÖN']}\n"
            telegram_yolla(rapor)
            last_report_time = current_time
        
        time.sleep(60)
