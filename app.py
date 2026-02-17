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

st.set_page_config(page_title="Alpha Sniper Balanced", layout="wide")
st.title("🛡️ ALPHA SNIPER V42: BALANCED ELITE")

if 'bot_active' not in st.session_state:
    st.session_state.bot_active = False

c1, c2 = st.columns(2)
with c1:
    if st.button("🚀 SİSTEMİ BAŞLAT"):
        st.session_state.bot_active = True
        telegram_yolla("✅ *Sistem Dengeli Modda Başlatıldı!* \n- 100 Puan: *Anlık Bildirim*\n- Piyasa Özeti: *20 Dakikada Bir*")

with c2:
    if st.button("🛑 SİSTEMİ DURDUR"):
        st.session_state.bot_active = False
        telegram_yolla("⚠️ *Sistem Kapatıldı!*")

# Takip Listesi
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
                
                # --- DENGELİ TEKNİK ANALİZ ---
                df['EMA200'] = ta.ema(df['c'], length=200)
                df['RSI'] = ta.rsi(df['c'], length=14)
                stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
                df = pd.concat([df, stoch], axis=1)
                
                l, p = df.iloc[-1], df.iloc[-2]
                sk = [col for col in df.columns if 'STOCHRSIk' in col][0]
                sd = [col for col in df.columns if 'STOCHRSId' in col][0]
                
                skor = 0
                # 1. Trend (30 Puan): Fiyat EMA200 üzerinde mi?
                if l['c'] > l['EMA200']: 
                    skor += 30
                
                # 2. Stoch RSI (40 Puan): 40 seviyesinin altında yukarı kesişim?
                if p[sk] < p[sd] and l[sk] > l[sd] and l[sk] < 40: 
                    skor += 40
                
                # 3. Hacim (20 Puan): Son mum hacmi ortalamanın üzerinde mi?
                if l['v'] > df['v'].tail(15).mean(): 
                    skor += 20
                
                # 4. RSI (10 Puan): RSI sağlıklı bölgede mi?
                if 35 <= l['RSI'] <= 65: 
                    skor += 10
                
                results.append({"COIN": s.replace('/USDT',''), "SKOR": skor, "FİYAT": l['c'], "RSI": int(l['RSI'])})
                
                if skor >= 100:
                    telegram_yolla(f"🎯 *SİNYAL: 100 PUAN!* 🎯\n\n*Coin:* {s}\n*Fiyat:* {l['c']}\n*Skor:* 100/100\n_Filtrelerden başarıyla geçti!_")
            except:
                continue
        
        final_df = pd.DataFrame(results).sort_values(by="SKOR", ascending=False)
        
        with placeholder.container():
            st.write(f"⏱️ Son Tarama: {datetime.now().strftime('%H:%M:%S')}")
            st.table(final_df)
        
        # 20 Dakikalık Rapor
        if (current_time - last_report_time) >= 1200:
            rapor = "📋 *Piyasa Özeti*\n"
            for _, row in final_df.head(10).iterrows():
                emoji = "🟢" if row['SKOR'] >= 70 else "🟡" if row['SKOR'] >= 30 else "⚪"
                rapor += f"{emoji} *{row['COIN']}*: {row['SKOR']} Puan | F: {row['FİYAT']}\n"
            telegram_yolla(rapor)
            last_report_time = current_time
        
        time.sleep(60)
