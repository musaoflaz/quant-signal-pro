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

st.set_page_config(page_title="Alpha Sniper 7/24 Guardian", layout="wide")
st.title("🛡️ ALPHA SNIPER V42: GUARDIAN")

if 'bot_active' not in st.session_state:
    st.session_state.bot_active = False

c1, c2 = st.columns(2)
with c1:
    if st.button("🚀 SİSTEMİ 7/24 BAŞLAT"):
        st.session_state.bot_active = True
        telegram_yolla("✅ *Gözcü Sistemi Başlatıldı!*\n- 100 Puan: *Anlık Bildirim*\n- Piyasa Özeti: *20 Dakikada Bir*")

with c2:
    if st.button("🛑 SİSTEMİ DURDUR"):
        st.session_state.bot_active = False
        telegram_yolla("⚠️ *Sistem Durduruldu!*")

# Takip Edilecek Geniş Liste
symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'FET/USDT', 'SUI/USDT', 'NEAR/USDT', 'RNDR/USDT', 'TIA/USDT', 'ARB/USDT', 'OP/USDT', 'LINK/USDT', 'DOT/USDT']

placeholder = st.empty()

if st.session_state.bot_active:
    last_report_time = 0 # İlk raporu hemen atması için
    
    while st.session_state.bot_active:
        current_time = time.time()
        results = []
        
        for s in symbols:
            try:
                bars = exchange.fetch_ohlcv(s, timeframe='1h', limit=200)
                df = pd.DataFrame(bars, columns=['t','o','h','l','c','v'])
                
                # --- ULTRA SERT ANALİZ ---
                df['EMA200'] = ta.ema(df['c'], length=200)
                df['RSI'] = ta.rsi(df['c'], length=14)
                stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
                df = pd.concat([df, stoch], axis=1)
                
                l, p = df.iloc[-1], df.iloc[-2]
                sk = [col for col in df.columns if 'STOCHRSIk' in col][0]
                sd = [col for col in df.columns if 'STOCHRSId' in col][0]
                
                skor = 0
                if l['c'] > l['EMA200'] and l['EMA200'] > df['EMA200'].iloc[-5]: skor += 30
                if p[sk] < p[sd] and l[sk] > l[sd] and l[sk] < 25: skor += 40
                if l['v'] > (df['v'].tail(15).mean() * 1.5): skor += 20
                if 45 <= l['RSI'] <= 60: skor += 10
                
                results.append({"COIN": s.replace('/USDT',''), "SKOR": skor, "FİYAT": l['c']})
                
                # --- ACİL DURUM: 100 PUAN ---
                if skor >= 100:
                    telegram_yolla(f"🚨 *ACİL SİNYAL: 100 PUAN!* 🚨\n\n*Coin:* {s}\n*Fiyat:* {l['c']}\n*Durum:* Kusursuz Giriş Şartları Oluştu!")
            except:
                continue
        
        final_df = pd.DataFrame(results).sort_values(by="SKOR", ascending=False)
        
        # Streamlit Ekranını Güncelle
        with placeholder.container():
            st.write(f"⏱️ Son Tarama: {datetime.now().strftime('%H:%M:%S')}")
            st.table(final_df)
        
        # --- 20 DAKİKADA BİR TABLO RAPORU ---
        # 20 dakika = 1200 saniye
        if (current_time - last_report_time) >= 1200:
            rapor = "📋 *20 Dakikalık Piyasa Özeti*\n"
            rapor += "--------------------------\n"
            for _, row in final_df.head(10).iterrows():
                # Skorlara göre emoji ekleyelim
                emoji = "⚪"
                if row['SKOR'] >= 70: emoji = "🔥"
                elif row['SKOR'] >= 30: emoji = "👀"
                
                rapor += f"{emoji} *{row['COIN']}*: {row['SKOR']} Puan | {row['FİYAT']}\n"
            
            telegram_yolla(rapor)
            last_report_time = current_time
        
        # Tarama aralığı (Dakikada bir piyasayı kontrol eder ama raporu 20 dk'da bir atar)
        time.sleep(60)
