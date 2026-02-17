import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

st.set_page_config(layout="wide", page_title="Alpha Sniper | High Confidence")

exchange = ccxt.kucoin({'enableRateLimit': True, 'timeout': 60000})

st.title("🏛️ QUANT ALPHA: YÜKSEK GÜVENLİ SİNYAL")
st.write("Sistem sadece 'Yüksek Skor' (80-100) onayı alan coinleri ön plana çıkarır.")

def high_confidence_scanner():
    results = []
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'AVAX/USDT', 
               'DOGE/USDT', 'LINK/USDT', 'NEAR/USDT', 'TIA/USDT', 'SUI/USDT']
    
    progress = st.progress(0)
    
    for idx, symbol in enumerate(symbols):
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=150)
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # Analiz
            df['EMA200'] = ta.ema(df['c'], length=200)
            stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
            df = pd.concat([df, stoch], axis=1)
            rsi_val = ta.rsi(df['c'], length=14).iloc[-1]
            
            l, p = df.iloc[-1], df.iloc[-2]
            sk, sd = "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"
            
            skor = 0
            komut = "BEKLE"
            
            # --- LONG SKORLAMA ---
            if l['c'] > l['EMA200']: # Trend Pozitif
                skor += 50
                if p[sk] < p[sd] and l[sk] > l[sd]: # Kesişim Onayı
                    skor += 40
                if rsi_val < 65: # Şişmemişlik Bonusu
                    skor += 10
                
                if skor >= 90: komut = "🚀 GÜÇLÜ LONG"
                elif skor >= 50: komut = "📈 LONG PUSU"

            # --- SHORT SKORLAMA ---
            elif l['c'] < l['EMA200']: # Trend Negatif
                skor += 50
                if p[sk] > p[sd] and l[sk] < l[sd]: # Kesişim Onayı
                    skor += 40
                if rsi_val > 35: # Aşırı Satım Değilse Bonusu
                    skor += 10
                
                if skor >= 90: komut = "💥 GÜÇLÜ SHORT"
                elif skor >= 50: komut = "📉 SHORT PUSU"

            results.append({
                "COIN": symbol,
                "FİYAT": f"{l['c']:.4f}",
                "GÜVEN SKORU": skor,
                "KOMUT": komut,
                "RSI": int(rsi_val)
            })
            time.sleep(0.2)
        except: continue
        progress.progress((idx + 1) / len(symbols))
    
    return pd.DataFrame(results)

if st.button('🎯 YÜKSEK GÜVENLİ TARAMAYI BAŞLAT'):
    data = high_confidence_scanner()
    if not data.empty:
        # Görsel Filtreleme
        def highlight_high(row):
            if row['GÜVEN SKORU'] >= 90:
                return ['background-color: #11381b; color: #52ff8f'] * len(row) if "LONG" in row['KOMUT'] else ['background-color: #3b0d0d; color: #ff6e6e'] * len(row)
            return [''] * len(row)

        st.dataframe(data.sort_values('GÜVEN SKORU', ascending=False).style.apply(highlight_high, axis=1), use_container_width=True)
    else:
        st.error("Sinyal üretilemedi.")
