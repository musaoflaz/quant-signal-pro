import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

st.set_page_config(layout="wide", page_title="Alpha Sniper | Final Shield")

# KuCoin: IP engeli olmayan en stabil borsa
exchange = ccxt.kucoin({'enableRateLimit': True, 'timeout': 60000})

st.title("🛡️ QUANT ALPHA: FINAL SHIELD")
st.write("Geniş havuz taraması (25 Altcoin) ile yüksek skorlu sinyal avı.")

def final_scanner():
    results = []
    # Genişletilmiş liste (Yüksek skor ihtimalini artırır)
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'AVAX/USDT', 
        'DOGE/USDT', 'LINK/USDT', 'NEAR/USDT', 'ADA/USDT', 'DOT/USDT',
        'LTC/USDT', 'SHIB/USDT', 'TRX/USDT', 'UNI/USDT', 'PEPE/USDT',
        'TIA/USDT', 'SUI/USDT', 'OP/USDT', 'ARB/USDT', 'APT/USDT',
        'RNDR/USDT', 'INJ/USDT', 'FET/USDT', 'KAS/USDT', 'STX/USDT'
    ]
    
    progress = st.progress(0)
    
    for idx, symbol in enumerate(symbols):
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=200)
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # Göstergeler
            df['EMA200'] = ta.ema(df['c'], length=200)
            stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
            df = pd.concat([df, stoch], axis=1)
            rsi_val = ta.rsi(df['c'], length=14).iloc[-1]
            
            l, p = df.iloc[-1], df.iloc[-2]
            sk, sd = "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"
            
            skor = 0
            komut = "İZLE ⌛"
            
            # --- SKORLAMA (Terste Kalmama Filtresi) ---
            if l['c'] > l['EMA200']: # Trend Yukarı
                skor += 50
                if p[sk] < p[sd] and l[sk] > l[sd]: # Kesişim
                    skor += 40
                if rsi_val < 65: # Şişmemişlik
                    skor += 10
                
                if skor >= 90: komut = "🚀 GÜÇLÜ LONG"
                elif skor >= 50: komut = "📈 LONG PUSU"
            
            elif l['c'] < l['EMA200']: # Trend Aşağı
                skor += 50
                if p[sk] > p[sd] and l[sk] < l[sd]: # Kesişim
                    skor += 40
                if rsi_val > 35: # Dip Değil
                    skor += 10
                
                if skor >= 90: komut = "💥 GÜÇLÜ SHORT"
                elif skor >= 50: komut = "📉 SHORT PUSU"

            results.append({
                "COIN": symbol,
                "FİYAT": f"{l['c']:.4f}",
                "KOMUT": komut,
                "SKOR": skor,
                "RSI": int(rsi_val)
            })
            time.sleep(0.1)
        except: continue
        progress.progress((idx + 1) / len(symbols))
    
    return pd.DataFrame(results)

# --- Arayüz ---
if st.button('🛡️ ANALİZİ BAŞLAT (FINAL)'):
    data = final_scanner()
    
    if not data.empty:
        # Hatalı olan stil kısmını düzelttim:
        def style_rows(row):
            if row['SKOR'] >= 90:
                color = '#0c3e1e' if "LONG" in row['KOMUT'] else '#4b0a0a'
                return [f'background-color: {color}; color: white; font-weight: bold'] * len(row)
            return [''] * len(row)

        st.subheader("🎯 Canlı İşlem Sinyalleri")
        # TypeError vermemesi için apply(style_rows, axis=1) şeklinde düzelttik
        st.dataframe(data.sort_values('SKOR', ascending=False).style.apply(style_rows, axis=1), use_container_width=True)
    else:
        st.error("Borsa verisi alınamadı.")
