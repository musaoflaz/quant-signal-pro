import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

st.set_page_config(layout="wide", page_title="Quant Alpha | V21 Final")

# Bybit: Bulut sunucuları için en güvenli liman
exchange = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'linear'}})

st.markdown("# 🏛️ QUANT ALPHA: FINAL WATCH")

def get_symbols():
    try:
        tickers = exchange.fetch_tickers()
        df = pd.DataFrame.from_dict(tickers, orient='index')
        return df[df['symbol'].str.contains('USDT')].sort_values('quoteVolume', ascending=False).head(25).index.tolist()
    except:
        return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']

def ultra_scanner():
    symbols = get_symbols()
    results = []
    progress = st.progress(0)
    
    for idx, symbol in enumerate(symbols):
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=240)
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # Göstergeler
            df['EMA200'] = ta.ema(df['c'], length=200)
            stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
            df = pd.concat([df, stoch], axis=1)
            
            l, p = df.iloc[-1], df.iloc[-2]
            sk, sd = "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"
            
            # --- PUANLAMA ---
            skor = 0
            trend = "YUKARI" if l['c'] > l['EMA200'] else "AŞAĞI"
            
            # 1. Trend Onayı (50 Puan)
            skor += 50
            
            # 2. Kesişim Onayı (+40 Puan)
            cross_up = p[sk] < p[sd] and l[sk] > l[sd]
            cross_down = p[sk] > p[sd] and l[sk] < l[sd]
            
            if trend == "YUKARI" and cross_up: skor += 40
            elif trend == "AŞAĞI" and cross_down: skor += 40

            # Durum Belirleme
            if skor >= 90:
                eylem = "🚀 LONG GİR" if trend == "YUKARI" else "💥 SHORT GİR"
            else:
                eylem = "📉 PUSUDA BEKLE" if trend == "AŞAĞI" else "📈 PUSUDA BEKLE"

            results.append({
                "COIN": symbol,
                "FİYAT": f"{l['c']:.4f}",
                "DURUM": eylem,
                "SKOR": skor,
                "TREND": trend
            })
            time.sleep(0.1)
        except: continue
        progress.progress((idx + 1) / len(symbols))
    
    return pd.DataFrame(results).sort_values('SKOR', ascending=False)

if st.button('🎯 SİNYAL AVINI BAŞLAT'):
    data = ultra_scanner()
    if not data.empty:
        st.dataframe(data.style.apply(lambda x: ['background-color: #0c3e1e' if 'GİR' in str(v) else '' for v in x], axis=1), use_container_width=True)
    else:
        st.error("Borsa bağlantısı başarısız.")
