import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

st.set_page_config(layout="wide", page_title="Alpha Sniper Nexus")

# Borsa Havuzu (Hata vermemesi için sırayla deneyecek)
exchanges = {
    'bybit': ccxt.bybit({'enableRateLimit': True}),
    'okx': ccxt.okx({'enableRateLimit': True}),
    'kucoin': ccxt.kucoin({'enableRateLimit': True})
}

st.title("🌐 NEXUS MULTI-SCANNER (V32)")
st.info("Bybit, OKX ve KuCoin üzerinden Binance coinleri taranıyor...")

def get_data(symbol):
    # Borsaları sırayla dene
    for name, ex in exchanges.items():
        try:
            bars = ex.fetch_ohlcv(symbol, timeframe='1h', limit=150)
            if bars:
                return pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v']), name
        except:
            continue
    return None, None

def nexus_scanner():
    results = []
    # Binance'te olan ama diğerlerinde de bulunan geniş liste
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'LTC/USDT', 'AVAX/USDT', 
        'LINK/USDT', 'FET/USDT', 'TIA/USDT', 'RNDR/USDT', 'NEAR/USDT',
        'ARB/USDT', 'OP/USDT', 'INJ/USDT', 'SUI/USDT', 'PEPE/USDT',
        'ORDI/USDT', 'SEI/USDT', 'STX/USDT', 'BEAM/USDT', 'KAS/USDT'
    ]
    
    progress = st.progress(0)
    for idx, symbol in enumerate(symbols):
        df, active_ex = get_data(symbol)
        
        if df is not None:
            try:
                # Başarılı V29 Stratejisi
                df['EMA200'] = ta.ema(df['c'], length=200)
                stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
                df = pd.concat([df, stoch], axis=1)
                rsi_val = ta.rsi(df['c'], length=14).iloc[-1]
                
                l, p = df.iloc[-1], df.iloc[-2]
                sk, sd = "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"
                
                skor = 0
                if l['c'] > l['EMA200']: # TREND YUKARI (LONG)
                    skor += 40
                    if p[sk] < p[sd] and l[sk] > l[sd]: skor += 50
                    if rsi_val < 65: skor += 10
                    komut = "🚀 LONG" if skor >= 90 else "📈 PUSU"
                else: # TREND AŞAĞI (SHORT)
                    skor += 40
                    if p[sk] > p[sd] and l[sk] < l[sd]: skor += 50
                    if rsi_val > 35: skor += 10
                    komut = "💥 SHORT" if skor >= 90 else "📉 PUSU"

                results.append({
                    "COIN": symbol,
                    "SKOR": skor,
                    "DURUM": komut,
                    "RSI": int(rsi_val),
                    "BORSA": active_ex.upper()
                })
            except: pass
        progress.progress((idx + 1) / len(symbols))
    return pd.DataFrame(results)

if st.button('🎯 DEV HAVUZU TARA'):
    data = nexus_scanner()
    if not data.empty:
        # Renklendirme ve Görselleştirme
        st.dataframe(data.sort_values('SKOR', ascending=False), use_container_width=True)
    else:
        st.error("Veri çekilemedi, bağlantını kontrol et.")
