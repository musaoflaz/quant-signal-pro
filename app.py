import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

# Sayfa Genişliği
st.set_page_config(layout="wide", page_title="Alpha Sniper Guardian")

# KuCoin Bağlantısı (Daha yüksek limit ve hata payı ile)
exchange = ccxt.kucoin({'enableRateLimit': True, 'timeout': 30000})

st.title("🛡️ ALPHA GUARDIAN (V36)")
st.info("Kusursuz veri çekme ve ultra hassas filtreleme devrede.")

def guardian_scanner():
    results = []
    # En stabil 25 Coin
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'LTC/USDT', 'AVAX/USDT', 
        'LINK/USDT', 'FET/USDT', 'TIA/USDT', 'RNDR/USDT', 'NEAR/USDT',
        'ARB/USDT', 'OP/USDT', 'INJ/USDT', 'SUI/USDT', 'PEPE/USDT',
        'ADA/USDT', 'DOT/USDT', 'XRP/USDT', 'DOGE/USDT', 'SHIB/USDT',
        'APT/USDT', 'STX/USDT', 'JUP/USDT', 'WIF/USDT', 'BONK/USDT'
    ]
    
    progress = st.progress(0)
    for idx, symbol in enumerate(symbols):
        try:
            # Daha stabil veri çekme (limit 100 yeterli)
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
            if not bars: continue
            
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # İNDİKATÖRLER
            df['EMA200'] = ta.ema(df['c'], length=200) or 0
            df['RSI'] = ta.rsi(df['c'], length=14)
            stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
            df = pd.concat([df, stoch], axis=1)
            # ADX ve Bollinger (Veri yetersizliğine karşı hata kontrolüyle)
            adx_df = ta.adx(df['h'], df['l'], df['c'], length=14)
            df['ADX'] = adx_df['ADX_14'] if adx_df is not None else 0
            bb = ta.bbands(df['c'], length=20)
            
            l = df.iloc[-1]
            p = df.iloc[-2]
            sk, sd = "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"
            
            skor = 0
            # --- ZORLAŞTIRILMIŞ PUANLAMA ---
            if l['c'] > l['EMA200']: # LONG
                skor += 30
                if p[sk] < p[sd] and l[sk] > l[sd]: skor += 30 # Kesişim
                if l['ADX'] > 20: skor += 20 # Trend Gücü
                if 35 <= l['RSI'] <= 65: skor += 20 # RSI Güvenli Bölge
                # Bollinger Ceza
                if bb is not None and l['c'] > bb['BBU_20_2.0'].iloc[-1]: skor -= 40
            
            elif l['c'] < l['EMA200']: # SHORT
                skor += 30
                if p[sk] > p[sd] and l[sk] < l[sd]: skor += 30
                if l['ADX'] > 20: skor += 20
                if 35 <= l['RSI'] <= 65: skor += 20
                if bb is not None and l['c'] < bb['BBL_20_2.0'].iloc[-1]: skor -= 40

            results.append({
                "COIN": symbol,
                "SKOR": max(0, int(skor)),
                "ADX": int(l['ADX']),
                "RSI": int(l['RSI']),
                "SİNYAL": "🔥 GİR" if skor >= 90 else "⌛ BEKLE"
            })
            time.sleep(0.1) # Borsa banlamasın diye kısa bekleme
        except: continue
        progress.progress((idx + 1) / len(symbols))
    return pd.DataFrame(results)

if st.button('🛡️ GUARDIAN TARAMAYI BAŞLAT'):
    data = guardian_scanner()
    if not data.empty:
        st.dataframe(data.sort_values('SKOR', ascending=False), use_container_width=True)
    else:
        st.error("Veri çekilemedi. Lütfen internetini veya borsa bağlantısını kontrol et.")
