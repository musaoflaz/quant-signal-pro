import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

st.set_page_config(layout="wide", page_title="Alpha Sniper Fortress")

exchange = ccxt.kucoin({'enableRateLimit': True})

st.title("🛡️ ALPHA FORTRESS: ULTRA GÜVENLİ FİLTRE (V35)")
st.warning("Skor almak artık çok zor. 90-100 arası 'Gerçek Sinyal' kabul edilir.")

def fortress_scanner():
    results = []
    # Geniş liste (Binance uyumlu)
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'LTC/USDT', 'AVAX/USDT', 
        'LINK/USDT', 'FET/USDT', 'TIA/USDT', 'RNDR/USDT', 'NEAR/USDT',
        'ARB/USDT', 'OP/USDT', 'INJ/USDT', 'SUI/USDT', 'PEPE/USDT',
        'ADA/USDT', 'DOT/USDT', 'XRP/USDT', 'DOGE/USDT', 'SHIB/USDT',
        'APT/USDT', 'FIL/USDT', 'STX/USDT', 'GRT/USDT', 'ATOM/USDT',
        'SEI/USDT', 'JUP/USDT', 'WIF/USDT', 'BONK/USDT', 'LDO/USDT'
    ]
    
    progress = st.progress(0)
    for idx, symbol in enumerate(symbols):
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=150)
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # --- GELİŞMİŞ İNDİKATÖR SETİ ---
            df['EMA200'] = ta.ema(df['c'], length=200) # Ana Trend
            stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
            df = pd.concat([df, stoch], axis=1)
            df['RSI'] = ta.rsi(df['c'], length=14)
            df['ADX'] = ta.adx(df['h'], df['l'], df['c'], length=14)['ADX_14'] # Trend Gücü
            bb = ta.bbands(df['c'], length=20, std=2) # Bollinger Bantları
            
            l = df.iloc[-1] # Son mum
            p = df.iloc[-2] # Önceki mum
            sk, sd = "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"
            
            skor = 0
            # 1. Engel: Ana Trend Onayı (30 Puan)
            if l['c'] > l['EMA200']:
                skor += 30
                # 2. Engel: Stoch RSI Kesişimi (30 Puan)
                if p[sk] < p[sd] and l[sk] > l[sd]:
                    skor += 30
                # 3. Engel: Trend Gücü (ADX > 25 ise +20 Puan)
                if l['ADX'] > 25:
                    skor += 20
                # 4. Engel: Güvenli Bölge (RSI 40-60 arası ise +20 Puan)
                if 40 <= l['RSI'] <= 60:
                    skor += 20
                # Ceza Puanı: Bollinger Üst Bandı aşılmışsa (Aşırı şişmiş)
                if l['c'] >= bb['BBU_20_2.0'].iloc[-1]:
                    skor -= 40
                
                eylem = "🚀 KESİN LONG" if skor >= 90 else "📈 TREND OLUMLU"
            
            elif l['c'] < l['EMA200']: # SHORT SENARYOSU
                skor += 30
                if p[sk] > p[sd] and l[sk] < l[sd]:
                    skor += 30
                if l['ADX'] > 25:
                    skor += 20
                if 40 <= l['RSI'] <= 60:
                    skor += 20
                if l['c'] <= bb['BBL_20_2.0'].iloc[-1]:
                    skor -= 40
                
                eylem = "💥 KESİN SHORT" if skor >= 90 else "📉 TREND NEGATİF"

            results.append({
                "COIN": symbol, 
                "SKOR": max(0, skor), 
                "SİNYAL": eylem, 
                "ADX(GÜÇ)": int(l['ADX']), 
                "RSI": int(l['RSI'])
            })
        except: continue
        progress.progress((idx + 1) / len(symbols))
    return pd.DataFrame(results)

if st.button('🛡️ ULTRA GÜVENLİ TARA'):
    data = fortress_scanner()
    if not data.empty:
        # Görsel Filtre: Sadece 90-100 arası parlasın
        def style_fortress(row):
            if row['SKOR'] >= 90:
                return ['background-color: #ffd700; color: black; font-weight: bold'] * len(row)
            return [''] * len(row)
        
        st.dataframe(data.sort_values('SKOR', ascending=False).style.apply(style_fortress, axis=1), use_container_width=True)
    else:
        st.error("Veri alınamadı.")
