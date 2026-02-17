import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

st.set_page_config(layout="wide", page_title="Alpha Sniper | Bybit Pro")

# 1. Bybit Bağlantısı (Bulut sunucularında daha stabil çalışır)
exchange = ccxt.bybit({
    'enableRateLimit': True, 
    'options': {'defaultType': 'linear'}, # Vadeli (Futures) işlemler için
    'timeout': 60000
})

st.markdown("# 🏛️ QUANT ALPHA: BYBIT PRO FUTURES")
st.write("---")

def get_bybit_symbols():
    try:
        markets = exchange.fetch_markets()
        # Sadece USDT vadeli çiftlerini al
        symbols = [m['symbol'] for m in markets if m['active'] and m['quote'] == 'USDT' and m['linear']]
        return symbols[:30] # İlk 30 aktif çift
    except Exception as e:
        return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'AVAX/USDT']

def bybit_scanner():
    symbols = get_bybit_symbols()
    results = []
    progress = st.progress(0)
    status = st.empty()
    
    for idx, symbol in enumerate(symbols):
        status.info(f"🛡️ Analiz Ediliyor: **{symbol}**")
        try:
            # 1 Saatlik Veri
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=250)
            if len(bars) < 200: continue
            
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # Strateji Hesaplama
            df['EMA200'] = ta.ema(df['c'], length=200)
            stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
            df = pd.concat([df, stoch], axis=1)
            df['RSI'] = ta.rsi(df['c'], length=14)
            
            l, p = df.iloc[-1], df.iloc[-2]
            sk, sd = "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"
            
            puan = 0
            trend = "LONG" if l['c'] > l['EMA200'] else "SHORT"
            
            # Onay Puanları
            puan += 40 # Ana Trend puanı
            
            # Kesişim Kontrolü
            if trend == "LONG" and p[sk] < p[sd] and l[sk] > l[sd]:
                puan += 40
            elif trend == "SHORT" and p[sk] > p[sd] and l[sk] < l[sd]:
                puan += 40
                
            # Momentum
            if (trend == "LONG" and l['RSI'] < 65) or (trend == "SHORT" and l['RSI'] > 35):
                puan += 20

            komut = "İZLE"
            if puan >= 80:
                komut = "🚀 LONG GİRİŞ" if trend == "LONG" else "💥 SHORT GİRİŞ"

            results.append({
                "COIN": symbol,
                "FİYAT": f"{l['c']:.4f}",
                "KOMUT": komut,
                "SKOR": puan,
                "TREND": trend,
                "RSI": int(l['RSI'])
            })
            time.sleep(0.2)
        except: continue
        progress.progress((idx + 1) / len(symbols))
    
    status.empty()
    if not results: return pd.DataFrame()
    return pd.DataFrame(results).sort_values('SKOR', ascending=False)

# --- Arayüz ---
if st.button('🚀 ANALİZİ BAŞLAT (BYBIT)'):
    data = bybit_scanner()
    
    if not data.empty:
        def style_pro(row):
            color = ''
            if row['SKOR'] >= 80:
                color = 'background-color: #0c3e1e; color: #52ff8f' if "LONG" in row['KOMUT'] else 'background-color: #4b0a0a; color: #ff6e6e'
            return [color]*len(row)

        st.dataframe(data.style.apply(style_pro, axis=1), use_container_width=True, height=600)
    else:
        st.warning("⚠️ Şu an kriterlere uyan sinyal yok veya borsa meşgul. Lütfen tekrar deneyin.")
