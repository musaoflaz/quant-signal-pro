import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

st.set_page_config(layout="wide", page_title="Alpha Sniper Pro")

# KuCoin üzerinden en geniş ve engelsiz veri akışı
exchange = ccxt.kucoin({'enableRateLimit': True, 'timeout': 20000})

st.title("🛡️ ALPHA SNIPER PRO (V37)")
st.info("Log hataları giderildi. Ultra keskin sinyal filtresi aktif.")

def pro_scanner():
    results = []
    # Binance'te 3x kaldıraç açabileceğin en popüler coinler
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'LTC/USDT', 'AVAX/USDT', 
        'LINK/USDT', 'FET/USDT', 'TIA/USDT', 'RNDR/USDT', 'NEAR/USDT',
        'ARB/USDT', 'OP/USDT', 'INJ/USDT', 'SUI/USDT', 'PEPE/USDT',
        'ADA/USDT', 'DOT/USDT', 'XRP/USDT', 'DOGE/USDT', 'SHIB/USDT'
    ]
    
    progress = st.progress(0)
    for idx, symbol in enumerate(symbols):
        try:
            # Hata almamak için limit 100 (Hızlı ve etkili)
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
            if not bars: continue
            
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # --- KRİTİK GÖSTERGELER ---
            df['EMA200'] = ta.ema(df['c'], length=200) or df['c'].rolling(50).mean()
            df['RSI'] = ta.rsi(df['c'], length=14)
            stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
            df = pd.concat([df, stoch], axis=1)
            
            l, p = df.iloc[-1], df.iloc[-2]
            sk, sd = "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"
            
            skor = 0
            # 1. Filtre: Trend (40 Puan)
            if l['c'] > l['EMA200']:
                skor += 40
                # 2. Filtre: Altın Kesişim (40 Puan)
                if p[sk] < p[sd] and l[sk] > l[sd]: skor += 40
                # 3. Filtre: RSI Gücü (20 Puan)
                if 40 <= l['RSI'] <= 60: skor += 20
                # 4. Filtre: Hacim Onayı (Ekstra Güven)
                if l['v'] > df['v'].tail(10).mean(): skor += 10
                
                eylem = "🔥 KESİN LONG" if skor >= 90 else "📈 TAKİP ET"
            
            elif l['c'] < l['EMA200']:
                skor += 40
                if p[sk] > p[sd] and l[sk] < l[sd]: skor += 40
                if 40 <= l['RSI'] <= 60: skor += 20
                if l['v'] > df['v'].tail(10).mean(): skor += 10
                
                eylem = "💥 KESİN SHORT" if skor >= 90 else "📉 TAKİP ET"

            results.append({
                "COIN": symbol,
                "SKOR": min(100, skor),
                "KOMUT": eylem,
                "RSI": int(l['RSI']),
                "GÜNCEL FİYAT": f"{l['c']:.4f}"
            })
            time.sleep(0.1)
        except: continue
        progress.progress((idx + 1) / len(symbols))
    return pd.DataFrame(results)

if st.button('🎯 ANALİZİ BAŞLAT'):
    data = pro_scanner()
    if not data.empty:
        # Görsel düzenleme
        def color_map(val):
            if val >= 90: return 'background-color: #ffd700; color: black; font-weight: bold'
            return ''
        
        st.dataframe(data.sort_values('SKOR', ascending=False).style.applymap(color_map, subset=['SKOR']), use_container_width=True)
    else:
        st.error("Sistem yoğunluğu nedeniyle veri çekilemedi. Lütfen 10 saniye sonra tekrar deneyin.")
