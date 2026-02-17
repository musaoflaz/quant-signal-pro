import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

st.set_page_config(layout="wide", page_title="Quant Alpha | V20 Global")

# BYBIT Bağlantısı (Bulut sunucuları için en garantisi)
exchange = ccxt.bybit({
    'enableRateLimit': True, 
    'options': {'defaultType': 'linear'}, 
    'timeout': 60000
})

st.markdown("# 🏛️ QUANT ALPHA: GLOBAL SNIPER")
st.write("---")

def get_pro_symbols():
    try:
        markets = exchange.fetch_markets()
        # Sadece popüler ve hacimli USDT çiftleri
        symbols = [m['symbol'] for m in markets if m['active'] and m['quote'] == 'USDT' and m['linear']]
        return symbols[:30]
    except:
        return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']

def global_scanner():
    symbols = get_pro_symbols()
    results = []
    progress = st.progress(0)
    status = st.empty()
    
    for idx, symbol in enumerate(symbols):
        status.info(f"🔍 Analiz: **{symbol}**")
        try:
            # Tek seferde 1 saatlik veri çekimi (Hız için)
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=250)
            if len(bars) < 200: continue
            
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # İndikatörler
            df['EMA200'] = ta.ema(df['c'], length=200)
            stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
            df = pd.concat([df, stoch], axis=1)
            df['RSI'] = ta.rsi(df['c'], length=14)
            
            l, p = df.iloc[-1], df.iloc[-2]
            sk, sd = "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"
            
            # --- GELİŞMİŞ SKORLAMA ---
            skor = 0
            
            # 1. Trend Onayı (40 Puan)
            trend_yukari = l['c'] > l['EMA200']
            skor += 40 
            
            # 2. Kesişim Onayı (40 Puan)
            up_cross = p[sk] < p[sd] and l[sk] > l[sd]
            down_cross = p[sk] > p[sd] and l[sk] < l[sd]
            
            if trend_yukari and up_cross: skor += 40
            elif not trend_yukari and down_cross: skor += 40
            
            # 3. RSI Güvenlik Filtresi (20 Puan)
            if (trend_yukari and l['RSI'] < 70) or (not trend_yukari and l['RSI'] > 30):
                skor += 20

            # Komut Belirleme
            komut = "⌛ BEKLE"
            if skor >= 80:
                komut = "🚀 LONG GİR" if trend_yukari else "💥 SHORT GİR"
            elif skor >= 40:
                komut = "📈 LONG TAKİP" if trend_yukari else "📉 SHORT TAKİP"

            results.append({
                "COIN": symbol,
                "FİYAT": f"{l['c']:.4f}",
                "EYLEM": komut,
                "SKOR": skor,
                "TREND": "YUKARI" if trend_yukari else "AŞAĞI",
                "RSI": int(l['RSI'])
            })
            time.sleep(0.2)
        except: continue
        progress.progress((idx + 1) / len(symbols))
    
    status.empty()
    progress.empty()
    
    # Boş liste hatasını kökten çözen kontrol
    if not results:
        return pd.DataFrame(columns=["COIN", "FİYAT", "EYLEM", "SKOR", "TREND", "RSI"])
        
    return pd.DataFrame(results).sort_values('SKOR', ascending=False)

# --- Arayüz ---
if st.button('🎯 GLOBAL TARAMAYI BAŞLAT'):
    data = global_scanner()
    
    if not data.empty:
        # Renklendirme
        def style_v20(row):
            bg = ''
            if row['SKOR'] >= 80:
                bg = 'background-color: #0c3e1e; color: #52ff8f' if "LONG" in row['EYLEM'] else 'background-color: #4b0a0a; color: #ff6e6e'
            return [bg]*len(row)

        st.subheader("📊 Canlı Sinyal Masası")
        st.dataframe(data.style.apply(style_v20, axis=1), use_container_width=True, height=600)
    else:
        st.warning("⚠️ Şu an kriterlere uyan bir coin bulunamadı. Lütfen birkaç dakika sonra tekrar deneyin.")
