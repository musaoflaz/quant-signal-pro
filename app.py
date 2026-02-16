import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

# Sayfa Ayarları
st.set_page_config(layout="wide", page_title="Quant Pro | Alpha Sniper")

# Borsa Bağlantısı
exchange = ccxt.bybit({'enableRateLimit': True, 'timeout': 60000})

st.markdown("# 🏛️ QUANT PRO - SİNYAL TERMİNALİ")
st.write("---")

symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT', 'AVAX/USDT', 'SUI/USDT', 'PEPE/USDT']

def pro_scanner():
    rows = []
    
    # --- İŞTE BEKLEMENİ ENGELLEYECEK GÖSTERGELER ---
    status_text = st.empty() # Dinamik yazı alanı
    progress_bar = st.progress(0) # İlerleme çubuğu
    
    for idx, symbol in enumerate(symbols):
        # Hangi coin taranıyor göster
        status_text.info(f"🔍 Şu an analiz ediliyor: **{symbol}** ({idx+1}/{len(symbols)})")
        
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=200)
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # Analizler (Trend + Momentum + Volatilite)
            df['EMA200'] = ta.ema(df['c'], length=200)
            df['RSI'] = ta.rsi(df['c'], length=14)
            st_data = ta.supertrend(df['h'], df['l'], df['c'], length=10, multiplier=3)
            df = pd.concat([df, st_data], axis=1)
            
            last = df.iloc[-1]
            score = 0
            # Long Şartları
            if last['c'] > last['EMA200']: score += 25
            if last['SUPERTd_10_3.0'] == 1: score += 25
            if last['RSI'] < 45: score += 50
            
            # Short Şartları
            short_score = 0
            if last['c'] < last['EMA200']: short_score += 25
            if last['SUPERTd_10_3.0'] == -1: short_score += 25
            if last['RSI'] > 55: short_score += 50

            eylem = "⚪ BEKLE"
            if score >= 75: eylem = "🚀 GÜÇLÜ LONG"
            elif short_score >= 75: eylem = "💥 GÜÇLÜ SHORT"

            rows.append({
                "VARLIK": symbol,
                "FİYAT": f"{last['c']:.4f}",
                "TREND": "BOĞA" if last['c'] > last['EMA200'] else "AYI",
                "GÜVEN": f"%{max(score, short_score)}",
                "İŞLEM EYLEMİ": eylem
            })
            
            # İlerleme çubuğunu güncelle
            progress_bar.progress((idx + 1) / len(symbols))
            time.sleep(0.3) # API banlanmaması için
            
        except Exception as e:
            st.error(f"{symbol} hatası: {e}")
            continue
            
    # Tarama bitince göstergeleri temizle
    status_text.empty()
    progress_bar.empty()
    
    return pd.DataFrame(rows)

# Tabloyu Renkli Bas
def style_action(val):
    if "LONG" in str(val): return 'background-color: #0c3e1e; color: #52ff8f; font-weight: bold'
    if "SHORT" in str(val): return 'background-color: #4b0a0a; color: #ff6e6e; font-weight: bold'
    return ''

data = pro_scanner()

if not data.empty:
    st.dataframe(data.style.applymap(style_action, subset=['İŞLEM EYLEMİ']), use_container_width=True)
    st.success("✅ Tüm piyasa başarıyla tarandı!")
else:
    st.warning("Veri çekilemedi.")

if st.sidebar.button('🔄 Derin Analizi Başlat'):
    st.rerun()
