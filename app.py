import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

# Sayfa Ayarları
st.set_page_config(layout="wide", page_title="Quant Pro | Anti-Block Terminal")

# --- ENGELİ AŞMAK İÇİN KUCOIN BAĞLANTISI ---
# KuCoin bulut sunucularına Bybit ve Binance'den daha fazla tolerans gösterir.
exchange = ccxt.kucoin({
    'enableRateLimit': True,
    'timeout': 60000,
    'options': {'adjustForTimeDifference': True}
})

st.markdown("# 🏛️ QUANT PRO - SİNYAL TERMİNALİ")
st.write("---")

# KuCoin formatında varlık listesi
symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT', 'ADA/USDT', 'SUI/USDT']

def pro_scanner():
    rows = []
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    for idx, symbol in enumerate(symbols):
        status_text.info(f"🔍 Analiz Ediliyor: **{symbol}**...")
        
        try:
            # KuCoin'den veri çekme
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=150)
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # Teknik Göstergeler
            df['EMA200'] = ta.ema(df['c'], length=100) # Daha hızlı tepki için 100
            df['RSI'] = ta.rsi(df['c'], length=14)
            
            last = df.iloc[-1]
            c, rsi, ema = last['c'], last['RSI'], last['EMA200']
            
            # Sinyal Skoru
            score = 0
            if c > ema: score += 50  # Trend Boğa
            if rsi < 40: score += 50 # Aşırı Satım
            
            short_score = 0
            if c < ema: short_score += 50 # Trend Ayı
            if rsi > 60: short_score += 50 # Aşırı Alım

            eylem = "⚪ BEKLE"
            if score >= 100: eylem = "🚀 GÜÇLÜ LONG"
            elif short_score >= 100: eylem = "💥 GÜÇLÜ SHORT"

            rows.append({
                "VARLIK": symbol,
                "FİYAT": f"{c:.4f}",
                "TREND": "BOĞA" if c > ema else "AYI",
                "GÜVEN": f"%{max(score, short_score)}",
                "İŞLEM EYLEMİ": eylem
            })
            
            progress_bar.progress((idx + 1) / len(symbols))
            time.sleep(1) # Borsa bizi robot sanmasın diye bekliyoruz
            
        except Exception as e:
            # Hata mesajını sadeleştirip kullanıcıya bildir
            st.warning(f"⚠️ {symbol} için bağlantı denemesi başarısız. Diğerine geçiliyor.")
            continue
            
    status_text.empty()
    progress_bar.empty()
    return pd.DataFrame(rows)

# Renklendirme
def style_action(val):
    if "LONG" in str(val): return 'background-color: #0c3e1e; color: #52ff8f; font-weight: bold'
    if "SHORT" in str(val): return 'background-color: #4b0a0a; color: #ff6e6e; font-weight: bold'
    return ''

data = pro_scanner()

if not data.empty:
    st.dataframe(data.style.applymap(style_action, subset=['İŞLEM EYLEMİ']), use_container_width=True)
else:
    st.error("❌ Tüm borsalar erişimi reddetti. Lütfen bir süre sonra tekrar deneyin.")

if st.sidebar.button('🔄 Yeniden Tara'):
    st.rerun()
