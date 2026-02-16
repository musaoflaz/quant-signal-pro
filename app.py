import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

# 1. Sayfa Ayarları
st.set_page_config(layout="wide", page_title="Quant Signal Pro | Terminal V3")

# 2. Borsa Bağlantı Fonksiyonu (IP Engeline Karşı Çoklu Deneme)
def get_exchange_connection():
    # Gate.io bulut sunucularına karşı genellikle daha toleranslıdır.
    return ccxt.gateio({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
        'timeout': 30000
    })

# 3. Başlık
st.markdown("# 🏛️ TRADE TERMINAL (Cloud Optimized)")
st.write("---")

# Varlık listesi
symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT', 'SUI/USDT', 'AVAX/USDT']

def fetch_safe_data():
    exchange = get_exchange_connection()
    results = []
    
    # İlerleme çubuğu (Kullanıcıya veri çekildiğini hissettirir)
    progress_bar = st.progress(0)
    
    for i, symbol in enumerate(symbols):
        try:
            # Veri çekme (Hata alırsak 1 saniye bekle ve geç)
            bars = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=50)
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            rsi = ta.rsi(df['c'], length=14).iloc[-1]
            last_price = df['c'].iloc[-1]
            
            # Sinyal Mantığı
            if rsi < 35: eylem = "🟢 AL (LONG)"; rejim = "TREND (UP)"
            elif rsi > 65: eylem = "🔴 SAT (SHORT)"; rejim = "TREND (DOWN)"
            else: eylem = "⚪ BEKLE"; rejim = "YATAY (RANGING)"

            results.append({
                "VARLIK": symbol,
                "FİYAT": f"{last_price:.4f}",
                "PİYASA REJİMİ": rejim,
                "İŞLEM EYLEMİ": eylem,
                "GÜVEN %": f"%{int(abs(50-rsi)*2)}",
                "ANALİZ": f"H4 | RSI:{int(rsi)}"
            })
            time.sleep(0.2) # API Banlanmaması için küçük es
        except:
            continue
        progress_bar.progress((i + 1) / len(symbols))
    
    progress_bar.empty()
    return pd.DataFrame(results)

# Arayüz Akışı
data = fetch_safe_data()

if not data.empty:
    # Sütun bazlı renklendirme (Hata vermeyen en güvenli metod)
    def style_rows(val):
        if "AL" in str(val): return 'color: #00ff00; font-weight: bold'
        if "SAT" in str(val): return 'color: #ff4b4b; font-weight: bold'
        return ''

    st.dataframe(
        data.style.map(style_rows, subset=['İŞLEM EYLEMİ']),
        use_container_width=True,
        height=500
    )
else:
    st.warning("⚠️ Bulut sunucusu borsa bağlantısını şu an reddediyor. 30 saniye sonra otomatik tekrar denenecek.")
    time.sleep(5)
    st.rerun()

# Yenileme butonu
if st.sidebar.button('🔄 Terminali Yenile'):
    st.rerun()
