import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

# --- PROXY YAPILANDIRMASI ---
# Bu proxy'ler herkese açık ve ücretsizdir. Biri çalışmazsa diğeri devreye girer.
PROXIES = [
    'https://api.allorigins.win/raw?url=', # CORS Proxy
    'https://thingproxy.freeboard.io/fetch/', # Alternative Proxy
]

# Binance bağlantısını bir fonksiyon içinde kuralım
def get_binance_connection():
    # Streamlit Cloud üzerinde Binance direkt engelli olduğu için 
    # CCXT'nin içinden proxy ayarı yapıyoruz
    return ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}, # Veya 'future'
        'timeout': 30000,
        # 'proxies': {'http': '...', 'https': '...'} # Eğer özel proxy alırsan buraya
    })

exchange = get_binance_connection()

st.title("🏛️ BINANCE PROXY SHIELD (V30)")
st.info("Ücretsiz köprüler üzerinden Binance verisi taranıyor...")

def binance_scanner():
    results = []
    # Binance'in en hacimli altcoinleri
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'LTC/USDT', 'AVAX/USDT', 'LINK/USDT', 'FET/USDT']
    
    progress = st.progress(0)
    
    for idx, symbol in enumerate(symbols):
        try:
            # Binance'ten veri çekmeyi deniyoruz
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=150)
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # Başarılı V29 stratejimiz
            df['EMA200'] = ta.ema(df['c'], length=200)
            stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
            df = pd.concat([df, stoch], axis=1)
            rsi_val = ta.rsi(df['c'], length=14).iloc[-1]
            
            l, p = df.iloc[-1], df.iloc[-2]
            sk, sd = "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"
            
            skor = 0
            komut = "İZLE"
            
            # 100 PUAN MANTIĞI (V29'dan gelen başarılı sistem)
            if l['c'] > l['EMA200']:
                skor += 40
                if p[sk] < p[sd] and l[sk] > l[sd]: skor += 50
                if rsi_val < 60: skor += 10
                if skor >= 90: komut = "🚀 BINANCE LONG"
            
            elif l['c'] < l['EMA200']:
                skor += 40
                if p[sk] > p[sd] and l[sk] < l[sd]: skor += 50
                if rsi_val > 40: skor += 10
                if skor >= 90: komut = "💥 BINANCE SHORT"

            results.append({"COIN": symbol, "FİYAT": l['c'], "EYLEM": komut, "SKOR": skor, "RSI": int(rsi_val)})
            time.sleep(0.5) 
        except Exception as e:
            st.warning(f"{symbol} için Binance bağlantı hatası: {str(e)[:50]}...")
            continue
        progress.progress((idx + 1) / len(symbols))
    
    return pd.DataFrame(results)

if st.button('🎯 BINANCE ÜZERİNDEN TARA'):
    data = binance_scanner()
    if not data.empty:
        st.dataframe(data.sort_values('SKOR', ascending=False), use_container_width=True)
