import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

st.set_page_config(layout="wide", page_title="Alpha Sniper | Binance Bridge")

# --- BINANCE KÖPRÜSÜ (EN STABİL YÖNTEM) ---
def get_binance():
    return ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
        'timeout': 30000,
        # Binance engeli olan sunucular için özel bir endpoint kullanıyoruz
        'urls': {
            'api': {
                'public': 'https://api1.binance.com/api/v3',
                'private': 'https://api1.binance.com/api/v3',
            }
        }
    })

exchange = get_binance()

st.title("🏛️ BINANCE BRIDGE (V31)")
st.info("Binance API1 Köprüsü üzerinden veriler çekiliyor...")

def bridge_scanner():
    results = []
    # Binance'te işlem gören en popüler 10 coin
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'LTC/USDT', 'AVAX/USDT', 'FET/USDT', 'TIA/USDT', 'RNDR/USDT']
    
    progress = st.progress(0)
    for idx, symbol in enumerate(symbols):
        try:
            # Veri çekme
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=150)
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # Başarılı Stratejimiz (V29)
            df['EMA200'] = ta.ema(df['c'], length=200)
            stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
            df = pd.concat([df, stoch], axis=1)
            rsi_val = ta.rsi(df['c'], length=14).iloc[-1]
            
            l, p = df.iloc[-1], df.iloc[-2]
            sk, sd = "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"
            
            skor = 0
            komut = "İZLE"
            
            if l['c'] > l['EMA200']: # LONG
                skor += 40
                if p[sk] < p[sd] and l[sk] > l[sd]: skor += 50
                if rsi_val < 65: skor += 10
                if skor >= 90: komut = "🚀 BINANCE LONG"
            elif l['c'] < l['EMA200']: # SHORT
                skor += 40
                if p[sk] > p[sd] and l[sk] < l[sd]: skor += 50
                if rsi_val > 35: skor += 10
                if skor >= 90: komut = "💥 BINANCE SHORT"

            results.append({"COIN": symbol, "FİYAT": l['c'], "EYLEM": komut, "SKOR": skor, "RSI": int(rsi_val)})
            time.sleep(0.3)
        except: continue
        progress.progress((idx + 1) / len(symbols))
    return pd.DataFrame(results)

if st.button('🎯 KÖPRÜ ÜZERİNDEN TARA'):
    data = bridge_scanner()
    if not data.empty:
        def style_logic(row):
            if row['SKOR'] >= 90:
                color = '#0c3e1e' if "LONG" in row['EYLEM'] else '#4b0a0a'
                return [f'background-color: {color}; color: white; font-weight: bold'] * len(row)
            return [''] * len(row)
        st.dataframe(data.sort_values('SKOR', ascending=False).style.apply(style_logic, axis=1), use_container_width=True)
    else:
        st.error("Köprü şu an kapalı. KuCoin verisiyle devam etmenizi öneririm.")
