import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

st.set_page_config(layout="wide", page_title="Alpha Sniper | Gold Standard")

# Binance Futures Bağlantısı
exchange = ccxt.binance({
    'enableRateLimit': True, 
    'options': {'defaultType': 'future'},
    'timeout': 60000
})

st.markdown("# 🏛️ QUANT ALPHA: GOLD STANDARD")
st.write("---")

def get_pro_symbols():
    try:
        tickers = exchange.fetch_tickers()
        df_t = pd.DataFrame.from_dict(tickers, orient='index')
        df_t = df_t[df_t['symbol'].str.contains('USDT')]
        # Hacmi en yüksek 30 coin (Okyanusun en büyük balıkları)
        return df_t.sort_values('quoteVolume', ascending=False).head(30).index.tolist()
    except:
        return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']

def heavy_gold_scanner():
    symbols = get_pro_symbols()
    results = []
    progress = st.progress(0)
    status = st.empty()
    
    for idx, symbol in enumerate(symbols):
        status.info(f"🛡️ Analiz Ediliyor: **{symbol}**")
        try:
            # H4 Ana Trend
            bars_h4 = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=200)
            df_h4 = pd.DataFrame(bars_h4, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            ema200_h4 = ta.ema(df_h4['c'], length=200).iloc[-1]
            last_c = df_h4['c'].iloc[-1]
            ana_trend = "BOĞA (H4)" if last_c > ema200_h4 else "AYI (H4)"

            # H1 Sinyal
            bars_h1 = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=150)
            df_h1 = pd.DataFrame(bars_h1, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            stoch = ta.stochrsi(df_h1['c'], length=14, rsi_length=14, k=3, d=3)
            df_h1 = pd.concat([df_h1, stoch], axis=1)
            rsi = ta.rsi(df_h1['c'], length=14).iloc[-1]
            vol_avg = df_h1['v'].rolling(20).mean().iloc[-1]
            
            l_h1, p_h1 = df_h1.iloc[-1], df_h1.iloc[-2]
            sk, sd = "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"
            
            # --- PROFESYONEL PUANLAMA ---
            puan = 0
            
            # Trend Puanı (40 Puan)
            if (ana_trend == "BOĞA (H4)" and l_h1['c'] > ema200_h4): puan += 40
            elif (ana_trend == "AYI (H4)" and l_h1['c'] < ema200_h4): puan += 40
            
            # Kesişim Puanı (40 Puan)
            long_cross = p_h1[sk] < p_h1[sd] and l_h1[sk] > l_h1[sd]
            short_cross = p_h1[sk] > p_h1[sd] and l_h1[sk] < l_h1[sd]
            
            if ana_trend == "BOĞA (H4)" and long_cross: puan += 40
            if ana_trend == "AYI (H4)" and short_cross: puan += 40
            
            # Hacim ve RSI Bonusu (20 Puan)
            if l_h1['v'] > vol_avg: puan += 10
            if (ana_trend == "BOĞA (H4)" and rsi < 65) or (ana_trend == "AYI (H4)" and rsi > 35): puan += 10

            # Komut Belirleme
            komut = "İZLE"
            if puan >= 80: komut = "🚀 LONG" if ana_trend == "BOĞA (H4)" else "💥 SHORT"
            elif puan >= 50: komut = "📈 GÖZLEM" if ana_trend == "BOĞA (H4)" else "📉 GÖZLEM"

            results.append({
                "COIN": symbol,
                "FİYAT": f"{l_h1['c']:.4f}",
                "KOMUT": komut,
                "SKOR": puan,
                "TREND": ana_trend,
                "RSI": int(rsi)
            })
            time.sleep(0.4)
        except: continue
        progress.progress((idx + 1) / len(symbols))
    
    status.empty()
    return pd.DataFrame(results).sort_values('SKOR', ascending=False)

# --- Arayüz ---
if st.button('🛡️ GOLD ANALİZİ BAŞLAT'):
    data = heavy_gold_scanner()
    
    if not data.empty:
        def style_gold(row):
            color = ''
            if row['SKOR'] >= 80:
                color = 'background-color: #11381b; color: #52ff8f; font-weight: bold' if "LONG" in row['KOMUT'] else 'background-color: #3b0d0d; color: #ff6e6e; font-weight: bold'
            return [color]*len(row)

        st.dataframe(
            data.style.apply(style_gold, axis=1), 
            use_container_width=True, 
            height=650
        )
    else:
        st.error("Veri çekilemedi.")
