import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

st.set_page_config(layout="wide", page_title="Alpha Sniper | Pro-Confirmation")

# Binance Futures Bağlantısı
exchange = ccxt.binance({
    'enableRateLimit': True, 
    'options': {'defaultType': 'future'},
    'timeout': 60000
})

st.markdown("# 🏛️ QUANT ALPHA: AĞIR ANALİZ (MULTI-CONFIRM)")
st.info("Sistem her coini H4 ve H1 zaman dilimlerinde çapraz kontrole alıyor. Bu işlem biraz zaman alır ama daha güvenlidir.")

def get_pro_symbols():
    try:
        # Sadece hacmi en yüksek ilk 25 coini al (Kalite > Nicelik)
        tickers = exchange.fetch_tickers()
        df_t = pd.DataFrame.from_dict(tickers, orient='index')
        df_t = df_t[df_t['symbol'].str.contains('USDT')]
        return df_t.sort_values('quoteVolume', ascending=False).head(25).index.tolist()
    except:
        return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']

def heavy_pro_scanner():
    symbols = get_pro_symbols()
    results = []
    progress = st.progress(0)
    status = st.empty()
    
    for idx, symbol in enumerate(symbols):
        status.info(f"🛡️ Derin Analiz Yapılıyor: **{symbol}**")
        try:
            # 1. KATMAN: H4 (4 Saatlik) ANA TREND KONTROLÜ
            bars_h4 = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=200)
            df_h4 = pd.DataFrame(bars_h4, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            ema200_h4 = ta.ema(df_h4['c'], length=200).iloc[-1]
            last_c = df_h4['c'].iloc[-1]
            
            ana_trend = "UP" if last_c > ema200_h4 else "DOWN"

            # 2. KATMAN: H1 (1 Saatlik) SİNYAL KONTROLÜ
            bars_h1 = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=150)
            df_h1 = pd.DataFrame(bars_h1, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # Göstergeler (H1)
            stoch = ta.stochrsi(df_h1['c'], length=14, rsi_length=14, k=3, d=3)
            df_h1 = pd.concat([df_h1, stoch], axis=1)
            rsi = ta.rsi(df_h1['c'], length=14).iloc[-1]
            vol_avg = df_h1['v'].rolling(20).mean().iloc[-1]
            
            l_h1, p_h1 = df_h1.iloc[-1], df_h1.iloc[-2]
            sk, sd = "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"
            
            # --- GARANTİ SKOR HESAPLAMA ---
            final_skor = 0
            komut = "BEKLE"
            
            # LONG İÇİN ÇOKLU ONAY
            if ana_trend == "UP":
                final_skor += 40 # Ana trend yönünde puan
                if p_h1[sk] < p_h1[sd] and l_h1[sk] > l_h1[sd]: # Kesişim
                    final_skor += 40
                if l_h1['v'] > vol_avg: # Hacim Onayı
                    final_skor += 10
                if rsi < 65: # Aşırı alım değilse
                    final_skor += 10
                
                if final_skor >= 80: komut = "🚀 GÜÇLÜ LONG"
                elif final_skor >= 50: komut = "📈 LONG TAKİP"

            # SHORT İÇİN ÇOKLU ONAY
            elif ana_trend == "DOWN":
                final_skor += 40
                if p_h1[sk] > p_h1[sd] and l_h1[sk] < l_h1[sd]:
                    final_skor += 40
                if l_h1['v'] > vol_avg:
                    final_skor += 10
                if rsi > 35:
                    final_skor += 10
                
                if final_skor >= 80: komut = "💥 GÜÇLÜ SHORT"
                elif final_skor >= 50: komut = "📉 SHORT TAKİP"

            results.append({
                "COIN": symbol,
                "FİYAT": f"{l_h1['c']:.4f}",
                "KOMUT": komut,
                "GÜVEN SKORU": final_skor,
                "TREND (H4)": ana_trend,
                "HACİM": "YÜKSEK" if l_h1['v'] > vol_avg else "NORMAL"
            })
            time.sleep(0.5) # Ağır döngü, borsa koruması
        except: continue
        progress.progress((idx + 1) / len(symbols))
    
    status.empty()
    return pd.DataFrame(results).sort_values('GÜVEN SKORU', ascending=False)

# --- Arayüz ---
if st.button('🛡️ DERİN ANALİZİ BAŞLAT (GARANTİ MOD)'):
    data = heavy_pro_scanner()
    
    if not data.empty:
        def style_pro(row):
            color = ''
            if row['GÜVEN SKORU'] >= 80:
                color = 'background-color: #11381b; color: #52ff8f; font-weight: bold' if "LONG" in row['KOMUT'] else 'background-color: #3b0d0d; color: #ff6e6e; font-weight: bold'
            return [color]*len(row)

        st.dataframe(data.style.apply(style_pro, axis=1), use_container_width=True)
    else:
        st.error("Veri çekilemedi.")
