import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

st.set_page_config(layout="wide", page_title="Quant Alpha | V12 Final")

# Borsa Bağlantısı - Timeout süresi artırıldı
exchange = ccxt.kucoin({
    'enableRateLimit': True, 
    'timeout': 60000,
    'options': {'adjustForTimeDifference': True}
})

st.markdown("# 🏛️ QUANT ALPHA: AKILLI ANALİZ")
st.write("---")

def get_symbols():
    try:
        tickers = exchange.fetch_tickers()
        df_t = pd.DataFrame.from_dict(tickers, orient='index')
        df_t = df_t[df_t['symbol'].str.contains('/USDT')]
        # Sayıyı 30'a düşürdük (Daha stabil olması için)
        return df_t.sort_values('quoteVolume', ascending=False).head(30).index.tolist()
    except:
        return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT']

def ultra_scanner():
    symbols = get_symbols()
    results = []
    
    progress = st.progress(0)
    status = st.empty()
    
    for idx, symbol in enumerate(symbols):
        status.info(f"🔍 Analiz: **{symbol}** ({idx+1}/{len(symbols)})")
        try:
            # Daha az veri çekerek hızı artırıyoruz ama EMA200 için yeterli (250 limit)
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=250)
            if len(bars) < 200: continue # Yeterli veri yoksa pas geç
            
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # Teknik Analiz
            df['EMA200'] = ta.ema(df['c'], length=200)
            df['RSI'] = ta.rsi(df['c'], length=14)
            stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
            df = pd.concat([df, stoch], axis=1)
            
            l, p = df.iloc[-1], df.iloc[-2]
            
            # KRİTERLER
            score = 0
            label = "GÖZLEM"
            
            # Stoch RSI Sütun isimlerini güvenli alalım
            sk = "STOCHRSIk_14_14_3_3"
            sd = "STOCHRSId_14_14_3_3"
            
            up_cross = p[sk] < p[sd] and l[sk] > l[sd]
            down_cross = p[sk] > p[sd] and l[sk] < l[sd]

            if l['c'] > l['EMA200'] and up_cross:
                label = "🚀 GÜÇLÜ AL (LONG)"
                score = 90
            elif l['c'] < l['EMA200'] and down_cross:
                label = "💥 GÜÇLÜ SAT (SHORT)"
                score = 90
            elif l['c'] > l['EMA200']:
                label = "🟢 TREND YUKARI"
                score = 50
            else:
                label = "🔴 TREND AŞAĞI"
                score = 40

            results.append({
                "COIN": symbol,
                "FİYAT": f"{l['c']:.4f}",
                "DURUM": label,
                "SKOR": score,
                "RSI": int(l['RSI'])
            })
            time.sleep(0.5) # Borsa bloklamasın diye yarım saniye bekle
        except Exception as e:
            continue
        progress.progress((idx + 1) / len(symbols))
    
    status.empty()
    progress.empty()
    
    # EĞER HİÇ VERİ YOKSA BOŞ TABLO YERİNE ÖRNEK SATIR OLUŞTUR (Hata engelleyici)
    if not results:
        return pd.DataFrame(columns=["COIN", "FİYAT", "DURUM", "SKOR", "RSI"])
    
    return pd.DataFrame(results).sort_values(by='SKOR', ascending=False)

# --- Arayüz ---
if st.button('🎯 ANALİZİ BAŞLAT'):
    data = ultra_scanner()
    
    if not data.empty and 'SKOR' in data.columns:
        # Sinyaller
        signals = data[data['SKOR'] >= 80]
        if not signals.empty:
            st.subheader("🔥 KRİTERLERE UYANLAR")
            st.table(signals)
        else:
            st.warning("Şu an tam uyumlu sinyal yok, piyasayı izle.")

        # Tüm Liste
        st.write("---")
        st.subheader("👀 TÜM LİSTE")
        st.dataframe(data, use_container_width=True)
    else:
        st.error("⚠️ Borsaya bağlanılamadı. Lütfen butona tekrar basarak tazeleyin.")
