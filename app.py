import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

st.set_page_config(layout="wide", page_title="Alpha Sniper | Binance Futures")

# 1. Binance Futures Bağlantısı (Özel Yapılandırma)
exchange = ccxt.binance({
    'enableRateLimit': True, 
    'options': {'defaultType': 'future'}, # Vadeli işlemler için zorunlu
    'timeout': 60000
})

st.markdown("# 🏛️ QUANT ALPHA: BINANCE FUTURES")
st.write("---")

def get_binance_symbols():
    try:
        markets = exchange.fetch_markets()
        # Sadece USDT vadeli çiftleri ve aktif olanları al
        symbols = [m['symbol'] for m in markets if m['active'] and m['quote'] == 'USDT' and m['contract']]
        # En popüler ilk 30 çifti seç (Hız ve stabilite için)
        return symbols[:30]
    except Exception as e:
        st.error(f"Sembol listesi alınamadı: {e}")
        return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'AVAX/USDT']

def binance_pro_scanner():
    symbols = get_binance_symbols()
    results = []
    progress = st.progress(0)
    status = st.empty()
    
    for idx, symbol in enumerate(symbols):
        status.info(f"🛡️ Binance Analiz: **{symbol}**")
        try:
            # 1 Saatlik (H1) verileri çekiyoruz
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=250)
            if len(bars) < 200: continue
            
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # İndikatör Hesaplamaları
            df['EMA200'] = ta.ema(df['c'], length=200)
            stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
            df = pd.concat([df, stoch], axis=1)
            df['RSI'] = ta.rsi(df['c'], length=14)
            
            l = df.iloc[-1]  # Son mum
            p = df.iloc[-2]  # Bir önceki mum
            sk, sd = "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"
            
            puan = 0
            trend = "YUKARI" if l['c'] > l['EMA200'] else "AŞAĞI"
            
            # PUANLAMA MANTIĞI (Senin Stratejin)
            # 1. Trend Uyumu (40 Puan)
            puan += 40 
            
            # 2. Kesişim Kontrolü (40 Puan)
            if trend == "YUKARI" and p[sk] < p[sd] and l[sk] > l[sd]:
                puan += 40
            elif trend == "AŞAĞI" and p[sk] > p[sd] and l[sk] < l[sd]:
                puan += 40
                
            # 3. Momentum Bonusu (20 Puan)
            if (trend == "YUKARI" and l['RSI'] < 65) or (trend == "AŞAĞI" and l['RSI'] > 35):
                puan += 20

            # Komut Oluşturma
            komut = "İZLE"
            if puan >= 80:
                komut = "🚀 LONG GİRİŞ" if trend == "YUKARI" else "💥 SHORT GİRİŞ"
            elif puan >= 50:
                komut = "📈 LONG PUSU" if trend == "YUKARI" else "📉 SHORT PUSU"

            results.append({
                "COIN": symbol,
                "FİYAT": f"{l['c']:.4f}",
                "KOMUT": komut,
                "SKOR": puan,
                "TREND (EMA200)": trend,
                "RSI": int(l['RSI'])
            })
            # Binance banlamasın diye küçük bir nefes
            time.sleep(0.2)
        except:
            continue
        progress.progress((idx + 1) / len(symbols))
    
    status.empty()
    progress.empty()
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results).sort_values('SKOR', ascending=False)

# --- Arayüz ---
if st.button('🚀 BINANCE FUTURES ANALİZİ BAŞLAT'):
    data = binance_pro_scanner()
    
    if not data.empty:
        def style_futures(row):
            color = ''
            if row['SKOR'] >= 80:
                color = 'background-color: #0c3e1e; color: #52ff8f' if "LONG" in row['KOMUT'] else 'background-color: #4b0a0a; color: #ff6e6e'
            return [color]*len(row)

        st.subheader("🎯 Binance İşlem Sinyalleri")
        st.dataframe(data.style.apply(style_futures, axis=1), use_container_width=True, height=600)
    else:
        st.warning("⚠️ Şu an kriterlere uyan sinyal bulunamadı. Lütfen piyasayı takipte kalın.")
