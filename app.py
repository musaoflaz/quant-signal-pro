import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

# Sayfa Ayarları
st.set_page_config(layout="wide", page_title="Quant Alpha | Final V11")

# Borsa Bağlantısı (KuCoin - Stabil)
exchange = ccxt.kucoin({'enableRateLimit': True, 'timeout': 30000})

st.markdown("# 🏛️ QUANT ALPHA: AKILLI ANALİZ TERMİNALİ")
st.write("---")

def get_symbols():
    try:
        tickers = exchange.fetch_tickers()
        df_t = pd.DataFrame.from_dict(tickers, orient='index')
        # Sadece USDT çiftleri ve hacmi en yüksek 40 coini al
        df_t = df_t[df_t['symbol'].str.contains('/USDT')]
        return df_t.sort_values('quoteVolume', ascending=False).head(40).index.tolist()
    except:
        return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT', 'AVAX/USDT']

def final_scanner():
    symbols = get_symbols()
    results = []
    
    progress = st.progress(0)
    status = st.empty()
    
    for idx, symbol in enumerate(symbols):
        status.info(f"🔍 Kriter Denetimi: **{symbol}**")
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=150)
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # Teknik Analiz
            df['EMA200'] = ta.ema(df['c'], length=200)
            df['RSI'] = ta.rsi(df['c'], length=14)
            stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
            df = pd.concat([df, stoch], axis=1)
            
            l, p = df.iloc[-1], df.iloc[-2]
            
            # --- KRİTERLER (Senin Lokal Başarı Referansın) ---
            score = 0
            label = "GÖZLEM"
            
            # Kesişim ve Trend Kontrolü
            up_cross = p['STOCHRSIk_14_14_3_3'] < p['STOCHRSId_14_14_3_3'] and l['STOCHRSIk_14_14_3_3'] > l['STOCHRSId_14_14_3_3']
            down_cross = p['STOCHRSIk_14_14_3_3'] > p['STOCHRSId_14_14_3_3'] and l['STOCHRSIk_14_14_3_3'] < l['STOCHRSId_14_14_3_3']

            if l['c'] > l['EMA200'] and up_cross:
                label = "🚀 GÜÇLÜ AL (LONG)"
                score = 90
            elif l['c'] < l['EMA200'] and down_cross:
                label = "💥 GÜÇLÜ SAT (SHORT)"
                score = 90
            elif l['c'] > l['EMA200']:
                label = "🟢 TREND YUKARI"
                score = 50
            elif l['c'] < l['EMA200']:
                label = "🔴 TREND AŞAĞI"
                score = 50

            results.append({
                "COIN": symbol,
                "FİYAT": f"{l['c']:.4f}",
                "DURUM": label,
                "SKOR": score,
                "RSI": int(l['RSI'])
            })
            time.sleep(0.3)
        except: continue
        progress.progress((idx + 1) / len(symbols))
    
    status.empty()
    progress.empty()
    
    # Boş liste kontrolü (KeyError'u engelleyen kritik nokta)
    if not results:
        return pd.DataFrame()
    
    df_res = pd.DataFrame(results)
    if 'SKOR' in df_res.columns:
        return df_res.sort_values(by='SKOR', ascending=False)
    return df_res

# --- Arayüz Kontrolü ---
if st.button('🎯 PİYASAYI ANALİZ ET'):
    data = final_scanner()
    
    if not data.empty:
        # 1. Gerçek Sinyaller (Skor 90 olanlar)
        if 'SKOR' in data.columns:
            signals = data[data['SKOR'] >= 80]
            if not signals.empty:
                st.subheader("🔥 KRİTERLERE TAM UYAN SİNYALLER")
                st.success(f"{len(signals)} adet fırsat yakalandı!")
                st.table(signals[['COIN', 'FİYAT', 'DURUM', 'RSI']])
            else:
                st.warning("⚠️ Şu an senin kriterlerine (Trend + Kesişim) tam uyan bir giriş sinyali yok.")

        # 2. Genel Sıralama (Gözlem Listesi)
        st.write("---")
        st.subheader("👀 TÜM PİYASA DURUMU (TOP 20)")
        
        def color_map(val):
            if "GÜÇLÜ" in str(val): return 'background-color: #1a4d2e; color: #52ff8f; font-weight: bold'
            return ''
        
        st.dataframe(data.head(20).style.applymap(color_map, subset=['DURUM']), use_container_width=True)
    else:
        st.error("Veriler alınırken bir sorun oluştu veya borsa yanıt vermedi. Lütfen tekrar deneyin.")
