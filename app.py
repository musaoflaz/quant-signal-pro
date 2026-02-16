import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

# 1. Sayfa Ayarları
st.set_page_config(layout="wide", page_title="Quant Alpha | Smart Signal Filter")

# 2. Borsa Bağlantısı
exchange = ccxt.kucoin({'enableRateLimit': True, 'timeout': 30000})

st.markdown("# 🏛️ QUANT ALPHA: AKILLI SİNYAL FİLTRESİ")
st.write("---")

def get_symbols():
    try:
        tickers = exchange.fetch_tickers()
        df_t = pd.DataFrame.from_dict(tickers, orient='index')
        df_t = df_t[df_t['symbol'].str.contains('/USDT')]
        # En hacimli 40 coin (Piyasayı temsil eder)
        return df_t.sort_values('quoteVolume', ascending=False).head(40).index.tolist()
    except:
        return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT', 'SUI/USDT']

def smart_scanner():
    symbols = get_symbols()
    results = []
    
    progress = st.progress(0)
    status = st.empty()
    
    for idx, symbol in enumerate(symbols):
        status.info(f"🔍 Kriter Denetimi Yapılıyor: **{symbol}**")
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=150)
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # --- TEKNİK ANALİZ ---
            df['EMA200'] = ta.ema(df['c'], length=200)
            df['RSI'] = ta.rsi(df['c'], length=14)
            stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
            df = pd.concat([df, stoch], axis=1)
            
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Değişkenler
            c, rsi, ema = last['c'], last['RSI'], last['EMA200']
            k, d = last['STOCHRSIk_14_14_3_3'], last['STOCHRSId_14_14_3_3']
            pk, pd_val = prev['STOCHRSIk_14_14_3_3'], prev['STOCHRSId_14_14_3_3']
            
            # --- KRİTER DENETİMİ (Senin Referansın) ---
            durum = "KRİTER DIŞI"
            skor = 0
            
            # LONG KRİTERİ: Trend Üstü + Stoch Kesişimi
            if c > ema:
                if pk < pd_val and k > d:
                    durum = "🚀 GÜÇLÜ AL (LONG)"
                    skor = 90
                elif k > d:
                    durum = "🟢 TREND YUKARI (GÖZLEM)"
                    skor = 60
            
            # SHORT KRİTERİ: Trend Altı + Stoch Kesişimi
            elif c < ema:
                if pk > pd_val and k < d:
                    durum = "💥 GÜÇLÜ SAT (SHORT)"
                    skor = 90
                elif k < d:
                    durum = "🔴 TREND AŞAĞI (GÖZLEM)"
                    skor = 60

            results.append({
                "COIN": symbol,
                "FİYAT": f"{c:.4f}",
                "ANALİZ SONUCU": durum,
                "SKOR": skor,
                "RSI": int(rsi)
            })
            time.sleep(0.2)
        except: continue
        progress.progress((idx + 1) / len(symbols))
    
    status.empty()
    progress.empty()
    
    # Sonuçları listele (Önce en yüksek skorlar)
    df_res = pd.DataFrame(results).sort_values('SKOR', ascending=False)
    return df_res

# --- Arayüz ---
if st.button('🎯 PİYASAYI TARA VE ANALİZ ET'):
    data = smart_scanner()
    
    if not data.empty:
        # Sinyalleri ayır (Kriterlere uyanlar ve uymayanlar)
        guclu_sinyaller = data[data['SKOR'] >= 80]
        gozlem_listesi = data[(data['SKOR'] < 80) & (data['SKOR'] > 0)]
        
        if not guclu_sinyaller.empty:
            st.subheader("🔥 KRİTERLERE TAM UYAN SİNYALLER")
            st.success(f"{len(guclu_sinyaller)} adet güçlü fırsat yakalandı!")
            st.table(guclu_sinyaller)
        else:
            st.warning("Şu an kriterlerine (EMA + Stoch Kesişimi) tam uyan bir fırsat yok.")

        if not gozlem_listesi.empty:
            st.subheader("👀 GÖZLEM LİSTESİ (Potansiyel Trendler)")
            st.dataframe(gozlem_listesi, use_container_width=True)
            
    else:
        st.error("Veri çekilemedi, lütfen tekrar deneyin.")
