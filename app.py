import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

st.set_page_config(layout="wide", page_title="Quant Alpha | Mega Scanner")

# Borsa Bağlantısı
exchange = ccxt.kucoin({'enableRateLimit': True, 'timeout': 60000})

st.markdown("# 🏛️ QUANT ALPHA: TOP 20 OPPORTUNITIES")
st.info("Piyasa taranıyor, en yüksek skorlu 20 sinyal listeleniyor...")

def get_top_symbols():
    try:
        markets = exchange.fetch_tickers()
        # Sadece USDT çiftlerini ve hacmi yüksek olanları al
        df_m = pd.DataFrame.from_dict(markets, orient='index')
        df_m = df_m[df_m['symbol'].str.contains('/USDT')]
        # En yüksek hacimli ilk 50-60 coini tara (Hız ve IP engeli için dengeli rakam)
        top_hacim = df_m.sort_values('quoteVolume', ascending=False).head(60).index.tolist()
        return top_hacim
    except:
        return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT', 'SUI/USDT', 'AVAX/USDT']

def scan_and_score():
    symbols = get_top_symbols()
    all_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, symbol in enumerate(symbols):
        status_text.write(f"🔍 Analiz Ediliyor: {symbol}")
        try:
            # Veri çekme (H1)
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=150)
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # Göstergeler
            df['EMA200'] = ta.ema(df['c'], length=200)
            df['RSI'] = ta.rsi(df['c'], length=14)
            stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
            df = pd.concat([df, stoch], axis=1)
            df['VOL_AVG'] = df['v'].rolling(window=20).mean()
            
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Değerler
            c, rsi, ema = last['c'], last['RSI'], last['EMA200']
            k, d = last['STOCHRSIk_14_14_3_3'], last['STOCHRSId_14_14_3_3']
            pk, pd_val = prev['STOCHRSIk_14_14_3_3'], prev['STOCHRSId_14_14_3_3']
            
            # --- SKORLAMA MANTIĞI ---
            long_score = 0
            short_score = 0
            
            # Trend Puanı
            if c > ema: long_score += 30
            else: short_score += 30
            
            # Momentum Puanı (Kesişim)
            if pk < pd_val and k > d: long_score += 40
            if pk > pd_val and k < d: short_score += 40
            
            # RSI & Hacim Puanı
            if rsi < 40: long_score += 20
            if rsi > 60: short_score += 20
            if last['v'] > last['VOL_AVG']: 
                long_score += 10
                short_score += 10

            final_score = long_score if long_score > short_score else short_score
            yon = "LONG" if long_score > short_score else "SHORT"
            
            all_results.append({
                "SYMBOL": symbol,
                "FİYAT": f"{c:.4f}",
                "YÖN": yon,
                "SKOR": final_score,
                "RSI": int(rsi),
                "VOL": "YÜKSEK" if last['v'] > last['VOL_AVG'] else "NORMAL"
            })
            
            time.sleep(0.2) # Rate limit koruması
        except:
            continue
        progress_bar.progress((idx + 1) / len(symbols))
        
    status_text.empty()
    progress_bar.empty()
    
    # Skorlara göre sırala ve en iyi 20'yi al
    full_df = pd.DataFrame(all_results)
    if not full_df.empty:
        return full_df.sort_values('SKOR', ascending=False).head(20)
    return full_df

# Tabloyu Renklendir
def highlight_signals(s):
    if s.YÖN == 'LONG' and s.SKOR >= 70:
        return ['background-color: #11381b; color: #52ff8f']*len(s)
    elif s.YÖN == 'SHORT' and s.SKOR >= 70:
        return ['background-color: #3b0d0d; color: #ff6e6e']*len(s)
    return ['']*len(s)

# Sonuçları Göster
top_20 = scan_and_score()

if not top_20.empty:
    st.dataframe(top_20.style.apply(highlight_signals, axis=1), use_container_width=True, height=700)
    st.success(f"✅ Piyasa tarandı. En potansiyelli {len(top_20)} sinyal yukarıda.")
else:
    st.warning("Veri işlenirken hata oluştu veya piyasa çok stabil.")

if st.sidebar.button('🔄 Tüm Piyasayı Yeniden Tara'):
    st.rerun()
