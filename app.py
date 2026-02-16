import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

# Sayfa Ayarları
st.set_page_config(layout="wide", page_title="Quant Alpha | Mega Scanner V7")

# Borsa Bağlantısı (Stabilite için optimize edildi)
exchange = ccxt.kucoin({
    'enableRateLimit': True, 
    'timeout': 30000,
    'options': {'adjustForTimeDifference': True}
})

st.markdown("# 🏛️ QUANT ALPHA: TOP 20 OPPORTUNITIES")
st.info("Piyasa taranıyor... En yüksek skorlu sinyaller listelenecek.")

def get_filtered_symbols():
    try:
        tickers = exchange.fetch_tickers()
        df_tickers = pd.DataFrame.from_dict(tickers, orient='index')
        # Sadece USDT çiftleri ve hacmi en yüksek 40 coini al (Hız ve güvenlik dengesi)
        df_tickers = df_tickers[df_tickers['symbol'].str.contains('/USDT')]
        top_40 = df_tickers.sort_values('quoteVolume', ascending=False).head(40).index.tolist()
        return top_40
    except:
        return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT', 'AVAX/USDT', 'SUI/USDT', 'PEPE/USDT']

def alpha_scanner():
    symbols = get_filtered_symbols()
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, symbol in enumerate(symbols):
        status_text.info(f"🔍 Analiz Ediliyor ({idx+1}/{len(symbols)}): **{symbol}**")
        try:
            # Veri Çekme (H1)
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=150)
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # --- PROFESYONEL İNDİKATÖR SETİ (Referans Modu) ---
            df['EMA200'] = ta.ema(df['c'], length=200)
            df['RSI'] = ta.rsi(df['c'], length=14)
            stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
            df = pd.concat([df, stoch], axis=1)
            df['VOL_AVG'] = df['v'].rolling(window=20).mean()
            
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            # --- SKORLAMA MOTORU ---
            score = 0
            yon = "NÖTR"
            
            # Trend ve Momentum Onayı
            ema_onay = last['c'] > last['EMA200']
            stoch_cross_up = prev['STOCHRSIk_14_14_3_3'] < prev['STOCHRSId_14_14_3_3'] and last['STOCHRSIk_14_14_3_3'] > last['STOCHRSId_14_14_3_3']
            stoch_cross_down = prev['STOCHRSIk_14_14_3_3'] > prev['STOCHRSId_14_14_3_3'] and last['STOCHRSIk_14_14_3_3'] < last['STOCHRSId_14_14_3_3']
            
            # LONG Skoru
            if ema_onay: score += 30
            if stoch_cross_up: score += 40
            if last['RSI'] < 45: score += 20
            if last['v'] > last['VOL_AVG']: score += 10
            
            # SHORT Skoru (Eğer trend aşağıysa)
            s_score = 0
            if not ema_onay: s_score += 30
            if stoch_cross_down: s_score += 40
            if last['RSI'] > 55: s_score += 20
            if last['v'] > last['VOL_AVG']: s_score += 10

            final_score = score if score >= s_score else s_score
            yon = "LONG" if score >= s_score else "SHORT"
            
            if final_score >= 50: # Sadece kayda değer sinyalleri ekle
                results.append({
                    "SYMBOL": symbol,
                    "FİYAT": f"{last['c']:.4f}",
                    "YÖN": yon,
                    "GÜVEN SKORU": final_score,
                    "RSI": int(last['RSI']),
                    "HACİM": "GÜÇLÜ" if last['v'] > last['VOL_AVG'] else "ZAYIF"
                })
            
            time.sleep(0.4) # API Ban koruması
        except:
            continue
        progress_bar.progress((idx + 1) / len(symbols))
        
    status_text.empty()
    progress_bar.empty()
    
    # En yüksek skorlu 20 sinyali getir
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        return res_df.sort_values('GÜVEN SKORU', ascending=False).head(20)
    return pd.DataFrame()

# Görsel Stil
def color_rows(row):
    color = ''
    if row['YÖN'] == 'LONG' and row['GÜVEN SKORU'] >= 70: color = 'background-color: #0c3e1e; color: #52ff8f'
    elif row['YÖN'] == 'SHORT' and row['GÜVEN SKORU'] >= 70: color = 'background-color: #4b0a0a; color: #ff6e6e'
    return [color]*len(row)

# Tarayıcıyı Çalıştır
top_20 = alpha_scanner()

if not top_20.empty:
    st.dataframe(top_20.style.apply(color_rows, axis=1), use_container_width=True, height=650)
    st.success(f"🎯 Tarama Tamamlandı. En yüksek olasılıklı {len(top_20)} sinyal listelendi.")
else:
    st.warning("⚠️ Şu an kriterlere uygun güçlü bir sinyal bulunamadı. Lütfen 5 dk sonra tekrar deneyin.")

if st.sidebar.button('🔄 Tüm Piyasayı Taramayı Başlat'):
    st.rerun()
