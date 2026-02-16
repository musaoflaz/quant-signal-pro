import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

# 1. Sayfa Ayarları
st.set_page_config(layout="wide", page_title="Quant Pro | Trend & Signal Tracker")

# 2. Borsa Bağlantısı (Stabilite için Gate.io)
exchange = ccxt.gateio({'enableRateLimit': True, 'timeout': 30000})

st.markdown("# 🏛️ QUANT PRO - TREND VE SİNYAL TERMİNALİ")
st.write("---")

# İzleme Listesi
symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT', 'SUI/USDT', 'AVAX/USDT', 'LINK/USDT', 'PEPE/USDT']

def trend_ve_sinyal_analizi():
    rows = []
    progress = st.progress(0)
    
    for idx, symbol in enumerate(symbols):
        try:
            # Hem H4 (Ana Trend) hem H1 (Giriş Sinyali) verisi çekilebilir ancak stabilite için H1 üzerinden gidelim
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=200)
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # --- TREND İNDİKATÖRLERİ ---
            # 1. EMA 200 (Ana Yön)
            df['EMA200'] = ta.ema(df['close'], length=200)
            # 2. ADX (Trendin Gücü: 25 üstü güçlü trenddir)
            adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
            df = pd.concat([df, adx_df], axis=1)
            # 3. SuperTrend (Yön Takibi)
            st_df = ta.supertrend(df['high'], df['low'], df['close'], length=7, multiplier=3)
            df = pd.concat([df, st_df], axis=1)
            # 4. RSI (Giriş Zamanlaması)
            df['RSI'] = ta.rsi(df['close'], length=14)
            
            last = df.iloc[-1]
            last_close = last['c']
            last_rsi = last['RSI']
            ema200 = last['EMA200']
            adx = last['ADX_14']
            st_direction = last['SUPERTd_7_3.0'] # 1 ise Boğa, -1 ise Ayı
            
            # --- TREND VE SİNYAL MANTIĞI ---
            trend_durumu = ""
            if last_close > ema200 and st_direction == 1:
                trend_durumu = "📈 GÜÇLÜ BOĞA"
            elif last_close < ema200 and st_direction == -1:
                trend_durumu = "📉 GÜÇLÜ AYI"
            elif last_close > ema200:
                trend_durumu = "↗️ YUKARI"
            else:
                trend_durumu = "↘️ AŞAĞI"
            
            # Sinyal Üretimi
            eylem = "BEKLE"
            if trend_durumu.startswith("📈") and last_rsi < 40:
                eylem = "🔥 TREND LONG"
            elif trend_durumu.startswith("📉") and last_rsi > 60:
                eylem = "💥 TREND SHORT"
            
            # Trend Gücü Notu
            guc_notu = "Zayıf"
            if adx > 25: guc_notu = "Güçlü"
            if adx > 40: guc_notu = "Çok Güçlü"

            rows.append({
                "VARLIK": symbol,
                "FİYAT": f"{last_close:.4f}",
                "TREND YÖNÜ": trend_durumu,
                "TREND GÜCÜ": guc_notu,
                "EYLEM": eylem,
                "RSI": int(last_rsi)
            })
            time.sleep(0.1)
        except:
            continue
        progress.progress((idx + 1) / len(symbols))
    
    progress.empty()
    return pd.DataFrame(rows)

# Stil Fonksiyonu
def style_results(val):
    if "LONG" in str(val): return 'background-color: #004d1a; color: white; font-weight: bold'
    if "SHORT" in str(val): return 'background-color: #4d0000; color: white; font-weight: bold'
    return ''

# Tabloyu Bas
data = trend_ve_sinyal_analizi()

if not data.empty:
    st.dataframe(
        data.style.applymap(style_results, subset=['EYLEM']),
        use_container_width=True,
        height=600
    )
else:
    st.error("Borsa verileri işlenirken bir hata oluştu.")

if st.sidebar.button('🔄 Trendleri Tara'):
    st.rerun()
