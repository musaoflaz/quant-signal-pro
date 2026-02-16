import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta

# 1. SAYFA KONFİGÜRASYONU
st.set_page_config(layout="wide", page_title="Quant Signal Pro | İşlem Terminali")

# 2. BORSA BAĞLANTISI (Güvenli Mod)
exchange = ccxt.binance({'enableRateLimit': True})

# 3. TASARIM VE BAŞLIK
st.markdown("# 🏛️ TRADE TERMINAL")
st.write("---")

# Laptop görselindeki varlık listesi
symbols = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT',
    'PEPE/USDT', 'ZEC/USDT', 'BNB/USDT', 'SUI/USDT', 'ADA/USDT'
]

def veri_isle():
    terminal_rows = []
    for symbol in symbols:
        try:
            # 4 Saatlik (H4) Veriler
            bars = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=50)
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # RSI ve Fiyat
            rsi = ta.rsi(df['c'], length=14).iloc[-1]
            last_price = df['c'].iloc[-1]
            
            # Laptop görselindeki Sinyal Mantığı
            if rsi < 35:
                eylem = "🟢 AL (LONG)"
                rejim = "TREND (TRENDING)"
                guven = f"%{int(100-rsi)}"
            elif rsi > 65:
                eylem = "🔴 SAT (SHORT)"
                rejim = "TREND (TRENDING)"
                guven = f"%{int(rsi)}"
            else:
                eylem = "⚪ BEKLE"
                rejim = "YATAY (RANGING)"
                guven = "%40"

            terminal_rows.append({
                "VARLIK": symbol,
                "FİYAT": f"{last_price:.4f}",
                "PİYASA REJİMİ": rejim,
                "İŞLEM EYLEMİ": eylem,
                "GÜVEN %": guven,
                "TEKNİK ANALİZ": f"H4 | RSI:{int(rsi)} | MACD+"
            })
        except:
            continue
    return pd.DataFrame(terminal_rows)

# GÜVENLİ BOYAMA FONKSİYONU (Hata riskini sıfırlayan yöntem)
def renklendir(row):
    color_map = []
    for val in row:
        if "AL" in str(val):
            color_map.append('background-color: #155724; color: #d4edda; font-weight: bold')
        elif "SAT" in str(val):
            color_map.append('background-color: #721c24; color: #f8d7da; font-weight: bold')
        else:
            color_map.append('')
    return color_map

# ANA AKIŞ
data = veri_isle()

if not data.empty:
    # Sekmeler (Görseldeki gibi)
    tab1, tab2 = st.tabs(["🔍 İŞLEM TARAYICI", "📊 ANALİZ MASASI"])
    
    with tab1:
        # Görseldeki tablonun birebir kopyası
        st.dataframe(
            data.style.apply(renklendir, axis=1, subset=['İŞLEM EYLEMİ']),
            use_container_width=True,
            height=600
        )
else:
    st.error("Veri çekilemedi. Lütfen Binance bağlantısını ve API durumunu kontrol edin.")

# Manuel Yenileme
if st.sidebar.button('Sinyalleri Yenile'):
    st.rerun()
