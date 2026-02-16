import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta

# 1. Sayfa Ayarları (Geniş ekran)
st.set_page_config(layout="wide", page_title="Quant Signal Pro | Multi-Exchange Terminal")

# 2. ALTERNATİF BORSA BAĞLANTISI (Bybit)
# Binance'de sorun varsa Bybit bulutta daha stabil çalışır.
exchange = ccxt.bybit({'enableRateLimit': True})

# 3. Başlık Tasarımı
st.markdown("# 🏛️ TRADE TERMINAL (Multi-Exchange)")
st.info("Veri Kaynağı: Bybit (Binance Alternatifi)")
st.write("---")

# Laptop görselindeki varlık listesi (Bybit uyumlu format)
symbols = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT',
    'PEPE/USDT', 'BNB/USDT', 'SUI/USDT', 'AVAX/USDT', 'LINK/USDT'
]

def veri_analizi_yedek():
    all_rows = []
    for symbol in symbols:
        try:
            # 4 Saatlik (H4) veriler
            bars = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=50)
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # RSI Hesaplama
            rsi = ta.rsi(df['c'], length=14).iloc[-1]
            last_price = df['c'].iloc[-1]
            
            # Lokalindeki Sinyal Mantığı
            if rsi < 35:
                eylem = "🟢 AL (LONG)"
                rejim = "TREND (UP)"
                guven = f"%{int(100-rsi)}"
            elif rsi > 65:
                eylem = "🔴 SAT (SHORT)"
                rejim = "TREND (DOWN)"
                guven = f"%{int(rsi)}"
            else:
                eylem = "⚪ BEKLE"
                rejim = "YATAY (RANGING)"
                guven = "%45"

            all_rows.append({
                "VARLIK": symbol,
                "FİYAT": f"{last_price:.4f}",
                "PİYASA REJİMİ": rejim,
                "İŞLEM EYLEMİ": eylem,
                "GÜVEN %": guven,
                "TEKNİK ANALİZ": f"H4 | RSI:{int(rsi)}"
            })
        except Exception as e:
            # Eğer bir borsa hata verirse diğerine geçmek için burayı kullanabiliriz
            continue
    return pd.DataFrame(all_rows)

# Renklendirme (Hata vermeyen güvenli metod)
def style_apply(val):
    if "AL" in str(val):
        return 'background-color: #0c3e1e; color: #52ff8f; font-weight: bold'
    if "SAT" in str(val):
        return 'background-color: #4b0a0a; color: #ff6e6e; font-weight: bold'
    return ''

# Veriyi çek ve göster
data = veri_analizi_yedek()

if not data.empty:
    # Sütun ismini sabit kullanarak KeyError hatasını önlüyoruz
    st.dataframe(
        data.style.map(style_apply, subset=['İŞLEM EYLEMİ']),
        use_container_width=True,
        height=650
    )
else:
    st.error("Alternatif borsalardan (Bybit/OKX) veri çekilemedi. Lütfen bağlantınızı kontrol edin.")

# Manuel Yenileme
if st.sidebar.button('Sinyalleri Yenile'):
    st.rerun()
