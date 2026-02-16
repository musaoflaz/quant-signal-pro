import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

st.set_page_config(layout="wide", page_title="Quant Alpha | Trade Commands")

# Borsa Bağlantısı
exchange = ccxt.kucoin({'enableRateLimit': True, 'timeout': 60000})

st.markdown("# 🏛️ QUANT ALPHA: İŞLEM TERMİNALİ")
st.write("---")

def get_symbols():
    try:
        tickers = exchange.fetch_tickers()
        df_t = pd.DataFrame.from_dict(tickers, orient='index')
        df_t = df_t[df_t['symbol'].str.contains('/USDT')]
        return df_t.sort_values('quoteVolume', ascending=False).head(30).index.tolist()
    except:
        return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']

def command_scanner():
    symbols = get_symbols()
    results = []
    progress = st.progress(0)
    
    for idx, symbol in enumerate(symbols):
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=250)
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # İndikatörler
            df['EMA200'] = ta.ema(df['c'], length=200)
            stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
            df = pd.concat([df, stoch], axis=1)
            
            l, p = df.iloc[-1], df.iloc[-2]
            sk, sd = "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"
            
            # --- NET KOMUT MANTIĞI ---
            komut = "⌛ BEKLE"
            aciklama = "Sinyal Bekleniyor"
            renk_kodu = 0 # 0:Nötr, 1:Long, 2:Short
            
            # LONG Şartı: EMA Üstü + Stoch Yukarı Kesişim
            if l['c'] > l['EMA200']:
                if p[sk] < p[sd] and l[sk] > l[sd]:
                    komut = "🚀 LONG (ALIM YAP)"
                    aciklama = "Trend Pozitif + Alım Kesişimi Geldi"
                    renk_kodu = 1
                else:
                    komut = "📈 LONG YÖNLÜ İZLE"
                    aciklama = "Fiyat EMA200 üzerinde, kesişim bekleniyor"
            
            # SHORT Şartı: EMA Altı + Stoch Aşağı Kesişim
            elif l['c'] < l['EMA200']:
                if p[sk] > p[sd] and l[sk] < l[sd]:
                    komut = "💥 SHORT (SATIŞ YAP)"
                    aciklama = "Trend Negatif + Satış Kesişimi Geldi"
                    renk_kodu = 2
                else:
                    komut = "📉 SHORT YÖNLÜ İZLE"
                    aciklama = "Fiyat EMA200 altında, kesişim bekleniyor"

            results.append({
                "COIN": symbol,
                "FİYAT": f"{l['c']:.4f}",
                "EYLEM / KOMUT": komut,
                "NEDEN": aciklama,
                "RENK": renk_kodu
            })
            time.sleep(0.4)
        except: continue
        progress.progress((idx + 1) / len(symbols))
    
    return pd.DataFrame(results)

# --- Arayüz ve Görselleştirme ---
if st.button('🎯 PİYASAYI TARA VE KOMUTLARI AL'):
    data = command_scanner()
    
    if not data.empty:
        # Renklendirme Fonksiyonu
        def style_commands(row):
            bg = ''
            if "🚀 LONG" in row['EYLEM / KOMUT']: bg = 'background-color: #0c3e1e; color: #52ff8f; font-weight: bold'
            elif "💥 SHORT" in row['EYLEM / KOMUT']: bg = 'background-color: #4b0a0a; color: #ff6e6e; font-weight: bold'
            return [bg]*len(row)

        # Tabloyu göster
        st.subheader("📊 Güncel İşlem Sinyalleri")
        st.dataframe(
            data.sort_values('RENK', ascending=False).drop(columns=['RENK']).style.apply(style_commands, axis=1),
            use_container_width=True,
            height=600
        )
    else:
        st.error("Veri alınamadı, tekrar deneyin.")
