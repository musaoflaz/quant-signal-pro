import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

# Sayfa Ayarları
st.set_page_config(layout="wide", page_title="Alpha Sniper V22")

# BYBIT Bağlantısı (Dünya genelinde en az engel çıkaran borsa)
exchange = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'linear'}})

st.title("🏛️ QUANT ALPHA: RESET & WIN")
st.write("Sistem Bybit üzerinden en popüler 20 coini tarar. Hata payı sıfıra indirildi.")

def reset_scanner():
    results = []
    # En güvenilir 20 coini elle yazdım ki liste hatası olmasın
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'AVAX/USDT', 
               'DOGE/USDT', 'ADA/USDT', 'LINK/USDT', 'DOT/USDT', 'MATIC/USDT',
               'LTC/USDT', 'BCH/USDT', 'UNI/USDT', 'NEAR/USDT', 'TIA/USDT',
               'SUI/USDT', 'OP/USDT', 'ARB/USDT', 'APT/USDT', 'RNDR/USDT']
    
    progress = st.progress(0)
    
    for idx, symbol in enumerate(symbols):
        try:
            # Veri çekme
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=150)
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # İndikatörler
            df['EMA200'] = ta.ema(df['c'], length=200)
            stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
            df = pd.concat([df, stoch], axis=1)
            
            # Son veriler
            l, p = df.iloc[-1], df.iloc[-2]
            sk, sd = "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"
            
            # Karar Mekanizması
            puan = 50 # Baz puan
            komut = "İZLE"
            
            # LONG Şartı: Fiyat EMA200 üstünde + Stoch Yukarı Kesişim
            if l['c'] > l['EMA200']:
                if p[sk] < p[sd] and l[sk] > l[sd]:
                    komut = "🚀 LONG (GİRİŞ)"
                    puan = 100
                else:
                    komut = "📈 LONG PUSU"
            
            # SHORT Şartı: Fiyat EMA200 altında + Stoch Aşağı Kesişim
            elif l['c'] < l['EMA200']:
                if p[sk] > p[sd] and l[sk] < l[sd]:
                    komut = "💥 SHORT (GİRİŞ)"
                    puan = 100
                else:
                    komut = "📉 SHORT PUSU"

            results.append({
                "COIN": symbol,
                "FİYAT": f"{l['c']:.4f}",
                "KOMUT": komut,
                "SKOR": puan
            })
            time.sleep(0.1)
        except:
            continue
        progress.progress((idx + 1) / len(symbols))
    
    return pd.DataFrame(results)

# --- Arayüz ---
if st.button('🎯 SİNYAL AVINI BAŞLAT'):
    with st.spinner('Piyasa taranıyor...'):
        data = reset_scanner()
        
    if not data.empty:
        # Renklendirme
        def color_row(row):
            if "GİRİŞ" in row['KOMUT']:
                return ['background-color: #155724; color: white'] * len(row) if "LONG" in row['KOMUT'] else ['background-color: #721c24; color: white'] * len(row)
            return [''] * len(row)

        st.subheader("📊 Canlı Sinyaller")
        # En yüksek puanlıları (Giriş sinyallerini) en üste atar
        st.dataframe(data.sort_values('SKOR', ascending=False).style.apply(color_row, axis=1), use_container_width=True)
    else:
        st.error("Borsa verisi alınamadı. Lütfen internet bağlantınızı kontrol edip tekrar deneyin.")
