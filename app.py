import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

st.set_page_config(layout="wide", page_title="Alpha Sniper | Dynamic Opportunity")

# KuCoin: Sorunsuz bağlantı
exchange = ccxt.kucoin({'enableRateLimit': True, 'timeout': 60000})

st.title("🏛️ QUANT ALPHA: DİNAMİK FIRSAT TAYİNİ")
st.info("Piyasa yönüne bakılmaksızın (Long/Short) en güçlü kesişimler taranıyor.")

def dynamic_scanner():
    results = []
    # Geniş ve hacimli liste
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'AVAX/USDT', 
        'DOGE/USDT', 'LINK/USDT', 'NEAR/USDT', 'ADA/USDT', 'DOT/USDT',
        'LTC/USDT', 'SHIB/USDT', 'TRX/USDT', 'PEPE/USDT', 'SUI/USDT'
    ]
    
    progress = st.progress(0)
    
    for idx, symbol in enumerate(symbols):
        try:
            # 1 Saatlik Veri
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=150)
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # Gösterge Hesaplamaları
            df['EMA200'] = ta.ema(df['c'], length=200)
            stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
            df = pd.concat([df, stoch], axis=1)
            rsi_val = ta.rsi(df['c'], length=14).iloc[-1]
            
            l, p = df.iloc[-1], df.iloc[-2]
            sk, sd = "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"
            
            skor = 0
            komut = "BEKLE ⌛"
            
            # --- GELİŞMİŞ GİRİŞ MANTIĞI ---
            
            # 1. SENARYO: LONG (Yükseliş Onayı)
            if l['c'] > l['EMA200']: # Fiyat Trend Üstünde
                skor += 40
                if p[sk] < p[sd] and l[sk] > l[sd]: # Yukarı Kesişim (Net Giriş)
                    skor += 50
                if rsi_val < 60: # Hala alan var mı?
                    skor += 10
                
                if skor >= 90: komut = "🚀 LONG GİR"
                elif skor >= 40: komut = "📈 LONG PUSU"

            # 2. SENARYO: SHORT (Düşüş Onayı)
            elif l['c'] < l['EMA200']: # Fiyat Trend Altında
                skor += 40
                if p[sk] > p[sd] and l[sk] < l[sd]: # Aşağı Kesişim (Net Satış)
                    skor += 50
                if rsi_val > 40: # Çok mu düştü?
                    skor += 10
                
                if skor >= 90: komut = "💥 SHORT GİR"
                elif skor >= 40: komut = "📉 SHORT PUSU"

            results.append({
                "COIN": symbol,
                "FİYAT": f"{l['c']:.4f}",
                "EYLEM": komut,
                "SKOR": skor,
                "RSI": int(rsi_val)
            })
            time.sleep(0.1)
        except: continue
        progress.progress((idx + 1) / len(symbols))
    
    return pd.DataFrame(results)

if st.button('🎯 FIRSATLARI TARA (V29)'):
    data = dynamic_scanner()
    if not data.empty:
        def style_rows(row):
            if row['SKOR'] >= 90:
                color = '#0c3e1e' if "LONG" in row['EYLEM'] else '#4b0a0a'
                return [f'background-color: {color}; color: white; font-weight: bold'] * len(row)
            return [''] * len(row)

        st.subheader("📊 Canlı Sinyal Paneli")
        st.dataframe(data.sort_values('SKOR', ascending=False).style.apply(style_rows, axis=1), use_container_width=True)
    else:
        st.error("Veri hattı meşgul, tekrar dene.")
