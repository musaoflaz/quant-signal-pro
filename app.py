import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

# Sayfa Ayarları
st.set_page_config(layout="wide", page_title="Alpha Sniper | KuCoin Gold")

# KuCoin Bağlantısı
exchange = ccxt.kucoin({
    'enableRateLimit': True, 
    'timeout': 60000
})

st.title("🏛️ QUANT ALPHA: KUCOIN GOLD SHIELD")
st.markdown("---")

def get_kucoin_signals():
    results = []
    # KuCoin'deki en hacimli ve kaldıraca uygun majörler
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'AVAX/USDT', 
        'DOGE/USDT', 'LINK/USDT', 'NEAR/USDT', 'ADA/USDT', 'DOT/USDT',
        'LTC/USDT', 'SHIB/USDT', 'TRX/USDT', 'UNI/USDT', 'PEPE/USDT'
    ]
    
    progress = st.progress(0)
    status_text = st.empty()
    
    for idx, symbol in enumerate(symbols):
        status_text.text(f"🔍 Analiz Ediliyor: {symbol}")
        try:
            # 1 Saatlik Veri (Stratejimiz için en kararlı zaman dilimi)
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=200)
            if not bars: continue
            
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # 🛡️ 3'LÜ ONAY MEKANİZMASI
            # 1. EMA 200 (Ana Trend)
            df['EMA200'] = ta.ema(df['c'], length=200)
            # 2. Stochastic RSI (Giriş Zamanlaması)
            stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
            df = pd.concat([df, stoch], axis=1)
            # 3. RSI (Aşırı Alım/Satım Kontrolü)
            df['RSI'] = ta.rsi(df['c'], length=14)
            
            l, p = df.iloc[-1], df.iloc[-2]
            sk, sd = "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"
            
            skor = 0
            komut = "İZLE ⌛"
            
            # --- LONG SKORLAMA ---
            if l['c'] > l['EMA200']: # Trend Yukarısı
                skor += 50
                if p[sk] < p[sd] and l[sk] > l[sd]: # Altın Kesişim
                    skor += 40
                if l['RSI'] < 65: # Şişmemişlik Bonusu
                    skor += 10
                
                if skor >= 90: komut = "🚀 GÜÇLÜ LONG"
                elif skor >= 50: komut = "📈 LONG PUSU"

            # --- SHORT SKORLAMA ---
            elif l['c'] < l['EMA200']: # Trend Aşağısı
                skor += 50
                if p[sk] > p[sd] and l[sk] < l[sd]: # Ölüm Kesişimi
                    skor += 40
                if l['RSI'] > 35: # Dip Değil Bonusu
                    skor += 10
                
                if skor >= 90: komut = "💥 GÜÇLÜ SHORT"
                elif skor >= 50: komut = "📉 SHORT PUSU"

            results.append({
                "COIN": symbol,
                "FİYAT": f"{l['c']:.4f}",
                "KOMUT": komut,
                "SKOR": skor,
                "RSI": int(l['RSI'])
            })
            time.sleep(0.2)
        except:
            continue
        progress.progress((idx + 1) / len(symbols))
    
    status_text.empty()
    return pd.DataFrame(results)

# --- Arayüz ---
if st.button('🛡️ KUCOIN TARAMAYI BAŞLAT'):
    with st.spinner('Veriler borsadan çekiliyor...'):
        data = get_kucoin_signals()
        
    if not data.empty:
        # Renklendirme Mantığı
        def style_logic(row):
            style = ''
            if row['SKOR'] >= 90:
                style = 'background-color: #0c3e1e; color: #52ff8f; font-weight: bold' if "LONG" in row['KOMUT'] else 'background-color: #4b0a0a; color: #ff6e6e; font-weight: bold'
            return [style] * len(row)

        st.subheader("🎯 Canlı İşlem Sinyalleri")
        # En yüksek skorları en üste al
        st.dataframe(
            data.sort_values('SKOR', ascending=False).style.apply(style_rows=style_logic, axis=1), 
            use_container_width=True,
            height=600
        )
    else:
        st.error("Şu an borsa verisi çekilemedi. Lütfen tekrar deneyin.")

st.markdown("---")
st.caption("Not: Skor 90+ ise terste kalma riskiniz daha düşüktür. Her zaman Stop-Loss kullanın.")
