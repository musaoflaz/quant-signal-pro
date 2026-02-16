import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import plotly.graph_objects as go

# --- AYARLAR VE GÜVENLİ BAĞLANTI ---
st.set_page_config(layout="wide", page_title="Quant Signal Pro")

# Binance bağlantısını aşırı yüklenmeye (rate limit) karşı korumalı kuralım
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

def fetch_data(symbol='BTC/USDT', timeframe='1h'):
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
        if not bars:
            return pd.DataFrame()
            
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Teknik Göstergeler (Orijinal mantığın)
        df['RSI'] = ta.rsi(df['close'], length=14)
        
        # HATA ALAN SÜTUN İSMİNİ SABİTLEYELİM:
        # Kodun aradığı 'İŞLEM EYLEMİ' ismini burada tanımlıyoruz.
        df['İŞLEM EYLEMİ'] = 'BEKLE'
        df.loc[df['RSI'] < 30, 'İŞLEM EYLEMİ'] = 'AL'
        df.loc[df['RSI'] > 70, 'İŞLEM EYLEMİ'] = 'SAT'
        
        return df.dropna()
    except:
        return pd.DataFrame()

# Renklendirme Fonksiyonu
def style_action_color(val):
    if val == 'AL': return 'background-color: green; color: white'
    if val == 'SAT': return 'background-color: red; color: white'
    return ''

# --- ARAYÜZ ---
st.title("📊 Quant Signal Pro")

tab1, tab2 = st.tabs(["🔍 İŞLEM TARAYICI", "📈 ANALİZ MASASI"])

with tab1:
    df = fetch_data()
    
    if not df.empty:
        try:
            # En yeni veriyi en üstte görmek için tabloyu ters çeviriyoruz (iloc[::-1])
            st.subheader("Canlı Sinyaller")
            st.dataframe(
                df.iloc[::-1].style.applymap(style_action_color, subset=['İŞLEM EYLEMİ']), 
                height=600, 
                use_container_width=True
            )
        except Exception as e:
            # Beklenmedik bir isimlendirme hatasında tabloyu sade göster, çökme.
            st.dataframe(df.iloc[::-1], height=600, use_container_width=True)
    else:
        st.warning("Veri çekiliyor veya Binance şu an yanıt vermiyor. Lütfen birkaç saniye bekleyip sayfayı yenileyin.")
        if st.button('Verileri Tekrar Yükle'):
            st.rerun()

with tab2:
    if not df.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=df['timestamp'],
            open=df['open'], high=df['high'],
            low=df['low'], close=df['close']
        )])
        fig.update_layout(title="BTC/USDT Grafik", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
