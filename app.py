import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import plotly.graph_objects as go

# Sayfa ayarları
st.set_page_config(layout="wide", page_title="Quant Signal Pro")

# Binance bağlantısı (Hız limitleri için korumalı)
exchange = ccxt.binance({'enableRateLimit': True})

def fetch_data(symbol='BTC/USDT', timeframe='1h'):
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # RSI Sinyal Hesaplama
        df['RSI'] = ta.rsi(df['close'], length=14)
        
        # HATA ALAN SÜTUN İSMİNİ BURADA TANIMLIYORUZ:
        # İsmi güvenli olması için 'SINYAL' yapıyoruz
        df['SINYAL'] = 'BEKLE'
        df.loc[df['RSI'] < 30, 'SINYAL'] = 'AL'
        df.loc[df['RSI'] > 70, 'SINYAL'] = 'SAT'
        
        return df.dropna()
    except:
        return pd.DataFrame()

def style_color(val):
    if val == 'AL': return 'background-color: green; color: white'
    if val == 'SAT': return 'background-color: red; color: white'
    return ''

st.title("📊 Quant Signal Pro")

tab1, tab2 = st.tabs(["🔍 İŞLEM TARAYICI", "📈 ANALİZ MASASI"])

with tab1:
    df = fetch_data()
    if not df.empty:
        try:
            # Sütun ismini 'SINYAL' olarak güncelleyerek renklendirme yapıyoruz
            st.dataframe(df.iloc[::-1].style.applymap(style_color, subset=['SINYAL']), height=600, use_container_width=True)
        except:
            # Eğer yine bir görsel hata olursa tabloyu sade göster, çökme
            st.dataframe(df.iloc[::-1], height=600, use_container_width=True)
    else:
        st.warning("Veri çekilemedi. Binance bağlantısı bekleniyor...")
        if st.button('Yeniden Dene'):
            st.rerun()

with tab2:
    if not df.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=df['timestamp'], 
            open=df['open'], high=df['high'], 
            low=df['low'], close=df['close']
        )])
        fig.update_layout(xaxis_rangeslider_visible=False, title="BTC/USDT Mum Grafiği")
        st.plotly_chart(fig, use_container_width=True)
