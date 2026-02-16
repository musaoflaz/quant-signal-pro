import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import plotly.graph_objects as go

# 1. Sayfa Konfigürasyonu (En üstte olmalı)
st.set_page_config(layout="wide", page_title="Quant Signal Pro")

# 2. Borsa Bağlantısı (Hata almamak için limitli)
exchange = ccxt.binance({'enableRateLimit': True})

def get_crypto_data(symbol='BTC/USDT'):
    try:
        # Veri çekme
        bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['Tarih'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # RSI Hesaplama
        df['RSI'] = ta.rsi(df['close'], length=14)
        
        # Sinyal Mantığı (En basit ve hatasız haliyle)
        df['Sinyal'] = 'Bekle'
        df.loc[df['RSI'] < 30, 'Sinyal'] = 'AL'
        df.loc[df['RSI'] > 70, 'Sinyal'] = 'SAT'
        
        # Sadece ihtiyacımız olan sütunları alalım
        return df[['Tarih', 'open', 'high', 'low', 'close', 'RSI', 'Sinyal']].dropna()
    except Exception as e:
        return pd.DataFrame()

# 3. Renklendirme Fonksiyonu (Bulut uyumlu)
def color_signals(val):
    color = ''
    if val == 'AL': color = 'background-color: #00ff00; color: black'
    elif val == 'SAT': color = 'background-color: #ff0000; color: white'
    return color

# --- ARAYÜZ ---
st.title("🚀 Quant Signal Pro (V2)")

# Veriyi Çek
df = get_crypto_data()

tab1, tab2 = st.tabs(["📊 Sinyal Tablosu", "📈 Teknik Grafik"])

with tab1:
    if not df.empty:
        st.subheader("BTC/USDT - 1 Saatlik Veriler")
        # En güncel veriyi en üste alıyoruz
        latest_df = df.iloc[::-1]
        
        # Tabloyu basıyoruz (Hatayı önlemek için subset belirttik)
        st.dataframe(
            latest_df.style.applymap(color_signals, subset=['Sinyal']),
            use_container_width=True,
            height=600
        )
    else:
        st.error("Şu an Binance verilerine ulaşılamıyor. Lütfen sayfayı yenile.")

with tab2:
    if not df.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=df['Tarih'],
            open=df['open'], high=df['high'],
            low=df['low'], close=df['close']
        )])
        fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# Manuel Yenileme
if st.sidebar.button('Verileri Güncelle'):
    st.rerun()
