import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import plotly.graph_objects as go

# --- AYARLAR VE BAĞLANTI ---
st.set_page_config(layout="wide", page_title="Quant Signal Pro")

# Binance bağlantısını daha güvenli hale getirelim
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

# --- VERİ ÇEKME FONKSİYONU ---
def fetch_data(symbol='BTC/USDT', timeframe='1h'):
    try:
        # Veri çekme denemesi
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
        if not bars:
            return pd.DataFrame()
            
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Sinyal Hesaplamaları
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['Aksiyon'] = 'BEKLE'
        df.loc[df['RSI'] < 30, 'Aksiyon'] = 'AL'
        df.loc[df['RSI'] > 70, 'Aksiyon'] = 'SAT'
        
        # NaN (boş) değerleri temizleyelim ki tablo çökmesin
        df = df.dropna()
        return df
    except Exception as e:
        # Hata olursa ekrana teknik detay yazma, sadece boş dön
        return pd.DataFrame()

# --- RENKLENDİRME ---
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
            # Tabloyu en güncel veri en üstte olacak şekilde ters çevirip gösterelim
            st.dataframe(df.iloc[::-1].style.applymap(style_action_color, subset=['Aksiyon']), height=650, use_container_width=True)
        except:
            st.dataframe(df.iloc[::-1], height=650, use_container_width=True)
    else:
        st.warning("Binance bağlantısı bekleniyor... Lütfen 5 saniye sonra sayfayı yukarıdan aşağı kaydırarak yenileyin.")
        if st.button('Şimdi Tekrar Dene'):
            st.rerun()

with tab2:
    if not df.empty:
        fig = go.Figure(data=[go.Candlestick(x=df['timestamp'],
                open=df['open'], high=df['high'],
                low=df['low'], close=df['close'])])
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
