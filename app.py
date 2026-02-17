import streamlit as st
import pandas as pd
import ccxt
import telegram
import time
from datetime import datetime
import pytz

# --- 1. AYARLAR (Kendi Bilgilerini Buraya Gir) ---
TELEGRAM_TOKEN = "BURAYA_BOT_TOKENINI_YAZ"
TELEGRAM_CHAT_ID = "BURAYA_CHAT_IDNI_YAZ"
COINLER = ['BTC/USDT', 'ETH/USDT', 'NEAR/USDT', 'SOL/USDT', 'AVAX/USDT'] # İstediğin coinleri ekle
TARAMA_ARALIGI = 900 # 15 dakikada bir (saniye cinsinden)

# --- 2. FONKSİYONLAR ---
def skor_hesapla(symbol):
    """Senin o meşhur başarılı analiz mantığın burası"""
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Buraya senin özel stratejin/skorlama mantığın gelecek
        # Örnek basit bir skorlama (Seninkini buraya entegre edebilirsin):
        son_fiyat = df['close'].iloc[-1]
        onceki_fiyat = df['close'].iloc[-2]
        
        if son_fiyat > onceki_fiyat:
            skor = "80 (LONG)"
        else:
            skor = "20 (SHORT)"
            
        return skor, son_fiyat
    except:
        return "Hata", 0

def tablo_ve_gonder():
    """Analiz yapar, tabloyu basar ve Telegram'a yollar"""
    veriler = []
    mesaj = "🚀 **GÜNCEL SİNYAL RAPORU** 🚀\n\n"
    
    for coin in COINLER:
        skor, fiyat = skor_hesapla(coin)
        veriler.append({"Coin": coin, "Fiyat": fiyat, "Skor/Yön": skor})
        mesaj += f"🔹 {coin}: {fiyat} | Skor: {skor}\n"
    
    df_sonuc = pd.DataFrame(veriler)
    
    # Ekrana Tabloyu Bas
    st.table(df_sonuc)
    
    # Telegram'a Gönder
    try:
        bot = telegram.Bot(token=TELEGRAM_TOKEN)
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=mesaj, parse_mode='Markdown')
        st.success(f"✅ Telegram'a gönderildi: {datetime.now(pytz.timezone('Europe/Istanbul')).strftime('%H:%M')}")
    except Exception as e:
        st.error(f"Telegram hatası: {e}")

# --- 3. STREAMLIT ARAYÜZÜ VE OTOMATİK DÖNGÜ ---
st.set_page_config(page_title="7/24 Sniper Bot", layout="wide")
st.title("🤖 7/24 Full Otomatik Sniper")

# Otomatik çalışma mantığı
if 'last_run' not in st.session_state:
    st.session_state.last_run = 0

current_time = time.time()

# Eğer son çalışmadan bu yana 15 dakika geçtiyse veya bot ilk kez açılıyorsa
if current_time - st.session_state.last_run > TARAMA_ARALIGI:
    tablo_ve_gonder()
    st.session_state.last_run = current_time
    st.info("🔄 Tarama tamamlandı. 15 dakika sonra tekrar otomatik başlayacak.")
else:
    kalan = int((TARAMA_ARALIGI - (current_time - st.session_state.last_run)) / 60)
    st.write(f"⏳ Sistem uyanık. Bir sonraki taramaya **{kalan} dakika** kaldı.")
    st.write("UptimeRobot sayesinde bu sayfa kapansa da bot çalışmaya devam eder.")

# Sayfayı 5 dakikada bir yenile (UptimeRobot ile senkronizasyon için)
time.sleep(300)
st.rerun()
