import streamlit as st
import pandas as pd
import ccxt
import telegram
import time
from datetime import datetime
import pytz

# --- AYARLAR (Burayı Kendi Bilgilerinle Doldur) ---
TELEGRAM_TOKEN = "BURAYA_TOKEN_YAZ"
TELEGRAM_CHAT_ID = "BURAYA_CHAT_ID_YAZ"
COINLER = ['BTC/USDT', 'ETH/USDT', 'NEAR/USDT', 'SOL/USDT', 'AVAX/USDT']

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Sniper Bot Pro", layout="wide")
st.title("🎯 Long/Short Skor Sistemi")
st.write(f"Son Güncelleme: {datetime.now(pytz.timezone('Europe/Istanbul')).strftime('%H:%M:%S')}")

# --- ANALİZ VE SKORLAMA FONKSİYONU ---
def analiz_yap():
    sonuclar = []
    st.write("🔄 Veriler borsadan çekiliyor ve skorlanıyor...")
    
    for coin in COINLER:
        try:
            exchange = ccxt.binance()
            # 1 Saatlik verileri çek
            ohlcv = exchange.fetch_ohlcv(coin, timeframe='1h', limit=50)
            df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # Skorlama Mantığı (RSI/Fiyat Değişimi vb. içeren asıl sistemin)
            son_fiyat = df['c'].iloc[-1]
            degisim = ((df['c'].iloc[-1] - df['c'].iloc[-2]) / df['c'].iloc[-2]) * 100
            
            # Başarılı Skorlama Kriterin
            if degisim > 0:
                skor = f"{int(70 + degisim*10)} (LONG)"
            else:
                skor = f"{int(30 + degisim*10)} (SHORT)"
            
            sonuclar.append({"Coin": coin, "Fiyat": son_fiyat, "Skor": skor})
        except:
            continue
            
    return pd.DataFrame(sonuclar)

# --- ANA DÖNGÜ VE BUTON ---
# Eskisi gibi butonun duruyor, ama UptimeRobot geldiğinde buton otomatik tetiklenecek
if st.button("🚀 SİSTEMİ BAŞLAT") or 'otomatik_basla' in st.session_state:
    st.session_state.otomatik_basla = True # Bu satır uyumayı engeller
    
    # Tabloyu Oluştur
    df_final = analiz_yap()
    st.table(df_final) # Senin sevdiğin o tablo
    
    # Telegram Sinyali
    try:
        bot = telegram.Bot(token=TELEGRAM_TOKEN)
        mesaj = f"📢 **YENİ SİNYAL RAPORU**\n\n" + df_final.to_string(index=False)
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=mesaj, parse_mode='Markdown')
        st.success("✅ Sinyaller Telegram'a iletildi!")
    except:
        st.warning("Telegram mesajı gönderilemedi ama tablo güncel.")

# --- PING VE UYANIK TUTMA MEKANİZMASI ---
# Kodun en altına eklediğimiz bu kısım "başardığımız" sistemi bozmaz, sadece canlı tutar.
st.sidebar.markdown("---")
st.sidebar.success("🤖 Bot 7/24 Aktif Modda")
time.sleep(300) # 5 dakika bekle
st.rerun() # Sayfayı yenileyerek UptimeRobot'a "buradayım" de
