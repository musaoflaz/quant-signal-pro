import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time

st.set_page_config(layout="wide", page_title="Quant Alpha | Smart Filter V9")

# 1. Borsa Bağlantısı (Stabil ve Hızlı)
exchange = ccxt.kucoin({'enableRateLimit': True, 'timeout': 30000})

st.markdown("# 🏛️ QUANT ALPHA: AKILLI ANALİZ TERMİNALİ")
st.info("Piyasa taranıyor... Sadece kriterlere tam uyanlar 'Sinyal' olarak listelenir.")

def get_best_symbols():
    try:
        tickers = exchange.fetch_tickers()
        df_t = pd.DataFrame.from_dict(tickers, orient='index')
        df_t = df_t[df_t['symbol'].str.contains('/USDT')]
        # En hacimli 40 coin (Piyasa yönünü belirleyen ana grup)
        return df_t.sort_values('quoteVolume', ascending=False).head(40).index.tolist()
    except:
        return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT', 'AVAX/USDT', 'SUI/USDT']

def deep_scanner():
    symbols = get_best_symbols()
    all_results = []
    
    progress = st.progress(0)
    status = st.empty()
    
    for idx, symbol in enumerate(symbols):
        status.info(f"🔍 Kriter Kontrolü: **{symbol}**")
        try:
            # 1 Saatlik Veri (Sinyal için)
            bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=150)
            df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # --- TEKNİK ANALİZ MOTORU ---
            df['EMA200'] = ta.ema(df['c'], length=200)
            df['RSI'] = ta.rsi(df['c'], length=14)
            stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
            df = pd.concat([df, stoch], axis=1)
            
            l, p = df.iloc[-1], df.iloc[-2]
            
            # Veriler
            c, rsi, ema = l['c'], l['RSI'], l['EMA200']
            k, d = l['STOCHRSIk_14_14_3_3'], l['STOCHRSId_14_14_3_3']
            pk, pd_val = p['STOCHRSIk_14_14_3_3'], p['STOCHRSId_14_14_3_3']
            
            # --- ANALİZ VE SIRALAMA MANTIĞI ---
            skor = 0
            etiket = "GÖZLEM"
            
            # 🚀 LONG ŞARTLARI (Kriter Onayı)
            if c > ema:
                if pk < pd_val and k > d: # Altın Kesişim
                    etiket = "🚀 GÜÇLÜ AL (LONG)"
                    skor = 90
                elif k > d:
                    etiket = "🟢 TREND YUKARI"
                    skor = 60
            
            # 💥 SHORT ŞARTLARI (Kriter Onayı)
            elif c < ema:
                if pk > pd_val and k < d: # Ölüm Kesişimi
                    etiket = "💥 GÜÇLÜ SAT (SHORT)"
                    skor = 90
                elif k < d:
                    etiket = "🔴 TREND AŞAĞI"
                    skor = 60

            all_results.append({
                "COIN": symbol,
                "FİYAT": f"{c:.4f}",
                "DURUM": etiket,
                "SKOR": skor,
                "RSI": int(rsi)
            })
            time.sleep(0.2)
        except: continue
        progress.progress((idx + 1) / len(symbols))
    
    status.empty()
    progress.empty()
    
    return pd.DataFrame(all_results).sort_values('SKOR', ascending=False)

# --- Arayüz ---
if st.button('🎯 PİYASAYI ANALİZ ET'):
    data = deep_scanner()
    
    if not data.empty:
        # 1. Sinyal Bölümü (Skor >= 80 olanlar)
        sinyaller = data[data['SKOR'] >= 80]
        if not sinyaller.empty:
            st.subheader("🔥 KRİTERLERE UYAN GÜÇLÜ SİNYALLER")
            st.success("Bu coinler hem trend onayı hem de momentum kesişimi verdi!")
            st.table(sinyaller[['COIN', 'FİYAT', 'DURUM', 'RSI']])
        else:
            st.warning("⚠️ Şu an kriterlere %100 uyan bir giriş fırsatı yok.")

        # 2. Gözlem Bölümü
        st.write("---")
        st.subheader("👀 TÜM PİYASA SIRALAMASI (Gözlem Listesi)")
        
        def color_status(val):
            if "AL" in str(val): return 'background-color: #0c3e1e; color: #52ff8f'
            if "SAT" in str(val): return 'background-color: #4b0a0a; color: #ff6e6e'
            return ''
        
        st.dataframe(data.style.applymap(color_status, subset=['DURUM']), use_container_width=True)
    else:
        st.error("Borsa verisi alınamadı.")
