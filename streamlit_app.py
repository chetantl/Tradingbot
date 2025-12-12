import streamlit as st
from kiteconnect import KiteConnect
import threading
import time
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
from pytz import timezone

# ====== TIMEZONE ====== #
INDIA_TZ = timezone("Asia/Kolkata")

def now_ist():
    return datetime.now(INDIA_TZ)

# ====== PAGE CONFIG ====== #
st.set_page_config(
    page_title="Institutional Sniper",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== CUSTOM CSS ====== #
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f23 100%);
        border-right: 1px solid #333;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.1rem;
    }
    
    .stDataFrame {
        background: #1e1e3f;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Animation keyframes */
    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 0 20px rgba(46, 204, 113, 0.4); }
        50% { box-shadow: 0 0 40px rgba(46, 204, 113, 0.8); }
    }
    
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 20px rgba(231, 76, 60, 0.4); }
        50% { box-shadow: 0 0 40px rgba(231, 76, 60, 0.8); }
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    
    .pulse-green { animation: pulse-green 2s infinite; }
    .pulse-red { animation: pulse-red 2s infinite; }
    .blink { animation: blink 1s infinite; }
</style>
""", unsafe_allow_html=True)

# ====== CONFIGURATION ====== #
CONFIG = {
    'bucket_seconds': 5,
    'bar_seconds': 300,
    'rvol_threshold': 1.5,
    'trend_confirmation_bars': 3,
    'breakout_lookback_bars': 4,
    'pcr_min_bullish': 0.7,
    'pcr_max_bearish': 1.3,
    'rvol_lookback_bars': 750
}

# ====== SHARED DATA STORE ====== #
@st.cache_resource
def get_shared_store():
    return {
        'data': {},
        'logs': deque(maxlen=100),
        'is_running': False,
        'worker_alive': False,
        'update_count': 0,
        'error_count': 0,
        'stop_flag': False,
        'lock': threading.Lock(),
        'symbols': [],
        'kite': None
    }

STORE = get_shared_store()

def store_set_data(symbol: str, data: dict):
    with STORE['lock']:
        STORE['data'][symbol] = data
        STORE['update_count'] += 1

def store_get_all_data() -> dict:
    with STORE['lock']:
        return dict(STORE['data'])

def store_log(msg: str):
    with STORE['lock']:
        ts = now_ist().strftime("%H:%M:%S")
        STORE['logs'].appendleft(f"[{ts}] {msg}")

def store_get_logs() -> list:
    with STORE['lock']:
        return list(STORE['logs'])

def store_reset():
    with STORE['lock']:
        STORE['data'] = {}
        STORE['logs'].clear()
        STORE['error_count'] = 0
        STORE['update_count'] = 0
        STORE['stop_flag'] = False

# ====== HELPER FUNCTIONS ====== #
def safe_float(val, default=0.0):
    try:
        return float(val) if val is not None else default
    except:
        return default

def safe_int(val, default=0):
    try:
        return int(val) if val is not None else default
    except:
        return default

def get_instrument_token(kite, symbol):
    try:
        instruments = kite.instruments("NSE")
        sym = symbol.upper().strip()
        for inst in instruments:
            if inst["tradingsymbol"] == sym and inst["segment"] == "NSE":
                return inst["instrument_token"]
    except Exception as e:
        store_log(f"Token error {symbol}: {e}")
    return None

def get_expiry_date(expiry):
    if expiry is None:
        return None
    if hasattr(expiry, 'date'):
        return expiry.date()
    return expiry

def fetch_historical_volumes(kite, token, lookback_bars=750):
    try:
        now = now_ist().replace(tzinfo=None)
        from_date = now - timedelta(days=15)
        candles = kite.historical_data(token, from_date, now, "5minute")
        df = pd.DataFrame(candles)
        if df.empty:
            return []
        volumes = df['volume'].tolist()
        return volumes[-lookback_bars:]
    except Exception as e:
        store_log(f"Historical error: {e}")
        return []

def get_pcr(kite, symbol, spot_price):
    try:
        all_opts = [i for i in kite.instruments("NFO") 
                    if i.get("segment") == "NFO-OPT" and i.get("name") == symbol]
        
        if not all_opts:
            return 1.0
        
        today = now_ist().date()
        expiries = sorted({
            get_expiry_date(i["expiry"]) for i in all_opts 
            if get_expiry_date(i["expiry"]) and get_expiry_date(i["expiry"]) >= today
        })
        
        if not expiries:
            return 1.0
        
        nearest = expiries[0]
        scoped = [i for i in all_opts if get_expiry_date(i["expiry"]) == nearest]
        
        ce_opts = {i['strike']: i for i in scoped if i["instrument_type"] == "CE"}
        pe_opts = {i['strike']: i for i in scoped if i["instrument_type"] == "PE"}
        
        relevant = [s for s in set(ce_opts.keys()) | set(pe_opts.keys())
                   if spot_price * 0.95 <= s <= spot_price * 1.05]
        
        if not relevant:
            return 1.0
        
        ce_syms = [f"NFO:{ce_opts[s]['tradingsymbol']}" for s in relevant if s in ce_opts]
        pe_syms = [f"NFO:{pe_opts[s]['tradingsymbol']}" for s in relevant if s in pe_opts]
        
        ce_oi, pe_oi = 0, 0
        
        for i in range(0, len(ce_syms), 30):
            batch = ce_syms[i:i+30]
            try:
                quotes = kite.quote(batch)
                for q in quotes.values():
                    ce_oi += safe_int(q.get('oi'))
            except:
                pass
        
        for i in range(0, len(pe_syms), 30):
            batch = pe_syms[i:i+30]
            try:
                quotes = kite.quote(batch)
                for q in quotes.values():
                    pe_oi += safe_int(q.get('oi'))
            except:
                pass
        
        return round(pe_oi / ce_oi, 2) if ce_oi > 0 else 1.0
        
    except Exception as e:
        store_log(f"PCR error: {e}")
        return 1.0

# ====== VOLUME TRACKER ====== #
class VolumeTracker:
    def __init__(self, maxlen=750):
        self.history = deque(maxlen=maxlen)
        self.sum_vol = 0.0
    
    def seed(self, volumes):
        self.history.clear()
        self.sum_vol = 0.0
        for v in volumes:
            self.history.append(v)
            self.sum_vol += v
    
    def add(self, volume):
        if len(self.history) >= self.history.maxlen:
            self.sum_vol -= self.history[0]
        self.history.append(volume)
        self.sum_vol += volume
    
    def get_avg(self):
        return self.sum_vol / len(self.history) if self.history else 1.0
    
    def get_rvol(self, current):
        avg = self.get_avg()
        return current / avg if avg > 0 else 1.0
    
    def get_count(self):
        return len(self.history)

# ====== BAR DATA ====== #
class BarData:
    def __init__(self):
        self.reset(0, None)
    
    def reset(self, price, ts):
        self.open = price
        self.high = price
        self.low = price
        self.close = price
        self.volume = 0
        self.aggr_buy = 0
        self.aggr_sell = 0
        self.start_time = ts
    
    def update(self, price, volume, is_buy):
        self.high = max(self.high, price)
        self.low = min(self.low, price) if self.low > 0 else price
        self.close = price
        self.volume += volume
        if is_buy:
            self.aggr_buy += volume
        else:
            self.aggr_sell += volume
    
    def get_imbalance(self):
        total = self.aggr_buy + self.aggr_sell
        return (self.aggr_buy - self.aggr_sell) / total if total > 0 else 0.0

# ====== SYMBOL TRACKER ====== #
class SymbolTracker:
    def __init__(self, symbol, token):
        self.symbol = symbol
        self.token = token
        self.volume_tracker = VolumeTracker(750)
        self.current_bar = BarData()
        self.completed_bars = deque(maxlen=50)
        self.prev_volume = 0
        self.prev_ltp = 0.0
        self.pcr = 1.0
        self.last_bar_minute = -1
        self.initialized = False
    
    def get_bar_minute(self, ts):
        return (ts.minute // 5) * 5
    
    def process_quote(self, ltp, volume, vwap, bid, ask, ts):
        if not self.initialized:
            self.prev_volume = volume
            self.prev_ltp = ltp
            self.initialized = True
            self.current_bar.reset(ltp, ts)
            self.last_bar_minute = self.get_bar_minute(ts)
            return
        
        vol_delta = max(0, volume - self.prev_volume)
        self.prev_volume = volume
        
        current_bar_min = self.get_bar_minute(ts)
        if current_bar_min != self.last_bar_minute:
            if self.current_bar.volume > 0:
                self.completed_bars.append({
                    'open': self.current_bar.open,
                    'high': self.current_bar.high,
                    'low': self.current_bar.low,
                    'close': self.current_bar.close,
                    'volume': self.current_bar.volume,
                    'imbalance': self.current_bar.get_imbalance()
                })
                self.volume_tracker.add(self.current_bar.volume)
            
            self.current_bar.reset(ltp, ts)
            self.last_bar_minute = current_bar_min
        
        if vol_delta > 0:
            is_buy = ltp >= (bid + ask) / 2 if bid > 0 and ask > 0 else ltp > self.prev_ltp
            self.current_bar.update(ltp, vol_delta, is_buy)
        
        self.prev_ltp = ltp
    
    def get_rvol(self):
        return self.volume_tracker.get_rvol(self.current_bar.volume)
    
    def get_signal(self, vwap):
        bars = list(self.completed_bars)
        if len(bars) < CONFIG['breakout_lookback_bars']:
            return "NEUTRAL", 0, {}, 0, 0, 0
        
        lookback = CONFIG['breakout_lookback_bars']
        trend_bars_count = CONFIG['trend_confirmation_bars']
        
        recent = bars[-lookback:]
        trend_bars = bars[-trend_bars_count:]
        
        recent_high = max(b['high'] for b in recent)
        recent_low = min(b['low'] for b in recent)
        
        current_close = self.current_bar.close
        rvol = self.get_rvol()
        
        bullish_trend = all(b['close'] > vwap for b in trend_bars)
        bearish_trend = all(b['close'] < vwap for b in trend_bars)
        
        breaking_up = current_close > recent_high
        breaking_down = current_close < recent_low
        
        volume_good = rvol >= CONFIG['rvol_threshold']
        pcr_bullish = self.pcr > CONFIG['pcr_min_bullish']
        pcr_bearish = self.pcr < CONFIG['pcr_max_bearish']
        
        diagnostics = {
            'Trend': 'Bullish' if bullish_trend else 'Bearish' if bearish_trend else 'Choppy',
            'RVol': f"{rvol:.2f}x",
            'PCR': f"{self.pcr:.2f}",
            'Range': f"{recent_low:.1f} - {recent_high:.1f}",
            'Breakout': 'Up' if breaking_up else 'Down' if breaking_down else 'None'
        }
        
        if bullish_trend and breaking_up and volume_good and pcr_bullish:
            entry = current_close
            sl = recent_low
            target = current_close + (current_close - recent_low) * 2
            return "BUY CALL", 13, diagnostics, entry, sl, target
        
        elif bearish_trend and breaking_down and volume_good and pcr_bearish:
            entry = current_close
            sl = recent_high
            target = current_close - (recent_high - current_close) * 2
            return "BUY PUT", 13, diagnostics, entry, sl, target
        
        return "NEUTRAL", 0, diagnostics, 0, 0, 0

# ====== WORKER THREAD ====== #
def worker_thread(kite, symbols):
    store_log("🚀 Worker thread started")
    STORE['worker_alive'] = True
    STORE['is_running'] = True
    
    trackers = {}
    pcr_cache = {}
    last_pcr_time = 0
    
    for item in symbols:
        sym = item['symbol']
        token = item['token']
        
        store_log(f"Initializing {sym}...")
        tracker = SymbolTracker(sym, token)
        
        hist = fetch_historical_volumes(kite, token)
        if hist:
            tracker.volume_tracker.seed(hist)
            store_log(f"✅ {sym}: Loaded {len(hist)} bars")
        else:
            store_log(f"⚠️ {sym}: No historical data")
        
        trackers[sym] = tracker
    
    store_log(f"✅ All {len(trackers)} symbols ready!")
    
    while not STORE['stop_flag'] and STORE['is_running']:
        try:
            keys = [f"NSE:{item['symbol']}" for item in symbols]
            
            try:
                quotes = kite.quote(keys)
            except Exception as e:
                store_log(f"❌ Quote error: {e}")
                STORE['error_count'] += 1
                time.sleep(2)
                continue
            
            now_ts = time.time()
            update_pcr = (now_ts - last_pcr_time) > 180
            if update_pcr:
                last_pcr_time = now_ts
                store_log("📊 Updating PCR...")
            
            for item in symbols:
                sym = item['symbol']
                key = f"NSE:{sym}"
                
                q = quotes.get(key)
                if not q:
                    continue
                
                tracker = trackers.get(sym)
                if not tracker:
                    continue
                
                ltp = safe_float(q.get('last_price'))
                volume = safe_int(q.get('volume'))
                vwap = safe_float(q.get('average_price'), ltp)
                
                depth = q.get('depth', {})
                buy_depth = depth.get('buy', [{}])
                sell_depth = depth.get('sell', [{}])
                
                bid = safe_float(buy_depth[0].get('price')) if buy_depth else ltp
                ask = safe_float(sell_depth[0].get('price')) if sell_depth else ltp
                
                ts = now_ist()
                tracker.process_quote(ltp, volume, vwap, bid, ask, ts)
                
                if update_pcr:
                    tracker.pcr = get_pcr(kite, sym, ltp)
                    pcr_cache[sym] = tracker.pcr
                elif sym in pcr_cache:
                    tracker.pcr = pcr_cache[sym]
                
                signal_type, score, diagnostics, entry, sl, target = tracker.get_signal(vwap)
                rvol = tracker.get_rvol()
                
                store_set_data(sym, {
                    'symbol': sym,
                    'ltp': ltp,
                    'vwap': vwap,
                    'bid': bid,
                    'ask': ask,
                    'volume': volume,
                    'rvol': round(rvol, 2),
                    'pcr': tracker.pcr,
                    'signal_type': signal_type,
                    'score': score,
                    'diagnostics': diagnostics,
                    'entry': entry,
                    'sl': sl,
                    'target': target,
                    'bars': tracker.volume_tracker.get_count(),
                    'completed_bars': len(tracker.completed_bars),
                    'current_bar_vol': tracker.current_bar.volume,
                    'imbalance': round(tracker.current_bar.get_imbalance(), 2),
                    'initialized': tracker.initialized,
                    'updated': ts.strftime("%H:%M:%S")
                })
                
                if score == 13:
                    store_log(f"🎯 SIGNAL: {sym} {signal_type} @ ₹{ltp:.2f}")
            
            time.sleep(1)
            
        except Exception as e:
            store_log(f"❌ Worker error: {e}")
            STORE['error_count'] += 1
            time.sleep(2)
    
    STORE['worker_alive'] = False
    STORE['is_running'] = False
    store_log("🛑 Worker stopped")

# ====== CARD RENDERING WITH STREAMLIT COMPONENTS ====== #
def render_signal_card_streamlit(d: dict, is_active: bool = False):
    """Render card using native Streamlit components"""
    signal_type = d.get('signal_type', 'NEUTRAL')
    
    # Determine colors and styles
    if signal_type == "BUY CALL":
        border_color = "#2ecc71"
        bg_color = "rgba(46, 204, 113, 0.1)"
        icon = "🟢"
        badge_bg = "linear-gradient(90deg, #27ae60, #2ecc71)"
    elif signal_type == "BUY PUT":
        border_color = "#e74c3c"
        bg_color = "rgba(231, 76, 60, 0.1)"
        icon = "🔴"
        badge_bg = "linear-gradient(90deg, #c0392b, #e74c3c)"
    else:
        border_color = "#555"
        bg_color = "rgba(255, 255, 255, 0.05)"
        icon = "⚪"
        badge_bg = "#555"
    
    # Card container
    st.markdown(f"""
        <div style="
            background: linear-gradient(145deg, #1e1e3f, #2a2a4a);
            border: 2px solid {border_color};
            border-radius: 16px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <span style="font-size: 1.5rem; font-weight: 700; color: #fff;">{icon} {d['symbol']}</span>
                <span style="
                    background: {badge_bg};
                    padding: 6px 16px;
                    border-radius: 20px;
                    font-weight: 600;
                    font-size: 0.85rem;
                    color: white;
                ">{signal_type}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Metrics using Streamlit columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💰 LTP", f"₹{d['ltp']:.2f}")
    with col2:
        rvol_delta = "normal" if d['rvol'] < 1.5 else "off"
        st.metric("📊 RVol", f"{d['rvol']}x", delta=f"{'High' if d['rvol'] >= 1.5 else 'Normal'}")
    with col3:
        st.metric("📈 PCR", f"{d['pcr']:.2f}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📍 VWAP", f"₹{d['vwap']:.2f}")
    with col2:
        st.metric("📦 Volume", f"{d['volume']:,}")
    with col3:
        imb = d.get('imbalance', 0)
        st.metric("⚖️ Imbalance", f"{imb:+.2f}")
    
    # Entry/SL/Target for active signals
    if is_active and d.get('entry', 0) > 0:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
                <div style="background: rgba(52, 152, 219, 0.2); border: 1px solid #3498db; border-radius: 10px; padding: 15px; text-align: center;">
                    <div style="font-size: 0.75rem; color: #3498db; text-transform: uppercase;">📍 Entry</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #fff;">₹{d['entry']:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div style="background: rgba(231, 76, 60, 0.2); border: 1px solid #e74c3c; border-radius: 10px; padding: 15px; text-align: center;">
                    <div style="font-size: 0.75rem; color: #e74c3c; text-transform: uppercase;">🛑 Stop Loss</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #fff;">₹{d['sl']:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div style="background: rgba(46, 204, 113, 0.2); border: 1px solid #2ecc71; border-radius: 10px; padding: 15px; text-align: center;">
                    <div style="font-size: 0.75rem; color: #2ecc71; text-transform: uppercase;">🎯 Target</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #fff;">₹{d['target']:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
    
    # Diagnostics
    diag = d.get('diagnostics', {})
    if diag:
        diag_str = " | ".join([f"**{k}:** {v}" for k, v in diag.items()])
        st.caption(f"📊 {diag_str}")
    
    st.caption(f"🕐 Updated: {d['updated']} | Bars: {d['bars']}/750 | Completed: {d['completed_bars']}")

def render_watchlist_card_streamlit(d: dict):
    """Render compact watchlist card using Streamlit components"""
    signal_type = d.get('signal_type', 'NEUTRAL')
    
    if signal_type == "BUY CALL":
        border_color = "#2ecc71"
        icon = "🟢"
    elif signal_type == "BUY PUT":
        border_color = "#e74c3c"
        icon = "🔴"
    else:
        border_color = "#444"
        icon = "⚪"
    
    rvol_color = "#2ecc71" if d['rvol'] >= 1.5 else "#fff"
    imb = d.get('imbalance', 0)
    imb_color = "#2ecc71" if imb > 0 else "#e74c3c" if imb < 0 else "#fff"
    
    # Card with inline styles
    st.markdown(f"""
        <div style="
            background: linear-gradient(145deg, #1e1e3f, #2a2a4a);
            border: 1px solid {border_color};
            border-radius: 12px;
            padding: 15px;
            margin: 8px 0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 1.2rem; font-weight: 700; color: #fff;">{d['symbol']}</span>
                <span style="font-size: 0.8rem; color: #888;">{icon} {signal_type}</span>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.65rem; color: #888; text-transform: uppercase;">LTP</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #fff;">₹{d['ltp']:.2f}</div>
                </div>
                <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.65rem; color: #888; text-transform: uppercase;">RVol</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: {rvol_color};">{d['rvol']}x</div>
                </div>
                <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.65rem; color: #888; text-transform: uppercase;">PCR</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #fff;">{d['pcr']:.2f}</div>
                </div>
                <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.65rem; color: #888; text-transform: uppercase;">VWAP</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #fff;">₹{d['vwap']:.2f}</div>
                </div>
            </div>
            
            <div style="margin-top: 10px; font-size: 0.75rem; color: #666;">
                Vol: {d['volume']:,} | Imb: <span style="color: {imb_color};">{imb:+.2f}</span> | {d['updated']}
            </div>
        </div>
    """, unsafe_allow_html=True)

# ====== SESSION STATE ====== #
if 'connected' not in st.session_state:
    st.session_state.connected = False

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 15px 0;">
            <div style="font-size: 2rem;">🎯</div>
            <h2 style="margin: 5px 0; color: #fff;">Sniper</h2>
            <p style="color: #888; font-size: 0.8rem; margin: 0;">Institutional Flow Detector</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    if not st.session_state.connected:
        st.subheader("🔐 Kite Login")
        
        api_key = st.text_input("API Key", key="api_key")
        api_secret = st.text_input("API Secret", type="password", key="api_secret")
        
        if api_key and api_secret:
            try:
                temp_kite = KiteConnect(api_key=api_key)
                login_url = temp_kite.login_url()
                st.markdown(f"[👉 Click to login to Kite]({login_url})")
            except Exception as e:
                st.error(f"Error: {e}")
        
        request_token = st.text_input("Request Token", key="req_token")
        
        if st.button("🔌 Connect", type="primary", use_container_width=True):
            if api_key and api_secret and request_token:
                try:
                    with st.spinner("Connecting..."):
                        kite = KiteConnect(api_key=api_key)
                        session_data = kite.generate_session(request_token, api_secret=api_secret)
                        kite.set_access_token(session_data["access_token"])
                        STORE['kite'] = kite
                        st.session_state.connected = True
                        store_log("✅ Connected to Kite API")
                    st.success("✅ Connected!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Connection failed: {e}")
            else:
                st.warning("Please fill all fields")
    
    else:
        st.success("✅ Connected to Kite")
        
        st.divider()
        
        st.subheader("📊 Symbols")
        symbols_input = st.text_area(
            "Enter symbols (comma separated)",
            value="RELIANCE, HDFCBANK, INFY, TCS, SBIN",
            height=80,
            key="symbols_input"
        )
        
        st.divider()
        st.subheader("⚡ Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_disabled = STORE['is_running']
            if st.button("🚀 START", type="primary", use_container_width=True, disabled=start_disabled):
                store_reset()
                
                syms = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
                valid_symbols = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, sym in enumerate(syms):
                    status_text.text(f"Loading {sym}...")
                    token = get_instrument_token(STORE['kite'], sym)
                    if token:
                        valid_symbols.append({'symbol': sym, 'token': token})
                        store_log(f"✓ Found {sym}")
                    else:
                        store_log(f"✗ Not found: {sym}")
                    progress_bar.progress((i + 1) / len(syms))
                
                progress_bar.empty()
                status_text.empty()
                
                if valid_symbols:
                    STORE['symbols'] = valid_symbols
                    STORE['is_running'] = True
                    STORE['stop_flag'] = False
                    
                    t = threading.Thread(
                        target=worker_thread,
                        args=(STORE['kite'], valid_symbols),
                        daemon=True
                    )
                    t.start()
                    
                    st.success(f"Started {len(valid_symbols)} symbols!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("No valid symbols found!")
        
        with col2:
            stop_disabled = not STORE['is_running']
            if st.button("🛑 STOP", type="secondary", use_container_width=True, disabled=stop_disabled):
                STORE['stop_flag'] = True
                STORE['is_running'] = False
                store_log("🛑 Stopping...")
                time.sleep(1)
                st.rerun()
        
        st.divider()
        st.subheader("📈 Status")
        
        c1, c2 = st.columns(2)
        with c1:
            if STORE['is_running']:
                st.success("🟢 Running")
            else:
                st.error("🔴 Stopped")
        with c2:
            if STORE['worker_alive']:
                st.success("⚡ Active")
            else:
                st.warning("💤 Idle")
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Updates", STORE['update_count'])
        with m2:
            st.metric("Errors", STORE['error_count'])
        with m3:
            st.metric("Symbols", len(store_get_all_data()))
        
        st.divider()
        
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
        
        if st.button("🔌 Disconnect", use_container_width=True):
            STORE['stop_flag'] = True
            STORE['is_running'] = False
            st.session_state.connected = False
            STORE['kite'] = None
            st.rerun()

# ============================================
# MAIN DASHBOARD
# ============================================

# Header
st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(90deg, #f39c12, #e74c3c, #9b59b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        ">🎯 Institutional Sniper</h1>
        <p style="color: #888; margin: 5px 0 0 0;">Real-time VWAP Trend + Breakout Detection with RVol Analysis</p>
    </div>
""", unsafe_allow_html=True)

# Status Bar
st.markdown("""
    <div style="
        background: rgba(0,0,0,0.3);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0 20px 0;
        border: 1px solid #333;
    ">
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if STORE['is_running']:
        st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 10px; height: 10px; background: #2ecc71; border-radius: 50%; animation: blink 1s infinite;"></span>
                <span style="color: #2ecc71; font-weight: 600;">LIVE</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<span style="color: #e74c3c; font-weight: 600;">🔴 OFFLINE</span>', unsafe_allow_html=True)

with col2:
    st.metric("Symbols", len(store_get_all_data()))
with col3:
    st.metric("Updates", STORE['update_count'])
with col4:
    st.metric("Errors", STORE['error_count'])
with col5:
    st.metric("Time", now_ist().strftime("%H:%M:%S"))

st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# Main Content
if STORE['is_running']:
    all_data = store_get_all_data()
    
    if not all_data:
        st.markdown("""
            <div style="
                text-align: center;
                padding: 60px;
                background: linear-gradient(145deg, #1e1e3f, #2a2a4a);
                border-radius: 16px;
                border: 1px solid #333;
            ">
                <div style="font-size: 3rem; margin-bottom: 20px;">⏳</div>
                <h2 style="color: #fff; margin-bottom: 10px;">Loading Data...</h2>
                <p style="color: #888;">Please wait 5-10 seconds for first update</p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🔧 Debug Info"):
            st.write(f"Worker Alive: {STORE['worker_alive']}")
            st.write(f"Is Running: {STORE['is_running']}")
            st.write(f"Symbols Count: {len(STORE['symbols'])}")
            st.write(f"Update Count: {STORE['update_count']}")
            st.write("**Recent Logs:**")
            for log in store_get_logs()[:10]:
                st.code(log)
    
    else:
        # Separate signals
        active_signals = {k: v for k, v in all_data.items() if v.get('score', 0) == 13}
        watchlist = {k: v for k, v in all_data.items() if v.get('score', 0) != 13}
        
        # ===== ACTIVE SIGNALS =====
        st.subheader(f"🔥 Active Signals ({len(active_signals)})")
        
        if active_signals:
            for sym, d in active_signals.items():
                with st.container():
                    render_signal_card_streamlit(d, is_active=True)
                    st.divider()
        else:
            st.markdown("""
                <div style="
                    text-align: center;
                    padding: 40px;
                    background: linear-gradient(145deg, #1e1e3f, #2a2a4a);
                    border-radius: 16px;
                    border: 1px solid #333;
                ">
                    <div style="font-size: 2.5rem; margin-bottom: 15px;">📡</div>
                    <h3 style="color: #fff; margin-bottom: 10px;">Scanning for Signals...</h3>
                    <p style="color: #888; font-size: 0.9rem;">Signals appear when: RVol ≥ 1.5x + VWAP Trend + Breakout + PCR</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ===== WATCHLIST =====
        st.subheader(f"👀 Watchlist ({len(watchlist)})")
        
        if watchlist:
            cols = st.columns(3)
            for idx, (sym, d) in enumerate(watchlist.items()):
                with cols[idx % 3]:
                    render_watchlist_card_streamlit(d)
        else:
            st.info("No symbols in watchlist")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ===== DATA TABLE =====
        st.subheader("📋 Data Table")
        
        df = pd.DataFrame(list(all_data.values()))
        display_cols = ['symbol', 'ltp', 'signal_type', 'rvol', 'pcr', 'vwap', 'volume', 'imbalance', 'updated']
        available = [c for c in display_cols if c in df.columns]
        
        st.dataframe(
            df[available],
            use_container_width=True,
            hide_index=True,
            column_config={
                'symbol': 'Symbol',
                'ltp': st.column_config.NumberColumn('LTP', format="₹%.2f"),
                'signal_type': 'Signal',
                'rvol': st.column_config.NumberColumn('RVol', format="%.2fx"),
                'pcr': st.column_config.NumberColumn('PCR', format="%.2f"),
                'vwap': st.column_config.NumberColumn('VWAP', format="₹%.2f"),
                'volume': st.column_config.NumberColumn('Volume', format="%d"),
                'imbalance': st.column_config.NumberColumn('Imbalance', format="%.2f"),
                'updated': 'Updated'
            }
        )

elif st.session_state.connected:
    st.markdown("""
        <div style="
            text-align: center;
            padding: 80px 40px;
            background: linear-gradient(145deg, #1e1e3f, #2a2a4a);
            border-radius: 20px;
            border: 1px solid #333;
        ">
            <div style="font-size: 4rem; margin-bottom: 20px;">👈</div>
            <h2 style="color: #fff; margin-bottom: 15px;">Ready to Start</h2>
            <p style="color: #888; font-size: 1.1rem;">Click <b>START</b> in the sidebar to begin monitoring</p>
        </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
        <div style="
            text-align: center;
            padding: 80px 40px;
            background: linear-gradient(145deg, #1e1e3f, #2a2a4a);
            border-radius: 20px;
            border: 1px solid #333;
        ">
            <div style="font-size: 4rem; margin-bottom: 20px;">🔐</div>
            <h2 style="color: #fff; margin-bottom: 15px;">Connect to Kite API</h2>
            <p style="color: #888; font-size: 1.1rem;">Enter your credentials in the sidebar to get started</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("📖 How to Use")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Step 1: Get API Credentials**
        - Go to [Kite Developer Console](https://developers.kite.trade/)
        - Create an app and get your API Key and Secret
        
        **Step 2: Login**
        1. Enter API Key and Secret in sidebar
        2. Click the login link
        3. Authorize the app
        4. Copy request_token from URL
        5. Click Connect
        """)
    
    with col2:
        st.markdown("""
        **Step 3: Start Monitoring**
        1. Enter symbols (comma separated)
        2. Click START
        3. Watch for signals!
        
        **Signal Conditions:**
        - 🟢 **BUY CALL:** RVol ≥ 1.5x + Above VWAP + Breaking High + PCR > 0.7
        - 🔴 **BUY PUT:** RVol ≥ 1.5x + Below VWAP + Breaking Low + PCR < 1.3
        """)

# ===== LOGS =====
st.divider()
with st.expander("📋 System Logs", expanded=False):
    logs = store_get_logs()
    if logs:
        for log in logs[:30]:
            st.code(log, language=None)
    else:
        st.info("No logs yet")

# ===== AUTO REFRESH =====
if STORE['is_running']:
    time.sleep(2)
    st.rerun()
