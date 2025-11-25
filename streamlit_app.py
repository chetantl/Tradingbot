import streamlit as st
import requests
import threading
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import time
import urllib.parse
import json

# Page config and styling
st.set_page_config(page_title="Institutional Sniper", page_icon="🎯", layout="wide")
st.markdown("""
<style>
.stMetric {
    background-color: #1e293b; padding: 20px; border-radius: 10px; border: 2px solid #334155;
}
.accumulation {
    background-color: #065f46 !important; border: 3px solid #10b981 !important; animation: pulse-green 2s infinite;
}
.distribution {
    background-color: #7f1d1d !important; border: 3px solid #ef4444 !important; animation: pulse-red 2s infinite;
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
    50% { box-shadow: 0 0 20px 10px rgba(16, 185, 129, 0.1); }
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
    50% { box-shadow: 0 0 20px 10px rgba(239, 68, 68, 0.1); }
}
.signal-badge {
    padding: 8px 16px; border-radius: 6px; font-weight: 700; display: inline-block; margin: 10px 0;
}
.signal-accumulation {
    background: #10b981; color: white;
}
.signal-distribution {
    background: #ef4444; color: white;
}
.signal-idle {
    background: #334155; color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# Thread-safe globals
MONITORING_ACTIVE = threading.Event()
DATA_LOCK = threading.Lock()
SHARED_DATA = {}

# Config for thresholds and persistence
CONFIG = {
    'rvol_threshold': 1.5,         # RVol threshold to pass filter
    'ratio_threshold': 2.5,        # Absorption ratio for signal
    'strong_ratio': 4.0,           # Strong absorption ratio
    'persistence_count': 3,        # Ticks to persist for normal ratio
    'strong_persistence': 1        # Ticks to persist for strong ratio
}

COMMON_STOCKS = {
    'RELIANCE': 'INE002A01018',
    'TCS': 'INE467B01029',
    'INFY': 'INE009A01021',
    'HDFCBANK': 'INE040A01034',
    'SBIN': 'INE062A01020',
    'ICICIBANK': 'INE090A01021'
}

def safe_float(x):
    try:
        return float(x) if x is not None else 0.0
    except:
        return 0.0

def safe_int(x):
    try:
        return int(x) if x is not None else 0
    except:
        return 0

def log(message):
    if 'logs' not in st.session_state:
        st.session_state['logs'] = deque(maxlen=50)
    st.session_state.logs.appendleft(f"{datetime.now().strftime('%H:%M:%S')} - {message}")

def get_upstox_headers():
    return {"Accept": "application/json", "Authorization": f"Bearer {st.session_state.access_token}"}

def get_isin(symbol):
    symbol = symbol.upper().strip()
    if symbol in COMMON_STOCKS:
        return f"NSE_EQ|{COMMON_STOCKS[symbol]}", symbol
    if symbol in st.session_state.get('instrument_map', {}):
        return f"NSE_EQ|{st.session_state.instrument_map[symbol]}", symbol
    try:
        url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            import gzip, io
            with gzip.open(io.BytesIO(resp.content), 'rt', encoding='utf-8') as f:
                instruments = json.load(f)
            for i in instruments:
                if i.get('trading_symbol', '') == symbol and i.get('instrument_type') == 'EQ':
                    isin = i.get('instrument_key', '').split('|')[1]
                    instrument_map = st.session_state.get('instrument_map', {})
                    instrument_map[symbol] = isin
                    st.session_state['instrument_map'] = instrument_map
                    return f"NSE_EQ|{isin}", symbol
    except Exception as e:
        log(f"ISIN lookup failed for {symbol}: {str(e)}")
    return None, None

def fetch_5min_candles(inst_key):
    encoded_key = urllib.parse.quote(inst_key, safe='')
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d")
    url = f"https://api.upstox.com/v2/historical-candle/{encoded_key}/5minute/{to_date}/{from_date}"
    try:
        resp = requests.get(url, headers=get_upstox_headers(), timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if 'data' in data and 'candles' in data['data']:
                return data['data']['candles']
    except Exception as e:
        log(f"5-min candle fetch failed ({inst_key}): {str(e)}")
    return []

def calculate_rvol(symbol, inst_key):
    candles = fetch_5min_candles(inst_key)
    if not candles:
        log(f"No 5-min candles found for {symbol}")
        return 0.0
    # Calculate slot index based on current time of day
    now = datetime.now()
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    slot_idx = int((now - market_open).total_seconds() // 300)
    if slot_idx < 0 or slot_idx >= 79:
        log(f"Market closed - slot idx {slot_idx}")
        return 0.0
    # Group candle volumes per slot over days
    slot_volumes = [[] for _ in range(79)]
    for c in candles:
        ts = datetime.fromtimestamp(c[0] // 1000)
        idx = (ts.hour - 9) * 12 + (ts.minute - 15) // 5
        if 0 <= idx < 79:
            slot_volumes[idx].append(c[5])
    # Calculate avg volume in same slot last 10 days
    avg_vol = np.mean(slot_volumes[slot_idx][-10:]) if len(slot_volumes[slot_idx]) >= 10 else 0
    curr_vol = slot_volumes[slot_idx][-1] if len(slot_volumes[slot_idx]) > 0 else 0
    rvol = curr_vol / avg_vol if avg_vol > 0 else 0
    return rvol

def fetch_market_data(inst_keys):
    encoded_keys = [urllib.parse.quote(k, safe='') for k in inst_keys]
    url = f"https://api.upstox.com/v2/market-quote/quotes?symbol={','.join(encoded_keys)}"
    try:
        resp = requests.get(url, headers=get_upstox_headers(), timeout=10)
        if resp.status_code == 200:
            return resp.json().get('data', {})
    except Exception as e:
        log(f"Market data fetch failed: {str(e)}")
    return {}

def fetch_option_chain(inst_key, expiry):
    encoded_key = urllib.parse.quote(inst_key, safe='')
    url = f"https://api.upstox.com/v2/option/chain?instrument_key={encoded_key}&expiry_date={expiry}"
    try:
        resp = requests.get(url, headers=get_upstox_headers(), timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        log(f"Option chain fetch failed: {str(e)}")
    return {}

def find_strike_indices(strikes, ltp):
    strikes = sorted(set(strikes))
    atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - ltp))
    one_itm_idx = max(atm_idx - 1, 0)
    two_otm_idxs = [i for i in [atm_idx + 1, atm_idx + 2] if i < len(strikes)]
    return atm_idx, one_itm_idx, two_otm_idxs

def get_expiry():
    now = datetime.now()
    offset = (4 - now.weekday()) % 7
    return (now + timedelta(days=offset)).strftime("%Y-%m-%d")

def monitoring_engine(stock_list, token):
    log(f"Monitoring started for: {stock_list}")
    while MONITORING_ACTIVE.is_set():
        try:
            inst_keys = [inst['key'] for inst in stock_list]
            market_data = fetch_market_data(inst_keys)
            for stock in stock_list:
                sym = stock['symbol']
                key = stock['key']
                if key not in market_
                    continue
                q = market_data[key]
                ohlc = q.get('ohlc', {})
                ltp = safe_float(ohlc.get('close', ohlc.get('open', 0)))
                vol = safe_int(q.get('volume', 0))
                depth = q.get('depth', {})
                buy_qty = sum(safe_int(b.get('quantity', 0)) for b in depth.get('buy', []))
                sell_qty = sum(safe_int(s.get('quantity', 0)) for s in depth.get('sell', []))
                now = time.time()
                with DATA_LOCK:
                    data = SHARED_DATA.setdefault(sym, {
                        'ratios': deque(maxlen=20), 'persist_count': 0,
                        'prev_vol': 0, 'prev_buy': 0, 'prev_sell': 0,
                        'prev_time': now, 'signal': 'IDLE', 'confirmed': False, 'details': {}
                    })
                delta_t = max(0.5, now - data['prev_time'])
                vol_rate = abs(vol - data['prev_vol']) / delta_t
                ob_rate = (abs(buy_qty - data['prev_buy']) + abs(sell_qty - data['prev_sell'])) / delta_t if delta_t > 0 else 1
                ratio = vol_rate / ob_rate if ob_rate > 0 else 0
                data['ratios'].append(ratio)
                avg_ratio = np.mean(data['ratios'])
                candles = fetch_5min_candles(key)
                vwap = safe_float(np.sum([c[4]*c[5] for c in candles]) / np.sum([c[5] for c in candles])) if candles else ltp
                with DATA_LOCK:
                    data.update({
                        'ltp': ltp, 'vwap': vwap, 'vol': vol, 'ratio': avg_ratio,
                        'prev_vol': vol, 'prev_buy': buy_qty, 'prev_sell': sell_qty, 'prev_time': now
                    })
                signal = None
                if avg_ratio > CONFIG['ratio_threshold']:
                    if sell_qty > 1.5 * buy_qty and ltp >= vwap:
                        signal = 'ACCUMULATION'
                    elif buy_qty > 1.5 * sell_qty and ltp < vwap:
                        signal = 'DISTRIBUTION'
                persist_needed = CONFIG['strong_persistence'] if avg_ratio > CONFIG['strong_ratio'] else CONFIG['persistence_count']
                with DATA_LOCK:
                    if signal == data.get('signal'):
                        data['persist_count'] = data.get('persist_count', 0) + 1
                    else:
                        data['persist_count'] = 1
                    data['signal'] = signal or 'IDLE'
                    if data['persist_count'] >= persist_needed and signal is not None:
                        expiry = get_expiry()
                        opts = fetch_option_chain(key, expiry)
                        details = {'confirmed': False}
                        if opts and 'data' in opts:
                            calls = opts['data'].get('call_options', [])
                            puts = opts['data'].get('put_options', [])
                            strikes = [c.get('strike_price') for c in calls]
                            if strikes:
                                atm_idx, one_itm, otm_idxs = find_strike_indices(strikes, ltp)
                                ce = [calls[i] for i in [one_itm, atm_idx] + otm_idxs if i < len(calls)]
                                pe = [puts[i] for i in [one_itm, atm_idx] + otm_idxs if i < len(puts)]
                                call_oi = sum(safe_int(x.get('open_interest', 0)) for x in ce)
                                put_oi = sum(safe_int(x.get('open_interest', 0)) for x in pe)
                                call_vol = sum(safe_int(x.get('volume', 0)) for x in ce)
                                put_vol = sum(safe_int(x.get('volume', 0)) for x in pe)
                                confirm = False
                                if signal == 'ACCUMULATION':
                                    confirm = (call_vol > 1.5 * put_vol) and (call_oi > put_oi)
                                elif signal == 'DISTRIBUTION':
                                    confirm = (put_vol > 1.5 * call_vol) and (put_oi > call_oi)
                                details.update({
                                    'atm': strikes[atm_idx], 'call_oi': call_oi, 'put_oi': put_oi,
                                    'call_vol': call_vol, 'put_vol': put_vol, 'confirmed': confirm
                                })
                        data['details'] = details
                        data['options_confirmed'] = details.get('confirmed', False)
                        data['confirmed'] = True
                    else:
                        data['options_confirmed'] = False
                        data['confirmed'] = False
            time.sleep(2)
        except Exception as e:
            log(f"Engine error: {e}")
            time.sleep(3)

# ============ STREAMLIT UI and CONTROL =============
st.sidebar.title('🔐 Authentication')
st.session_state.api_key = st.sidebar.text_input("API Key", type="password", value=st.session_state.api_key)
st.session_state.api_secret = st.sidebar.text_input("API Secret", type="password", value=st.session_state.api_secret)
st.session_state.redirect = st.sidebar.text_input("Redirect URI", value=st.session_state.get('redirect', 'https://127.0.0.1'))

if st.sidebar.button("Generate Login URL"):
    if st.session_state.api_key and st.session_state.api_secret:
        uri_enc = urllib.parse.quote(st.session_state.redirect, safe='')
        url = f"https://api.upstox.com/v2/login/authorization/dialog?client_id={st.session_state.api_key}&redirect_uri={uri_enc}&response_type=code"
        st.sidebar.code(url)

auth_code = st.sidebar.text_input("Authorization Code")
if st.sidebar.button("Connect"):
    if auth_code:
        try:
            r = requests.post("https://api.upstox.com/v2/login/authorization/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={"code": auth_code, "client_id": st.session_state.api_key,
                      "client_secret": st.session_state.api_secret,
                      "redirect_uri": st.session_state.redirect,
                      "grant_type": "authorization_code"})
            if r.status_code == 200:
                st.session_state.access_token = r.json()['access_token']
                st.session_state.authenticated = True
                st.sidebar.success("✅ Connected")
                time.sleep(1)
                st.experimental_rerun()
            else:
                st.sidebar.error("❌ Authentication failed")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")

st.sidebar.success("🟢 LIVE" if st.session_state.get('authenticated') else "🔴 Disconnected")

st.sidebar.title("📊 Watchlist")
watchlist = st.sidebar.text_area("Enter symbols (comma-separated)", value="RELIANCE,TCS,INFY")

if st.sidebar.button("Start Monitoring"):
    if not st.session_state.get('authenticated'):
        st.sidebar.error("Please authenticate first!")
    else:
        syms = [s.strip().upper() for s in watchlist.split(",") if s.strip()]
        if syms:
            valids = []
            for sym in syms:
                inst_key, validated = get_isin(sym)
                if not inst_key:
                    st.sidebar.error(f"Symbol {sym} invalid/not found")
                    continue
                rvol = calculate_slot_based_rvol(inst_key, st.session_state.access_token)
                if rvol >= CONFIG['rvol_threshold']:
                    valids.append({'key': inst_key, 'symbol': validated})
                    st.sidebar.success(f"{sym} passed RVol filter: {rvol:.2f}")
            if valids:
                SHARED_DATA.clear()
                MONITORING_ACTIVE.set()
                threading.Thread(target=monitoring_engine, args=(valids, st.session_state.access_token), daemon=True).start()
                st.session_state.active_instruments = valids
                st.success(f"Monitoring {len(valids)} symbols")
                time.sleep(1)
                st.experimental_rerun()
            else:
                st.sidebar.error("No stocks passed RVol filter")
        else:
            st.sidebar.error("Enter at least one symbol")

if st.sidebar.button("Stop Monitoring"):
    MONITORING_ACTIVE.clear()
    SHARED_DATA.clear()
    st.session_state.active_instruments = []

st.title("🎯 Institutional Sniper Dashboard")

if st.session_state.get('active_instruments') and MONITORING_ACTIVE.is_set():
    columns = st.columns(3)
    for i, stock in enumerate(st.session_state.active_instruments):
        sym = stock['symbol']
        col = columns[i % 3]
        data = SHARED_DATA.get(sym, {})
        ltp = data.get('ltp', 0)
        vwap = data.get('vwap', 0)
        ratio = data.get('ratio', 0)
        vol = data.get('vol', 0)
        signal = data.get('signal', 'IDLE')
        confirmed = data.get('confirmed', False)
        options_confirmed = data.get('options_confirmed', False)
        details = data.get('details', {})
        card_class = "accumulation" if signal == "ACCUMULATION" and confirmed else "distribution" if signal == "DISTRIBUTION" and confirmed else ""
        with col:
            st.markdown(f'<div class="stMetric {card_class}">', unsafe_allow_html=True)
            st.metric(f"{sym}", f"₹{ltp:.2f}")
            st.markdown(f'<div class="signal-badge signal-{signal.lower()}">{signal}</div>', unsafe_allow_html=True)
            st.text(f"Ratio: {ratio:.2f}x | VWAP: ₹{vwap:.2f}")
            st.text(f"Volume: {vol:,}")
            if confirmed:
                st.text(f"Options Confirmed: {'Yes' if options_confirmed else 'No'}")
                if details:
                    st.text(f"ATM Strike: {details.get('atm', '-')}")
                    st.text(f"Call OI: {details.get('call_oi', '-')} Put OI: {details.get('put_oi', '-')}")
                    st.text(f"Call Vol: {details.get('call_vol', '-')} Put Vol: {details.get('put_vol', '-')}")
            st.markdown("</div>", unsafe_allow_html=True)
    time.sleep(2)
    st.experimental_rerun()
else:
    st.info("Authenticate, enter symbols and start monitoring.")

st.markdown("---")
st.subheader("Logs")
if st.session_state.logs:
    st.text_area("Logs", "\n".join(st.session_state.logs), height=150, disabled=True)
