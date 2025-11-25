"""
🎯 INSTITUTIONAL SNIPER - FINAL WORKING VERSION
"""

import streamlit as st
import requests
import threading
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import time
import urllib.parse
import json
import sys

st.set_page_config(page_title="Institutional Sniper", page_icon="🎯", layout="wide")

st.markdown("""
<style>
.stMetric{background-color:#1e293b;padding:20px;border-radius:10px;border:2px solid #334155;}
.accumulation{background-color:#065f46!important;border:3px solid #10b981!important;}
.distribution{background-color:#7f1d1d!important;border:3px solid #ef4444!important;}
.signal-badge{padding:8px 16px;border-radius:6px;font-weight:700;display:inline-block;margin:10px 0;}
.signal-accumulation{background:#10b981;color:white;}
.signal-distribution{background:#ef4444;color:white;}
.signal-idle{background:#334155;color:#94a3b8;}
</style>
""", unsafe_allow_html=True)

# =================== GLOBALS ======================================
LOCK = threading.Lock()
ACTIVE = {'value': False}
DATA = {}
LOGS = deque(maxlen=50)

STOCKS = {
    'RELIANCE': 'INE002A01018', 'TCS': 'INE467B01029', 'INFY': 'INE009A01021',
    'HDFCBANK': 'INE040A01034', 'SBIN': 'INE062A01020', 'ICICIBANK': 'INE090A01021',
    'SBICARD': 'INE018E01016', 'TATAMOTORS': 'INE155A01022', 'BAJFINANCE': 'INE296A01024'
}

# =================== INIT ======================================
for k in ['auth', 'token', 'key', 'secret', 'redirect', 'stocks']:
    if k not in st.session_state:
        st.session_state[k] = False if k == 'auth' else '' if k != 'redirect' else 'https://127.0.0.1' if k != 'stocks' else []

# =================== UTILS ======================================
def log(msg):
    with LOCK:
        LOGS.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def safe_float(x): 
    try: return float(x)
    except: return 0.0

def safe_int(x):
    try: return int(x)
    except: return 0

# =================== API ======================================
def get_quotes(symbols, token):
    try:
        keys = ','.join([urllib.parse.quote(f"NSE_EQ:{s}", safe='') for s in symbols])
        r = requests.get(f"https://api.upstox.com/v2/market-quote/quotes?symbol={keys}",
            headers={"Authorization": f"Bearer {token}"}, timeout=10)
        if r.status_code == 200:
            return r.json().get('data', {})
        log(f"Quote error: {r.status_code}")
    except Exception as e:
        log(f"Exception: {e}")
    return {}

# =================== ENGINE ======================================
def engine(stocks, token):
    log(f"🚀 Engine started: {', '.join(stocks)}")
    cycle = 0
    
    while ACTIVE['value']:
        try:
            cycle += 1
            quotes = get_quotes(stocks, token)
            
            if not quotes:
                time.sleep(5)
                continue
            
            for sym in stocks:
                key = f"NSE_EQ:{sym}"
                if key not in quotes:
                    continue
                
                q = quotes[key]
                ltp = safe_float(q.get('ohlc', {}).get('close', 0))
                vol = safe_int(q.get('volume', 0))
                depth = q.get('depth', {})
                buy = sum(safe_int(b.get('quantity', 0)) for b in depth.get('buy', []))
                sell = sum(safe_int(s.get('quantity', 0)) for s in depth.get('sell', []))
                
                with LOCK:
                    if sym not in DATA:
                        DATA[sym] = {'buf': deque(maxlen=10), 'prev_vol': 0, 'prev_time': time.time()}
                    d = DATA[sym]
                    
                    dt = time.time() - d['prev_time']
                    if dt > 0:
                        vol_rate = abs(vol - d['prev_vol']) / dt
                        ob_rate = max(1, (abs(buy) + abs(sell)) / dt)
                        ratio = vol_rate / ob_rate
                        d['buf'].append(ratio)
                    
                    avg_ratio = np.mean(d['buf']) if d['buf'] else 0
                    signal = 'IDLE'
                    
                    if avg_ratio > 2.5:
                        if sell > 1.5 * buy:
                            signal = 'ACCUMULATION'
                        elif buy > 1.5 * sell:
                            signal = 'DISTRIBUTION'
                    
                    DATA[sym] = {
                        'ltp': ltp, 'vol': vol, 'buy': buy, 'sell': sell,
                        'ratio': avg_ratio, 'signal': signal, 'buf': d['buf'],
                        'prev_vol': vol, 'prev_time': time.time(), 'updated': datetime.now()
                    }
            
            if cycle % 10 == 0:
                log(f"Cycle {cycle}: Monitoring {len(stocks)} stocks")
            
            time.sleep(3)
        
        except Exception as e:
            log(f"Error: {e}")
            time.sleep(5)
    
    log("🛑 Engine stopped")

# =================== UI ======================================
st.title("🎯 Institutional Sniper")

# Sidebar
with st.sidebar:
    st.header("🔐 Authentication")
    
    st.session_state.key = st.text_input("API Key", type="password", value=st.session_state.key)
    st.session_state.secret = st.text_input("API Secret", type="password", value=st.session_state.secret)
    st.session_state.redirect = st.text_input("Redirect URI", value=st.session_state.redirect)
    
    if st.button("🔗 Generate URL"):
        if st.session_state.key:
            uri = urllib.parse.quote(st.session_state.redirect, safe='')
            url = f"https://api.upstox.com/v2/login/authorization/dialog?client_id={st.session_state.key}&redirect_uri={uri}&response_type=code"
            st.code(url, language=None)
    
    code = st.text_input("Auth Code")
    if st.button("🚀 Connect"):
        if code:
            try:
                r = requests.post("https://api.upstox.com/v2/login/authorization/token",
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    data={
                        "code": code,
                        "client_id": st.session_state.key,
                        "client_secret": st.session_state.secret,
                        "redirect_uri": st.session_state.redirect,
                        "grant_type": "authorization_code"
                    }, timeout=10)
                
                if r.status_code == 200:
                    st.session_state.token = r.json()['access_token']
                    st.session_state.auth = True
                    st.success("✅ Connected!")
                    log("Authentication successful")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Failed: {r.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown("---")
    st.metric("Status", "🟢 LIVE" if st.session_state.auth else "🔴 OFF")
    
    st.markdown("---")
    st.header("📊 Watchlist")
    
    symbols = st.text_area("Symbols (comma-separated)", 
        value="SBICARD,RELIANCE,TCS", height=80)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ Start", use_container_width=True):
            if not st.session_state.auth:
                st.error("Auth first!")
            else:
                syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
                if syms:
                    # Stop old
                    ACTIVE['value'] = False
                    time.sleep(2)
                    
                    # Clear
                    with LOCK:
                        DATA.clear()
                    
                    # Start new
                    st.session_state.stocks = syms
                    ACTIVE['value'] = True
                    
                    t = threading.Thread(
                        target=engine,
                        args=(syms, st.session_state.token),
                        daemon=True
                    )
                    t.start()
                    
                    st.success(f"Started: {', '.join(syms)}")
                    log(f"Started monitoring: {', '.join(syms)}")
                    time.sleep(1)
                    st.rerun()
    
    with col2:
        if st.button("⏹️ Stop", use_container_width=True):
            ACTIVE['value'] = False
            st.session_state.stocks = []
            log("Stopped monitoring")
            time.sleep(1)
            st.rerun()
    
    # Kill switch
    st.markdown("---")
    if st.button("🔴 EMERGENCY STOP", use_container_width=True):
        ACTIVE['value'] = False
        with LOCK:
            DATA.clear()
        st.session_state.stocks = []
        st.warning("Emergency stop activated!")
        time.sleep(1)
        st.rerun()

# Main dashboard
if st.session_state.stocks and ACTIVE['value']:
    st.subheader("📊 Live Monitor")
    
    with LOCK:
        data_copy = dict(DATA)
    
    cols = st.columns(3)
    
    for i, sym in enumerate(st.session_state.stocks):
        d = data_copy.get(sym, {})
        
        ltp = d.get('ltp', 0)
        vol = d.get('vol', 0)
        buy = d.get('buy', 0)
        sell = d.get('sell', 0)
        ratio = d.get('ratio', 0)
        signal = d.get('signal', 'IDLE')
        updated = d.get('updated')
        
        card = "accumulation" if signal == "ACCUMULATION" else "distribution" if signal == "DISTRIBUTION" else ""
        
        with cols[i % 3]:
            st.markdown(f'<div class="stMetric {card}">', unsafe_allow_html=True)
            st.metric(f"📈 {sym}", f"₹{ltp:.2f}" if ltp > 0 else "Loading...")
            st.markdown(f'<div class="signal-badge signal-{signal.lower()}">{signal}</div>', unsafe_allow_html=True)
            st.text(f"Ratio: {ratio:.2f}x")
            st.text(f"Vol: {vol:,}")
            st.text(f"Buy: {buy:,} | Sell: {sell:,}")
            if updated:
                st.caption(f"{updated.strftime('%H:%M:%S')}")
            st.markdown("</div>", unsafe_allow_html=True)
    
    time.sleep(3)
    st.rerun()

else:
    st.info("👆 Connect → Enter symbols → Start")
    
    with st.expander("📖 Instructions"):
        st.markdown("""
        **Setup:**
        1. Get API credentials from upstox.com/developer
        2. Enter API Key and Secret
        3. Click "Generate URL" and open in browser
        4. Authorize and copy the code from redirect URL
        5. Paste code and click Connect
        
        **Usage:**
        1. Enter stock symbols (comma-separated)
        2. Click Start
        3. Monitor signals in real-time
        
        **Signals:**
        - 🟢 ACCUMULATION: Institutional buying
        - 🔴 DISTRIBUTION: Institutional selling
        - ⚪ IDLE: No significant activity
        """)

# Logs
st.markdown("---")
with st.expander("📋 Activity Logs"):
    with LOCK:
        logs = list(LOGS)
    if logs:
        st.text("\n".join(logs))
    else:
        st.info("No logs yet")

st.caption("🎯 Institutional Sniper v2.0")