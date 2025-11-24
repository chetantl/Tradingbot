import streamlit as st
import requests
import threading
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np
import time
import urllib.parse

# =================== CONFIGURATION AND HELPERS ==============================

st.set_page_config(page_title="Institutional Sniper", page_icon="🎯", layout="wide")
st.markdown("""
<style>
.stMetric { background-color: #1e293b; padding: 20px; border-radius: 10px; border: 2px solid #334155;}
.accumulation {background-color:#065f46 !important;border:3px solid #10b981 !important;}
.distribution {background-color:#7f1d1d !important;border:3px solid #ef4444 !important;}
.signal-badge {padding:8px 16px; border-radius:6px; font-weight:700;display:inline-block;margin:10px 0;}
.signal-accumulation {background:#10b981;color:white;}
.signal-distribution {background:#ef4444;color:white;}
.signal-idle {background:#334155;color:#94a3b8;}
</style>
""", unsafe_allow_html=True)

def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    if 'logs' not in st.session_state: st.session_state.logs = []
    st.session_state.logs.insert(0, f"[{timestamp}] {level}: {msg}")
    if len(st.session_state.logs) > 100: st.session_state.logs.pop()

def safe_float(x):
    try: return float(x) if x is not None else 0.
    except: return 0.

def safe_int(x):
    try: return int(x) if x is not None else 0
    except: return 0

def get_upstox_auth_header():
    return {"Accept": "application/json", "Authorization": f"Bearer {st.session_state.access_token}"}

# ==================== SESSION STATE INIT ====================================

for k, v in [
    ('authenticated', False), ('access_token', None), ('stock_data', {}),
    ('active_instruments', []), ('equity_thread', None), ('monitoring_active', False),
    ('logs', []), ('api_key', ''), ('api_secret', ''), ('redirect_uri', 'https://127.0.0.1'),
    ('option_cache', defaultdict(dict))
]:
    if k not in st.session_state: st.session_state[k] = v

CONFIG = {
    'rvol_threshold': 2.0,
    'ratio_threshold': 2.5,
    'strong_ratio': 4.0,
    'persistence_count': 3,
    'strong_persistence': 1
}

# ====================== AUTHENTICATION SIDEBAR ==============================

st.sidebar.title("🔐 Authentication")
st.session_state.api_key = st.sidebar.text_input("API Key", type="password", value=st.session_state.api_key)
st.session_state.api_secret = st.sidebar.text_input("API Secret", type="password", value=st.session_state.api_secret)
st.session_state.redirect_uri = st.sidebar.text_input("Redirect URI", value=st.session_state.redirect_uri)

if st.sidebar.button("Generate Login URL"):
    if st.session_state.api_key and st.session_state.api_secret:
        encoded_uri = urllib.parse.quote(st.session_state.redirect_uri, safe='')
        url = f"https://api.upstox.com/v2/login/authorization/dialog?client_id={st.session_state.api_key}&redirect_uri={encoded_uri}&response_type=code"
        st.sidebar.code(url, language=None)
        st.sidebar.info("📋 Copy URL, login in browser, paste auth code below")
    else:
        st.sidebar.error("Enter API credentials first")

auth_code = st.sidebar.text_input("Authorization Code")
if st.sidebar.button("🚀 Connect to LIVE Market"):
    if not auth_code:
        st.sidebar.error("Enter auth code")
    else:
        try:
            with st.spinner("Authenticating..."):
                response = requests.post(
                    "https://api.upstox.com/v2/login/authorization/token",
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    data={
                        "code": auth_code,
                        "client_id": st.session_state.api_key,
                        "client_secret": st.session_state.api_secret,
                        "redirect_uri": st.session_state.redirect_uri,
                        "grant_type": "authorization_code"
                    }
                )
                if response.status_code == 200:
                    token = response.json()['access_token']
                    st.session_state.access_token = token
                    st.session_state.authenticated = True
                    st.sidebar.success("✅ Connected to LIVE MARKET")
                    st.rerun()
                else:
                    st.sidebar.error(f"❌ Auth failed: {response.text}")
                    log(f"Authentication failed: {response.text}", "ERROR")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {e}")
            log(f"Authentication error: {e}", "ERROR")
if st.session_state.authenticated:
    st.sidebar.success("🟢 LIVE MARKET CONNECTED")
else:
    st.sidebar.warning("🔴 Not Connected")

# =================== API: HISTORICAL AND MARKET QUOTE =======================

def fetch_5min_historical(symbol):
    # Fetch last 10 trading days of 5m candles
    inst_key = f"NSE_EQ|{symbol}"
    today = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d')
    url = f"https://api.upstox.com/v2/historical-candle/{inst_key}/5minute/{today}/{from_date}"
    headers = get_upstox_auth_header()
    response = requests.get(url, headers=headers, timeout=12)
    if response.status_code == 200:
        data = response.json()
        if 'data' in data and 'candles' in data['data']:
            return data['data']['candles']
    return []

def fetch_market_quote(symbols):
    keys = [f"NSE_EQ|{sym}" for sym in symbols]
    encoded = [urllib.parse.quote(k, safe='') for k in keys]
    url = f"https://api.upstox.com/v2/market-quote/quotes?symbol={','.join(encoded)}"
    response = requests.get(url, headers=get_upstox_auth_header())
    return response.json()['data'] if response.status_code == 200 else {}

def fetch_option_chain(symbol, expiry):
    # Cache expiry/option data for speed if polled a lot
    now = int(time.time())
    cachekey = (symbol, expiry)
    cache = st.session_state.option_cache
    if cachekey in cache and now - cache[cachekey].get('t', 0) < 10:  # 10 secs cache
        return cache[cachekey]['d']
    inst_key = f"NSE_EQ|{symbol}"
    url = "https://api.upstox.com/v2/option/chain"
    params = {'instrument_key': inst_key, 'expiry_date': expiry}
    response = requests.get(url, params=params, headers=get_upstox_auth_header(), timeout=12)
    data = response.json() if response.status_code == 200 else {}
    cache[cachekey] = {'t': now, 'd': data}
    return data

# =============== RVOL FILTER - ADVANCED HISTORICAL 5M LOGIC =================

def get_current_5min_index():
    now = datetime.now()
    # NSE opens at 9:15; first bin 9:15-9:20 is slot 0, and so on
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    slot = int(((now - market_open).total_seconds()) // 300)
    return slot if slot >= 0 else 0

def rvol_candidates(symbol_list):
    """Screen symbols with RVol >= 2.0 for the *current 5-min slot* over last 10 days"""
    rvols = []
    slot = get_current_5min_index()
    for sym in symbol_list:
        candles = fetch_5min_historical(sym)
        if not candles or len(candles) < slot+1: continue
        volumes_by_slot = [[] for _ in range(79)]  # 79 bins per day 9:15–15:30
        for candle in candles:
            c_time = datetime.fromtimestamp(candle[0]//1000)
            idx = ((c_time.hour-9)*12 + (c_time.minute-15)//5) if c_time.hour >= 9 else None
            if idx is not None and 0 <= idx < 79:
                volumes_by_slot[idx].append(candle[5])
        # Current time's bin
        ref_vols = volumes_by_slot[slot][-10:]  # Last 10 trading days for this slot
        avg_vol = np.mean(ref_vols) if ref_vols else 0
        # Current live 5-min candle in slot
        curr_candle = candles[-1] if len(candles) > 0 else [0,0,0,0,0,0]
        curr_vol = curr_candle[5]
        rvol = (curr_vol/avg_vol) if avg_vol else 0
        rvols.append({'symbol': sym, 'rvol': rvol, 'curr_vol': curr_vol, 'avg_vol': avg_vol})
    return [x['symbol'] for x in rvols if x['rvol'] >= CONFIG['rvol_threshold']]

# =============== LIVE ABSORPTION ENGINE + OPTIONS CONFIRMATION ==============

def get_next_expiry(symbol):
    # Placeholder: In real use, parse earliest expiry date from option chain
    # Common format: next Thursday/Friday, or closest available future
    now = datetime.now()
    weekday = now.weekday()
    # Friday expiry for stocks
    offset = (4-weekday)%7
    expiry_date = now + timedelta(days=offset)
    return expiry_date.strftime("%Y-%m-%d")

def find_multi_strike_indices(strikes, ltp):
    strikes = sorted(list(set(strikes)))
    atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i]-ltp))
    one_itm = max(atm_idx-1, 0)
    two_otm = [atm_idx+1, atm_idx+2]
    otm_idxs = [i for i in two_otm if i < len(strikes)]
    return atm_idx, one_itm, otm_idxs

def equity_and_option_engine(symbols):
    """Main live polling thread (market data + option chain confirmation)"""
    state = st.session_state
    min_delay = 2
    while state.monitoring_active:
        qdata = fetch_market_quote(symbols)
        for sym in symbols:
            api_key = f"NSE_EQ|{sym}"
            if api_key not in q:
                 continue
            quote_data = qdata[api_key]
            sdata = state.stock_data.setdefault(sym, {})
            ohlc = quote_data.get('ohlc', {})
            depth = quote_data.get('depth', {})
            ltp = safe_float(ohlc.get('close', ohlc.get('open', 0)))
            vol = safe_int(quote_data.get('volume', 0))
            buy_orders = depth.get('buy', [])
            sell_orders = depth.get('sell', [])
            buy_qty = sum(safe_int(b.get('quantity', 0)) for b in buy_orders)
            sell_qty = sum(safe_int(s.get('quantity', 0)) for s in sell_orders)
            high = safe_float(ohlc.get('high', ltp)); low = safe_float(ohlc.get('low', ltp))
            vwap = (high+low)/2 if (high and low) else ltp
            now = time.time()
            delta_t = max(0.5, now - sdata.get('prev_time', now-2))
            vol_rate = abs(vol-sdata.get('prev_vol', 0))/delta_t
            ob_delta = abs(buy_qty-sdata.get('prev_buy', 0))+abs(sell_qty-sdata.get('prev_sell', 0))
            ob_rate = ob_delta/delta_t if ob_delta else 1
            ratio = vol_rate/ob_rate if ob_rate else 0
            sdata.setdefault('ratio_buffer', deque(maxlen=20)).append(ratio)
            avg_ratio = np.mean(sdata['ratio_buffer'])
            sdata.update({'ltp': ltp, 'vwap': vwap, 'ratio': avg_ratio})
            # Signal logic (your rules)
            signal=None
            if avg_ratio > CONFIG['ratio_threshold'] and sell_qty > 1.5*buy_qty and ltp>=vwap:
                signal = "ACCUMULATION"
            elif avg_ratio > CONFIG['ratio_threshold'] and buy_qty > 1.5*sell_qty and ltp<vwap:
                signal = "DISTRIBUTION"
            persist = CONFIG['strong_persistence'] if avg_ratio>CONFIG['strong_ratio'] else CONFIG['persistence_count']
            if signal:
                sdata['persist_count'] = sdata.get('persist_count',0)+1
                if sdata['persist_count'] >= persist:
                    sdata['signal'] = signal; sdata['confirmed'] = True
                    # ========== LIVE OPTION CHAIN CONFIRMATION (1ITM+2OTM) ==========
                    expiry = get_next_expiry(sym)
                    oc = fetch_option_chain(sym, expiry)
                    details = {'confirmed': False} # What we'll populate
                    if oc and 'data' in oc:
                        # Find ATM, 1ITM, 2OTM indices
                        calls = oc['data'].get('call_options', [])
                        puts = oc['data'].get('put_options', [])
                        strikes = [c.get('strike_price') for c in calls]
                        atm_idx, one_itm_idx, otm_idxs = find_multi_strike_indices(strikes, ltp)
                        # Accumulation: Calls (ITM,ATM,OTM), Distribution: Put (ITM,ATM,OTM)
                        ce = [calls[i] for i in [one_itm_idx, atm_idx]+otm_idxs if i<len(calls)]
                        pe = [puts[i] for i in [one_itm_idx, atm_idx]+otm_idxs if i<len(puts)]
                        call_oi = sum(safe_int(x.get('open_interest',0)) for x in ce)
                        put_oi = sum(safe_int(x.get('open_interest',0)) for x in pe)
                        call_vol = sum(safe_int(x.get('volume',0)) for x in ce)
                        put_vol = sum(safe_int(x.get('volume',0)) for x in pe)
                        confirm = False
                        if signal=="ACCUMULATION":
                            confirm = (call_vol > 1.5*put_vol) and (call_oi > put_oi)
                        elif signal=="DISTRIBUTION":
                            confirm = (put_vol > 1.5*call_vol) and (put_oi > call_oi)
                        details.update(dict(
                            atm_strike=strikes[atm_idx],
                            ITM_C=ce[0].get('strike_price') if ce else None,
                            OTM_C=[c.get('strike_price') for c in ce[2:]] if len(ce)>2 else [],
                            ITM_P=pe[0].get('strike_price') if pe else None,
                            OTM_P=[p.get('strike_price') for p in pe[2:]] if len(pe)>2 else [],
                            call_oi=call_oi, put_oi=put_oi, call_vol=call_vol, put_vol=put_vol,
                            confirmed=confirm
                        ))
                        sdata['options_confirmed'] = confirm
                        sdata['details'] = details
                        log(f"{sym} {signal} OptConf={confirm} [CALL OI {call_oi} vol {call_vol} | PUT OI {put_oi} vol {put_vol}]", "INFO")
            else:
                sdata['persist_count'] = 0
                sdata['confirmed'] = False
                sdata['options_confirmed'] = False
            sdata['prev_vol'] = vol
            sdata['prev_buy'] = buy_qty
            sdata['prev_sell'] = sell_qty
            sdata['prev_time'] = now
        time.sleep(min_delay)

# ============ DASHBOARD ==============

st.title("🎯 Institutional Sniper Dashboard")
st.markdown("**Strategy:** Absorption Trading - Real-time iceberg, options-confirmed, persistent")

if not st.session_state.monitoring_active:
    st.sidebar.title("📊 Watchlist & RVol Filter")
    watchlist_input = st.sidebar.text_area("Enter symbols (comma)", value="RELIANCE,TCS,INFY,SBIN", height=70)
    if st.sidebar.button("Start With Smart RVol Filter"):
        syms = [s.strip().upper() for s in watchlist_input.split(",") if s.strip()]
        st.info("Screening by 5-min RVol, please wait…")
        candidates = rvol_candidates(syms)
        if candidates:
            st.session_state.active_instruments = candidates
            for s in candidates:
                st.session_state.stock_data[s] = {}
            st.success(f"{len(candidates)} stocks pass 5-min RVol≥2 filter and will be scanned.")
            st.session_state.monitoring_active = True
            threading.Thread(target=equity_and_option_engine, args=(candidates,), daemon=True).start()
            st.rerun()
        else:
            st.error("No stocks met RVol filter for this time slot!")
else:
    stocks = st.session_state.active_instruments
    cols = st.columns(3)
    for i, sym in enumerate(stocks):
        col = cols[i % 3]
        d = st.session_state.stock_data.get(sym, {})
        ltp = d.get('ltp',0); vwap=d.get('vwap',0); ratio = d.get('ratio',0)
        signal = d.get('signal','IDLE')
        confirmed = d.get('confirmed',False)
        opts_conf = d.get('options_confirmed',False)
        det = d.get('details', {})
        class_str = "accumulation" if signal=="ACCUMULATION" and confirmed else "distribution" if signal=="DISTRIBUTION" and confirmed else ""
        badge_class = f"signal-{signal.lower()}"
        status = (
            f"✅ ACCUMULATION" if signal=="ACCUMULATION" and confirmed else
            f"✅ DISTRIBUTION" if signal=="DISTRIBUTION" and confirmed else
            "⏳ Monitoring…"
        )
        substat = (f"Options Confirmed" if opts_conf else f"Options Weak") if confirmed else ""
        with col:
            st.markdown(f'<div class="stMetric {class_str}">', unsafe_allow_html=True)
            st.metric(label=f"{sym}", value=f"₹{ltp:.2f}")
            st.markdown(f'<div class="signal-badge {badge_class}">{signal}</div>', unsafe_allow_html=True)
            st.text(f"Ratio: {ratio:.2f} | VWAP: ₹{vwap:.2f}")
            st.text(status)
            st.text(substat)
            if det: st.text(f"ITM:{det.get('ITM_C','-') or det.get('ITM_P','-')}, ATM:{det.get('atm_strike','-')}, OTM:{','.join(map(str, det.get('OTM_C',[])+det.get('OTM_P',[])))}")
            st.text(f"Call OI:{det.get('call_oi','-')} Put OI:{det.get('put_oi','-')} | Call Vol:{det.get('call_vol','-')} Put Vol:{det.get('put_vol','-')}")
            st.markdown("</div>", unsafe_allow_html=True)
    st.button("⏹️ Stop Monitoring", on_click=lambda: st.session_state.update({'monitoring_active':False}))
    st.markdown("---")

st.subheader("📋 Logs")
if st.session_state.get('logs'): st.text_area("Recent Activity", value="\n".join(st.session_state.logs[:25]), height=220, disabled=True)
else: st.info("No logs yet.")
