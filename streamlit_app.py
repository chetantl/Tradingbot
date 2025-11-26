import streamlit as st
from kiteconnect import KiteConnect
import threading
import time
from datetime import datetime, timedelta
from collections import deque
import pandas as pd 
import copy

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Institutional Sniper - Full Strategy",
    page_icon="🎯",
    layout="wide",
)

st.markdown("""
<style>
/* Streamlit UI Overrides */
.stMetric { background-color: #111827; border: 1px solid #374151; border-radius: 8px; padding: 10px; }
.metric-value { font-size: 1.2rem; font-weight: bold; color: #f3f4f6; }
.badge { padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem; }
/* Signal Colors */
.accum { background: #064e3b; color: #6ee7b7; border: 1px solid #059669; }
.dist { background: #7f1d1d; color: #fca5a5; border: 1px solid #dc2626; }
.idle { background: #374151; color: #d1d5db; border: 1px solid #4b5563; }
/* Confirmed Border */
.conf { border: 3px solid #fbbf24; box-shadow: 0 0 12px #fbbf24; }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
CONFIG = {
    "rvol_threshold": 1.5,          # now only for labeling (High/Low), not gating
    "ratio_threshold": 2.5,
    "strong_ratio_threshold": 4.0,
    "persistence_count": 3,
    "strong_persistence": 1,
    "imbalance_mult": 1.5,
    "history_days": 10,
    "redirect_url": "http://127.0.0.1/"
}

# --- UTILITIES ---

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

# --- SINGLETON STATE MANAGEMENT ---

class SniperManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.logs = deque(maxlen=50)
        self.data = {}           # symbol -> state dict
        self.active_symbols = [] # list of {"symbol":..., "token":...}
        self.is_running = False
        self.stop_event = threading.Event()
        self.kite = None
        self.last_beat = 0       # heartbeat for worker

    def log(self, msg, level="INFO"):
        ts = datetime.now().strftime('%H:%M:%S')
        with self.lock:
            self.logs.appendleft(f"[{ts}] {level}: {msg}")

@st.cache_resource
def get_manager():
    return SniperManager()

manager = get_manager()

# hotfix for stale cache
if not hasattr(manager, "last_beat"):
    manager.last_beat = 0

# --- STRATEGY FUNCTIONS ---

def get_instrument_token(kite, symbol):
    """Fetch NSE instrument token for a given symbol."""
    try:
        instruments = kite.instruments("NSE")
        search = symbol.upper().strip()
        for i in instruments:
            if i["tradingsymbol"] == search and i["segment"] == "NSE":
                return i["instrument_token"]
    except:
        pass
    return None

def fetch_time_slot_rvol(kite, token, days=10):
    """RVol for current 5‑min slot vs same slot over history."""
    try:
        now = datetime.now()
        minute = (now.minute // 5) * 5
        start_time = now.replace(minute=minute, second=0, microsecond=0)

        from_date = now - timedelta(days=days + 5)
        candles = kite.historical_data(token, from_date, now, "5minute")
        df = pd.DataFrame(candles)
        if df.empty:
            return 0.0, 0

        df["date"] = pd.to_datetime(df["date"])
        target_time = start_time.time()
        slot_df = df[df["date"].dt.time == target_time]

        if len(slot_df) < 2:
            return 1.0, 0

        historical_vols = slot_df["volume"].iloc[:-1]
        avg_vol = historical_vols.mean()
        current_vol = slot_df["volume"].iloc[-1]

        rvol = current_vol / max(1, avg_vol)
        return rvol, current_vol
    except Exception as e:
        manager.log(f"RVol calc failed: {e}", "ERROR")
        return 0.0, 0

def get_option_confirmation(kite, symbol, signal_type, spot_price):
    """Option OI/volume confirmation for ACCUMULATION/DISTRIBUTION."""
    try:
        opts = [
            i for i in kite.instruments("NFO")
            if i.get("name") == symbol and i.get("segment") == "NFO-OPT"
        ]
        if not opts:
            return False, 0.0

        today = datetime.now().date()
        expiries = sorted(
            {i["expiry"].date() for i in opts if i["expiry"].date() >= today}
        )
        if not expiries:
            return False, 0.0

        curr_expiry = expiries[0]
        opts = [i for i in opts if i["expiry"].date() == curr_expiry]

        strikes = sorted({i["strike"] for i in opts})
        atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
        atm_idx = strikes.index(atm_strike)

        if signal_type == "ACCUMULATION":
            target_indices = [atm_idx - 1, atm_idx + 1, atm_idx + 2]
            opt_type = "CE"
        else:
            target_indices = [atm_idx + 1, atm_idx - 1, atm_idx - 2]
            opt_type = "PE"

        q_params = []
        for idx in target_indices:
            if 0 <= idx < len(strikes):
                strike = strikes[idx]
                inst = next(
                    (
                        i for i in opts
                        if i["strike"] == strike and i["instrument_type"] == opt_type
                    ),
                    None,
                )
                if inst:
                    q_params.append(f"NFO:{inst['tradingsymbol']}")

        if not q_params:
            return False, 0.0

        quotes = kite.quote(q_params)
        total_power = 0.0
        for k, v in quotes.items():
            oi_chg = safe_int(v.get("oi", 0)) - safe_int(
                v.get("ohlc", {}).get("open_oi", 0)
            )
            vol = safe_int(v.get("volume", 0))
            total_power += (oi_chg * vol) / 1_000_000

        return total_power > 0.5, total_power
    except Exception as e:
        manager.log(f"Opt logic error: {e}", "ERROR")
        return False, 0.0

# --- CORE WORKER THREAD ---

def sniper_worker(kite):
    manager.is_running = True
    manager.log("Sniper Engine Started", "SUCCESS")

    try:
        kite.instruments("NFO")
    except:
        pass

    while not manager.stop_event.is_set():
        try:
            manager.last_beat = time.time()

            with manager.lock:
                active_list = list(manager.active_symbols)

            if not active_list:
                time.sleep(1)
                continue

            query = [f"NSE:{i['symbol']}" for i in active_list]

            try:
                quotes = kite.quote(query)
            except Exception as e:
                manager.log(f"Network Error (Retrying): {e}", "WARNING")
                time.sleep(2)
                continue

            now_ts = time.time()

            with manager.lock:
                for item in manager.active_symbols:
                    sym = item["symbol"]
                    key = f"NSE:{sym}"
                    
                    # --- FIXED SYNTAX ERROR AREA ---
                    # The logic below replaces any incomplete lines
                    if key not in quotes or not quotes[key]:
                        continue
                    # -------------------------------
                    
                    q = quotes[key]

                    ltp = safe_float(q.get("last_price"))
                    vol = safe_int(q.get("volume"))
                    depth = q.get("depth", {})
                    buy_q = sum(safe_int(x.get("quantity")) for x in depth.get("buy", []))
                    sell_q = sum(
                        safe_int(x.get("quantity")) for x in depth.get("sell", [])
                    )

                    if sym not in manager:
                        continue

                    stt = manager.data[sym]
                    dt = now_ts - stt.get("prev_ts", now_ts - 2)
                    if dt < 1.0:
                        continue

                    d_vol = vol - stt.get("prev_vol", vol)
                    if d_vol < 0:
                        d_vol = 0

                    stt["cum_pv"] += ltp * d_vol
                    stt["cum_v"] += d_vol
                    vwap = (
                        stt["cum_pv"] / max(1, stt["cum_v"])
                        if stt["cum_v"] > 0
                        else ltp
                    )

                    d_ob = abs(buy_q - stt.get("prev_buy", buy_q)) + abs(
                        sell_q - stt.get("prev_sell", sell_q)
                    )

                    vol_rate = d_vol / dt
                    ob_rate = d_ob / max(1.0, dt)
                    ratio = vol_rate / max(1, ob_rate)

                    raw_signal = "IDLE"
                    if ratio > CONFIG["ratio_threshold"]:
                        if (
                            sell_q > buy_q * CONFIG["imbalance_mult"]
                            and ltp >= vwap
                        ):
                            raw_signal = "ACCUMULATION"
                        elif (
                            buy_q > sell_q * CONFIG["imbalance_mult"]
                            and ltp <= vwap
                        ):
                            raw_signal = "DISTRIBUTION"

                    req_persistence = CONFIG["persistence_count"]
                    if ratio > CONFIG["strong_ratio_threshold"]:
                        req_persistence = CONFIG["strong_persistence"]

                    if raw_signal == stt["signal"] and raw_signal != "IDLE":
                        stt["persist"] += 1
                    elif raw_signal != stt["signal"]:
                        stt["persist"] = 1
                        stt["signal"] = raw_signal
                        stt["confirmed"] = False
                        stt["opt_power"] = 0.0

                    if (
                        stt["signal"] != "IDLE"
                        and stt["persist"] >= req_persistence
                        and not stt["confirmed"]
                    ):
                        confirmed, power = get_option_confirmation(
                            kite, sym, stt["signal"], ltp
                        )
                        stt["confirmed"] = confirmed
                        stt["opt_power"] = power
                        if confirmed:
                            manager.log(
                                f"CONFIRMED {stt['signal']} on {sym} | OptPower: {power:.2f}",
                                "SUCCESS",
                            )

                    stt.update(
                        {
                            "ltp": ltp,
                            "vol": vol,
                            "vwap": vwap,
                            "ratio": ratio if d_vol > 0 else stt.get("ratio", 0.0),
                            "buy_q": buy_q,
                            "sell_q": sell_q,
                            "prev_vol": vol,
                            "prev_buy": buy_q,
                            "prev_sell": sell_q,
                            "prev_ts": now_ts,
                        }
                    )

            time.sleep(2)

        except Exception as e:
            manager.log(f"Worker Fatal Error: {e}", "ERROR")
            time.sleep(2)

# --- STREAMLIT UI: AUTH ---

st.sidebar.title("🔐 Zerodha Kite Connect")

if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "api_sec" not in st.session_state:
    st.session_state.api_sec = ""

api_key = st.sidebar.text_input("API Key", value=st.session_state.api_key)
api_sec = st.sidebar.text_input("API Secret", value=st.session_state.api_sec, type="password")
req_token = st.sidebar.text_input("Request Token")

st.sidebar.markdown("---")
st.sidebar.markdown("**Step 1: Get Request Token**")

if api_key:
    try:
        temp_kite = KiteConnect(api_key=api_key)
        login_link = temp_kite.login_url()
        st.sidebar.markdown(f"[Click here to log in]({login_link})")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

if st.sidebar.button("🚀 Connect"):
    if api_key and api_sec and req_token:
        try:
            k = KiteConnect(api_key=api_key)
            d = k.generate_session(req_token, api_secret=api_sec)
            k.set_access_token(d["access_token"])
            manager.kite = k
            st.session_state.api_key = api_key
            st.session_state.api_sec = api_sec
            st.sidebar.success("✅ Connected")
        except Exception as e:
            st.sidebar.error(f"Auth failed: {e}")
            manager.kite = None
    else:
        st.sidebar.warning("Fill all fields.")

st.sidebar.markdown("---")

# --- MONITORING CONTROLS ---

user_input = st.sidebar.text_area("Symbols (F&O Stocks)", "RELIANCE, TCS, HDFCBANK")

col1, col2 = st.sidebar.columns(2)

if col1.button("▶ START SNIPER"):
    if not manager.kite:
        st.error("Login first")
    elif manager.is_running:
        st.warning("Already running!")
    else:
        syms = [s.strip().upper() for s in user_input.split(",") if s.strip()]
        valid = []

        with manager.lock:
            manager.active_symbols = []
            manager.data = {}

        st.toast("Initializing symbols & calculating RVol...")

        for s in syms:
            tkn = get_instrument_token(manager.kite, s)
            if not tkn:
                manager.log(f"No NSE instrument for {s}", "WARNING")
                continue

            rvol, _ = fetch_time_slot_rvol(
                manager.kite, tkn, days=CONFIG["history_days"]
            )

            # Just log whether RVol is high or low; DO NOT block monitoring
            lvl = "High" if rvol >= CONFIG["rvol_threshold"] else "Low"
            manager.log(f"Init {s}: RVol {rvol:.2f} ({lvl})", "INFO")

            valid.append({"symbol": s, "token": tkn})

            try:
                q = manager.kite.quote(f"NSE:{s}")[f"NSE:{s}"]
                with manager.lock:
                    manager.data[s] = {
                        "ltp": safe_float(q.get("last_price")),
                        "vol": safe_int(q.get("volume")),
                        "vwap": safe_float(q.get("last_price")),
                        "ratio": 0.0,
                        "signal": "IDLE",
                        "confirmed": False,
                        "opt_power": 0.0,
                        "prev_vol": safe_int(q.get("volume")),
                        "prev_buy": 0,
                        "prev_sell": 0,
                        "prev_ts": time.time(),
                        "cum_pv": 0.0,
                        "cum_v": 0.0,
                        "persist": 0,
                        "buy_q": 0,
                        "sell_q": 0,
                        "rvol": rvol,
                    }
            except Exception as e:
                manager.log(f"Init quote failed for {s}: {e}", "WARNING")

        if not valid:
            st.error("No valid NSE instruments found for given symbols.")
        else:
            with manager.lock:
                manager.active_symbols = valid
                manager.stop_event.clear()

            t = threading.Thread(target=sniper_worker, args=(manager.kite,), daemon=True)
            t.start()
            st.success(f"Monitoring {len(valid)} stocks.")
            time.sleep(1)
            st.rerun()

if col2.button("⏹ STOP"):
    manager.stop_event.set()
    manager.is_running = False
    st.rerun()

# --- DASHBOARD ---

st.title("🎯 Institutional Sniper Dashboard")

last_beat = manager.last_beat
delta = time.time() - last_beat
hb_color = "#22c55e" if delta < 5 else "#ef4444"
hb_text = "RUNNING" if delta < 5 else "STALLED / STOPPED"
ts_text = datetime.fromtimestamp(last_beat).strftime("%H:%M:%S") if last_beat > 0 else "Never"
st.markdown(
    f"**Status:** <span style='color:{hb_color}; font-weight:bold'>● {hb_text}</span> "
    f"(Last Update: {ts_text})",
    unsafe_allow_html=True,
)

with manager.lock:
    display_data = [(k, v.copy()) for k, v in manager.data.items()]

if display_
    cols = st.columns(3)
    items = sorted(
        display_data,
        key=lambda x: (x[1].get("confirmed", False), x[1].get("ratio", 0.0)),
        reverse=True,
    )

    for i, (sym, d) in enumerate(items):
        with cols[i % 3]:
            sig = d["signal"]
            conf = d["confirmed"]
            css_class = (
                "accum" if sig == "ACCUMULATION"
                else "dist" if sig == "DISTRIBUTION"
                else "idle"
            )
            border_cls = "conf" if conf else ""

            st.markdown(
                f"""
            <div class="stMetric {border_cls}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:1.4em; font-weight:bold; color:white">{sym}</span>
                    <span class="badge {css_class}">{sig} {'✅' if conf else ''}</span>
                </div>
                <div style="margin-top:10px; display:flex; justify-content:space-between;">
                    <div>
                        <div style="color:#9ca3af; font-size:0.8em">LTP</div>
                        <div class="metric-value">₹{d['ltp']:.2f}</div>
                    </div>
                    <div>
                        <div style="color:#9ca3af; font-size:0.8em">Absorp. Ratio</div>
                        <div class="metric-value">{d['ratio']:.2f}x</div>
                    </div>
                </div>
                <hr style="border-color:#374151; margin: 5px 0;">
                <div style="font-size:0.8em; color:#d1d5db">
                    VWAP: ₹{d['vwap']:.2f} | RVOL: {d['rvol']:.2f}x <br>
                    Vol: {d['vol']:,} | Opt Power: {d['opt_power']:.2f} <br>
                    <span style="color:#6ee7b7">Buy: {d['buy_q']:,}</span> vs <span style="color:#fca5a5">Sell: {d['sell_q']:,}</span>
                    <br>Persist: {d['persist']}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

st.write("---")
st.text_area("System Logs", "\n".join(manager.logs), height=200, disabled=True)

if manager.is_running:
    time.sleep(2)
    st.rerun()
