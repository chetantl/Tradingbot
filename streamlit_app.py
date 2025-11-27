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
.stMetric { background-color: #111827; border: 1px solid #374151; border-radius: 8px; padding: 10px; }
.metric-value { font-size: 1.2rem; font-weight: bold; color: #f3f4f6; }
.badge { padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem; }
.accum { background: #064e3b; color: #6ee7b7; border: 1px solid #059669; }
.dist { background: #7f1d1d; color: #fca5a5; border: 1px solid #dc2626; }
.idle { background: #374151; color: #d1d5db; border: 1px solid #4b5563; }
.conf { border: 3px solid #fbbf24; box-shadow: 0 0 12px #fbbf24; }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
CONFIG = {
    "rvol_threshold": 1.5,
    "ratio_threshold": 2.5,
    "strong_ratio_threshold": 4.0,
    "persistence_count": 3,
    "strong_persistence": 1,
    "imbalance_mult": 1.5,
    "history_days": 10,
}
IDLE_DROP_COUNT = 2  # consecutive raw IDLE needed to clear signal


# --- UTILITIES ---
def safe_float(val, default=0.0):
    try:
        return float(val) if val is not None else default
    except Exception:
        return default


def safe_int(val, default=0):
    try:
        return int(val) if val is not None else default
    except Exception:
        return default


# --- STATE MANAGEMENT ---
class SniperManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.logs = deque(maxlen=200)
        self.data = {}            # symbol -> state dict
        self.active_symbols = []  # list of {"symbol":..., "token":...}
        self.is_running = False
        self.stop_event = threading.Event()
        self.kite = None
        self.last_beat = 0.0

    def log(self, msg, level="INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        with self.lock:
            self.logs.appendleft(f"[{ts}] {level}: {msg}")


@st.cache_resource
def get_manager():
    return SniperManager()


manager = get_manager()


# --- KITE HELPERS ---
def get_instrument_token(kite, symbol):
    try:
        instruments = kite.instruments("NSE")
        search = symbol.upper().strip()
        for i in instruments:
            if i["tradingsymbol"] == search and i["segment"] == "NSE":
                return i["instrument_token"]
    except Exception:
        pass
    return None


# --- RVOL: current 5m / avg of last 750 (5m) ---
def calc_rvol(kite, token, days=10):
    try:
        now = datetime.now()
        from_date = now - timedelta(days=days + 5)
        candles = kite.historical_data(token, from_date, now, "5minute")
        df = pd.DataFrame(candles)
        if df.empty or len(df) < 2:
            return 0.0, 0, 0.0

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        current_vol = int(df["volume"].iloc[-1])
        hist = df["volume"].iloc[:-1]
        if len(hist) > 750:
            hist = hist.iloc[-750:]
        avg_vol = float(hist.mean()) if len(hist) > 0 else 0.0
        if avg_vol <= 0:
            return 0.0, current_vol, avg_vol

        rvol = current_vol / avg_vol
        return float(rvol), current_vol, avg_vol
    except Exception as e:
        manager.log(f"RVol calc failed: {e}", "ERROR")
        return 0.0, 0, 0.0


# --- OPTIONS POWER: all CE vs all PE (nearest expiry) ---
def scan_opt_power(kite, symbol):
    try:
        all_opts = [
            i for i in kite.instruments("NFO")
            if i.get("segment") == "NFO-OPT" and i.get("name") == symbol
        ]
        if not all_opts:
            return 0.0, 0.0, 0.0

        today = datetime.now().date()
        expiries = sorted({i["expiry"].date() for i in all_opts if i["expiry"].date() >= today})
        if not expiries:
            return 0.0, 0.0, 0.0

        nearest_exp = expiries[0]
        scoped = [i for i in all_opts if i["expiry"].date() == nearest_exp]

        ce_syms = [f"NFO:{i['tradingsymbol']}" for i in scoped if i["instrument_type"] == "CE"]
        pe_syms = [f"NFO:{i['tradingsymbol']}" for i in scoped if i["instrument_type"] == "PE"]

        ce_vol = 0
        pe_vol = 0
        if ce_syms:
            q_ce = kite.quote(ce_syms)
            for v in q_ce.values():
                ce_vol += safe_int(v.get("volume", 0))
        if pe_syms:
            q_pe = kite.quote(pe_syms)
            for v in q_pe.values():
                pe_vol += safe_int(v.get("volume", 0))

        ce_m = round(ce_vol / 1_000_000, 2)
        pe_m = round(pe_vol / 1_000_000, 2)
        net_m = round(ce_m - pe_m, 2)
        return net_m, ce_m, pe_m
    except Exception as e:
        manager.log(f"Option scan failed for {symbol}: {e}", "ERROR")
        return 0.0, 0.0, 0.0


# --- RAW ABSORPTION CLASSIFICATION ---
def classify_raw_signal(ratio, ltp, vwap, buy_q, sell_q):
    raw = "IDLE"
    if ratio > CONFIG["ratio_threshold"]:
        strong = ratio > CONFIG["strong_ratio_threshold"]
        sell_dom = sell_q > buy_q * CONFIG["imbalance_mult"]
        buy_dom = buy_q > sell_q * CONFIG["imbalance_mult"]

        # ACCUMULATION above VWAP
        if ltp >= vwap and (sell_dom or strong):
            raw = "ACCUMULATION"
        # DISTRIBUTION below VWAP
        elif ltp <= vwap and (buy_dom or strong):
            raw = "DISTRIBUTION"

    return raw


# --- PERSISTENCE & STICKY BASE SIGNAL ---
def apply_persistence(stt, raw_signal, ratio):
    prev_raw = stt.get("raw_signal", "IDLE")
    if raw_signal == prev_raw:
        stt["persist"] = stt.get("persist", 0) + 1
    else:
        stt["persist"] = 1

    stt["raw_signal"] = raw_signal

    needed = CONFIG["persistence_count"]
    if ratio > CONFIG["strong_ratio_threshold"]:
        needed = CONFIG["strong_persistence"]

    base_signal = stt.get("base_signal", "IDLE")

    # Promote ACCUM/DIST when persistence reached
    if raw_signal in ("ACCUMULATION", "DISTRIBUTION") and stt["persist"] >= needed:
        base_signal = raw_signal
        stt["idle_persist"] = 0

    # Handle IDLE persistence to drop signal
    elif raw_signal == "IDLE":
        stt["idle_persist"] = stt.get("idle_persist", 0) + 1
        if stt["idle_persist"] >= IDLE_DROP_COUNT:
            base_signal = "IDLE"
            stt["persist"] = 0
    else:
        stt["idle_persist"] = 0

    stt["base_signal"] = base_signal
    return base_signal


# --- WORKER LOOP ---
def sniper_worker(kite):
    manager.is_running = True
    manager.log("Sniper Engine Started", "SUCCESS")

    last_cum_vol = {}
    last_buy = {}
    last_sell = {}
    last_opt_check = 0.0

    while not manager.stop_event.is_set():
        try:
            manager.last_beat = time.time()

            with manager.lock:
                active = list(manager.active_symbols)

            if not active:
                time.sleep(1)
                continue

            symbols = [f"NSE:{i['symbol']}" for i in active]
            try:
                quotes = kite.quote(symbols)
            except Exception as e:
                manager.log(f"Quote error: {e}", "WARNING")
                time.sleep(2)
                continue

            now_ts = time.time()
            check_options = (now_ts - last_opt_check) > 10
            if check_options:
                last_opt_check = now_ts

            with manager.lock:
                for item in manager.active_symbols:
                    sym = item["symbol"]
                    key = f"NSE:{sym}"
                    q = quotes.get(key)
                    if not q:
                        continue

                    ltp = safe_float(q.get("last_price"))
                    cum_vol = safe_int(q.get("volume"))
                    vwap = safe_float(q.get("average_price"), ltp)

                    depth = q.get("depth", {})
                    buy_q = sum(safe_int(x.get("quantity")) for x in depth.get("buy", [])[:5])
                    sell_q = sum(safe_int(x.get("quantity")) for x in depth.get("sell", [])[:5])

                    stt = manager.data.get(sym)
                    if not stt:
                        continue

                    prev_ts = stt.get("prev_ts", now_ts - 2)
                    dt = now_ts - prev_ts if now_ts > prev_ts else 1.0

                    if dt < 0.5:
                        stt["ltp"] = ltp
                        stt["vwap"] = vwap
                        continue

                    prev_cum = last_cum_vol.get(sym, cum_vol)
                    d_vol = cum_vol - prev_cum
                    if d_vol < 0:
                        d_vol = 0
                    last_cum_vol[sym] = cum_vol

                    prev_b = last_buy.get(sym, buy_q)
                    prev_s = last_sell.get(sym, sell_q)
                    d_ob = abs(buy_q - prev_b) + abs(sell_q - prev_s)
                    last_buy[sym] = buy_q
                    last_sell[sym] = sell_q

                    # Track 5-minute bar volume from tick deltas
                    now_dt = datetime.now()
                    minute_slot = (now_dt.minute // 5) * 5
                    curr_bar_start = now_dt.replace(minute=minute_slot, second=0, microsecond=0).timestamp()
                    prev_bar_start = stt.get("bar_start", curr_bar_start)

                    if curr_bar_start != prev_bar_start:
                        stt["bar_start"] = curr_bar_start
                        stt["bar_vol"] = d_vol
                    else:
                        stt["bar_vol"] = stt.get("bar_vol", 0) + d_vol

                    # Absorption ratio ~ ΔVol / ΔOB
                    ratio = d_vol / max(1, d_ob) if d_ob > 0 else 0.0

                    raw = classify_raw_signal(ratio, ltp, vwap, buy_q, sell_q)
                    base_signal = apply_persistence(stt, raw, ratio)

                    # Options power (all CE vs PE)
                    if check_options:
                        net, ce_m, pe_m = scan_opt_power(kite, sym)
                        stt["opt_power"] = net
                        stt["opt_ce"] = ce_m
                        stt["opt_pe"] = pe_m

                    call_bias = stt.get("opt_ce", 0.0) > stt.get("opt_pe", 0.0) * 1.2
                    put_bias = stt.get("opt_pe", 0.0) > stt.get("opt_ce", 0.0) * 1.2

                    trade_signal = "IDLE"
                    if base_signal == "ACCUMULATION" and call_bias:
                        trade_signal = "BUY CALL"
                    elif base_signal == "DISTRIBUTION" and put_bias:
                        trade_signal = "BUY PUT"

                    if trade_signal != stt.get("trade_signal"):
                        stt["trade_signal"] = trade_signal
                        if trade_signal != "IDLE":
                            manager.log(f"{sym}: {trade_signal} (base={base_signal}, ratio={ratio:.2f})", "INFO")

                    stt.update({
                        "ltp": ltp,
                        "vwap": vwap,
                        "ratio": ratio,
                        "buy_q": buy_q,
                        "sell_q": sell_q,
                        "vol": stt.get("bar_vol", 0),
                        "prev_ts": now_ts,
                    })

                    # Tape line
                    tape_line = (
                        f"{datetime.now().strftime('%H:%M:%S')} | "
                        f"ΔVol={d_vol:,} ΔOB={d_ob:,} ratio={ratio:.2f} | "
                        f"raw={raw} base={base_signal} trade={trade_signal}"
                    )
                    tape = stt.get("tape", [])
                    tape.insert(0, tape_line)
                    stt["tape"] = tape[:15]

            time.sleep(2)

        except Exception as e:
            manager.log(f"Worker Fatal Error: {e}", "ERROR")
            time.sleep(2)

    manager.is_running = False
    manager.log("Sniper Engine Stopped", "INFO")


# --- SIDEBAR: AUTH WITH LOGIN LINK ---
st.sidebar.title("🔐 Zerodha Kite Connect")

if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "api_sec" not in st.session_state:
    st.session_state.api_sec = ""
if "req_token" not in st.session_state:
    st.session_state.req_token = ""

api_key = st.sidebar.text_input("API Key", value=st.session_state.api_key)
api_sec = st.sidebar.text_input("API Secret", value=st.session_state.api_sec, type="password")
req_token = st.sidebar.text_input("Request Token", value=st.session_state.req_token)

# Login URL
if api_key:
    try:
        temp_kite = KiteConnect(api_key=api_key)
        login_url = temp_kite.login_url()
        st.sidebar.markdown("**Step 1: Get Request Token**")
        st.sidebar.markdown(f"[Click here to login to Zerodha]({login_url})")
    except Exception as e:
        st.sidebar.error(f"Login URL error: {e}")

if st.sidebar.button("🚀 Connect"):
    if api_key and api_sec and req_token:
        try:
            k = KiteConnect(api_key=api_key)
            sess = k.generate_session(req_token, api_secret=api_sec)
            k.set_access_token(sess["access_token"])
            manager.kite = k
            st.session_state.api_key = api_key
            st.session_state.api_sec = api_sec
            st.session_state.req_token = req_token
            st.sidebar.success("✅ Connected")
        except Exception as e:
            st.sidebar.error(f"Auth failed: {e}")
            manager.kite = None
    else:
        st.sidebar.warning("Fill API key, secret & request token")

st.sidebar.markdown("---")

# --- SYMBOL CONTROLS ---
user_input = st.sidebar.text_area("Symbols (F&O Stocks)", "EICHERMOT, TCS, RELIANCE")

col1, col2 = st.sidebar.columns(2)

if col1.button("▶ START SNIPER"):
    if not manager.kite:
        st.error("Login first")
    elif manager.is_running:
        st.warning("Already running")
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

            rvol, rvol_curr, rvol_avg = calc_rvol(manager.kite, tkn, days=CONFIG["history_days"])
            manager.log(f"Init {s}: RVol {rvol:.2f}x", "INFO")

            valid.append({"symbol": s, "token": tkn})

            try:
                q = manager.kite.quote(f"NSE:{s}")[f"NSE:{s}"]
                official_vwap = safe_float(q.get("average_price"), safe_float(q.get("last_price")))

                now_dt = datetime.now()
                minute_slot = (now_dt.minute // 5) * 5
                bar_start = now_dt.replace(minute=minute_slot, second=0, microsecond=0).timestamp()

                with manager.lock:
                    manager.data[s] = {
                        "ltp": safe_float(q.get("last_price")),
                        "vol": 0,
                        "vwap": official_vwap,
                        "ratio": 0.0,
                        "base_signal": "IDLE",
                        "trade_signal": "IDLE",
                        "raw_signal": "IDLE",
                        "persist": 0,
                        "idle_persist": 0,
                        "confirmed": False,
                        "opt_power": 0.0,
                        "opt_ce": 0.0,
                        "opt_pe": 0.0,
                        "prev_ts": time.time(),
                        "buy_q": 0,
                        "sell_q": 0,
                        "rvol": rvol,
                        "rvol_curr": rvol_curr,
                        "rvol_avg": rvol_avg,
                        "bar_start": bar_start,
                        "bar_vol": 0,
                        "tape": [],
                    }
            except Exception as e:
                manager.log(f"Init quote failed for {s}: {e}", "WARNING")

        if not valid:
            st.error("No valid instruments found.")
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

if display_data:
    cols = st.columns(3)
    # Sort: trade signal present first, then higher ratio
    items = sorted(
        display_data,
        key=lambda x: (
            0 if x[1].get("trade_signal") in ("BUY CALL", "BUY PUT") else 1,
            -x[1].get("ratio", 0.0),
        )
    )

    for i, (sym, d) in enumerate(items):
        with cols[i % 3]:
            trade_sig = d.get("trade_signal", "IDLE")
            base_sig = d.get("base_signal", "IDLE")

            if trade_sig == "BUY CALL":
                css_class = "accum"
            elif trade_sig == "BUY PUT":
                css_class = "dist"
            else:
                css_class = "idle"

            st.markdown(
                f"""
            <div class="stMetric">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:1.4em; font-weight:bold; color:white">{sym}</span>
                    <span class="badge {css_class}">{trade_sig}</span>
                </div>
                <div style="margin-top:10px; display:flex; justify-content:space-between;">
                    <div>
                        <div style="color:#9ca3af; font-size:0.8em">LTP</div>
                        <div class="metric-value">₹{d.get('ltp',0):.2f}</div>
                    </div>
                    <div>
                        <div style="color:#9ca3af; font-size:0.8em">Absorp. Ratio</div>
                        <div class="metric-value">{d.get('ratio',0.0):.2f}x</div>
                    </div>
                </div>
                <hr style="border-color:#374151; margin: 5px 0;">
                <div style="font-size:0.8em; color:#d1d5db">
                    Base: {base_sig} | RVOL: {d.get('rvol',0.0):.2f}x<br>
                    VWAP: ₹{d.get('vwap',0.0):.2f} | Vol(5m): {d.get('vol',0):,}<br>
                    Buy: {d.get('buy_q',0):,} vs Sell: {d.get('sell_q',0):,}<br>
                    CE Vol: {d.get('opt_ce',0.0):.2f}M | PE Vol: {d.get('opt_pe',0.0):.2f}M<br>
                    Net Opt: {d.get('opt_power',0.0):.2f}M | Persist: {d.get('persist',0)} | Idle: {d.get('idle_persist',0)}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Debug
    with st.expander("🔍 Debug – RVol, Volume, Tape"):
        for sym, d in items:
            st.markdown(f"### {sym}")
            st.markdown(
                f"""
- RVOL: **{d.get('rvol',0.0):.2f}x**  
- Current 5m vol (RVol numerator): **{d.get('rvol_curr',0):,}**  
- Avg last 750×5m vol: **{d.get('rvol_avg',0.0):,.0f}**  
- Live bar vol (this 5m from ticks): **{d.get('vol',0):,}**  
- Depth: Buy **{d.get('buy_q',0):,}**, Sell **{d.get('sell_q',0):,}**  
- Raw: **{d.get('raw_signal','IDLE')}**, Base: **{d.get('base_signal','IDLE')}**, Trade: **{d.get('trade_signal','IDLE')}**  
- Persist: **{d.get('persist',0)}**, IdlePersist: **{d.get('idle_persist',0)}**  
- Opt CE: **{d.get('opt_ce',0.0):.2f}M**, Opt PE: **{d.get('opt_pe',0.0):.2f}M**, Net: **{d.get('opt_power',0.0):.2f}M**
                """
            )
            tape_lines = d.get("tape", [])
            if tape_lines:
                st.markdown("**Recent tape:**")
                st.text("\n".join(tape_lines[:10]))
            st.markdown("---")

st.write("---")
st.text_area("System Logs", "\n".join(manager.logs), height=200, disabled=True)

if manager.is_running:
    time.sleep(2)
    st.rerun()

