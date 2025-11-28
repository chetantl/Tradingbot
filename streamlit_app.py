import streamlit as st
from kiteconnect import KiteConnect
import threading
import time
from datetime import datetime, timedelta
from collections import deque, defaultdict
import pandas as pd
import copy
from pytz import timezone

# ====== TIMEZONE (INDIA) ====== #
INDIA_TZ = timezone("Asia/Kolkata")

# ====== PAGE CONFIG ====== #
st.set_page_config(
    page_title="Institutional Sniper (5m Footprint)",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
<style>
.stMetric {
    background-color: #111827;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 12px;
}
.metric-value {
    font-size: 1.2rem;
    font-weight: bold;
    color: #f3f4f6;
}
.badge {
    padding: 4px 8px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 0.8rem;
}
.accum { background: #064e3b; color: #6ee7b7; border: 1px solid #059669; }
.dist  { background: #7f1d1d; color: #fca5a5; border: 1px solid #dc2626; }
.idle  { background: #374151; color: #d1d5db; border: 1px solid #4b5563; }
.card-container {
    background-color: #111827;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ====== CONFIG ====== #
CONFIG = {
    "ratio_threshold": 2.5,
    "strong_ratio_threshold": 4.0,
    "persistence_count": 2,
    "strong_persistence": 1,
    "imbalance_mult": 1.5,
    "history_days": 10,
}
IDLE_DROP_COUNT = 2
POLL_INTERVAL_SEC = 2


# ====== HELPERS ====== #
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


def now_ist():
    return datetime.now(INDIA_TZ)


def current_bar_id_5m():
    """Return an integer ID for the current 5-min bar, aligned to IST."""
    now = now_ist()
    minute_slot = (now.minute // 5) * 5
    bar_start = now.replace(minute=minute_slot, second=0, microsecond=0)
    return int(bar_start.timestamp()), bar_start


# ====== MANAGER ====== #
class SniperManager:
    def __init__(self):
        self.lock = threading.RLock()  # Changed to RLock for reentrant locking
        self.logs = deque(maxlen=200)
        self.data = {}  # Changed from defaultdict to regular dict
        self.active_symbols = []
        self.is_running = False
        self.stop_event = threading.Event()
        self.kite = None
        self.last_beat = 0.0
        self.initialized = False

    def log(self, msg, level="INFO"):
        ts = now_ist().strftime("%H:%M:%S")
        with self.lock:
            self.logs.appendleft(f"[{ts}] {level}: {msg}")

    def get_data_snapshot(self):
        """Thread-safe way to get data snapshot"""
        with self.lock:
            return copy.deepcopy(self.data)

    def get_logs_snapshot(self):
        """Thread-safe way to get logs snapshot"""
        with self.lock:
            return list(self.logs)

    def set_symbol_data(self, symbol, data):
        """Thread-safe way to set symbol data"""
        with self.lock:
            self.data[symbol] = data

    def get_symbol_data(self, symbol):
        """Thread-safe way to get symbol data"""
        with self.lock:
            return self.data.get(symbol, {}).copy()


@st.cache_resource
def get_manager():
    return SniperManager()


manager = get_manager()


# ====== KITE HELPERS ====== #
def get_instrument_token(kite, symbol):
    try:
        instruments = kite.instruments("NSE")
        sym = symbol.upper().strip()
        for inst in instruments:
            if inst["tradingsymbol"] == sym and inst["segment"] == "NSE":
                return inst["instrument_token"]
    except Exception as e:
        manager.log(f"Instrument lookup error for {symbol}: {e}", "ERROR")
    return None


def calc_rvol(kite, token, days=10):
    """RVol = last 5m candle volume / average volume of last 750 5m candles."""
    try:
        now = now_ist().replace(tzinfo=None)
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
        avg_750 = float(hist.mean()) if len(hist) > 0 else 0.0
        rvol = current_vol / avg_750 if avg_750 > 0 else 0.0
        return round(rvol, 2), current_vol, round(avg_750, 2)
    except Exception as e:
        manager.log(f"RVol calc failed: {e}", "ERROR")
        return 0.0, 0, 0.0


def scan_opt_power(kite, symbol):
    """Returns (net_million, ce_m, pe_m) using nearest expiry CE/PE volume."""
    try:
        all_opts = [
            i for i in kite.instruments("NFO")
            if i.get("segment") == "NFO-OPT" and i.get("name") == symbol
        ]
        if not all_opts:
            return 0.0, 0.0, 0.0

        today = now_ist().date()
        expiries = sorted({i["expiry"].date() for i in all_opts if i["expiry"].date() >= today})
        if not expiries:
            return 0.0, 0.0, 0.0

        nearest = expiries[0]
        scoped = [i for i in all_opts if i["expiry"].date() == nearest]

        ce_syms = [f"NFO:{i['tradingsymbol']}" for i in scoped if i["instrument_type"] == "CE"]
        pe_syms = [f"NFO:{i['tradingsymbol']}" for i in scoped if i["instrument_type"] == "PE"]

        ce_vol = 0
        pe_vol = 0

        if ce_syms:
            # Process in batches to avoid API limits
            for i in range(0, len(ce_syms), 50):
                batch = ce_syms[i:i+50]
                q_ce = kite.quote(batch)
                for v in q_ce.values():
                    prem = safe_float(v.get("last_price", 0.0))
                    if prem >= 20:
                        ce_vol += safe_int(v.get("volume", 0))

        if pe_syms:
            for i in range(0, len(pe_syms), 50):
                batch = pe_syms[i:i+50]
                q_pe = kite.quote(batch)
                for v in q_pe.values():
                    prem = safe_float(v.get("last_price", 0.0))
                    if prem >= 20:
                        pe_vol += safe_int(v.get("volume", 0))

        ce_m = round(ce_vol / 1_000_000, 2)
        pe_m = round(pe_vol / 1_000_000, 2)
        net_m = round(ce_m - pe_m, 2)
        return net_m, ce_m, pe_m
    except Exception as e:
        manager.log(f"Option scan failed for {symbol}: {e}", "ERROR")
        return 0.0, 0.0, 0.0


# ====== FOOTPRINT LOGIC (BAR-BASED) ====== #
def classify_raw_signal(ratio, ltp, vwap, buy_q, sell_q):
    raw = "IDLE"
    if ratio > CONFIG["ratio_threshold"]:
        strong = ratio > CONFIG["strong_ratio_threshold"]
        sell_dom = sell_q > buy_q * CONFIG["imbalance_mult"]
        buy_dom = buy_q > sell_q * CONFIG["imbalance_mult"]

        if ltp >= vwap and (sell_dom or strong):
            raw = "ACCUMULATION"
        elif ltp <= vwap and (buy_dom or strong):
            raw = "DISTRIBUTION"
    return raw


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

    base = stt.get("base_signal", "IDLE")

    if raw_signal in ("ACCUMULATION", "DISTRIBUTION") and stt["persist"] >= needed:
        base = raw_signal
        stt["idle_persist"] = 0
    elif raw_signal == "IDLE":
        stt["idle_persist"] = stt.get("idle_persist", 0) + 1
        if stt["idle_persist"] >= IDLE_DROP_COUNT:
            base = "IDLE"
            stt["persist"] = 0
    else:
        stt["idle_persist"] = 0

    stt["base_signal"] = base
    return base


# ====== WORKER THREAD (5m BAR ENGINE) ====== #
def sniper_worker(kite):
    manager.is_running = True
    manager.log("Sniper worker started (5m footprint, IST aligned)", "INFO")

    last_cum_vol = {}
    last_buy = {}
    last_sell = {}
    last_opt_scan = 0.0

    curr_bar_id, curr_bar_start = current_bar_id_5m()

    while not manager.stop_event.is_set():
        try:
            now_ts = time.time()
            manager.last_beat = now_ts

            with manager.lock:
                active_list = list(manager.active_symbols)

            if not active_list:
                time.sleep(1)
                continue

            new_bar_id, new_bar_start = current_bar_id_5m()
            bar_changed = new_bar_id != curr_bar_id

            keys = [f"NSE:{row['symbol']}" for row in active_list]
            try:
                quotes = kite.quote(keys)
            except Exception as e:
                manager.log(f"Quote error: {e}", "WARNING")
                time.sleep(2)
                continue

            do_opt_scan = (now_ts - last_opt_scan) > 60
            if do_opt_scan:
                last_opt_scan = now_ts

            for row in active_list:
                sym = row["symbol"]
                key = f"NSE:{sym}"
                q = quotes.get(key)
                if not q:
                    continue

                stt = manager.get_symbol_data(sym)

                ltp = safe_float(q.get("last_price"))
                vwap = safe_float(q.get("average_price"), ltp)
                depth = q.get("depth", {})

                buy_q = sum(safe_int(x.get("quantity")) for x in depth.get("buy", [])[:5])
                sell_q = sum(safe_int(x.get("quantity")) for x in depth.get("sell", [])[:5])

                cum_vol = safe_int(q.get("volume"))

                prev_cum = last_cum_vol.get(sym, cum_vol)
                d_vol = max(0, cum_vol - prev_cum)
                last_cum_vol[sym] = cum_vol

                prev_b = last_buy.get(sym, buy_q)
                prev_s = last_sell.get(sym, sell_q)
                d_ob = abs(buy_q - prev_b) + abs(sell_q - prev_s)
                last_buy[sym] = buy_q
                last_sell[sym] = sell_q

                if not stt:
                    stt = {
                        "ltp": ltp,
                        "vwap": vwap,
                        "buy_q": buy_q,
                        "sell_q": sell_q,
                        "bar_id": curr_bar_id,
                        "bar_d_vol": d_vol,
                        "bar_d_ob": d_ob,
                        "ratio_bar": 0.0,
                        "raw_signal": "IDLE",
                        "base_signal": "IDLE",
                        "persist": 0,
                        "idle_persist": 0,
                        "opt_power": 0.0,
                        "opt_ce": 0.0,
                        "opt_pe": 0.0,
                        "rvol": 0.0,
                        "rvol_avg": 0.0,
                        "rvol_curr": 0,
                        "trade_signal": "IDLE",
                    }

                if bar_changed and stt.get("bar_id", curr_bar_id) == curr_bar_id:
                    bar_vol = stt.get("bar_d_vol", 0)
                    bar_ob = stt.get("bar_d_ob", 0)
                    ratio_bar = bar_vol / max(1, bar_ob) if bar_ob > 0 else 0.0

                    raw = classify_raw_signal(ratio_bar, stt["ltp"], stt["vwap"],
                                              stt["buy_q"], stt["sell_q"])
                    base_signal = apply_persistence(stt, raw, ratio_bar)

                    stt["ratio_bar"] = ratio_bar

                    avg_vol = stt.get("rvol_avg", 0.0)
                    if avg_vol > 0:
                        stt["rvol"] = round(bar_vol / avg_vol, 2)

                    manager.log(
                        f"{sym} BAR CLOSE @ {curr_bar_start.strftime('%H:%M')} "
                        f"Vol={bar_vol} Absorp={ratio_bar:.2f} "
                        f"Raw={raw} Base={base_signal}",
                        "INFO",
                    )

                    stt["bar_id"] = new_bar_id
                    stt["bar_d_vol"] = d_vol
                    stt["bar_d_ob"] = d_ob
                else:
                    stt["bar_id"] = curr_bar_id
                    stt["bar_d_vol"] = stt.get("bar_d_vol", 0) + d_vol
                    stt["bar_d_ob"] = stt.get("bar_d_ob", 0) + d_ob

                if do_opt_scan:
                    net, ce_m, pe_m = scan_opt_power(kite, sym)
                    stt["opt_power"] = net
                    stt["opt_ce"] = ce_m
                    stt["opt_pe"] = pe_m

                ce_m = stt.get("opt_ce", 0.0)
                pe_m = stt.get("opt_pe", 0.0)

                call_bias = ce_m > pe_m * 1.2
                put_bias = pe_m > ce_m * 1.2

                base_signal = stt.get("base_signal", "IDLE")
                trade_signal = "IDLE"
                if base_signal == "ACCUMULATION" and call_bias:
                    trade_signal = "BUY CALL"
                elif base_signal == "DISTRIBUTION" and put_bias:
                    trade_signal = "BUY PUT"

                stt["trade_signal"] = trade_signal
                stt["ltp"] = ltp
                stt["vwap"] = vwap
                stt["buy_q"] = buy_q
                stt["sell_q"] = sell_q

                manager.set_symbol_data(sym, stt)

            if bar_changed:
                curr_bar_id = new_bar_id
                curr_bar_start = new_bar_start

            time.sleep(POLL_INTERVAL_SEC)

        except Exception as e:
            manager.log(f"Worker error: {e}", "ERROR")
            time.sleep(2)

    manager.is_running = False
    manager.log("Sniper worker stopped", "INFO")


# ====== SIDEBAR AUTH ====== #
st.sidebar.title("🔐 Zerodha Kite Connect (IST)")

if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "api_sec" not in st.session_state:
    st.session_state.api_sec = ""
if "req_token" not in st.session_state:
    st.session_state.req_token = ""

api_key = st.sidebar.text_input("API Key", value=st.session_state.api_key)
api_sec = st.sidebar.text_input("API Secret", value=st.session_state.api_sec, type="password")
req_token = st.sidebar.text_input("Request Token", value=st.session_state.req_token)

if api_key:
    try:
        temp_kite = KiteConnect(api_key=api_key)
        login_url = temp_kite.login_url()
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Step 1: Get Request Token**")
        st.sidebar.markdown(f"[Click here to login to Zerodha]({login_url})")
        st.sidebar.markdown(
            "After login, copy the `request_token` from the redirect URL and paste above."
        )
    except Exception as e:
        st.sidebar.error(f"Login URL error: {e}")

if st.sidebar.button("🚀 Connect"):
    if api_key and api_sec and req_token:
        try:
            kite = KiteConnect(api_key=api_key)
            ses = kite.generate_session(req_token, api_secret=api_sec)
            kite.set_access_token(ses["access_token"])
            manager.kite = kite
            st.session_state.api_key = api_key
            st.session_state.api_sec = api_sec
            st.session_state.req_token = req_token
            manager.log("Connected to Kite", "SUCCESS")
            st.sidebar.success("✅ Connected")
        except Exception as e:
            st.sidebar.error(f"Auth failed: {e}")
            manager.kite = None
    else:
        st.sidebar.warning("Please fill API key, secret & request token")

st.sidebar.markdown("---")

symbols_text = st.sidebar.text_area(
    "Symbols (F&O stocks, comma separated)",
    "M&M, PAYTM, GAIL, KEI, TVSMOTOR"
)

btn1, btn2 = st.sidebar.columns(2)

if btn1.button("▶ START"):
    if not manager.kite:
        st.error("Login first")
    elif manager.is_running:
        st.warning("Already running")
    else:
        syms = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
        valid = []

        with manager.lock:
            manager.active_symbols = []
            manager.data = {}

        with st.spinner("Initializing symbols & pre-computing RVol..."):
            for s in syms:
                tkn = get_instrument_token(manager.kite, s)
                if not tkn:
                    manager.log(f"No NSE instrument for {s}", "WARNING")
                    continue

                rvol, rvol_curr, rvol_avg = calc_rvol(manager.kite, tkn, days=CONFIG["history_days"])
                valid.append({"symbol": s, "token": tkn})

                try:
                    q = manager.kite.quote(f"NSE:{s}")[f"NSE:{s}"]
                    ltp = safe_float(q.get("last_price"))
                    vwap = safe_float(q.get("average_price"), ltp)

                    stt = {
                        "ltp": ltp,
                        "vwap": vwap,
                        "buy_q": 0,
                        "sell_q": 0,
                        "bar_id": current_bar_id_5m()[0],
                        "bar_d_vol": 0,
                        "bar_d_ob": 0,
                        "ratio_bar": 0.0,
                        "raw_signal": "IDLE",
                        "base_signal": "IDLE",
                        "persist": 0,
                        "idle_persist": 0,
                        "opt_power": 0.0,
                        "opt_ce": 0.0,
                        "opt_pe": 0.0,
                        "rvol": rvol,
                        "rvol_curr": rvol_curr,
                        "rvol_avg": rvol_avg,
                        "trade_signal": "IDLE",
                    }
                    manager.set_symbol_data(s, stt)
                except Exception as e:
                    manager.log(f"Init quote failed for {s}: {e}", "WARNING")

        if not valid:
            st.error("No valid instruments found.")
        else:
            with manager.lock:
                manager.active_symbols = valid
                manager.stop_event.clear()
                manager.initialized = True
            t = threading.Thread(target=sniper_worker, args=(manager.kite,), daemon=True)
            t.start()
            manager.is_running = True
            st.success(f"Monitoring {len(valid)} stocks (IST 5m bars).")

if btn2.button("⏹ STOP"):
    manager.stop_event.set()
    manager.is_running = False
    manager.initialized = False
    st.rerun()

# ====== DASHBOARD ====== #
st.title("🎯 Institutional Sniper – 5m Footprint (IST)")

last_beat = manager.last_beat
delta = time.time() - last_beat
hb_color = "#22c55e" if delta < 5 else "#ef4444"
hb_text = "LIVE" if delta < 5 else "STOPPED"
beat_time = (
    datetime.fromtimestamp(last_beat, INDIA_TZ).strftime("%H:%M:%S")
    if last_beat > 0 else "Never"
)

st.markdown(
    f"""
<div class="card-container">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div style="display:flex;align-items:center;gap:10px;">
      <div style="width:18px;height:18px;border-radius:50%;background:{hb_color};"></div>
      <div>
        <div style="color:#9ca3af;font-size:0.8em;">Engine Status (IST)</div>
        <div style="font-size:1.2em;color:white;font-weight:bold;">{hb_text}</div>
      </div>
    </div>
    <div style="color:#9ca3af;font-size:0.8em;">
      Last beat: {beat_time}
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Get snapshot safely
snapshot = manager.get_data_snapshot()

# Debug info
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Debug:** Data count: {len(snapshot)}")
st.sidebar.markdown(f"**Running:** {manager.is_running}")
st.sidebar.markdown(f"**Initialized:** {manager.initialized}")

if snapshot:
    st.subheader(f"📊 Monitoring {len(snapshot)} Symbols")
    
    cols = st.columns(3)
    items = sorted(
        snapshot.items(),
        key=lambda x: (
            0 if x[1].get("trade_signal") in ("BUY CALL", "BUY PUT") else 1,
            -x[1].get("ratio_bar", 0.0),
        ),
    )

    for i, (sym, d) in enumerate(items):
        with cols[i % 3]:
            trade_sig = d.get("trade_signal", "IDLE")
            base_sig = d.get("base_signal", "IDLE")
            raw_sig = d.get("raw_signal", "IDLE")

            if trade_sig == "BUY CALL":
                cls = "accum"
                badge_style = "background:#064e3b;color:#6ee7b7;border:1px solid #059669;"
            elif trade_sig == "BUY PUT":
                cls = "dist"
                badge_style = "background:#7f1d1d;color:#fca5a5;border:1px solid #dc2626;"
            else:
                cls = "idle"
                badge_style = "background:#374151;color:#d1d5db;border:1px solid #4b5563;"

            card_html = f"""
<div class="card-container">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span style="font-size:1.3em;font-weight:bold;color:white;">{sym}</span>
    <span style="padding:4px 8px;border-radius:4px;font-weight:bold;font-size:0.8rem;{badge_style}">{trade_sig}</span>
  </div>
  <div style="margin-top:8px;color:#9ca3af;font-size:0.85em;">
    LTP ₹{d.get('ltp',0):.2f} | VWAP ₹{d.get('vwap',0):.2f}<br>
    Absorp (last bar): {d.get('ratio_bar',0.0):.2f}x | RVol: {d.get('rvol',0.0):.2f}x<br>
    5m Vol: {d.get('bar_d_vol',0):,} | Depth Buy {d.get('buy_q',0):,} vs Sell {d.get('sell_q',0):,}<br>
    CE {d.get('opt_ce',0.0):.2f}M vs PE {d.get('opt_pe',0.0):.2f}M (Net {d.get('opt_power',0.0):.2f}M)<br>
    Raw: {raw_sig} | Base: {base_sig}<br>
    Persist: {d.get('persist',0)} | IdlePersist: {d.get('idle_persist',0)}
  </div>
</div>
"""
            st.markdown(card_html, unsafe_allow_html=True)
else:
    st.info("👆 Enter symbols and click START to begin monitoring")

st.write("---")

# Logs section
logs_list = manager.get_logs_snapshot()
st.subheader("📝 System Logs")
log_text = "\n".join(logs_list) if logs_list else "No logs yet..."
st.text_area("", log_text, height=220, disabled=True, key="logs_area")

# Auto-refresh only when running
if manager.is_running:
    time.sleep(3)
    st.rerun()