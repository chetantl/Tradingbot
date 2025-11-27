import streamlit as st
from kiteconnect import KiteConnect
import threading
import time
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import copy
from collections import defaultdict

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Institutional Sniper - Full Strategy",
    page_icon="🎯",
    layout="wide",
)

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

IDLE_DROP_COUNT = 2  # consecutive IDLE raw reads needed to reset final signal

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

# --- STATE MANAGEMENT ---
class SniperManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.logs = deque(maxlen=200)
        self.data = {}
        self.active_symbols = []
        self.is_running = False
        self.stop_event = threading.Event()
        self.kite = None
        self.last_beat = 0

    def log(self, msg, level="INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        with self.lock:
            self.logs.appendleft(f"[{ts}] {level}: {msg}")

@st.cache_resource
def get_manager():
    return SniperManager()

manager = get_manager()

# --- RVOL CALC: current 5m vs 750 previous 5m candles ---
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

        # Latest 5m candle volume
        current_vol = int(df["volume"].iloc[-1])
        history = df["volume"].iloc[:-1]

        # Take max 750 previous candles
        if len(history) > 750:
            history = history.iloc[-750:]
        avg_vol = float(history.mean()) if len(history) > 0 else 0.0

        # Compute RVol
        rvol = (current_vol / avg_vol) if avg_vol > 0 else 0.0
        return float(rvol), current_vol, avg_vol
    except Exception as e:
        manager.log(f"RVol calculation failed: {e}", "ERROR")
        return 0.0, 0, 0.0

# --- FULL OPTIONS POWER SCAN ---
def scan_opt_power(kite, symbol):
    """Scan ALL CE & PE strikes for nearest expiry and return net millions, CE millions, PE millions."""
    try:
        all_opts = [i for i in kite.instruments("NFO") if i["segment"]=="NFO-OPT" and i["name"]==symbol]
        if not all_opts:
            return 0.0, 0.0, 0.0

        today = datetime.now().date()
        expiries = sorted({o["expiry"].date() for o in all_opts if o["expiry"].date() >= today})
        if not expiries:
            return 0.0, 0.0, 0.0

        nearest_exp = expiries[0]
        scoped = [o for o in all_opts if o["expiry"].date()==nearest_exp]

        ce_syms = [f"NFO:{o['tradingsymbol']}" for o in scoped if o["instrument_type"]=="CE"]
        pe_syms = [f"NFO:{o['tradingsymbol']}" for o in scoped if o["instrument_type"]=="PE"]

        ce_vol = 0
        pe_vol = 0
        
        if ce_syms:
            qc = kite.quote(ce_syms)
            for v in qc.values():
                ce_vol += safe_int(v.get("volume", 0))
        if pe_syms:
            qp = kite.quote(pe_syms)
            for v in qp.values():
                pe_vol += safe_int(v.get("volume", 0))

        ce_m = round(ce_vol/1_000_000,2)
        pe_m = round(pe_vol/1_000_000,2)
        net_m = round(ce_m - pe_m,2)
        return net_m, ce_m, pe_m
    except Exception as e:
        manager.log(f"Options power scan failed: {e}", "ERROR")
        return 0.0, 0.0, 0.0

# --- CLASSIFY RAW ABSORPTION SIGNAL ---
def classify_raw_signal(sym, ratio, ltp, vwap, buy_q, sell_q):
    raw = "IDLE"
    if ratio > CONFIG["ratio_threshold"]:
        strong = ratio > CONFIG["strong_ratio_threshold"]
        if ltp > vwap and (sell_q > buy_q * CONFIG["imbalance_mult"] or strong):
            raw = "ACCUMULATION"
        elif ltp < vwap and (buy_q > sell_q * CONFIG["imbalance_mult"] or strong):
            raw = "DISTRIBUTION"
    return raw

# --- SIGNAL PROMOTION (ensures stickiness and persistence) ---
def promote_signal(state, raw, ratio):
    last_raw = state.get("raw_signal", raw)

    if raw == last_raw:
        state["persist"] = state.get("persist", 0) + 1
    else:
        state["persist"] = 1

    state["raw_signal"] = raw

    # Promote only after persist threshold reached
    needed = CONFIG["persistence_count"]
    if ratio > CONFIG["strong_ratio_threshold"]:
        needed = CONFIG["strong_persistence"]

    if state["persist"] >= needed and raw != "IDLE":
        state["signal"] = raw

    # Drop to IDLE only after 2 consecutive raw=IDLE
    if raw == "IDLE":
        state["idle_persist"] = state.get("idle_persist", 0) + 1
        if state["idle_persist"] >= IDLE_DROP_COUNT:
            state["signal"] = "IDLE"
            state["persist"] = 0
            state["already_logged_conf"] = False
    else:
        state["idle_persist"] = 0

    return state["signal"]

# --- WORKER ENGINE ---
def sniper_worker_loop(kite):
    manager.is_running = True
    manager.log("Sniper engine started", "SUCCESS")

    last_vol = {}
    last_book = {}

    while not manager.stop_event.is_set():
        try:
            manager.last_beat = time.time()

            symbols = list(manager.active_symbols)
            if not symbols:
                time.sleep(1)
                continue

            batch = [f"NSE:{s}" for s in symbols]
            quotes = kite.quote(batch)

            with manager.lock:
                for sym in symbols:
                    key = f"NSE:{sym}"
                    q = quotes.get(key)
                    if not q:
                        continue

                    ltp = safe_float(q.get("last_price"))
                    cum_vol = safe_int(q.get("volume"))
                    vwap = safe_float(q.get("average_price"), ltp)

                    depth = q.get("depth", {})
                    buy_q = sum(safe_int(x.get("quantity")) for x in depth.get("buy",[])[:5])
                    sell_q = sum(safe_int(x.get("quantity")) for x in depth.get("sell",[])[:5])

                    now = time.time()
                    prev_ts = last_book.get(sym,{}).get("ts", now - 2)
                    dt = now - prev_ts if now>prev_ts else 1

                    # Compute deltas
                    prev_vol = last_vol.get(sym, cum_vol)
                    d_vol = cum_vol - prev_vol
                    if d_vol < 0: d_vol = 0
                    last_vol[sym] = cum_vol

                    prev_buy = last_book.get(sym,{}).get("buy", buy_q)
                    prev_sell = last_book.get(sym,{}).get("sell", sell_q)
                    d_ob = abs(buy_q - prev_buy) + abs(sell_q - prev_sell)

                    last_book[sym] = {"buy":buy_q, "sell":sell_q, "ts":now, "ts":now}

                    # Absorption Ratio ≈ ΔVol/ΔOB
                    ratio = d_vol / max(1, (d_ob := d_ob));

                    # Raw footprint classification
                    raw = classify_raw_signal(sym, ratio,ltp, vwap,buy_q, sell_q)

                    # Persistence & sticky promotion
                    stt = manager.data.setdefault(sym, {})
                    prev_signal = stt.get("signal", "IDLE")

                    final_signal = promote_signal(stt, raw, ratio)

                    # Option power confirmation (stored separately for CE/PE)
                    if now - stt.get("last_opt_ts",0) > 10:
                        stt["last_opt_ts"] = now
                        net, ce, pe = scan_opt_power(kite, sym,ltp)
                        stt["opt_power"] = net
                        stt["opt_power_ce"] = ce
                        stt["opt_power_pe"] = pe

                    call_bias = stt["opt_power_ce"] > stt["opt_power_pe"] * 1.2
                    put_bias  = stt["opt_power_pe"] > stt["opt_power_ce"] * 1.2
                    
                    # Convert final signal into actual trade label
                    trade_label = prev_signal
                    if final_signal == "ACCUMULATION" and call_bias:
                        trade_label = "BUY CALL"
                    elif final_signal == "DISTRIBUTION" and put_bias:
                        trade_label = "BUY PUT"
                    else:
                        trade_label = "IDLE"

                    if trade_label != prev_signal:
                        stt["signal"] = trade_label
                        stt["confirmed"] = False
                        if trade_label != "IDLE":
                            manager.log(f"{sym} → {trade_label} promoted after persistence", "INFO")

                    # Finally update state
                    stt.update({
                        "ltp":ltp,
                        "vwap":vwap,
                        "ratio":ratio,
                        "buy_q":buy_q,
                        "sell_q":sell_q,
                        "opt_power":stt.get("opt_power",0.0),
                        "opt_power_ce":stt.get("opt_power_ce",0.0),
                        "opt_power_pe":stt.get("opt_power_pe",0.0),
                        "persist":stt.get("persist",0),
                        "idle_persist":stt.get("persist",0),
                    })

                    # Tape log for debug
                    line = f"{datetime.now().strftime('%H:%M:%S')} | ΔVol={d_vol:,} ΔOB={d_ob:,} ratio={ratio:.2f} raw={raw} final={manager.data[sym]['signal']}"
                    t = manager.data[sym].get("tape", [])
                    t.insert(0, line)
                    manager.data[sym]["tape"] = t[:15]

            time.sleep(2)
        except Exception as e:
            manager.log(f"Worker loop error: {e}", "ERROR")
            time.sleep(2)

    manager.is_running = False
    manager.log("Engine stopped","INFO")


# --- STREAMLIT UI ---
st.sidebar.title("🔐 Zerodha Kite Connect")
api_key = st.sidebar.text_input("API Key")
api_sec = st.sidebar.text_input("API Secret", type="password")
req_token = st.sidebar.text_input("Request Token")

if st.sidebar.button("🚀 Connect"):
    try:
        manager.kite = KiteConnect(api_key=api_key)
        ss = manager.kite.generate_session(req_token, api_secret=api_sec)
        manager.kite.set_access_token(ss["access_token"])
        st.sidebar.success("✅ Connected")
    except Exception as e:
        st.sidebar.error(str(e))

symbols = st.sidebar.text_area("Stocks to snipe", "EICHERMOT, TCS, RELIANCE").split(",")
if st.sidebar.button("▶ START"):
    for sym in symbols:
        sym_clean = sym.strip().upper()
        token = get_instrument_secret(manager.kite,sym_clean)
        if token:
            rvol, curr5, av = calc_rvol(manager.kite,token)
            manager.active_symbols.append(sym_clean)
            manager.data[sym_clean].update({
                "rvol":round(rvol,2),
                "rvol_curr":curr5,
                "rvol_avg":av
            })
    t = threading.Thread(target=sniper_engine, args=(manager.kite,), daemon=True)
    t.start()
    st.rerun()

if st.sidebar.button("🟥 STOP"):
    manager.stop_event.set()


# --- DASHBOARD UI ---
st.title("🎯 Institutional Sniper Dashboard")

with manager.lock:
    items = copy.deepcopy(list(manager.data.items()))

if items:
    cols = st.cols = st.columns = st.columns(3)
    sorted_syms = sorted(items, key=lambda x:(x[1].get("confirmed",False), x[1].get("ratio",0.0)), reverse=True)

    for i,(sym,d) in enumerate(sorted_syms):
         with cols[i%3]:
              css_class = "accum" if d.get("signal")=="BUY CALL" else "dist" if d.get("signal")=="BUY PUT" else "idle"
              st.markdown(f"""
        <div class="stMetric {"conf" if d.get("confirmed") else ""}">
            <div style="display:flex;justify-content:space-between;">
                <span>{sym}</span>
                <span class="badge {sig_badge}">{txt}</span>
            </div>
            <small>₹{d.get("ltp", 0):.2f} | VWAP ₹{d.get("vwap", 0):.2f} | Ratio {d.get("ratio", 0):.2f}x<br>
            CE Vol {d.get("opt_power_ce",0):.2f} vs PE Vol {d.get("opt_power_pe",0):.2f}<br>
            OptBias {d.get("opt_power",0):.2f} M | Persist {d.get("persist",0)} | IdlePersist {d.get("idle_persist",0)}</small>
        </div>
        """,unsafe_allow_html=True)

    with st.expander("📟 DEBUG TAPE"):
         for sym,d in sorted_syms:
              st.write(sym)
              st.text("\n".join(d.get("tape","")[:10]))

st.text_area("System logs","\n".join([l for l in manager.logs]),height=200)

if manager.is_running:
    time.sleep(2)
    st.rerun()
