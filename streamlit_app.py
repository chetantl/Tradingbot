import streamlit as st
from kiteconnect import KiteConnect
import threading
import time
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import copy
from pytz import timezone

# ====== TIMEZONE (INDIA) ====== #
INDIA_TZ = timezone("Asia/Kolkata")

# ====== PAGE CONFIG ====== #
st.set_page_config(
    page_title="Institutional Sniper",
    page_icon="🎯",
    layout="wide",
)

# ====== CUSTOM CSS FOR COMPACT CARDS ====== #
st.markdown("""
<style>
    /* Reduce overall padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Compact metric styling */
    [data-testid="stMetricValue"] {
        font-size: 0.9rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.7rem !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.65rem !important;
    }
    
    /* Smaller expander text */
    .streamlit-expanderHeader {
        font-size: 0.8rem !important;
    }
    
    /* Compact card style */
    .compact-card {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #444;
    }
    
    .card-bullish {
        border-left-color: #00c853 !important;
    }
    
    .card-bearish {
        border-left-color: #ff1744 !important;
    }
    
    .card-warning {
        border-left-color: #ffc107 !important;
    }
    
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    
    .symbol-name {
        font-size: 1.1rem;
        font-weight: bold;
        color: #ffffff;
    }
    
    .signal-badge {
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: bold;
    }
    
    .badge-call {
        background-color: #1b5e20;
        color: #a5d6a7;
    }
    
    .badge-put {
        background-color: #b71c1c;
        color: #ef9a9a;
    }
    
    .badge-conflict {
        background-color: #e65100;
        color: #ffcc80;
    }
    
    .badge-idle {
        background-color: #37474f;
        color: #b0bec5;
    }
    
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 6px;
        margin-bottom: 8px;
    }
    
    .metric-box {
        background-color: #2d2d2d;
        padding: 6px;
        border-radius: 4px;
        text-align: center;
    }
    
    .metric-label {
        font-size: 0.6rem;
        color: #888;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    
    .metric-value {
        font-size: 0.85rem;
        font-weight: bold;
        color: #fff;
    }
    
    .metric-sub {
        font-size: 0.55rem;
        color: #666;
    }
    
    .bullish {
        color: #4caf50 !important;
    }
    
    .bearish {
        color: #f44336 !important;
    }
    
    .neutral {
        color: #9e9e9e !important;
    }
    
    .oi-analysis {
        background-color: #252525;
        padding: 6px;
        border-radius: 4px;
        font-size: 0.65rem;
        color: #aaa;
        margin-top: 6px;
    }
    
    .footer-info {
        font-size: 0.55rem;
        color: #555;
        margin-top: 4px;
    }
    
    /* Status bar styling */
    .status-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #1a1a1a;
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 15px;
    }
    
    .status-item {
        text-align: center;
    }
    
    .status-label {
        font-size: 0.6rem;
        color: #666;
    }
    
    .status-value {
        font-size: 0.85rem;
        font-weight: bold;
    }
    
    .live {
        color: #4caf50;
    }
    
    .stopped {
        color: #f44336;
    }
</style>
""", unsafe_allow_html=True)

# ====== CONFIG ====== #
CONFIG = {
    "ratio_threshold": 2.5,
    "strong_ratio_threshold": 4.0,
    "persistence_count": 2,
    "strong_persistence": 1,
    "imbalance_mult": 1.5,
    "history_days": 10,
    "oi_change_threshold": 0.05,
    "pcr_bullish_threshold": 0.7,
    "pcr_bearish_threshold": 1.3,
    "atm_range_pct": 0.03,
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


def get_expiry_date(expiry):
    if expiry is None:
        return None
    if hasattr(expiry, 'date'):
        return expiry.date()
    return expiry


def current_bar_id_5m():
    now = now_ist()
    minute_slot = (now.minute // 5) * 5
    bar_start = now.replace(minute=minute_slot, second=0, microsecond=0)
    return int(bar_start.timestamp()), bar_start


# ====== OI ANALYSIS ====== #
class OISnapshot:
    def __init__(self):
        self.timestamp = None
        self.ce_oi_total = 0
        self.pe_oi_total = 0
        self.ce_oi_atm = 0
        self.pe_oi_atm = 0
        self.max_ce_oi_strike = 0
        self.max_pe_oi_strike = 0
        self.max_ce_oi = 0
        self.max_pe_oi = 0
        self.strike_data = {}


class OIAnalysis:
    @staticmethod
    def calculate_pcr(pe_oi, ce_oi):
        if ce_oi == 0:
            return 0.0
        return round(pe_oi / ce_oi, 2)
    
    @staticmethod
    def analyze_oi_change(prev, curr, ltp, vwap):
        if prev is None or curr is None:
            return "NEUTRAL", 0.3, "Waiting for OI data..."
        
        if prev.ce_oi_total == 0 and prev.pe_oi_total == 0:
            return "NEUTRAL", 0.3, "Previous OI empty"
        
        if curr.ce_oi_total == 0 and curr.pe_oi_total == 0:
            return "NEUTRAL", 0.3, "Current OI empty"
        
        parts = []
        
        prev_pcr = OIAnalysis.calculate_pcr(prev.pe_oi_total, prev.ce_oi_total)
        curr_pcr = OIAnalysis.calculate_pcr(curr.pe_oi_total, curr.ce_oi_total)
        pcr_change = curr_pcr - prev_pcr
        
        ce_oi_change = curr.ce_oi_total - prev.ce_oi_total
        pe_oi_change = curr.pe_oi_total - prev.pe_oi_total
        
        ce_pct = (ce_oi_change / prev.ce_oi_total * 100) if prev.ce_oi_total > 0 else 0
        pe_pct = (pe_oi_change / prev.pe_oi_total * 100) if prev.pe_oi_total > 0 else 0
        
        ce_atm_chg = curr.ce_oi_atm - prev.ce_oi_atm
        pe_atm_chg = curr.pe_oi_atm - prev.pe_oi_atm
        
        sup_shift = curr.max_pe_oi_strike - prev.max_pe_oi_strike
        res_shift = curr.max_ce_oi_strike - prev.max_ce_oi_strike
        
        bull = 0
        bear = 0
        
        if pe_oi_change > 0 and ltp >= vwap:
            bull += 2
            parts.append(f"Put+{pe_pct:.1f}%")
        
        if ce_oi_change < 0 and ce_atm_chg < 0:
            bull += 2
            parts.append(f"CallCover{ce_pct:.1f}%")
        
        if curr_pcr < CONFIG["pcr_bullish_threshold"] and pcr_change > 0.05:
            bull += 1
            parts.append("PCR↑")
        
        if sup_shift > 0:
            bull += 1
            parts.append("Sup↑")
        
        if ltp > curr.max_pe_oi_strike:
            bull += 1
        
        if ce_oi_change > 0 and ltp <= vwap:
            bear += 2
            parts.append(f"Call+{ce_pct:.1f}%")
        
        if pe_oi_change < 0 and pe_atm_chg < 0:
            bear += 2
            parts.append(f"PutUnwind{pe_pct:.1f}%")
        
        if curr_pcr > CONFIG["pcr_bearish_threshold"] and pcr_change < -0.05:
            bear += 1
            parts.append("PCR↓")
        
        if res_shift < 0:
            bear += 1
            parts.append("Res↓")
        
        if ltp < curr.max_ce_oi_strike:
            bear += 1
        
        net = bull - bear
        
        if net >= 3:
            sig = "STRONG_BULLISH"
            conf = min(0.9, 0.5 + net * 0.1)
        elif net >= 1:
            sig = "BULLISH"
            conf = min(0.7, 0.4 + net * 0.1)
        elif net <= -3:
            sig = "STRONG_BEARISH"
            conf = min(0.9, 0.5 + abs(net) * 0.1)
        elif net <= -1:
            sig = "BEARISH"
            conf = min(0.7, 0.4 + abs(net) * 0.1)
        else:
            sig = "NEUTRAL"
            conf = 0.3
            if not parts:
                parts.append("No change")
        
        return sig, conf, " | ".join(parts) if parts else "Analyzing..."


# ====== MANAGER CLASS ====== #
class SniperManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.logs = deque(maxlen=200)
        self.data = {}
        self.active_symbols = []
        self.is_running = False
        self.stop_event = threading.Event()
        self.kite = None
        self.last_beat = 0.0
        self.initialized = False
        self.oi_snapshots = {}
        self.bar_trackers = {}

    def log(self, msg, level="INFO"):
        ts = now_ist().strftime("%H:%M:%S")
        with self.lock:
            self.logs.appendleft(f"[{ts}] {level}: {msg}")

    def get_data_snapshot(self):
        with self.lock:
            return copy.deepcopy(self.data)

    def get_logs_snapshot(self):
        with self.lock:
            return list(self.logs)

    def set_symbol_data(self, symbol, data):
        with self.lock:
            self.data[symbol] = data

    def get_symbol_data(self, symbol):
        with self.lock:
            return self.data.get(symbol, {}).copy()
    
    def update_oi_snapshot(self, symbol, snapshot):
        with self.lock:
            if symbol not in self.oi_snapshots:
                self.oi_snapshots[symbol] = {'prev': None, 'curr': None}
            self.oi_snapshots[symbol]['prev'] = self.oi_snapshots[symbol]['curr']
            self.oi_snapshots[symbol]['curr'] = snapshot
    
    def get_oi_snapshots(self, symbol):
        with self.lock:
            if symbol in self.oi_snapshots:
                return (self.oi_snapshots[symbol]['prev'], self.oi_snapshots[symbol]['curr'])
            return None, None
    
    def init_bar_tracker(self, symbol, bar_id):
        with self.lock:
            self.bar_trackers[symbol] = {
                'bar_id': bar_id,
                'total_vol': 0,
                'total_ob_change': 0,
                'tick_count': 0,
                'last_cum_vol': None,
                'last_buy_q': None,
                'last_sell_q': None,
            }
    
    def update_bar_tracker(self, symbol, bar_id, cum_vol, buy_q, sell_q):
        with self.lock:
            if symbol not in self.bar_trackers:
                self.init_bar_tracker(symbol, bar_id)
            
            tracker = self.bar_trackers[symbol]
            
            if tracker['bar_id'] != bar_id:
                prev_stats = {
                    'total_vol': tracker['total_vol'],
                    'total_ob_change': tracker['total_ob_change'],
                    'tick_count': tracker['tick_count'],
                }
                self.bar_trackers[symbol] = {
                    'bar_id': bar_id,
                    'total_vol': 0,
                    'total_ob_change': 0,
                    'tick_count': 1,
                    'last_cum_vol': cum_vol,
                    'last_buy_q': buy_q,
                    'last_sell_q': sell_q,
                }
                return prev_stats, True
            
            if tracker['last_cum_vol'] is not None:
                d_vol = max(0, cum_vol - tracker['last_cum_vol'])
                tracker['total_vol'] += d_vol
            
            if tracker['last_buy_q'] is not None and tracker['last_sell_q'] is not None:
                d_ob = abs(buy_q - tracker['last_buy_q']) + abs(sell_q - tracker['last_sell_q'])
                tracker['total_ob_change'] += d_ob
            
            tracker['last_cum_vol'] = cum_vol
            tracker['last_buy_q'] = buy_q
            tracker['last_sell_q'] = sell_q
            tracker['tick_count'] += 1
            
            return {
                'total_vol': tracker['total_vol'],
                'total_ob_change': tracker['total_ob_change'],
                'tick_count': tracker['tick_count'],
            }, False


def get_fresh_manager():
    return SniperManager()


if 'manager' not in st.session_state:
    st.session_state.manager = get_fresh_manager()

manager = st.session_state.manager


# ====== KITE HELPERS ====== #
def get_instrument_token(kite, symbol):
    try:
        instruments = kite.instruments("NSE")
        sym = symbol.upper().strip()
        for inst in instruments:
            if inst["tradingsymbol"] == sym and inst["segment"] == "NSE":
                return inst["instrument_token"]
    except Exception as e:
        manager.log(f"Token error {symbol}: {e}", "ERROR")
    return None


def calc_rvol(kite, token, days=10):
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
        manager.log(f"RVol error: {e}", "ERROR")
        return 0.0, 0, 0.0


def scan_option_chain_oi(kite, symbol, spot_price):
    try:
        all_opts = [
            i for i in kite.instruments("NFO")
            if i.get("segment") == "NFO-OPT" and i.get("name") == symbol
        ]
        if not all_opts:
            return None

        today = now_ist().date()
        
        expiries = sorted({
            get_expiry_date(i["expiry"])
            for i in all_opts
            if get_expiry_date(i["expiry"]) is not None and get_expiry_date(i["expiry"]) >= today
        })
        
        if not expiries:
            return None

        nearest = expiries[0]
        scoped = [i for i in all_opts if get_expiry_date(i["expiry"]) == nearest]

        ce_opts = {i['strike']: i for i in scoped if i["instrument_type"] == "CE"}
        pe_opts = {i['strike']: i for i in scoped if i["instrument_type"] == "PE"}
        
        all_strikes = sorted(set(ce_opts.keys()) | set(pe_opts.keys()))
        
        if not all_strikes:
            return None
        
        relevant_strikes = [s for s in all_strikes if spot_price * 0.9 <= s <= spot_price * 1.1]
        if not relevant_strikes:
            relevant_strikes = all_strikes[:20]
        
        atm_low = spot_price * (1 - CONFIG["atm_range_pct"])
        atm_high = spot_price * (1 + CONFIG["atm_range_pct"])
        
        ce_syms = [f"NFO:{ce_opts[s]['tradingsymbol']}" for s in relevant_strikes if s in ce_opts]
        pe_syms = [f"NFO:{pe_opts[s]['tradingsymbol']}" for s in relevant_strikes if s in pe_opts]
        
        ce_quotes = {}
        pe_quotes = {}
        
        for i in range(0, len(ce_syms), 40):
            batch = ce_syms[i:i+40]
            try:
                q = kite.quote(batch)
                ce_quotes.update(q)
                time.sleep(0.1)
            except:
                pass
        
        for i in range(0, len(pe_syms), 40):
            batch = pe_syms[i:i+40]
            try:
                q = kite.quote(batch)
                pe_quotes.update(q)
                time.sleep(0.1)
            except:
                pass
        
        snapshot = OISnapshot()
        snapshot.timestamp = now_ist()
        
        max_ce_oi = 0
        max_pe_oi = 0
        
        for strike in relevant_strikes:
            ce_oi = 0
            pe_oi = 0
            
            if strike in ce_opts:
                ce_sym = f"NFO:{ce_opts[strike]['tradingsymbol']}"
                if ce_sym in ce_quotes:
                    ce_oi = safe_int(ce_quotes[ce_sym].get('oi', 0))
            
            if strike in pe_opts:
                pe_sym = f"NFO:{pe_opts[strike]['tradingsymbol']}"
                if pe_sym in pe_quotes:
                    pe_oi = safe_int(pe_quotes[pe_sym].get('oi', 0))
            
            snapshot.strike_data[strike] = {'ce_oi': ce_oi, 'pe_oi': pe_oi}
            snapshot.ce_oi_total += ce_oi
            snapshot.pe_oi_total += pe_oi
            
            if atm_low <= strike <= atm_high:
                snapshot.ce_oi_atm += ce_oi
                snapshot.pe_oi_atm += pe_oi
            
            if ce_oi > max_ce_oi:
                max_ce_oi = ce_oi
                snapshot.max_ce_oi_strike = int(strike)
                snapshot.max_ce_oi = ce_oi
            
            if pe_oi > max_pe_oi:
                max_pe_oi = pe_oi
                snapshot.max_pe_oi_strike = int(strike)
                snapshot.max_pe_oi = pe_oi
        
        return snapshot
        
    except Exception as e:
        manager.log(f"OI scan error {symbol}: {e}", "ERROR")
        return None


# ====== FOOTPRINT LOGIC ====== #
def classify_raw_signal(ratio, ltp, vwap, buy_q, sell_q):
    raw = "IDLE"
    reason = ""
    
    if ratio > CONFIG["ratio_threshold"]:
        strong = ratio > CONFIG["strong_ratio_threshold"]
        sell_dom = sell_q > buy_q * CONFIG["imbalance_mult"]
        buy_dom = buy_q > sell_q * CONFIG["imbalance_mult"]

        if ltp >= vwap and (sell_dom or strong):
            raw = "ACCUMULATION"
            reason = "Price≥VWAP"
        elif ltp <= vwap and (buy_dom or strong):
            raw = "DISTRIBUTION"
            reason = "Price≤VWAP"
        else:
            reason = f"R:{ratio:.1f}x"
    else:
        reason = f"Low R:{ratio:.1f}x"
    
    return raw, reason


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


def generate_trade_signal(base_signal, oi_signal):
    trade_signal = "IDLE"
    signal_strength = "WEAK"
    
    if base_signal == "ACCUMULATION":
        if oi_signal in ("STRONG_BULLISH", "BULLISH"):
            trade_signal = "BUY CALL"
            signal_strength = "STRONG" if oi_signal == "STRONG_BULLISH" else "CONFIRMED"
        elif oi_signal == "NEUTRAL":
            trade_signal = "BUY CALL"
            signal_strength = "UNCONFIRMED"
        else:
            trade_signal = "CONFLICTING"
            signal_strength = "AVOID"
    
    elif base_signal == "DISTRIBUTION":
        if oi_signal in ("STRONG_BEARISH", "BEARISH"):
            trade_signal = "BUY PUT"
            signal_strength = "STRONG" if oi_signal == "STRONG_BEARISH" else "CONFIRMED"
        elif oi_signal == "NEUTRAL":
            trade_signal = "BUY PUT"
            signal_strength = "UNCONFIRMED"
        else:
            trade_signal = "CONFLICTING"
            signal_strength = "AVOID"
    
    return trade_signal, signal_strength


# ====== WORKER THREAD ====== #
def sniper_worker(kite, mgr):
    mgr.is_running = True
    mgr.log("Worker started", "INFO")

    last_oi_scan = 0.0
    curr_bar_id, curr_bar_start = current_bar_id_5m()

    while not mgr.stop_event.is_set():
        try:
            now_ts = time.time()
            mgr.last_beat = now_ts

            with mgr.lock:
                active_list = list(mgr.active_symbols)

            if not active_list:
                time.sleep(1)
                continue

            new_bar_id, new_bar_start = current_bar_id_5m()
            bar_changed = new_bar_id != curr_bar_id

            keys = [f"NSE:{row['symbol']}" for row in active_list]
            try:
                quotes = kite.quote(keys)
            except Exception as e:
                mgr.log(f"Quote error: {e}", "WARNING")
                time.sleep(2)
                continue

            do_oi_scan = (now_ts - last_oi_scan) > 120
            if do_oi_scan:
                last_oi_scan = now_ts

            for row in active_list:
                sym = row["symbol"]
                key = f"NSE:{sym}"
                q = quotes.get(key)
                if not q:
                    continue

                stt = mgr.get_symbol_data(sym)

                ltp = safe_float(q.get("last_price"))
                vwap = safe_float(q.get("average_price"), ltp)
                depth = q.get("depth", {})

                buy_q = sum(safe_int(x.get("quantity")) for x in depth.get("buy", [])[:5])
                sell_q = sum(safe_int(x.get("quantity")) for x in depth.get("sell", [])[:5])
                cum_vol = safe_int(q.get("volume"))

                bar_stats, is_new_bar = mgr.update_bar_tracker(sym, curr_bar_id, cum_vol, buy_q, sell_q)
                
                if not stt:
                    stt = {
                        "ltp": ltp, "vwap": vwap, "buy_q": buy_q, "sell_q": sell_q,
                        "bar_vol": 0, "bar_ob_change": 0, "ratio_bar": 0.0,
                        "raw_signal": "IDLE", "raw_reason": "Init",
                        "base_signal": "IDLE", "persist": 0, "idle_persist": 0,
                        "oi_signal": "NEUTRAL", "oi_confidence": 0.3,
                        "oi_analysis": "Waiting...", "pcr": 0.0,
                        "support_strike": 0, "resistance_strike": 0,
                        "ce_oi_total": 0, "pe_oi_total": 0,
                        "rvol": 0.0, "rvol_avg": 0.0,
                        "trade_signal": "IDLE", "signal_strength": "WEAK",
                        "tick_count": 0,
                    }

                stt["bar_vol"] = bar_stats['total_vol']
                stt["bar_ob_change"] = bar_stats['total_ob_change']
                stt["tick_count"] = bar_stats['tick_count']
                
                if bar_stats['total_ob_change'] > 0:
                    stt["ratio_bar"] = round(bar_stats['total_vol'] / bar_stats['total_ob_change'], 2)
                
                if bar_changed and is_new_bar:
                    bar_vol = bar_stats['total_vol']
                    bar_ob = bar_stats['total_ob_change']
                    ratio_bar = bar_vol / max(1, bar_ob) if bar_ob > 0 else 0.0
                    
                    raw, reason = classify_raw_signal(ratio_bar, ltp, vwap, buy_q, sell_q)
                    base_signal = apply_persistence(stt, raw, ratio_bar)
                    
                    stt["ratio_bar"] = round(ratio_bar, 2)
                    stt["raw_reason"] = reason
                    
                    mgr.log(f"{sym}: R={ratio_bar:.1f}x {raw}→{base_signal}", "INFO")
                
                if do_oi_scan:
                    oi_snapshot = scan_option_chain_oi(kite, sym, ltp)
                    
                    if oi_snapshot:
                        mgr.update_oi_snapshot(sym, oi_snapshot)
                        prev_snap, curr_snap = mgr.get_oi_snapshots(sym)
                        
                        if prev_snap and curr_snap:
                            oi_signal, oi_confidence, oi_analysis = OIAnalysis.analyze_oi_change(
                                prev_snap, curr_snap, ltp, vwap
                            )
                            stt["oi_signal"] = oi_signal
                            stt["oi_confidence"] = oi_confidence
                            stt["oi_analysis"] = oi_analysis
                        
                        if curr_snap:
                            stt["pcr"] = OIAnalysis.calculate_pcr(curr_snap.pe_oi_total, curr_snap.ce_oi_total)
                            stt["support_strike"] = int(curr_snap.max_pe_oi_strike)
                            stt["resistance_strike"] = int(curr_snap.max_ce_oi_strike)

                trade_signal, signal_strength = generate_trade_signal(
                    stt.get("base_signal", "IDLE"),
                    stt.get("oi_signal", "NEUTRAL")
                )
                
                stt["trade_signal"] = trade_signal
                stt["signal_strength"] = signal_strength
                stt["ltp"] = ltp
                stt["vwap"] = vwap
                stt["buy_q"] = buy_q
                stt["sell_q"] = sell_q

                mgr.set_symbol_data(sym, stt)

            if bar_changed:
                curr_bar_id = new_bar_id
                curr_bar_start = new_bar_start
                mgr.log(f"NEW BAR: {new_bar_start.strftime('%H:%M')}", "INFO")

            time.sleep(POLL_INTERVAL_SEC)

        except Exception as e:
            mgr.log(f"Error: {e}", "ERROR")
            time.sleep(2)

    mgr.is_running = False
    mgr.log("Worker stopped", "INFO")


# ====== RENDER COMPACT CARD ====== #
def render_compact_card(sym, d):
    trade_sig = d.get("trade_signal", "IDLE")
    signal_strength = d.get("signal_strength", "WEAK")
    base_sig = d.get("base_signal", "IDLE")
    oi_sig = d.get("oi_signal", "NEUTRAL")
    oi_conf = d.get("oi_confidence", 0.0)
    oi_analysis = d.get("oi_analysis", "")
    
    ltp = d.get('ltp', 0)
    vwap = d.get('vwap', 0)
    pcr = d.get('pcr', 0)
    ratio = d.get('ratio_bar', 0)
    support = d.get('support_strike', 0)
    resistance = d.get('resistance_strike', 0)
    bar_vol = d.get('bar_vol', 0)
    tick_count = d.get('tick_count', 0)
    
    # Determine card class
    if trade_sig == "BUY CALL":
        card_class = "card-bullish"
        badge_class = "badge-call"
    elif trade_sig == "BUY PUT":
        card_class = "card-bearish"
        badge_class = "badge-put"
    elif trade_sig == "CONFLICTING":
        card_class = "card-warning"
        badge_class = "badge-conflict"
    else:
        card_class = ""
        badge_class = "badge-idle"
    
    # Base signal color
    base_class = "bullish" if base_sig == "ACCUMULATION" else "bearish" if base_sig == "DISTRIBUTION" else "neutral"
    
    # OI signal color
    oi_class = "bullish" if "BULLISH" in oi_sig else "bearish" if "BEARISH" in oi_sig else "neutral"
    
    # Price indicator
    price_ind = "▲" if ltp > vwap else "▼" if ltp < vwap else "●"
    price_class = "bullish" if ltp > vwap else "bearish" if ltp < vwap else "neutral"
    
    card_html = f'''
    <div class="compact-card {card_class}">
        <div class="card-header">
            <span class="symbol-name">{sym}</span>
            <span class="signal-badge {badge_class}">{trade_sig} ({signal_strength})</span>
        </div>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">LTP</div>
                <div class="metric-value {price_class}">₹{ltp:.2f} {price_ind}</div>
                <div class="metric-sub">VWAP: {vwap:.2f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Footprint</div>
                <div class="metric-value {base_class}">{base_sig}</div>
                <div class="metric-sub">Ratio: {ratio:.2f}x</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">OI Signal</div>
                <div class="metric-value {oi_class}">{oi_sig}</div>
                <div class="metric-sub">Conf: {oi_conf:.0%} | PCR: {pcr:.2f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Levels</div>
                <div class="metric-value"><span class="bullish">S:{support}</span> <span class="bearish">R:{resistance}</span></div>
                <div class="metric-sub">Vol: {bar_vol:,}</div>
            </div>
        </div>
        <div class="oi-analysis">{oi_analysis}</div>
        <div class="footer-info">Ticks: {tick_count} | RVol: {d.get('rvol', 0):.2f}x</div>
    </div>
    '''
    
    st.markdown(card_html, unsafe_allow_html=True)


# ====== SIDEBAR ====== #
st.sidebar.title("🔐 Kite Connect")

if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "api_sec" not in st.session_state:
    st.session_state.api_sec = ""
if "req_token" not in st.session_state:
    st.session_state.req_token = ""

api_key = st.sidebar.text_input("API Key", value=st.session_state.api_key, key="api_key_input")
api_sec = st.sidebar.text_input("API Secret", value=st.session_state.api_sec, type="password", key="api_sec_input")
req_token = st.sidebar.text_input("Request Token", value=st.session_state.req_token, key="req_token_input")

if api_key:
    try:
        temp_kite = KiteConnect(api_key=api_key)
        st.sidebar.markdown(f"[🔗 Login]({temp_kite.login_url()})")
    except:
        pass

if st.sidebar.button("🚀 Connect", key="connect_btn"):
    if api_key and api_sec and req_token:
        try:
            kite = KiteConnect(api_key=api_key)
            ses = kite.generate_session(req_token, api_secret=api_sec)
            kite.set_access_token(ses["access_token"])
            manager.kite = kite
            st.session_state.api_key = api_key
            st.session_state.api_sec = api_sec
            st.session_state.req_token = req_token
            manager.log("Connected", "SUCCESS")
            st.sidebar.success("✅ Connected")
        except Exception as e:
            st.sidebar.error(f"Failed: {e}")

st.sidebar.markdown("---")

symbols_text = st.sidebar.text_area("Symbols", "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK", key="symbols_input")

col1, col2 = st.sidebar.columns(2)

if col1.button("▶️ START", key="start_btn"):
    if not manager.kite:
        st.error("Connect first")
    elif manager.is_running:
        st.warning("Already running")
    else:
        syms = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
        valid = []

        with manager.lock:
            manager.active_symbols = []
            manager.data = {}
            manager.oi_snapshots = {}
            manager.bar_trackers = {}

        with st.spinner("Initializing..."):
            for s in syms:
                tkn = get_instrument_token(manager.kite, s)
                if not tkn:
                    continue

                rvol, _, rvol_avg = calc_rvol(manager.kite, tkn)
                valid.append({"symbol": s, "token": tkn})

                try:
                    q = manager.kite.quote(f"NSE:{s}")[f"NSE:{s}"]
                    ltp = safe_float(q.get("last_price"))
                    vwap = safe_float(q.get("average_price"), ltp)

                    oi_snapshot = scan_option_chain_oi(manager.kite, s, ltp)
                    
                    pcr, support, resistance = 0.0, 0, 0
                    if oi_snapshot:
                        manager.update_oi_snapshot(s, oi_snapshot)
                        pcr = OIAnalysis.calculate_pcr(oi_snapshot.pe_oi_total, oi_snapshot.ce_oi_total)
                        support = int(oi_snapshot.max_pe_oi_strike)
                        resistance = int(oi_snapshot.max_ce_oi_strike)

                    curr_bar_id, _ = current_bar_id_5m()
                    manager.init_bar_tracker(s, curr_bar_id)

                    stt = {
                        "ltp": ltp, "vwap": vwap, "buy_q": 0, "sell_q": 0,
                        "bar_vol": 0, "bar_ob_change": 0, "ratio_bar": 0.0,
                        "raw_signal": "IDLE", "raw_reason": "Init",
                        "base_signal": "IDLE", "persist": 0, "idle_persist": 0,
                        "oi_signal": "NEUTRAL", "oi_confidence": 0.3,
                        "oi_analysis": f"PCR={pcr:.2f}",
                        "pcr": pcr, "support_strike": support, "resistance_strike": resistance,
                        "rvol": rvol, "rvol_avg": rvol_avg,
                        "trade_signal": "IDLE", "signal_strength": "WEAK",
                        "tick_count": 0,
                    }
                    manager.set_symbol_data(s, stt)
                except Exception as e:
                    manager.log(f"Init {s}: {e}", "WARNING")

        if valid:
            with manager.lock:
                manager.active_symbols = valid
                manager.stop_event.clear()
            t = threading.Thread(target=sniper_worker, args=(manager.kite, manager), daemon=True)
            t.start()
            st.success(f"Started {len(valid)} symbols")

if col2.button("⏹️ STOP", key="stop_btn"):
    manager.stop_event.set()
    manager.is_running = False
    time.sleep(1)
    st.rerun()

if st.sidebar.button("🔄 Reset", key="reset_btn"):
    st.session_state.manager = get_fresh_manager()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"Status: {'🟢 Running' if manager.is_running else '🔴 Stopped'}")


# ====== MAIN DASHBOARD ====== #
st.title("🎯 Institutional Sniper")

# Status bar
last_beat = manager.last_beat
delta = time.time() - last_beat
is_live = delta < 5

curr_bar_id, curr_bar_start = current_bar_id_5m()
bar_end = curr_bar_start + timedelta(minutes=5)
time_to_close = max(0, (bar_end - now_ist()).total_seconds())
beat_time = datetime.fromtimestamp(last_beat, INDIA_TZ).strftime("%H:%M:%S") if last_beat > 0 else "-"

status_html = f'''
<div class="status-bar">
    <div class="status-item">
        <div class="status-label">STATUS</div>
        <div class="status-value {'live' if is_live else 'stopped'}">{'🟢 LIVE' if is_live else '🔴 STOPPED'}</div>
    </div>
    <div class="status-item">
        <div class="status-label">CURRENT BAR</div>
        <div class="status-value">{curr_bar_start.strftime('%H:%M')} - {bar_end.strftime('%H:%M')}</div>
    </div>
    <div class="status-item">
        <div class="status-label">CLOSES IN</div>
        <div class="status-value" style="color: #ffc107;">{int(time_to_close)}s</div>
    </div>
    <div class="status-item">
        <div class="status-label">LAST BEAT</div>
        <div class="status-value">{beat_time}</div>
    </div>
</div>
'''
st.markdown(status_html, unsafe_allow_html=True)

# Symbol cards
snapshot = manager.get_data_snapshot()

if snapshot:
    st.markdown(f"### 📈 Monitoring {len(snapshot)} Symbols")
    
    items = sorted(
        snapshot.items(),
        key=lambda x: (
            0 if x[1].get("signal_strength") == "STRONG" else
            1 if x[1].get("trade_signal") in ("BUY CALL", "BUY PUT") else 2,
            -x[1].get("oi_confidence", 0.0),
        ),
    )
    
    col_left, col_right = st.columns(2)
    
    for i, (sym, d) in enumerate(items):
        if i % 2 == 0:
            with col_left:
                render_compact_card(sym, d)
        else:
            with col_right:
                render_compact_card(sym, d)
else:
    st.info("👆 Enter symbols and click START")

st.markdown("---")

# Guide
with st.expander("📖 Guide", expanded=False):
    st.markdown("""
    **Signals:** BUY CALL (STRONG) = Accumulation + Bullish OI | BUY PUT (STRONG) = Distribution + Bearish OI | CONFLICTING = Avoid
    
    **OI Patterns:** Put+ = Support | CallCover = Short covering | Call+ = Resistance | PutUnwind = Longs exiting
    
    **PCR:** <0.7 Oversold | 0.7-1.3 Neutral | >1.3 Overbought
    """)

# Logs
with st.expander("📝 Logs", expanded=False):
    logs_list = manager.get_logs_snapshot()
    st.text_area("Logs", "\n".join(logs_list[:50]) if logs_list else "No logs...", height=150, disabled=True, label_visibility="collapsed")

# Auto-refresh
if manager.is_running:
    time.sleep(3)
    st.rerun()
