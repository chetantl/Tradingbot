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
    page_title="Institutional Sniper (Absorption Model)",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
<style>
.card-container {
    background-color: #111827;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 12px;
}
.metric-box {
    background: #1f2937;
    border-radius: 4px;
    padding: 8px;
    margin: 4px 0;
}
.accum-glow {
    box-shadow: 0 0 15px rgba(34, 197, 94, 0.4);
    border: 1px solid #22c55e;
}
.dist-glow {
    box-shadow: 0 0 15px rgba(239, 68, 68, 0.4);
    border: 1px solid #ef4444;
}
</style>
""",
    unsafe_allow_html=True,
)

# ====== CONFIG ====== #
CONFIG = {
    # Base absorption thresholds (will be scaled per-symbol using avg_vol)
    "absorption_threshold": 50000,
    "strong_absorption": 100000,
    
    # Efficiency thresholds
    "low_efficiency_threshold": 0.00005,
    "high_efficiency_threshold": 0.0005,
    
    # Signal confirmation
    "absorption_bars_needed": 2,
    "score_threshold": 70,
    
    # Price movement minimum (base, will be made dynamic)
    "min_price_move": 0.01,
    
    # Tick history
    "default_max_ticks": 500,
    
    # Historical bars for average
    "avg_bars": 750,
    
    # Buy/sell estimation weights
    "tick_rule_weight": 0.7,
    "book_delta_weight": 0.3,
    
    # Minimum ticks for valid classification
    "min_ticks_for_signal": 5,
}

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
    now = now_ist()
    minute_slot = (now.minute // 5) * 5
    bar_start = now.replace(minute=minute_slot, second=0, microsecond=0)
    return int(bar_start.timestamp()), bar_start


def safe_quote_batch(kite, batch, manager_log_func, attempts=2, sleep_s=0.3):
    """
    Safely fetch quotes with retry logic.
    Returns dict of quotes (may be partial if some fail).
    """
    for attempt in range(attempts):
        try:
            q = kite.quote(batch)
            # Check completeness
            if len(q) < len(batch):
                manager_log_func(
                    f"Quote batch incomplete: got {len(q)}/{len(batch)} symbols",
                    "WARNING"
                )
            return q
        except Exception as e:
            if attempt < attempts - 1:
                time.sleep(sleep_s)
            else:
                manager_log_func(f"Quote batch failed after {attempts} attempts: {e}", "WARNING")
    return {}


# ====== ABSORPTION ENGINE (Thread-Safe, Adaptive) ====== #
class AbsorptionEngine:
    """
    Real institutional detection using:
    1. ABSORPTION = Volume absorbed without price impact
    2. EFFICIENCY = Price move per unit volume
    3. IMBALANCE = Aggressive volume vs price direction
    
    Thread-safe with adaptive parameters per symbol.
    """
    
    def __init__(self):
        self.tick_history = {}  # symbol -> deque of ticks
        self.lock = threading.RLock()
        self.default_max_ticks = CONFIG["default_max_ticks"]

    def _get_max_ticks_for_symbol(self, avg_vol):
        """Adaptive tick history size based on symbol liquidity."""
        if not avg_vol or avg_vol < 1000:
            return 200
        elif avg_vol < 5000:
            return 300
        elif avg_vol < 10000:
            return 400
        else:
            return self.default_max_ticks

    def record_tick(self, symbol, ltp, vol_delta, buy_delta, sell_delta, avg_vol=None):
        """Record each tick for analysis (thread-safe)."""
        with self.lock:
            if symbol not in self.tick_history:
                max_ticks = self._get_max_ticks_for_symbol(avg_vol or 0)
                self.tick_history[symbol] = deque(maxlen=max_ticks)
            
            self.tick_history[symbol].append({
                "ts": time.time(),
                "ltp": ltp,
                "vol_delta": vol_delta,
                "buy_delta": buy_delta,
                "sell_delta": sell_delta,
            })

    def reset_bar(self, symbol):
        """Reset tick history at bar close (thread-safe)."""
        with self.lock:
            if symbol in self.tick_history:
                self.tick_history[symbol].clear()

    def get_ticks_snapshot(self, symbol):
        """Get thread-safe copy of tick history."""
        with self.lock:
            return list(self.tick_history.get(symbol, []))

    def get_tick_count(self, symbol):
        """Get current tick count for a symbol."""
        with self.lock:
            return len(self.tick_history.get(symbol, []))

    def calculate_absorption_metrics(self, symbol, current_ltp, avg_vol=None, 
                                      abs_threshold=None, strong_abs=None):
        """
        Calculate absorption metrics from tick history.
        
        Args:
            symbol: Stock symbol
            current_ltp: Current LTP for dynamic calculations
            avg_vol: Average volume for adaptive thresholds
            abs_threshold: Per-symbol absorption threshold (if pre-computed)
            strong_abs: Per-symbol strong absorption threshold
        
        Returns:
            dict with absorption scores and classifications
        """
        ticks = self.get_ticks_snapshot(symbol)
        
        # Minimum ticks required for valid analysis
        min_ticks = CONFIG["min_ticks_for_signal"]
        if len(ticks) < min_ticks:
            return self._empty_metrics(tick_count=len(ticks))

        # ========== PRICE ANALYSIS ==========
        first_price = ticks[0]["ltp"]
        last_price = ticks[-1]["ltp"]
        price_change = last_price - first_price
        price_up = max(0, price_change)
        price_down = abs(min(0, price_change))
        
        prices = [t["ltp"] for t in ticks]
        price_high = max(prices)
        price_low = min(prices)
        price_range = price_high - price_low

        # ========== VOLUME ANALYSIS ==========
        total_volume = sum(t["vol_delta"] for t in ticks)
        total_buy_vol = sum(t["buy_delta"] for t in ticks)
        total_sell_vol = sum(t["sell_delta"] for t in ticks)

        # ========== DYNAMIC MIN PRICE MOVE ==========
        # Scale based on current price to handle both low and high-priced stocks
        base_min = CONFIG["min_price_move"]
        ltp_based_min = (current_ltp or 100.0) * 0.0002  # 0.02% of price
        
        # For high-volume stocks, use tighter threshold
        if avg_vol and avg_vol > 50000:
            ltp_based_min = (current_ltp or 100.0) * 0.0001  # 0.01%
        
        min_move = max(base_min, ltp_based_min)

        # ========== ABSORPTION METRICS ==========
        # Accumulation: High sell volume absorbed without price drop
        # Distribution: High buy volume absorbed without price rise
        
        accu_absorption = 0
        dist_absorption = 0
        
        if total_sell_vol > 0:
            accu_absorption = total_sell_vol / max(price_down, min_move)
        
        if total_buy_vol > 0:
            dist_absorption = total_buy_vol / max(price_up, min_move)

        # ========== EFFICIENCY ==========
        efficiency = 0
        inverse_efficiency = 0
        
        if total_volume > 0:
            efficiency = price_range / total_volume
            if efficiency > 0:
                inverse_efficiency = 1 / efficiency

        volume_imbalance = total_buy_vol - total_sell_vol

        # ========== ADAPTIVE THRESHOLDS ==========
        # Use passed thresholds or compute from avg_vol
        if abs_threshold is None:
            if avg_vol and avg_vol > 0:
                abs_threshold = max(500, avg_vol * 0.02)  # 2% of avg volume
            else:
                abs_threshold = CONFIG["absorption_threshold"]
        
        if strong_abs is None:
            if avg_vol and avg_vol > 0:
                strong_abs = max(abs_threshold * 2.5, avg_vol * 0.05)
            else:
                strong_abs = CONFIG["strong_absorption"]

        # Adaptive efficiency threshold (scale with price)
        low_eff_threshold = CONFIG["low_efficiency_threshold"]
        if current_ltp and current_ltp > 0:
            # Normalize: higher priced stocks have smaller relative moves
            low_eff_threshold = CONFIG["low_efficiency_threshold"] * (100 / current_ltp)

        # Adaptive volume threshold for bonus
        vol_threshold = max(10000, avg_vol * 0.05) if avg_vol else 10000

        # ========== SCORING ==========
        accu_score = 0
        dist_score = 0

        # Accumulation scoring
        if accu_absorption > abs_threshold:
            # Base score from absorption ratio
            accu_score += min(40, (accu_absorption / strong_abs) * 40)
            
            # Bonus: Price went UP despite heavy selling (strong absorption)
            if price_change > 0 and total_sell_vol > total_buy_vol:
                accu_score += 30
            # Bonus: Price stable despite selling
            elif price_change >= -min_move:
                accu_score += 20
            
            # Bonus: Low efficiency (volume not moving price)
            if efficiency < low_eff_threshold:
                accu_score += 20
            
            # Bonus: Significant volume
            if total_volume > vol_threshold:
                accu_score += 10

        # Distribution scoring
        if dist_absorption > abs_threshold:
            # Base score from absorption ratio
            dist_score += min(40, (dist_absorption / strong_abs) * 40)
            
            # Bonus: Price went DOWN despite heavy buying (strong distribution)
            if price_change < 0 and total_buy_vol > total_sell_vol:
                dist_score += 30
            # Bonus: Price stable despite buying
            elif price_change <= min_move:
                dist_score += 20
            
            # Bonus: Low efficiency
            if efficiency < low_eff_threshold:
                dist_score += 20
            
            # Bonus: Significant volume
            if total_volume > vol_threshold:
                dist_score += 10

        # Cap scores at 100
        accu_score = min(100, round(accu_score, 1))
        dist_score = min(100, round(dist_score, 1))

        # ========== CLASSIFICATION ==========
        classification = "NEUTRAL"
        threshold = CONFIG["score_threshold"]
        
        if accu_score >= threshold and accu_score > dist_score:
            classification = "ACCUMULATION"
        elif dist_score >= threshold and dist_score > accu_score:
            classification = "DISTRIBUTION"
        elif accu_score >= threshold * 0.7 and accu_score > dist_score:
            classification = "POSSIBLE_ACCUM"
        elif dist_score >= threshold * 0.7 and dist_score > accu_score:
            classification = "POSSIBLE_DIST"

        return {
            "accu_absorption": round(accu_absorption, 2),
            "dist_absorption": round(dist_absorption, 2),
            "accu_score": accu_score,
            "dist_score": dist_score,
            "efficiency": efficiency,
            "inverse_efficiency": round(inverse_efficiency, 2),
            "price_change": round(price_change, 2),
            "price_range": round(price_range, 2),
            "total_volume": total_volume,
            "buy_volume": total_buy_vol,
            "sell_volume": total_sell_vol,
            "volume_imbalance": volume_imbalance,
            "classification": classification,
            "tick_count": len(ticks),
            # Include thresholds used for debugging
            "abs_threshold_used": round(abs_threshold, 0),
            "min_move_used": round(min_move, 4),
        }

    def _empty_metrics(self, tick_count=0):
        return {
            "accu_absorption": 0,
            "dist_absorption": 0,
            "accu_score": 0,
            "dist_score": 0,
            "efficiency": 0,
            "inverse_efficiency": 0,
            "price_change": 0,
            "price_range": 0,
            "total_volume": 0,
            "buy_volume": 0,
            "sell_volume": 0,
            "volume_imbalance": 0,
            "classification": "NEUTRAL",
            "tick_count": tick_count,
            "abs_threshold_used": 0,
            "min_move_used": 0,
        }


# ====== SIGNAL PERSISTENCE ====== #
def apply_absorption_persistence(stt, current_classification, accu_score, dist_score):
    """
    Apply persistence logic based on absorption scores.
    Requires consecutive confirmations for signal generation.
    """
    prev_class = stt.get("raw_classification", "NEUTRAL")
    
    # Track consecutive classifications
    if current_classification == prev_class:
        stt["class_persist"] = stt.get("class_persist", 0) + 1
    else:
        stt["class_persist"] = 1
    
    stt["raw_classification"] = current_classification
    
    confirmed = stt.get("confirmed_signal", "NEUTRAL")
    needed = CONFIG["absorption_bars_needed"]
    
    # Strong signals (score > 85) need less confirmation
    max_score = max(accu_score, dist_score)
    if max_score >= 85:
        needed = 1
    elif max_score >= 75:
        needed = max(1, needed - 1)
    
    if current_classification in ("ACCUMULATION", "DISTRIBUTION"):
        if stt["class_persist"] >= needed:
            confirmed = current_classification
            stt["neutral_persist"] = 0
    elif current_classification == "NEUTRAL":
        stt["neutral_persist"] = stt.get("neutral_persist", 0) + 1
        if stt["neutral_persist"] >= 3:
            confirmed = "NEUTRAL"
            stt["class_persist"] = 0
    else:
        stt["neutral_persist"] = 0
    
    stt["confirmed_signal"] = confirmed
    return confirmed


# ====== MANAGER ====== #
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
        self.avg_volumes = {}  # symbol -> {avg_vol, bar_count, abs_threshold, strong_abs, ...}
        self.absorption_engine = AbsorptionEngine()

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

    def set_avg_volume(self, symbol, avg_vol, bar_count):
        """Store pre-calculated average volume and derived thresholds for a symbol."""
        with self.lock:
            # Compute adaptive thresholds
            if avg_vol and avg_vol > 0:
                abs_threshold = max(500, avg_vol * 0.02)  # 2% of avg volume
                strong_abs = max(abs_threshold * 2.5, avg_vol * 0.05)
            else:
                abs_threshold = CONFIG["absorption_threshold"]
                strong_abs = CONFIG["strong_absorption"]
            
            self.avg_volumes[symbol] = {
                "avg_vol": avg_vol,
                "bar_count": bar_count,
                "abs_threshold": abs_threshold,
                "strong_abs": strong_abs,
                "calculated_at": now_ist().strftime("%H:%M:%S")
            }

    def get_avg_volume(self, symbol):
        with self.lock:
            return self.avg_volumes.get(symbol, {}).get("avg_vol", 0.0)

    def get_symbol_thresholds(self, symbol):
        """Get per-symbol adaptive thresholds."""
        with self.lock:
            info = self.avg_volumes.get(symbol, {})
            return {
                "avg_vol": info.get("avg_vol", 0),
                "abs_threshold": info.get("abs_threshold", CONFIG["absorption_threshold"]),
                "strong_abs": info.get("strong_abs", CONFIG["strong_absorption"]),
            }


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


def calc_avg_volume_750(kite, token):
    """Calculate average volume of last 750 completed 5-min bars (once at start)."""
    try:
        now = now_ist().replace(tzinfo=None)
        from_date = now - timedelta(days=15)
        
        candles = kite.historical_data(token, from_date, now, "5minute")
        df = pd.DataFrame(candles)
        
        if df.empty:
            return 0.0, 0

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        
        completed_bars = df.iloc[:-1] if len(df) > 1 else df
        
        if len(completed_bars) > CONFIG["avg_bars"]:
            completed_bars = completed_bars.tail(CONFIG["avg_bars"])
        
        bar_count = len(completed_bars)
        
        if bar_count == 0:
            return 0.0, 0
        
        avg_vol = float(completed_bars["volume"].mean())
        
        return round(avg_vol, 2), bar_count
        
    except Exception as e:
        manager.log(f"Avg volume calc error: {e}", "ERROR")
        return 0.0, 0


def calc_live_rvol(current_bar_volume, avg_750_volume):
    """Calculate live RVol for current bar."""
    if avg_750_volume <= 0:
        return 0.0
    return round(current_bar_volume / avg_750_volume, 2)


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

        # Process CE options in batches
        if ce_syms:
            for i in range(0, len(ce_syms), 50):
                batch = ce_syms[i:i+50]
                q_ce = safe_quote_batch(kite, batch, manager.log)
                for v in q_ce.values():
                    prem = safe_float(v.get("last_price", 0.0))
                    if prem >= 20:
                        ce_vol += safe_int(v.get("volume", 0))

        # Process PE options in batches
        if pe_syms:
            for i in range(0, len(pe_syms), 50):
                batch = pe_syms[i:i+50]
                q_pe = safe_quote_batch(kite, batch, manager.log)
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


# ====== BUY/SELL VOLUME ESTIMATION ====== #
def estimate_buy_sell_volume(vol_delta, ltp, prev_ltp, buy_q, sell_q, prev_buy_q, prev_sell_q):
    """
    Improved buy/sell volume estimation combining:
    1. Tick rule (price direction)
    2. Orderbook delta (bid/ask quantity changes)
    
    Returns: (buy_vol_delta, sell_vol_delta)
    """
    if vol_delta <= 0:
        return 0, 0
    
    # ========== TICK RULE ==========
    if ltp > prev_ltp:
        # Price up = aggressive buying
        tick_buy = vol_delta
        tick_sell = 0
    elif ltp < prev_ltp:
        # Price down = aggressive selling
        tick_buy = 0
        tick_sell = vol_delta
    else:
        # Price unchanged - split based on orderbook imbalance
        total_q = buy_q + sell_q
        if total_q > 0:
            tick_buy = int(vol_delta * (buy_q / total_q))
            tick_sell = vol_delta - tick_buy
        else:
            tick_buy = vol_delta // 2
            tick_sell = vol_delta - tick_buy
    
    # ========== ORDERBOOK DELTA ==========
    # Positive delta = quantity increased (possible spoofing/real orders)
    # Negative delta = quantity consumed (real trades)
    book_buy_delta = max(0, prev_buy_q - buy_q)   # Buy qty decreased = buys executed
    book_sell_delta = max(0, prev_sell_q - sell_q)  # Sell qty decreased = sells executed
    
    # Normalize book deltas to vol_delta scale
    book_total = book_buy_delta + book_sell_delta
    if book_total > 0:
        book_buy_normalized = int(vol_delta * (book_buy_delta / book_total))
        book_sell_normalized = vol_delta - book_buy_normalized
    else:
        book_buy_normalized = 0
        book_sell_normalized = 0
    
    # ========== COMBINE WITH WEIGHTS ==========
    tick_weight = CONFIG["tick_rule_weight"]
    book_weight = CONFIG["book_delta_weight"]
    
    buy_vol_delta = int(tick_buy * tick_weight + book_buy_normalized * book_weight)
    sell_vol_delta = int(tick_sell * tick_weight + book_sell_normalized * book_weight)
    
    # Ensure total matches vol_delta
    total_estimated = buy_vol_delta + sell_vol_delta
    if total_estimated > 0 and total_estimated != vol_delta:
        # Scale to match
        scale = vol_delta / total_estimated
        buy_vol_delta = int(buy_vol_delta * scale)
        sell_vol_delta = vol_delta - buy_vol_delta
    
    return buy_vol_delta, sell_vol_delta


# ====== WORKER THREAD ====== #
def sniper_worker(kite):
    manager.is_running = True
    manager.log("Sniper worker started (Absorption Model v2)", "INFO")

    # Track cumulative volume from exchange
    last_cum_vol = {}
    bar_start_cum_vol = {}
    
    # Track for buy/sell estimation
    last_ltp = {}
    last_buy_q = {}
    last_sell_q = {}
    
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

            # Fetch quotes for all symbols
            keys = [f"NSE:{row['symbol']}" for row in active_list]
            quotes = safe_quote_batch(kite, keys, manager.log)
            
            if not quotes:
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
                thresholds = manager.get_symbol_thresholds(sym)

                ltp = safe_float(q.get("last_price"))
                depth = q.get("depth", {})

                buy_q = sum(safe_int(x.get("quantity")) for x in depth.get("buy", [])[:5])
                sell_q = sum(safe_int(x.get("quantity")) for x in depth.get("sell", [])[:5])

                cum_vol = safe_int(q.get("volume"))

                # ========== VOLUME DELTA ==========
                prev_cum = last_cum_vol.get(sym, cum_vol)
                vol_delta = max(0, cum_vol - prev_cum)
                last_cum_vol[sym] = cum_vol

                # ========== BUY/SELL ESTIMATION ==========
                prev_ltp = last_ltp.get(sym, ltp)
                prev_buy_q = last_buy_q.get(sym, buy_q)
                prev_sell_q = last_sell_q.get(sym, sell_q)
                
                buy_vol_delta, sell_vol_delta = estimate_buy_sell_volume(
                    vol_delta, ltp, prev_ltp, 
                    buy_q, sell_q, prev_buy_q, prev_sell_q
                )
                
                # Update tracking
                last_ltp[sym] = ltp
                last_buy_q[sym] = buy_q
                last_sell_q[sym] = sell_q

                # ========== RECORD TICK ==========
                avg_750 = thresholds["avg_vol"]
                if vol_delta > 0:
                    manager.absorption_engine.record_tick(
                        sym, ltp, vol_delta, buy_vol_delta, sell_vol_delta, avg_vol=avg_750
                    )

                # ========== BAR VOLUME TRACKING ==========
                if sym not in bar_start_cum_vol:
                    bar_start_cum_vol[sym] = cum_vol

                if bar_changed:
                    bar_start_cum_vol[sym] = last_cum_vol.get(sym, cum_vol)

                current_bar_vol = max(0, cum_vol - bar_start_cum_vol.get(sym, cum_vol))

                # ========== LIVE RVOL ==========
                live_rvol = calc_live_rvol(current_bar_vol, avg_750)

                # ========== ABSORPTION METRICS ==========
                absorption_metrics = manager.absorption_engine.calculate_absorption_metrics(
                    sym, ltp, 
                    avg_vol=avg_750,
                    abs_threshold=thresholds["abs_threshold"],
                    strong_abs=thresholds["strong_abs"]
                )

                # ========== INITIALIZE STATE ==========
                if not stt:
                    stt = {
                        "ltp": ltp,
                        "buy_q": buy_q,
                        "sell_q": sell_q,
                        "bar_id": curr_bar_id,
                        "bar_vol": current_bar_vol,
                        "raw_classification": "NEUTRAL",
                        "confirmed_signal": "NEUTRAL",
                        "class_persist": 0,
                        "neutral_persist": 0,
                        "opt_power": 0.0,
                        "opt_ce": 0.0,
                        "opt_pe": 0.0,
                        "rvol": live_rvol,
                        "avg_750": avg_750,
                        "trade_signal": "IDLE",
                        "last_bar_vol": 0,
                        "last_bar_rvol": 0.0,
                        "absorption": absorption_metrics,
                        "last_buy_q": buy_q,
                        "last_sell_q": sell_q,
                    }

                # ========== BAR CLOSE LOGIC ==========
                if bar_changed and stt.get("bar_id", curr_bar_id) == curr_bar_id:
                    # Get final absorption metrics for the bar
                    final_metrics = manager.absorption_engine.calculate_absorption_metrics(
                        sym, ltp,
                        avg_vol=avg_750,
                        abs_threshold=thresholds["abs_threshold"],
                        strong_abs=thresholds["strong_abs"]
                    )
                    
                    # Check if we have enough ticks for valid signal
                    tick_count = final_metrics["tick_count"]
                    min_ticks = CONFIG["min_ticks_for_signal"]
                    
                    if tick_count >= min_ticks:
                        # Apply persistence
                        confirmed = apply_absorption_persistence(
                            stt, 
                            final_metrics["classification"],
                            final_metrics["accu_score"],
                            final_metrics["dist_score"]
                        )
                    else:
                        confirmed = stt.get("confirmed_signal", "NEUTRAL")
                        manager.log(
                            f"{sym}: Low tick count ({tick_count}) at bar close, skipping classification",
                            "WARNING"
                        )

                    final_bar_vol = stt.get("bar_vol", 0)
                    final_rvol = calc_live_rvol(final_bar_vol, avg_750)

                    stt["last_bar_vol"] = final_bar_vol
                    stt["last_bar_rvol"] = final_rvol
                    stt["absorption"] = final_metrics

                    # Log bar close
                    accu_s = final_metrics["accu_score"]
                    dist_s = final_metrics["dist_score"]
                    cls = final_metrics["classification"]
                    
                    emoji = ""
                    if cls == "ACCUMULATION":
                        emoji = "🟢"
                    elif cls == "DISTRIBUTION":
                        emoji = "🔴"
                    elif "POSSIBLE" in cls:
                        emoji = "🟡"
                    
                    rvol_flag = "🔥" if final_rvol >= 2.0 else ""
                    
                    manager.log(
                        f"{emoji} {sym} BAR @ {curr_bar_start.strftime('%H:%M')} | "
                        f"Vol={final_bar_vol:,} RVol={final_rvol}x{rvol_flag} | "
                        f"Ticks={tick_count} | "
                        f"Accu={accu_s:.0f} Dist={dist_s:.0f} → {cls} "
                        f"[Confirmed: {confirmed}]",
                        "INFO",
                    )

                    # Reset absorption engine for new bar
                    manager.absorption_engine.reset_bar(sym)

                    # Reset for new bar
                    stt["bar_id"] = new_bar_id
                    stt["bar_vol"] = current_bar_vol

                else:
                    stt["bar_id"] = curr_bar_id
                    stt["bar_vol"] = current_bar_vol
                    stt["absorption"] = absorption_metrics

                # Update live values
                stt["rvol"] = live_rvol
                stt["avg_750"] = avg_750
                stt["last_buy_q"] = buy_q
                stt["last_sell_q"] = sell_q

                # ========== OPTIONS SCAN ==========
                if do_opt_scan:
                    net, ce_m, pe_m = scan_opt_power(kite, sym)
                    stt["opt_power"] = net
                    stt["opt_ce"] = ce_m
                    stt["opt_pe"] = pe_m

                # ========== TRADE SIGNAL LOGIC ==========
                ce_m = stt.get("opt_ce", 0.0)
                pe_m = stt.get("opt_pe", 0.0)
                call_bias = ce_m > pe_m * 1.2
                put_bias = pe_m > ce_m * 1.2

                confirmed_signal = stt.get("confirmed_signal", "NEUTRAL")
                trade_signal = "IDLE"
                
                if confirmed_signal == "ACCUMULATION" and call_bias:
                    trade_signal = "🟢 BUY CALL"
                elif confirmed_signal == "DISTRIBUTION" and put_bias:
                    trade_signal = "🔴 BUY PUT"
                elif confirmed_signal == "ACCUMULATION":
                    trade_signal = "ACCUM ⏳"
                elif confirmed_signal == "DISTRIBUTION":
                    trade_signal = "DIST ⏳"

                stt["trade_signal"] = trade_signal
                stt["ltp"] = ltp
                stt["buy_q"] = buy_q
                stt["sell_q"] = sell_q

                manager.set_symbol_data(sym, stt)

            if bar_changed:
                curr_bar_id = new_bar_id
                curr_bar_start = new_bar_start

            time.sleep(POLL_INTERVAL_SEC)

        except Exception as e:
            manager.log(f"Worker error: {e}", "ERROR")
            import traceback
            manager.log(traceback.format_exc(), "DEBUG")
            time.sleep(2)

    manager.is_running = False
    manager.log("Sniper worker stopped", "INFO")


# ====== SIDEBAR AUTH ====== #
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
            manager.avg_volumes = {}
            manager.absorption_engine = AbsorptionEngine()

        with st.spinner("Initializing symbols and calculating baselines..."):
            progress_bar = st.progress(0)
            
            for idx, s in enumerate(syms):
                tkn = get_instrument_token(manager.kite, s)
                if not tkn:
                    manager.log(f"No NSE instrument for {s}", "WARNING")
                    continue

                # Calculate 750-bar average and set adaptive thresholds
                avg_vol, bar_count = calc_avg_volume_750(manager.kite, tkn)
                manager.set_avg_volume(s, avg_vol, bar_count)
                
                thresholds = manager.get_symbol_thresholds(s)
                
                manager.log(
                    f"{s}: Avg Vol={avg_vol:,.0f} ({bar_count} bars) | "
                    f"AbsThresh={thresholds['abs_threshold']:,.0f} | "
                    f"StrongAbs={thresholds['strong_abs']:,.0f}",
                    "INFO"
                )

                valid.append({"symbol": s, "token": tkn})

                # Initialize symbol data
                try:
                    q = manager.kite.quote(f"NSE:{s}")[f"NSE:{s}"]
                    ltp = safe_float(q.get("last_price"))
                    depth = q.get("depth", {})
                    buy_q = sum(safe_int(x.get("quantity")) for x in depth.get("buy", [])[:5])
                    sell_q = sum(safe_int(x.get("quantity")) for x in depth.get("sell", [])[:5])

                    stt = {
                        "ltp": ltp,
                        "buy_q": buy_q,
                        "sell_q": sell_q,
                        "bar_id": current_bar_id_5m()[0],
                        "bar_vol": 0,
                        "raw_classification": "NEUTRAL",
                        "confirmed_signal": "NEUTRAL",
                        "class_persist": 0,
                        "neutral_persist": 0,
                        "opt_power": 0.0,
                        "opt_ce": 0.0,
                        "opt_pe": 0.0,
                        "rvol": 0.0,
                        "avg_750": avg_vol,
                        "trade_signal": "IDLE",
                        "last_bar_vol": 0,
                        "last_bar_rvol": 0.0,
                        "absorption": manager.absorption_engine._empty_metrics(),
                        "last_buy_q": buy_q,
                        "last_sell_q": sell_q,
                    }
                    manager.set_symbol_data(s, stt)
                except Exception as e:
                    manager.log(f"Init quote failed for {s}: {e}", "WARNING")
                
                progress_bar.progress((idx + 1) / len(syms))

            progress_bar.empty()

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
            st.success(f"✅ Monitoring {len(valid)} stocks (Absorption Model v2)")

if btn2.button("⏹ STOP"):
    manager.stop_event.set()
    manager.is_running = False
    manager.initialized = False
    st.rerun()

# ====== DASHBOARD ====== #
st.title("🎯 Institutional Sniper – Absorption Model v2")
st.caption("Detects hidden accumulation/distribution via absorption + efficiency + imbalance (adaptive thresholds)")

# Status indicator
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
        <div style="color:#9ca3af;font-size:0.8em;">Engine Status</div>
        <div style="font-size:1.2em;color:white;font-weight:bold;">{hb_text}</div>
      </div>
    </div>
    <div style="color:#9ca3af;font-size:0.8em;">Last beat: {beat_time}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Get snapshot
snapshot = manager.get_data_snapshot()

# Sidebar debug info
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Symbols:** {len(snapshot)}")
st.sidebar.markdown(f"**Running:** {manager.is_running}")

# Show thresholds
if manager.avg_volumes:
    with st.sidebar.expander("📊 Adaptive Thresholds"):
        for sym, info in manager.avg_volumes.items():
            st.markdown(
                f"**{sym}**  \n"
                f"Avg: {info['avg_vol']:,.0f}  \n"
                f"AbsThresh: {info['abs_threshold']:,.0f}  \n"
                f"StrongAbs: {info['strong_abs']:,.0f}"
            )
            st.markdown("---")

if snapshot:
    st.subheader(f"📊 Monitoring {len(snapshot)} Symbols")
    
    cols = st.columns(3)
    
    # Sort by max absorption score
    items = sorted(
        snapshot.items(),
        key=lambda x: max(
            x[1].get("absorption", {}).get("accu_score", 0),
            x[1].get("absorption", {}).get("dist_score", 0)
        ),
        reverse=True
    )

    for i, (sym, d) in enumerate(items):
        with cols[i % 3]:
            trade_sig = d.get("trade_signal", "IDLE")
            confirmed = d.get("confirmed_signal", "NEUTRAL")
            
            absorp = d.get("absorption", {})
            accu_score = absorp.get("accu_score", 0)
            dist_score = absorp.get("dist_score", 0)
            classification = absorp.get("classification", "NEUTRAL")
            tick_count = absorp.get("tick_count", 0)
            
            # Card styling
            card_class = ""
            if confirmed == "ACCUMULATION":
                badge_style = "background:#064e3b;color:#6ee7b7;border:1px solid #059669;"
                card_class = "accum-glow"
            elif confirmed == "DISTRIBUTION":
                badge_style = "background:#7f1d1d;color:#fca5a5;border:1px solid #dc2626;"
                card_class = "dist-glow"
            else:
                badge_style = "background:#374151;color:#d1d5db;border:1px solid #4b5563;"

            # Score bar colors
            accu_bar_color = "#22c55e" if accu_score >= 70 else "#4ade80" if accu_score >= 50 else "#6b7280"
            dist_bar_color = "#ef4444" if dist_score >= 70 else "#f87171" if dist_score >= 50 else "#6b7280"

            # RVol styling
            rvol = d.get('rvol', 0.0)
            rvol_emoji = "🔥" if rvol >= 2.0 else "⚡" if rvol >= 1.5 else ""

            # Tick count indicator
            tick_status = "🟢" if tick_count >= 10 else "🟡" if tick_count >= 5 else "🔴"

            card_html = f"""
<div class="card-container {card_class}">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span style="font-size:1.3em;font-weight:bold;color:white;">{sym}</span>
    <span style="padding:4px 8px;border-radius:4px;font-weight:bold;font-size:0.8rem;{badge_style}">{trade_sig}</span>
  </div>
  
  <div style="margin-top:10px;color:#9ca3af;font-size:0.85em;">
    <b>LTP</b> ₹{d.get('ltp',0):.2f} | <b>Ticks</b> {tick_status} {tick_count}
  </div>
  
  <!-- Absorption Scores -->
  <div class="metric-box">
    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
      <span style="color:#22c55e;">🟢 Accumulation</span>
      <span style="color:#22c55e;font-weight:bold;">{accu_score:.0f}/100</span>
    </div>
    <div style="background:#374151;border-radius:4px;height:8px;overflow:hidden;">
      <div style="background:{accu_bar_color};height:100%;width:{min(accu_score, 100)}%;transition:width 0.3s;"></div>
    </div>
  </div>
  
  <div class="metric-box">
    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
      <span style="color:#ef4444;">🔴 Distribution</span>
      <span style="color:#ef4444;font-weight:bold;">{dist_score:.0f}/100</span>
    </div>
    <div style="background:#374151;border-radius:4px;height:8px;overflow:hidden;">
      <div style="background:{dist_bar_color};height:100%;width:{min(dist_score, 100)}%;transition:width 0.3s;"></div>
    </div>
  </div>
  
  <div style="margin-top:8px;color:#9ca3af;font-size:0.8em;">
    <div style="display:flex;justify-content:space-between;">
      <span>Classification:</span>
      <span style="font-weight:bold;">{classification}</span>
    </div>
    <div style="display:flex;justify-content:space-between;">
      <span>Confirmed:</span>
      <span style="font-weight:bold;color:{'#22c55e' if confirmed=='ACCUMULATION' else '#ef4444' if confirmed=='DISTRIBUTION' else '#9ca3af'};">{confirmed}</span>
    </div>
    <div style="display:flex;justify-content:space-between;">
      <span>RVol:</span>
      <span>{rvol:.2f}x {rvol_emoji}</span>
    </div>
    <div style="display:flex;justify-content:space-between;">
      <span>Bar Vol:</span>
      <span>{d.get('bar_vol',0):,}</span>
    </div>
    <div style="display:flex;justify-content:space-between;">
      <span>Options:</span>
      <span>CE {d.get('opt_ce',0.0):.2f}M / PE {d.get('opt_pe',0.0):.2f}M</span>
    </div>
  </div>
  
  <!-- Absorption Details -->
  <details style="margin-top:8px;">
    <summary style="color:#6b7280;cursor:pointer;font-size:0.75em;">📊 Absorption Details</summary>
    <div style="color:#6b7280;font-size:0.7em;margin-top:4px;background:#0f172a;padding:6px;border-radius:4px;">
      <b>Metrics:</b><br>
      Accu Absorp: {absorp.get('accu_absorption', 0):,.0f}<br>
      Dist Absorp: {absorp.get('dist_absorption', 0):,.0f}<br>
      Efficiency: {absorp.get('efficiency', 0):.8f}<br>
      Inv Efficiency: {absorp.get('inverse_efficiency', 0):,.0f}<br>
      <br>
      <b>Volume:</b><br>
      Buy Vol: {absorp.get('buy_volume', 0):,}<br>
      Sell Vol: {absorp.get('sell_volume', 0):,}<br>
      Imbalance: {absorp.get('volume_imbalance', 0):,}<br>
      <br>
      <b>Price:</b><br>
      Price Δ: ₹{absorp.get('price_change', 0):.2f}<br>
      Range: ₹{absorp.get('price_range', 0):.2f}<br>
      <br>
      <b>Thresholds Used:</b><br>
      Abs Thresh: {absorp.get('abs_threshold_used', 0):,.0f}<br>
      Min Move: ₹{absorp.get('min_move_used', 0):.4f}
    </div>
  </details>
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

# Auto-refresh when running
if manager.is_running:
    time.sleep(3)
    st.rerun()