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
    page_title="Institutional Sniper (Net Absorption)",
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
.bias-bar {
    height: 12px;
    border-radius: 6px;
    background: linear-gradient(to right, #22c55e 0%, #22c55e 50%, #ef4444 50%, #ef4444 100%);
    position: relative;
    overflow: hidden;
}
.bias-indicator {
    position: absolute;
    width: 4px;
    height: 100%;
    background: white;
    border-radius: 2px;
    transition: left 0.3s;
}
</style>
""",
    unsafe_allow_html=True,
)

# ====== CONFIG ====== #
CONFIG = {
    # Net absorption thresholds
    "neutral_zone_pct": 0.3,          # If |net_bias| < 30% of max absorption, it's neutral
    "strong_signal_pct": 0.6,         # If |net_bias| > 60% of max absorption, it's strong
    
    # Minimum volume for valid signal
    "min_volume_for_signal": 5000,
    
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
    
    # Confirmation settings
    "bars_to_confirm": 2,             # Consecutive bars needed
    "bars_to_reset": 2,               # Neutral bars to reset signal
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
    """Safely fetch quotes with retry logic."""
    for attempt in range(attempts):
        try:
            q = kite.quote(batch)
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
                manager_log_func(f"Quote batch failed: {e}", "WARNING")
    return {}


# ====== ABSORPTION ENGINE (Net Bias Model) ====== #
class AbsorptionEngine:
    """
    Simplified institutional detection using NET ABSORPTION BIAS.
    
    Core Formula:
        net_bias = dist_absorption - accu_absorption
        
        net_bias > 0 → Distribution (hidden sellers absorbing buys)
        net_bias < 0 → Accumulation (hidden buyers absorbing sells)
        net_bias ≈ 0 → Neutral (no clear institutional activity)
    
    This eliminates confusion from dual-scoring systems.
    """
    
    def __init__(self):
        self.tick_history = {}
        self.lock = threading.RLock()
        self.default_max_ticks = CONFIG["default_max_ticks"]

    def _get_max_ticks_for_symbol(self, avg_vol):
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
        with self.lock:
            return list(self.tick_history.get(symbol, []))

    def get_tick_count(self, symbol):
        with self.lock:
            return len(self.tick_history.get(symbol, []))

    def calculate_net_absorption(self, symbol, current_ltp, avg_vol=None):
        """
        Calculate NET ABSORPTION BIAS - the ONLY metric that matters.
        
        Returns:
            dict with net_bias, classification, and supporting metrics
        """
        ticks = self.get_ticks_snapshot(symbol)
        
        min_ticks = CONFIG["min_ticks_for_signal"]
        if len(ticks) < min_ticks:
            return self._empty_metrics(tick_count=len(ticks), reason="Insufficient ticks")

        # ========== PRICE ANALYSIS ==========
        first_price = ticks[0]["ltp"]
        last_price = ticks[-1]["ltp"]
        price_change = last_price - first_price
        
        # Separate up and down movements
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

        # Check minimum volume
        if total_volume < CONFIG["min_volume_for_signal"]:
            return self._empty_metrics(
                tick_count=len(ticks), 
                reason=f"Low volume ({total_volume:,})",
                total_volume=total_volume,
                buy_volume=total_buy_vol,
                sell_volume=total_sell_vol
            )

        # ========== DYNAMIC MIN PRICE MOVE ==========
        base_min = CONFIG["min_price_move"]
        ltp_based_min = (current_ltp or 100.0) * 0.0002
        
        if avg_vol and avg_vol > 50000:
            ltp_based_min = (current_ltp or 100.0) * 0.0001
        
        min_move = max(base_min, ltp_based_min)

        # ========== ABSORPTION CALCULATION ==========
        # Accumulation Absorption: Sell volume absorbed per unit price drop
        # High value = hidden buyer absorbing sells
        accu_absorption = 0
        if total_sell_vol > 0:
            accu_absorption = total_sell_vol / max(price_down, min_move)
        
        # Distribution Absorption: Buy volume absorbed per unit price rise
        # High value = hidden seller absorbing buys
        dist_absorption = 0
        if total_buy_vol > 0:
            dist_absorption = total_buy_vol / max(price_up, min_move)

        # ========== NET ABSORPTION BIAS ==========
        # THE KEY METRIC: Which side is absorbing more?
        # Positive = Distribution dominates (sellers absorbing)
        # Negative = Accumulation dominates (buyers absorbing)
        net_bias = dist_absorption - accu_absorption
        
        # Max absorption for normalization
        max_absorption = max(accu_absorption, dist_absorption, 1)
        
        # Normalized bias (-100 to +100 scale)
        # -100 = Pure Accumulation
        # +100 = Pure Distribution
        # 0 = Neutral
        if max_absorption > 0:
            normalized_bias = (net_bias / max_absorption) * 100
        else:
            normalized_bias = 0
        
        normalized_bias = max(-100, min(100, normalized_bias))

        # ========== CLASSIFICATION (Simple and Clear) ==========
        neutral_threshold = CONFIG["neutral_zone_pct"] * 100  # 30
        strong_threshold = CONFIG["strong_signal_pct"] * 100  # 60
        
        if abs(normalized_bias) < neutral_threshold:
            classification = "NEUTRAL"
            strength = "none"
        elif normalized_bias > 0:
            # Distribution
            if normalized_bias >= strong_threshold:
                classification = "DISTRIBUTION"
                strength = "strong"
            else:
                classification = "DISTRIBUTION"
                strength = "moderate"
        else:
            # Accumulation
            if normalized_bias <= -strong_threshold:
                classification = "ACCUMULATION"
                strength = "strong"
            else:
                classification = "ACCUMULATION"
                strength = "moderate"

        # ========== EFFICIENCY (For reference only) ==========
        efficiency = price_range / total_volume if total_volume > 0 else 0

        return {
            # Primary metrics
            "net_bias": round(net_bias, 2),
            "normalized_bias": round(normalized_bias, 1),
            "classification": classification,
            "strength": strength,
            
            # Supporting metrics
            "accu_absorption": round(accu_absorption, 2),
            "dist_absorption": round(dist_absorption, 2),
            
            # Volume breakdown
            "total_volume": total_volume,
            "buy_volume": total_buy_vol,
            "sell_volume": total_sell_vol,
            "volume_delta": total_buy_vol - total_sell_vol,
            
            # Price metrics
            "price_change": round(price_change, 2),
            "price_range": round(price_range, 2),
            "price_up": round(price_up, 2),
            "price_down": round(price_down, 2),
            
            # Meta
            "efficiency": efficiency,
            "tick_count": len(ticks),
            "min_move_used": round(min_move, 4),
            "reason": None,
        }

    def _empty_metrics(self, tick_count=0, reason=None, **kwargs):
        base = {
            "net_bias": 0,
            "normalized_bias": 0,
            "classification": "NEUTRAL",
            "strength": "none",
            "accu_absorption": 0,
            "dist_absorption": 0,
            "total_volume": 0,
            "buy_volume": 0,
            "sell_volume": 0,
            "volume_delta": 0,
            "price_change": 0,
            "price_range": 0,
            "price_up": 0,
            "price_down": 0,
            "efficiency": 0,
            "tick_count": tick_count,
            "min_move_used": 0,
            "reason": reason,
        }
        base.update(kwargs)
        return base


# ====== SIGNAL CONFIRMATION (Simplified) ====== #
def apply_signal_confirmation(stt, current_classification, strength):
    """
    Simple confirmation logic that NEVER contradicts itself.
    
    Rules:
    1. Same classification for N bars → Confirmed
    2. Neutral for M bars → Reset to Neutral
    3. Classification change → Reset counter, start fresh
    
    Returns: confirmed_signal (same as classification once confirmed)
    """
    prev_class = stt.get("last_classification", "NEUTRAL")
    confirmed = stt.get("confirmed_signal", "NEUTRAL")
    
    # Track consecutive same classifications
    if current_classification == prev_class:
        stt["same_class_count"] = stt.get("same_class_count", 0) + 1
    else:
        # Classification changed - reset
        stt["same_class_count"] = 1
    
    stt["last_classification"] = current_classification
    
    # Confirmation logic
    bars_needed = CONFIG["bars_to_confirm"]
    
    # Strong signals need fewer bars
    if strength == "strong":
        bars_needed = 1
    
    if current_classification in ("ACCUMULATION", "DISTRIBUTION"):
        if stt["same_class_count"] >= bars_needed:
            confirmed = current_classification
            stt["neutral_count"] = 0
    elif current_classification == "NEUTRAL":
        stt["neutral_count"] = stt.get("neutral_count", 0) + 1
        if stt["neutral_count"] >= CONFIG["bars_to_reset"]:
            confirmed = "NEUTRAL"
            stt["same_class_count"] = 0
    
    stt["confirmed_signal"] = confirmed
    
    # IMPORTANT: Ensure classification and confirmed are NEVER contradictory
    # If current is strong and different from confirmed, update confirmed immediately
    if strength == "strong" and current_classification != "NEUTRAL":
        confirmed = current_classification
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
        self.avg_volumes = {}
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
        with self.lock:
            self.avg_volumes[symbol] = {
                "avg_vol": avg_vol,
                "bar_count": bar_count,
                "calculated_at": now_ist().strftime("%H:%M:%S")
            }

    def get_avg_volume(self, symbol):
        with self.lock:
            return self.avg_volumes.get(symbol, {}).get("avg_vol", 0.0)


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
    """Calculate average volume of last 750 completed 5-min bars."""
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

        if ce_syms:
            for i in range(0, len(ce_syms), 50):
                batch = ce_syms[i:i+50]
                q_ce = safe_quote_batch(kite, batch, manager.log)
                for v in q_ce.values():
                    prem = safe_float(v.get("last_price", 0.0))
                    if prem >= 20:
                        ce_vol += safe_int(v.get("volume", 0))

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
    """
    if vol_delta <= 0:
        return 0, 0
    
    # Tick rule
    if ltp > prev_ltp:
        tick_buy = vol_delta
        tick_sell = 0
    elif ltp < prev_ltp:
        tick_buy = 0
        tick_sell = vol_delta
    else:
        total_q = buy_q + sell_q
        if total_q > 0:
            tick_buy = int(vol_delta * (buy_q / total_q))
            tick_sell = vol_delta - tick_buy
        else:
            tick_buy = vol_delta // 2
            tick_sell = vol_delta - tick_buy
    
    # Orderbook delta
    book_buy_delta = max(0, prev_buy_q - buy_q)
    book_sell_delta = max(0, prev_sell_q - sell_q)
    
    book_total = book_buy_delta + book_sell_delta
    if book_total > 0:
        book_buy_normalized = int(vol_delta * (book_buy_delta / book_total))
        book_sell_normalized = vol_delta - book_buy_normalized
    else:
        book_buy_normalized = 0
        book_sell_normalized = 0
    
    # Combine with weights
    tick_weight = CONFIG["tick_rule_weight"]
    book_weight = CONFIG["book_delta_weight"]
    
    buy_vol_delta = int(tick_buy * tick_weight + book_buy_normalized * book_weight)
    sell_vol_delta = int(tick_sell * tick_weight + book_sell_normalized * book_weight)
    
    # Normalize to match vol_delta
    total_estimated = buy_vol_delta + sell_vol_delta
    if total_estimated > 0 and total_estimated != vol_delta:
        scale = vol_delta / total_estimated
        buy_vol_delta = int(buy_vol_delta * scale)
        sell_vol_delta = vol_delta - buy_vol_delta
    
    return buy_vol_delta, sell_vol_delta


# ====== WORKER THREAD ====== #
def sniper_worker(kite):
    manager.is_running = True
    manager.log("Sniper worker started (Net Absorption Model)", "INFO")

    last_cum_vol = {}
    bar_start_cum_vol = {}
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
                avg_750 = manager.get_avg_volume(sym)

                ltp = safe_float(q.get("last_price"))
                depth = q.get("depth", {})

                buy_q = sum(safe_int(x.get("quantity")) for x in depth.get("buy", [])[:5])
                sell_q = sum(safe_int(x.get("quantity")) for x in depth.get("sell", [])[:5])

                cum_vol = safe_int(q.get("volume"))

                # Volume delta
                prev_cum = last_cum_vol.get(sym, cum_vol)
                vol_delta = max(0, cum_vol - prev_cum)
                last_cum_vol[sym] = cum_vol

                # Buy/sell estimation
                prev_ltp = last_ltp.get(sym, ltp)
                prev_buy_q = last_buy_q.get(sym, buy_q)
                prev_sell_q = last_sell_q.get(sym, sell_q)
                
                buy_vol_delta, sell_vol_delta = estimate_buy_sell_volume(
                    vol_delta, ltp, prev_ltp, 
                    buy_q, sell_q, prev_buy_q, prev_sell_q
                )
                
                last_ltp[sym] = ltp
                last_buy_q[sym] = buy_q
                last_sell_q[sym] = sell_q

                # Record tick
                if vol_delta > 0:
                    manager.absorption_engine.record_tick(
                        sym, ltp, vol_delta, buy_vol_delta, sell_vol_delta, avg_vol=avg_750
                    )

                # Bar volume tracking
                if sym not in bar_start_cum_vol:
                    bar_start_cum_vol[sym] = cum_vol

                if bar_changed:
                    bar_start_cum_vol[sym] = last_cum_vol.get(sym, cum_vol)

                current_bar_vol = max(0, cum_vol - bar_start_cum_vol.get(sym, cum_vol))

                # Live RVol
                live_rvol = calc_live_rvol(current_bar_vol, avg_750)

                # Net Absorption calculation
                absorption_metrics = manager.absorption_engine.calculate_net_absorption(
                    sym, ltp, avg_vol=avg_750
                )

                # Initialize state
                if not stt:
                    stt = {
                        "ltp": ltp,
                        "buy_q": buy_q,
                        "sell_q": sell_q,
                        "bar_id": curr_bar_id,
                        "bar_vol": current_bar_vol,
                        "last_classification": "NEUTRAL",
                        "confirmed_signal": "NEUTRAL",
                        "same_class_count": 0,
                        "neutral_count": 0,
                        "opt_power": 0.0,
                        "opt_ce": 0.0,
                        "opt_pe": 0.0,
                        "rvol": live_rvol,
                        "avg_750": avg_750,
                        "trade_signal": "IDLE",
                        "last_bar_vol": 0,
                        "last_bar_rvol": 0.0,
                        "absorption": absorption_metrics,
                    }

                # Bar close logic
                if bar_changed and stt.get("bar_id", curr_bar_id) == curr_bar_id:
                    final_metrics = manager.absorption_engine.calculate_net_absorption(
                        sym, ltp, avg_vol=avg_750
                    )
                    
                    tick_count = final_metrics["tick_count"]
                    min_ticks = CONFIG["min_ticks_for_signal"]
                    
                    if tick_count >= min_ticks:
                        confirmed = apply_signal_confirmation(
                            stt, 
                            final_metrics["classification"],
                            final_metrics["strength"]
                        )
                    else:
                        confirmed = stt.get("confirmed_signal", "NEUTRAL")
                        manager.log(
                            f"{sym}: Low ticks ({tick_count}), keeping {confirmed}",
                            "WARNING"
                        )

                    final_bar_vol = stt.get("bar_vol", 0)
                    final_rvol = calc_live_rvol(final_bar_vol, avg_750)

                    stt["last_bar_vol"] = final_bar_vol
                    stt["last_bar_rvol"] = final_rvol
                    stt["absorption"] = final_metrics

                    # Log bar close
                    net_bias = final_metrics["normalized_bias"]
                    cls = final_metrics["classification"]
                    strength = final_metrics["strength"]
                    
                    emoji = "🟢" if cls == "ACCUMULATION" else "🔴" if cls == "DISTRIBUTION" else "⚪"
                    strength_emoji = "💪" if strength == "strong" else ""
                    rvol_flag = "🔥" if final_rvol >= 2.0 else ""
                    
                    manager.log(
                        f"{emoji} {sym} BAR @ {curr_bar_start.strftime('%H:%M')} | "
                        f"Bias={net_bias:+.0f} → {cls} {strength_emoji} | "
                        f"Confirmed={confirmed} | "
                        f"Vol={final_bar_vol:,} RVol={final_rvol}x{rvol_flag}",
                        "INFO",
                    )

                    manager.absorption_engine.reset_bar(sym)
                    stt["bar_id"] = new_bar_id
                    stt["bar_vol"] = current_bar_vol

                else:
                    stt["bar_id"] = curr_bar_id
                    stt["bar_vol"] = current_bar_vol
                    stt["absorption"] = absorption_metrics

                stt["rvol"] = live_rvol
                stt["avg_750"] = avg_750

                # Options scan
                if do_opt_scan:
                    net, ce_m, pe_m = scan_opt_power(kite, sym)
                    stt["opt_power"] = net
                    stt["opt_ce"] = ce_m
                    stt["opt_pe"] = pe_m

                # Trade signal logic
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
                    trade_signal = "🟢 ACCUM ⏳"
                elif confirmed_signal == "DISTRIBUTION":
                    trade_signal = "🔴 DIST ⏳"

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

        with st.spinner("Initializing symbols..."):
            progress_bar = st.progress(0)
            
            for idx, s in enumerate(syms):
                tkn = get_instrument_token(manager.kite, s)
                if not tkn:
                    manager.log(f"No NSE instrument for {s}", "WARNING")
                    continue

                avg_vol, bar_count = calc_avg_volume_750(manager.kite, tkn)
                manager.set_avg_volume(s, avg_vol, bar_count)
                
                manager.log(f"{s}: Avg Vol={avg_vol:,.0f} ({bar_count} bars)", "INFO")

                valid.append({"symbol": s, "token": tkn})

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
                        "last_classification": "NEUTRAL",
                        "confirmed_signal": "NEUTRAL",
                        "same_class_count": 0,
                        "neutral_count": 0,
                        "opt_power": 0.0,
                        "opt_ce": 0.0,
                        "opt_pe": 0.0,
                        "rvol": 0.0,
                        "avg_750": avg_vol,
                        "trade_signal": "IDLE",
                        "last_bar_vol": 0,
                        "last_bar_rvol": 0.0,
                        "absorption": manager.absorption_engine._empty_metrics(),
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
            st.success(f"✅ Monitoring {len(valid)} stocks (Net Absorption Model)")

if btn2.button("⏹ STOP"):
    manager.stop_event.set()
    manager.is_running = False
    manager.initialized = False
    st.rerun()

# ====== DASHBOARD ====== #
st.title("🎯 Institutional Sniper – Net Absorption Model")
st.caption("Single directional bias: Negative = Accumulation | Positive = Distribution")

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

# Sidebar debug
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Symbols:** {len(snapshot)}")
st.sidebar.markdown(f"**Running:** {manager.is_running}")

if snapshot:
    st.subheader(f"📊 Monitoring {len(snapshot)} Symbols")
    
    cols = st.columns(3)
    
    # Sort by absolute bias (strongest signals first)
    items = sorted(
        snapshot.items(),
        key=lambda x: abs(x[1].get("absorption", {}).get("normalized_bias", 0)),
        reverse=True
    )

    for i, (sym, d) in enumerate(items):
        with cols[i % 3]:
            trade_sig = d.get("trade_signal", "IDLE")
            confirmed = d.get("confirmed_signal", "NEUTRAL")
            
            absorp = d.get("absorption", {})
            net_bias = absorp.get("normalized_bias", 0)
            classification = absorp.get("classification", "NEUTRAL")
            strength = absorp.get("strength", "none")
            tick_count = absorp.get("tick_count", 0)
            reason = absorp.get("reason")
            
            # Card styling based on confirmed signal (NOT classification)
            card_class = ""
            if confirmed == "ACCUMULATION":
                badge_style = "background:#064e3b;color:#6ee7b7;border:1px solid #059669;"
                card_class = "accum-glow"
            elif confirmed == "DISTRIBUTION":
                badge_style = "background:#7f1d1d;color:#fca5a5;border:1px solid #dc2626;"
                card_class = "dist-glow"
            else:
                badge_style = "background:#374151;color:#d1d5db;border:1px solid #4b5563;"

            # Bias bar position (50% = neutral, 0% = full accum, 100% = full dist)
            bias_position = 50 + (net_bias / 2)  # Convert -100..+100 to 0..100
            bias_position = max(0, min(100, bias_position))
            
            # Bias color
            if net_bias < -30:
                bias_color = "#22c55e"  # Green for accumulation
                bias_text_color = "#22c55e"
            elif net_bias > 30:
                bias_color = "#ef4444"  # Red for distribution
                bias_text_color = "#ef4444"
            else:
                bias_color = "#6b7280"  # Gray for neutral
                bias_text_color = "#9ca3af"

            # RVol styling
            rvol = d.get('rvol', 0.0)
            rvol_emoji = "🔥" if rvol >= 2.0 else "⚡" if rvol >= 1.5 else ""

            # Tick indicator
            tick_status = "🟢" if tick_count >= 10 else "🟡" if tick_count >= 5 else "🔴"

            # Strength indicator
            strength_badge = ""
            if strength == "strong":
                strength_badge = '<span style="background:#fbbf24;color:#000;padding:2px 6px;border-radius:3px;font-size:0.7em;margin-left:5px;">STRONG</span>'

            card_html = f"""
<div class="card-container {card_class}">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span style="font-size:1.3em;font-weight:bold;color:white;">{sym}</span>
    <span style="padding:4px 8px;border-radius:4px;font-weight:bold;font-size:0.8rem;{badge_style}">{trade_sig}</span>
  </div>
  
  <div style="margin-top:10px;color:#9ca3af;font-size:0.85em;">
    <b>LTP</b> ₹{d.get('ltp',0):.2f} | <b>Ticks</b> {tick_status} {tick_count}
  </div>
  
  <!-- NET BIAS METER (The Only Metric That Matters) -->
  <div class="metric-box" style="margin-top:10px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
      <span style="color:#9ca3af;font-size:0.8em;">Net Absorption Bias</span>
      <span style="color:{bias_text_color};font-weight:bold;font-size:1.1em;">{net_bias:+.0f}</span>
    </div>
    
    <!-- Bias Bar -->
    <div style="position:relative;height:16px;background:linear-gradient(to right, #22c55e 0%, #22c55e 45%, #374151 45%, #374151 55%, #ef4444 55%, #ef4444 100%);border-radius:8px;">
      <!-- Center marker -->
      <div style="position:absolute;left:50%;top:-2px;width:2px;height:20px;background:#fff;transform:translateX(-50%);"></div>
      <!-- Bias indicator -->
      <div style="position:absolute;left:{bias_position}%;top:50%;width:12px;height:12px;background:{bias_color};border:2px solid #fff;border-radius:50%;transform:translate(-50%,-50%);box-shadow:0 0 8px {bias_color};"></div>
    </div>
    
    <div style="display:flex;justify-content:space-between;margin-top:4px;font-size:0.7em;color:#6b7280;">
      <span>🟢 ACCUM</span>
      <span>NEUTRAL</span>
      <span>DIST 🔴</span>
    </div>
  </div>
  
  <!-- Classification & Confirmed -->
  <div style="margin-top:10px;display:flex;gap:10px;">
    <div style="flex:1;background:#1f2937;padding:6px;border-radius:4px;text-align:center;">
      <div style="font-size:0.7em;color:#6b7280;">Current</div>
      <div style="font-weight:bold;color:{'#22c55e' if classification=='ACCUMULATION' else '#ef4444' if classification=='DISTRIBUTION' else '#9ca3af'};">
        {classification}{strength_badge}
      </div>
    </div>
    <div style="flex:1;background:#1f2937;padding:6px;border-radius:4px;text-align:center;">
      <div style="font-size:0.7em;color:#6b7280;">Confirmed</div>
      <div style="font-weight:bold;color:{'#22c55e' if confirmed=='ACCUMULATION' else '#ef4444' if confirmed=='DISTRIBUTION' else '#9ca3af'};">
        {confirmed}
      </div>
    </div>
  </div>
  
  <div style="margin-top:8px;color:#9ca3af;font-size:0.8em;">
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
  
  {f'<div style="margin-top:6px;padding:4px;background:#7f1d1d33;border-radius:4px;font-size:0.7em;color:#fca5a5;">⚠️ {reason}</div>' if reason else ''}
  
  <!-- Details -->
  <details style="margin-top:8px;">
    <summary style="color:#6b7280;cursor:pointer;font-size:0.75em;">📊 Absorption Details</summary>
    <div style="color:#6b7280;font-size:0.7em;margin-top:4px;background:#0f172a;padding:6px;border-radius:4px;">
      <b>Absorption:</b><br>
      Accu (buyers absorbing): {absorp.get('accu_absorption', 0):,.0f}<br>
      Dist (sellers absorbing): {absorp.get('dist_absorption', 0):,.0f}<br>
      Net Bias: {absorp.get('net_bias', 0):,.0f}<br>
      <br>
      <b>Volume:</b><br>
      Buy Vol: {absorp.get('buy_volume', 0):,}<br>
      Sell Vol: {absorp.get('sell_volume', 0):,}<br>
      Delta: {absorp.get('volume_delta', 0):+,}<br>
      <br>
      <b>Price:</b><br>
      Change: ₹{absorp.get('price_change', 0):+.2f}<br>
      Up Move: ₹{absorp.get('price_up', 0):.2f}<br>
      Down Move: ₹{absorp.get('price_down', 0):.2f}
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