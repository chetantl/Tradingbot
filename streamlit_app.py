import streamlit as st
from kiteconnect import KiteConnect
import threading
import time
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import copy
from pytz import timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# ====== TIMEZONE (INDIA) ====== #
INDIA_TZ = timezone("Asia/Kolkata")

# ====== PAGE CONFIG ====== #
st.set_page_config(
    page_title="Institutional Sniper (Intraday Signals)",
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
    box-shadow: 0 0 20px rgba(34, 197, 94, 0.5);
    border: 2px solid #22c55e;
}
.dist-glow {
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
    border: 2px solid #ef4444;
}
.signal-active {
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}
.ttl-bar {
    height: 6px;
    border-radius: 3px;
    background: #374151;
    overflow: hidden;
}
.ttl-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 1s linear;
}
</style>
""",
    unsafe_allow_html=True,
)

# ====== CONFIG ====== #
CONFIG = {
    # === Per-bar absorption settings ===
    "min_price_move": 0.01,
    "default_max_ticks": 500,
    "avg_bars": 750,
    
    # === Adaptive thresholds ===
    "min_volume_pct_of_avg": 0.02,       # 2% of time-adjusted avg
    "min_volume_floor": 500,              # Lower floor for illiquid
    "min_ticks_floor": 3,
    "min_ticks_cap": 50,
    
    # === INTRADAY AGGREGATION ===
    "agg_window_bars": 12,
    "agg_min_filled_pct": 0.5,
    "agg_bias_threshold": 40,
    "agg_strong_threshold": 60,
    "agg_rvol_threshold": 1.0,
    "agg_ema_alpha": 0.3,
    
    # === Signal lifecycle ===
    "signal_ttl_minutes": 90,
    "signal_extend_on_confirm": 30,
    "signal_max_ttl_minutes": 180,
    "bars_to_reverse": 3,
    
    # === Time filters ===
    "market_open_hour": 9,
    "market_open_minute": 15,
    "market_close_hour": 15,
    "market_close_minute": 30,
    "no_trade_first_minutes": 15,
    "no_trade_last_minutes": 15,
    
    # === Risk management ===
    "default_sl_pct": 0.5,
    "default_tp_pct": 1.0,
    
    # === Performance ===
    "parallel_batch_threshold": 20,      # Use parallel processing above this
    "max_workers": 3,
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
    """Always return timezone-aware datetime in IST."""
    return datetime.now(INDIA_TZ)


def today_ist():
    """Return today's date in IST."""
    return now_ist().date()


def current_bar_id_5m():
    """Return bar ID and bar start time (both timezone-aware IST)."""
    now = now_ist()
    minute_slot = (now.minute // 5) * 5
    bar_start = now.replace(minute=minute_slot, second=0, microsecond=0)
    return int(bar_start.timestamp()), bar_start


def get_market_open_time(date=None):
    """Get market open time for a specific date (IST, timezone-aware)."""
    if date is None:
        base = now_ist()
    else:
        base = datetime.combine(date, datetime.min.time())
        base = INDIA_TZ.localize(base)
    
    return base.replace(
        hour=CONFIG["market_open_hour"],
        minute=CONFIG["market_open_minute"],
        second=0,
        microsecond=0
    )


def get_market_close_time(date=None):
    """Get market close time for a specific date (IST, timezone-aware)."""
    if date is None:
        base = now_ist()
    else:
        base = datetime.combine(date, datetime.min.time())
        base = INDIA_TZ.localize(base)
    
    return base.replace(
        hour=CONFIG["market_close_hour"],
        minute=CONFIG["market_close_minute"],
        second=0,
        microsecond=0
    )


def is_market_hours():
    """Check if current time is within trading hours (all IST)."""
    now = now_ist()
    return get_market_open_time() <= now <= get_market_close_time()


def is_tradeable_time():
    """Check if current time is suitable for new signals."""
    if not is_market_hours():
        return False, "Market closed"
    
    now = now_ist()
    market_open = get_market_open_time()
    market_close = get_market_close_time()
    
    minutes_from_open = (now - market_open).total_seconds() / 60
    minutes_to_close = (market_close - now).total_seconds() / 60
    
    if minutes_from_open < CONFIG["no_trade_first_minutes"]:
        return False, f"First {CONFIG['no_trade_first_minutes']} min"
    
    if minutes_to_close < CONFIG["no_trade_last_minutes"]:
        return False, f"Last {CONFIG['no_trade_last_minutes']} min"
    
    return True, "OK"


def get_intraday_volume_factor():
    """
    Get volume adjustment factor based on time of day.
    Volume is typically 2-3x higher in first hour and last 30 min.
    """
    now = now_ist()
    hour = now.hour
    minute = now.minute
    
    # First hour (9:15-10:15): High activity
    if (hour == 9 and minute >= 15) or (hour == 10 and minute < 15):
        return 2.5
    
    # Last 30 min (15:00-15:30): High activity
    if hour == 15 and minute >= 0:
        return 2.0
    
    # Lunch period (11:30-13:30): Low activity
    if (hour == 11 and minute >= 30) or hour == 12 or (hour == 13 and minute < 30):
        return 0.6
    
    # Normal hours
    return 1.0


def safe_quote_batch(kite, batch, manager_log_func, attempts=3, sleep_s=0.5):
    """
    Safely fetch quotes with retry logic, exponential backoff, and missing symbol retry.
    """
    if not batch:
        return {}
    
    for attempt in range(attempts):
        try:
            q = kite.quote(batch)
            
            # Check for missing symbols
            missing = set(batch) - set(q.keys())
            
            if missing:
                manager_log_func(
                    f"Quote incomplete (attempt {attempt+1}): {len(q)}/{len(batch)}. "
                    f"Missing: {list(missing)[:5]}{'...' if len(missing) > 5 else ''}",
                    "WARNING"
                )
                
                # On final attempt, retry just the missing symbols
                if attempt == attempts - 1 and len(missing) <= 10:
                    time.sleep(sleep_s)
                    try:
                        retry_q = kite.quote(list(missing))
                        q.update(retry_q)
                        if retry_q:
                            manager_log_func(
                                f"Retry recovered {len(retry_q)} symbols",
                                "INFO"
                            )
                    except Exception as re:
                        manager_log_func(f"Retry failed: {re}", "WARNING")
            
            return q
            
        except Exception as e:
            backoff = sleep_s * (2 ** attempt)  # Exponential backoff
            if attempt < attempts - 1:
                manager_log_func(
                    f"Quote error (attempt {attempt+1}): {e}. Retrying in {backoff:.1f}s",
                    "WARNING"
                )
                time.sleep(backoff)
            else:
                manager_log_func(f"Quote failed after {attempts} attempts: {e}", "ERROR")
    
    return {}


# ====== VOLUME TRACKER (Fixed Day Rollover) ====== #
class VolumeTracker:
    """
    Tracks cumulative volume per symbol with proper market-open reset detection.
    Uses time-based detection at 9:15 AM instead of percentage-based.
    """
    
    def __init__(self):
        self.lock = threading.RLock()
        self.last_cum_vol = {}           # symbol -> last cumulative volume
        self.bar_start_cum_vol = {}      # symbol -> cumulative at bar start
        self.last_update_time = {}       # symbol -> datetime of last update
        self.last_ltp = {}
        self.last_bid = {}               # symbol -> last best bid price
        self.last_ask = {}               # symbol -> last best ask price
        self.last_buy_q = {}
        self.last_sell_q = {}
    
    def get_volume_delta(self, symbol, current_cum_vol):
        """
        Calculate volume delta with proper market-open reset detection.
        Uses time-based detection at 9:15 AM.
        """
        with self.lock:
            now = now_ist()
            market_open = get_market_open_time()
            
            prev_cum = self.last_cum_vol.get(symbol)
            last_update = self.last_update_time.get(symbol)
            
            # Case 1: First update ever for this symbol
            if prev_cum is None:
                self.last_cum_vol[symbol] = current_cum_vol
                self.last_update_time[symbol] = now
                return 0
            
            # Case 2: Crossed market open (9:15 AM) since last update
            if last_update and last_update < market_open <= now:
                self.last_cum_vol[symbol] = current_cum_vol
                self.last_update_time[symbol] = now
                return 0
            
            # Case 3: Detect unexpected reset (exchange issue or data gap)
            # Only trigger if drop is significant AND we're past market open
            if current_cum_vol < prev_cum:
                drop_pct = (prev_cum - current_cum_vol) / prev_cum if prev_cum > 0 else 0
                
                # If volume dropped more than 30%, treat as reset
                if drop_pct > 0.3:
                    self.last_cum_vol[symbol] = current_cum_vol
                    self.last_update_time[symbol] = now
                    return 0
                else:
                    # Small drop could be data correction, return 0 but don't reset
                    self.last_update_time[symbol] = now
                    return 0
            
            # Case 4: Normal delta
            delta = current_cum_vol - prev_cum
            self.last_cum_vol[symbol] = current_cum_vol
            self.last_update_time[symbol] = now
            
            return delta
    
    def set_bar_start_volume(self, symbol, cum_vol):
        """Mark the start of a new bar."""
        with self.lock:
            self.bar_start_cum_vol[symbol] = cum_vol
    
    def get_bar_volume(self, symbol, current_cum_vol):
        """Get volume accumulated in current bar."""
        with self.lock:
            bar_start = self.bar_start_cum_vol.get(symbol)
            
            if bar_start is None:
                self.bar_start_cum_vol[symbol] = current_cum_vol
                return 0
            
            # Handle reset within bar
            if current_cum_vol < bar_start:
                self.bar_start_cum_vol[symbol] = current_cum_vol
                return 0
            
            return current_cum_vol - bar_start
    
    def update_price_data(self, symbol, ltp, bid_price, ask_price):
        """Update and return previous price data."""
        with self.lock:
            prev_ltp = self.last_ltp.get(symbol, ltp)
            prev_bid = self.last_bid.get(symbol, bid_price)
            prev_ask = self.last_ask.get(symbol, ask_price)
            
            self.last_ltp[symbol] = ltp
            self.last_bid[symbol] = bid_price
            self.last_ask[symbol] = ask_price
            
            return prev_ltp, prev_bid, prev_ask
    
    def update_depth(self, symbol, buy_q, sell_q):
        """Update and return previous depth quantities."""
        with self.lock:
            prev_buy = self.last_buy_q.get(symbol, buy_q)
            prev_sell = self.last_sell_q.get(symbol, sell_q)
            self.last_buy_q[symbol] = buy_q
            self.last_sell_q[symbol] = sell_q
            return prev_buy, prev_sell
    
    def reset_symbol(self, symbol):
        """Fully reset tracking for a symbol."""
        with self.lock:
            for d in [self.last_cum_vol, self.bar_start_cum_vol, self.last_update_time,
                      self.last_ltp, self.last_bid, self.last_ask, 
                      self.last_buy_q, self.last_sell_q]:
                d.pop(symbol, None)
    
    def reset_all(self):
        """Reset all tracking."""
        with self.lock:
            self.last_cum_vol.clear()
            self.bar_start_cum_vol.clear()
            self.last_update_time.clear()
            self.last_ltp.clear()
            self.last_bid.clear()
            self.last_ask.clear()
            self.last_buy_q.clear()
            self.last_sell_q.clear()


# ====== ADAPTIVE THRESHOLD CALCULATOR (Time-Aware) ====== #
class AdaptiveThresholds:
    """
    Calculate per-symbol adaptive thresholds with intraday volume curve adjustment.
    """
    
    @staticmethod
    def get_min_volume(avg_vol):
        """
        Minimum volume for valid signal, adjusted for time of day.
        """
        if not avg_vol or avg_vol <= 0:
            return CONFIG["min_volume_floor"]
        
        time_factor = get_intraday_volume_factor()
        adjusted_avg = avg_vol * time_factor
        
        adaptive = adjusted_avg * CONFIG["min_volume_pct_of_avg"]
        return max(CONFIG["min_volume_floor"], int(adaptive))
    
    @staticmethod
    def get_min_ticks(avg_vol):
        """
        Minimum tick count, scaled with liquidity and time of day.
        """
        if not avg_vol or avg_vol <= 0:
            return CONFIG["min_ticks_floor"]
        
        time_factor = get_intraday_volume_factor()
        adjusted_avg = avg_vol * time_factor
        
        # 1 tick per 500 adjusted avg_vol
        adaptive = int(adjusted_avg / 500)
        return max(CONFIG["min_ticks_floor"], min(CONFIG["min_ticks_cap"], adaptive))
    
    @staticmethod
    def get_min_price_move(ltp, avg_vol=None):
        """Dynamic minimum price move based on LTP."""
        if not ltp or ltp <= 0:
            return CONFIG["min_price_move"]
        
        pct_based = ltp * 0.0002  # 0.02% base
        
        if avg_vol and avg_vol > 50000:
            pct_based = ltp * 0.0001  # Tighter for liquid
        
        return max(CONFIG["min_price_move"], pct_based)
    
    @staticmethod
    def get_max_ticks(avg_vol):
        """Adaptive tick history size."""
        if not avg_vol or avg_vol < 5000:
            return 200
        elif avg_vol < 20000:
            return 350
        else:
            return CONFIG["default_max_ticks"]


# ====== IMPROVED TICK CLASSIFICATION ====== #
def estimate_buy_sell_volume(vol_delta, ltp, prev_ltp, bid_price, ask_price,
                              buy_q, sell_q, prev_buy_q, prev_sell_q):
    """
    Enhanced tick classification using bid-ask prices and aggressive trade detection.
    
    Logic:
    - Trade at/above ask = aggressive buy (lifted offer)
    - Trade at/below bid = aggressive sell (hit bid)
    - Trade inside spread = use midpoint comparison
    - Fallback to tick rule if no bid/ask
    """
    if vol_delta <= 0:
        return 0, 0
    
    # Use bid-ask based classification if available
    if bid_price > 0 and ask_price > 0 and bid_price < ask_price:
        mid_price = (bid_price + ask_price) / 2
        
        # Aggressive buy: trade at or above ask (lifted the offer)
        if ltp >= ask_price:
            return vol_delta, 0
        
        # Aggressive sell: trade at or below bid (hit the bid)
        elif ltp <= bid_price:
            return 0, vol_delta
        
        # Trade inside spread - use position relative to midpoint
        elif ltp > mid_price:
            # Closer to ask = more likely buy
            spread = ask_price - bid_price
            distance_from_mid = ltp - mid_price
            buy_ratio = 0.5 + (distance_from_mid / spread) * 0.5  # 0.5 to 1.0
            buy_ratio = min(0.85, max(0.5, buy_ratio))
            buy_vol = int(vol_delta * buy_ratio)
            return buy_vol, vol_delta - buy_vol
        
        elif ltp < mid_price:
            # Closer to bid = more likely sell
            spread = ask_price - bid_price
            distance_from_mid = mid_price - ltp
            sell_ratio = 0.5 + (distance_from_mid / spread) * 0.5
            sell_ratio = min(0.85, max(0.5, sell_ratio))
            sell_vol = int(vol_delta * sell_ratio)
            return vol_delta - sell_vol, sell_vol
        
        else:
            # Exactly at midpoint - split evenly
            return vol_delta // 2, vol_delta - (vol_delta // 2)
    
    # Fallback to simple tick rule if no valid bid/ask
    if ltp > prev_ltp:
        return vol_delta, 0
    elif ltp < prev_ltp:
        return 0, vol_delta
    else:
        # Price unchanged - use order book imbalance as hint
        total_q = buy_q + sell_q
        if total_q > 0:
            buy_ratio = buy_q / total_q
            buy_vol = int(vol_delta * buy_ratio)
            return buy_vol, vol_delta - buy_vol
        else:
            return vol_delta // 2, vol_delta - (vol_delta // 2)


# ====== ABSORPTION ENGINE ====== #
class AbsorptionEngine:
    """Per-bar absorption calculation with net bias and adaptive thresholds."""
    
    def __init__(self):
        self.tick_history = {}
        self.lock = threading.RLock()

    def record_tick(self, symbol, ltp, vol_delta, buy_delta, sell_delta, avg_vol=None):
        """Record a tick (thread-safe)."""
        with self.lock:
            if symbol not in self.tick_history:
                max_ticks = AdaptiveThresholds.get_max_ticks(avg_vol)
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

    def calculate_net_absorption(self, symbol, current_ltp, avg_vol=None):
        """Calculate net absorption bias for current bar with adaptive thresholds."""
        ticks = self.get_ticks_snapshot(symbol)
        
        min_ticks = AdaptiveThresholds.get_min_ticks(avg_vol)
        
        if len(ticks) < min_ticks:
            return self._empty_metrics(
                tick_count=len(ticks),
                min_ticks_required=min_ticks,
                reason=f"Need {min_ticks} ticks, have {len(ticks)}"
            )

        first_price = ticks[0]["ltp"]
        last_price = ticks[-1]["ltp"]
        price_change = last_price - first_price
        price_up = max(0, price_change)
        price_down = abs(min(0, price_change))
        
        prices = [t["ltp"] for t in ticks]
        price_range = max(prices) - min(prices)

        total_volume = sum(t["vol_delta"] for t in ticks)
        total_buy_vol = sum(t["buy_delta"] for t in ticks)
        total_sell_vol = sum(t["sell_delta"] for t in ticks)

        min_volume = AdaptiveThresholds.get_min_volume(avg_vol)
        
        if total_volume < min_volume:
            return self._empty_metrics(
                tick_count=len(ticks),
                total_volume=total_volume,
                min_volume_required=min_volume,
                reason=f"Need vol {min_volume:,}, have {total_volume:,}"
            )

        min_move = AdaptiveThresholds.get_min_price_move(current_ltp, avg_vol)

        # Absorption calculation
        accu_absorption = total_sell_vol / max(price_down, min_move) if total_sell_vol > 0 else 0
        dist_absorption = total_buy_vol / max(price_up, min_move) if total_buy_vol > 0 else 0

        # Net bias
        net_bias = dist_absorption - accu_absorption
        max_absorption = max(accu_absorption, dist_absorption, 1)
        
        normalized_bias = (net_bias / max_absorption) * 100 if max_absorption > 0 else 0
        normalized_bias = max(-100, min(100, normalized_bias))

        return {
            "net_bias": round(net_bias, 2),
            "normalized_bias": round(normalized_bias, 1),
            "accu_absorption": round(accu_absorption, 2),
            "dist_absorption": round(dist_absorption, 2),
            "total_volume": total_volume,
            "buy_volume": total_buy_vol,
            "sell_volume": total_sell_vol,
            "price_change": round(price_change, 2),
            "price_range": round(price_range, 2),
            "tick_count": len(ticks),
            "min_ticks_required": min_ticks,
            "min_volume_required": min_volume,
            "min_move_used": round(min_move, 4),
            "reason": None,
        }

    def _empty_metrics(self, **kwargs):
        base = {
            "net_bias": 0, "normalized_bias": 0, "accu_absorption": 0,
            "dist_absorption": 0, "total_volume": 0, "buy_volume": 0,
            "sell_volume": 0, "price_change": 0, "price_range": 0,
            "tick_count": 0, "min_ticks_required": 0, "min_volume_required": 0,
            "min_move_used": 0, "reason": None,
        }
        base.update(kwargs)
        return base


# ====== INTRADAY SIGNAL AGGREGATOR (Fixed TTL) ====== #
class IntradaySignalAggregator:
    """
    Aggregates per-bar absorption into intraday signals with fixed TTL handling.
    Uses absolute expiry timestamps instead of relative durations.
    """
    
    def __init__(self):
        self.lock = threading.RLock()
        self.rolling_bias = {}
        self.bias_ema = {}
        self.opposite_count = {}
        self.active_signals = {}
    
    def initialize_symbol(self, symbol):
        """Initialize aggregation state for a symbol."""
        with self.lock:
            self.rolling_bias[symbol] = deque(maxlen=CONFIG["agg_window_bars"])
            self.bias_ema[symbol] = 0.0
            self.opposite_count[symbol] = 0
            self.active_signals[symbol] = self._empty_signal()
    
    def _empty_signal(self):
        return {
            "signal": "NEUTRAL",
            "direction": None,
            "start_time": None,
            "expiry_time": None,        # Use absolute expiry
            "entry_price": None,
            "sl_price": None,
            "tp_price": None,
            "current_strength": 0,
            "peak_strength": 0,
            "confirmations": 0,
            "is_strong": False,
        }
    
    def record_bar(self, symbol, normalized_bias, rvol, tick_count, ltp, avg_vol=None):
        """Record a completed bar's bias and update aggregated signal."""
        with self.lock:
            if symbol not in self.rolling_bias:
                self.initialize_symbol(symbol)
            
            # Store bar bias
            self.rolling_bias[symbol].append(normalized_bias)
            
            # Calculate rolling average
            rb = self.rolling_bias[symbol]
            min_bars = max(4, int(CONFIG["agg_window_bars"] * CONFIG["agg_min_filled_pct"]))
            
            if len(rb) >= min_bars:
                rolling_avg = sum(rb) / len(rb)
            else:
                rolling_avg = sum(rb) / max(1, len(rb))
            
            # Apply EMA smoothing
            prev_ema = self.bias_ema.get(symbol, 0.0)
            alpha = CONFIG["agg_ema_alpha"]
            ema = alpha * rolling_avg + (1 - alpha) * prev_ema
            self.bias_ema[symbol] = ema
            
            # Determine direction and magnitude
            bias_magnitude = abs(ema)
            if ema < -CONFIG["agg_bias_threshold"]:
                current_direction = "ACCUMULATION"
            elif ema > CONFIG["agg_bias_threshold"]:
                current_direction = "DISTRIBUTION"
            else:
                current_direction = "NEUTRAL"
            
            is_strong = bias_magnitude >= CONFIG["agg_strong_threshold"]
            
            # Check requirements
            rvol_ok = rvol >= CONFIG["agg_rvol_threshold"]
            min_ticks = AdaptiveThresholds.get_min_ticks(avg_vol)
            ticks_ok = tick_count >= min_ticks
            time_ok, time_reason = is_tradeable_time()
            
            # Get current active signal
            active = self.active_signals.get(symbol, self._empty_signal())
            now = now_ist()
            
            # Check expiry using absolute expiry_time
            if active["signal"] != "NEUTRAL" and active["expiry_time"]:
                if now >= active["expiry_time"]:
                    active = self._reset_signal(symbol)
            
            # Check for reversal
            if active["signal"] != "NEUTRAL":
                if current_direction != "NEUTRAL" and current_direction != active["signal"]:
                    self.opposite_count[symbol] = self.opposite_count.get(symbol, 0) + 1
                    if self.opposite_count[symbol] >= CONFIG["bars_to_reverse"]:
                        active = self._reset_signal(symbol)
                else:
                    self.opposite_count[symbol] = 0
            
            # Generate or extend signal
            if current_direction != "NEUTRAL" and rvol_ok and ticks_ok and time_ok:
                if active["signal"] == "NEUTRAL":
                    # NEW SIGNAL - use absolute expiry time
                    is_long = current_direction == "ACCUMULATION"
                    sl_mult = 1 - CONFIG["default_sl_pct"]/100 if is_long else 1 + CONFIG["default_sl_pct"]/100
                    tp_mult = 1 + CONFIG["default_tp_pct"]/100 if is_long else 1 - CONFIG["default_tp_pct"]/100
                    
                    expiry = now + timedelta(minutes=CONFIG["signal_ttl_minutes"])
                    
                    active = {
                        "signal": current_direction,
                        "direction": "LONG" if is_long else "SHORT",
                        "start_time": now,
                        "expiry_time": expiry,
                        "entry_price": ltp,
                        "sl_price": ltp * sl_mult,
                        "tp_price": ltp * tp_mult,
                        "current_strength": bias_magnitude,
                        "peak_strength": bias_magnitude,
                        "confirmations": 1,
                        "is_strong": is_strong,
                    }
                    self.active_signals[symbol] = active
                    
                elif active["signal"] == current_direction:
                    # RECONFIRMATION - extend expiry properly
                    active["confirmations"] += 1
                    active["current_strength"] = bias_magnitude
                    active["peak_strength"] = max(active["peak_strength"], bias_magnitude)
                    active["is_strong"] = is_strong or active.get("is_strong", False)
                    
                    # Calculate new expiry: extend by confirm amount, but cap at max from start
                    current_expiry = active["expiry_time"]
                    proposed_expiry = now + timedelta(minutes=CONFIG["signal_extend_on_confirm"])
                    max_expiry = active["start_time"] + timedelta(minutes=CONFIG["signal_max_ttl_minutes"])
                    
                    # Take the minimum of proposed and max, but at least current
                    new_expiry = min(proposed_expiry, max_expiry)
                    if new_expiry > current_expiry:
                        active["expiry_time"] = new_expiry
                    
                    self.active_signals[symbol] = active
            
            # Calculate remaining TTL
            remaining_ttl = 0
            ttl_pct = 0
            if active["signal"] != "NEUTRAL" and active["expiry_time"] and active["start_time"]:
                remaining_ttl = max(0, (active["expiry_time"] - now).total_seconds() / 60)
                total_duration = (active["expiry_time"] - active["start_time"]).total_seconds() / 60
                ttl_pct = (remaining_ttl / total_duration) * 100 if total_duration > 0 else 0
            
            return {
                "signal": active["signal"],
                "direction": active.get("direction"),
                "start_time": active.get("start_time"),
                "expiry_time": active.get("expiry_time"),
                "remaining_ttl": round(remaining_ttl, 1),
                "ttl_pct": round(ttl_pct, 1),
                "entry_price": active.get("entry_price"),
                "sl_price": active.get("sl_price"),
                "tp_price": active.get("tp_price"),
                "current_strength": round(bias_magnitude, 1),
                "peak_strength": active.get("peak_strength", 0),
                "confirmations": active.get("confirmations", 0),
                "is_strong": active.get("is_strong", False),
                "ema_bias": round(ema, 1),
                "rolling_avg": round(rolling_avg, 1),
                "window_filled": len(rb),
                "window_size": CONFIG["agg_window_bars"],
                "rvol_ok": rvol_ok,
                "ticks_ok": ticks_ok,
                "time_ok": time_ok,
                "time_reason": time_reason if not time_ok else None,
            }
    
    def get_live_state(self, symbol):
        """Get current signal state without recording a new bar."""
        with self.lock:
            if symbol not in self.active_signals:
                return {"signal": "NEUTRAL", "remaining_ttl": 0}
            
            active = self.active_signals[symbol]
            now = now_ist()
            
            remaining_ttl = 0
            ttl_pct = 0
            
            if active["signal"] != "NEUTRAL" and active["expiry_time"]:
                # Check if expired
                if now >= active["expiry_time"]:
                    active = self._reset_signal(symbol)
                else:
                    remaining_ttl = (active["expiry_time"] - now).total_seconds() / 60
                    if active["start_time"]:
                        total_duration = (active["expiry_time"] - active["start_time"]).total_seconds() / 60
                        ttl_pct = (remaining_ttl / total_duration) * 100 if total_duration > 0 else 0
            
            return {
                "signal": active["signal"],
                "direction": active.get("direction"),
                "remaining_ttl": round(remaining_ttl, 1),
                "ttl_pct": round(ttl_pct, 1),
                "entry_price": active.get("entry_price"),
                "sl_price": active.get("sl_price"),
                "tp_price": active.get("tp_price"),
                "confirmations": active.get("confirmations", 0),
                "is_strong": active.get("is_strong", False),
                "ema_bias": round(self.bias_ema.get(symbol, 0), 1),
            }
    
    def _reset_signal(self, symbol):
        """Reset signal to neutral."""
        with self.lock:
            self.opposite_count[symbol] = 0
            self.active_signals[symbol] = self._empty_signal()
            return self.active_signals[symbol]
    
    def reset_all(self):
        """Reset all aggregation state."""
        with self.lock:
            self.rolling_bias.clear()
            self.bias_ema.clear()
            self.opposite_count.clear()
            self.active_signals.clear()


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
        self.signal_aggregator = IntradaySignalAggregator()
        self.volume_tracker = VolumeTracker()

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
            self.avg_volumes[symbol] = {"avg_vol": avg_vol, "bar_count": bar_count}

    def get_avg_volume(self, symbol):
        with self.lock:
            return self.avg_volumes.get(symbol, {}).get("avg_vol", 0.0)

    def reset_all_state(self):
        """Reset all manager state."""
        with self.lock:
            self.data.clear()
            self.avg_volumes.clear()
            self.absorption_engine = AbsorptionEngine()
            self.signal_aggregator = IntradaySignalAggregator()
            self.volume_tracker = VolumeTracker()


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
        manager.log(f"Instrument lookup error: {e}", "ERROR")
    return None


def calc_avg_volume_750(kite, token):
    """Calculate average volume with proper timezone handling."""
    try:
        now_naive = now_ist().replace(tzinfo=None)
        from_date = now_naive - timedelta(days=15)
        
        candles = kite.historical_data(token, from_date, now_naive, "5minute")
        df = pd.DataFrame(candles)
        
        if df.empty:
            return 0.0, 0
        
        df = df.sort_values("date")
        completed = df.iloc[:-1] if len(df) > 1 else df
        
        if len(completed) > CONFIG["avg_bars"]:
            completed = completed.tail(CONFIG["avg_bars"])
        
        return round(float(completed["volume"].mean()), 2), len(completed)
    except Exception as e:
        manager.log(f"Avg volume error: {e}", "ERROR")
        return 0.0, 0


def calc_live_rvol(current_vol, avg_vol):
    return round(current_vol / avg_vol, 2) if avg_vol > 0 else 0.0


def scan_opt_power(kite, symbol):
    """Scan options with proper error handling."""
    try:
        all_opts = [i for i in kite.instruments("NFO") 
                    if i.get("segment") == "NFO-OPT" and i.get("name") == symbol]
        if not all_opts:
            return 0.0, 0.0, 0.0
        
        today = today_ist()
        expiries = sorted({i["expiry"].date() for i in all_opts if i["expiry"].date() >= today})
        if not expiries:
            return 0.0, 0.0, 0.0
        
        nearest = expiries[0]
        scoped = [i for i in all_opts if i["expiry"].date() == nearest]
        
        ce_syms = [f"NFO:{i['tradingsymbol']}" for i in scoped if i["instrument_type"] == "CE"]
        pe_syms = [f"NFO:{i['tradingsymbol']}" for i in scoped if i["instrument_type"] == "PE"]
        
        ce_vol, pe_vol = 0, 0
        
        for batch in [ce_syms[i:i+50] for i in range(0, len(ce_syms), 50)]:
            quotes = safe_quote_batch(kite, batch, manager.log)
            for v in quotes.values():
                if safe_float(v.get("last_price")) >= 20:
                    ce_vol += safe_int(v.get("volume"))
        
        for batch in [pe_syms[i:i+50] for i in range(0, len(pe_syms), 50)]:
            quotes = safe_quote_batch(kite, batch, manager.log)
            for v in quotes.values():
                if safe_float(v.get("last_price")) >= 20:
                    pe_vol += safe_int(v.get("volume"))
        
        return round((ce_vol - pe_vol) / 1e6, 2), round(ce_vol / 1e6, 2), round(pe_vol / 1e6, 2)
    except Exception as e:
        manager.log(f"Option scan error: {e}", "ERROR")
        return 0.0, 0.0, 0.0


# ====== SYMBOL PROCESSOR (for parallel processing) ====== #
def process_single_symbol(row, quotes, manager, curr_bar_id, bar_changed, do_opt_scan, kite):
    """
    Process a single symbol. Used for both sequential and parallel processing.
    Returns updated state dict or None if processing failed.
    """
    sym = row["symbol"]
    key = f"NSE:{sym}"
    q = quotes.get(key)
    
    if not q:
        return None, f"{sym}: No quote data"
    
    try:
        stt = manager.get_symbol_data(sym)
        avg_750 = manager.get_avg_volume(sym)

        ltp = safe_float(q.get("last_price"))
        depth = q.get("depth", {})
        
        # Extract bid/ask prices for improved tick classification
        buy_depth = depth.get("buy", [])
        sell_depth = depth.get("sell", [])
        
        bid_price = buy_depth[0].get("price", 0) if buy_depth else 0
        ask_price = sell_depth[0].get("price", 0) if sell_depth else 0
        
        buy_q = sum(safe_int(x.get("quantity")) for x in buy_depth[:5])
        sell_q = sum(safe_int(x.get("quantity")) for x in sell_depth[:5])
        
        cum_vol = safe_int(q.get("volume"))

        # Volume tracking
        vol_delta = manager.volume_tracker.get_volume_delta(sym, cum_vol)
        prev_ltp, prev_bid, prev_ask = manager.volume_tracker.update_price_data(sym, ltp, bid_price, ask_price)
        prev_buy_q, prev_sell_q = manager.volume_tracker.update_depth(sym, buy_q, sell_q)
        
        # Improved tick classification using bid/ask
        buy_vol, sell_vol = estimate_buy_sell_volume(
            vol_delta, ltp, prev_ltp, bid_price, ask_price,
            buy_q, sell_q, prev_buy_q, prev_sell_q
        )
        
        # Record tick
        if vol_delta > 0:
            manager.absorption_engine.record_tick(sym, ltp, vol_delta, buy_vol, sell_vol, avg_750)

        # Bar volume tracking
        if bar_changed:
            manager.volume_tracker.set_bar_start_volume(sym, cum_vol)
        
        current_bar_vol = manager.volume_tracker.get_bar_volume(sym, cum_vol)
        live_rvol = calc_live_rvol(current_bar_vol, avg_750)
        absorption = manager.absorption_engine.calculate_net_absorption(sym, ltp, avg_750)

        # Initialize state
        if not stt:
            manager.signal_aggregator.initialize_symbol(sym)
            stt = {
                "ltp": ltp, "buy_q": buy_q, "sell_q": sell_q,
                "bid_price": bid_price, "ask_price": ask_price,
                "bar_id": curr_bar_id, "bar_vol": current_bar_vol,
                "rvol": live_rvol, "avg_750": avg_750,
                "absorption": absorption,
                "intraday_signal": {"signal": "NEUTRAL"},
                "opt_ce": 0, "opt_pe": 0, "opt_power": 0,
                "trade_signal": "IDLE", "last_bar_vol": 0, "last_bar_rvol": 0,
            }

        # Bar close processing
        bar_close_log = None
        if bar_changed and stt.get("bar_id") == curr_bar_id:
            final_absorption = manager.absorption_engine.calculate_net_absorption(sym, ltp, avg_750)
            final_bar_vol = stt.get("bar_vol", 0)
            final_rvol = calc_live_rvol(final_bar_vol, avg_750)

            intraday_state = manager.signal_aggregator.record_bar(
                sym, final_absorption.get("normalized_bias", 0),
                final_rvol, final_absorption.get("tick_count", 0),
                ltp, avg_vol=avg_750
            )

            stt["intraday_signal"] = intraday_state
            stt["last_bar_vol"] = final_bar_vol
            stt["last_bar_rvol"] = final_rvol

            # Prepare log message
            sig = intraday_state["signal"]
            if sig != "NEUTRAL":
                ttl = intraday_state["remaining_ttl"]
                conf = intraday_state["confirmations"]
                emoji = "🟢" if sig == "ACCUMULATION" else "🔴"
                strength = "💪" if intraday_state.get("is_strong") else ""
                bar_close_log = (
                    f"{emoji} {sym} INTRADAY {sig} {strength} | "
                    f"TTL={ttl:.0f}m | Confirms={conf} | "
                    f"EMA={intraday_state['ema_bias']:+.0f} | RVol={final_rvol:.1f}x"
                )

            manager.absorption_engine.reset_bar(sym)
            stt["bar_id"] = None  # Will be set to new bar id after processing
            stt["bar_vol"] = 0
        else:
            stt["bar_vol"] = current_bar_vol
            stt["absorption"] = absorption
            stt["intraday_signal"] = manager.signal_aggregator.get_live_state(sym)

        stt["rvol"] = live_rvol
        stt["ltp"] = ltp
        stt["buy_q"] = buy_q
        stt["sell_q"] = sell_q
        stt["bid_price"] = bid_price
        stt["ask_price"] = ask_price

        # Options scan
        if do_opt_scan:
            net, ce, pe = scan_opt_power(kite, sym)
            stt["opt_power"], stt["opt_ce"], stt["opt_pe"] = net, ce, pe

        # Trade signal
        intraday = stt.get("intraday_signal", {})
        sig = intraday.get("signal", "NEUTRAL")
        ce_m, pe_m = stt.get("opt_ce", 0), stt.get("opt_pe", 0)
        call_bias = ce_m > pe_m * 1.2
        put_bias = pe_m > ce_m * 1.2

        if sig == "ACCUMULATION" and call_bias:
            stt["trade_signal"] = "🟢 BUY CALL"
        elif sig == "DISTRIBUTION" and put_bias:
            stt["trade_signal"] = "🔴 BUY PUT"
        elif sig == "ACCUMULATION":
            stt["trade_signal"] = "🟢 ACCUM (await CE)"
        elif sig == "DISTRIBUTION":
            stt["trade_signal"] = "🔴 DIST (await PE)"
        else:
            stt["trade_signal"] = "IDLE"

        return stt, bar_close_log
        
    except Exception as e:
        return None, f"{sym}: Error - {str(e)[:50]}"


# ====== WORKER THREAD ====== #
def sniper_worker(kite):
    manager.is_running = True
    manager.log("Worker started (Intraday Signal Mode v2)", "INFO")
    manager.log(f"Window: {CONFIG['agg_window_bars']} bars | TTL: {CONFIG['signal_ttl_minutes']}m", "INFO")

    curr_bar_id, curr_bar_start = current_bar_id_5m()
    last_opt_scan = 0.0

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

            # Fetch quotes
            keys = [f"NSE:{row['symbol']}" for row in active_list]
            quotes = safe_quote_batch(kite, keys, manager.log)
            
            if not quotes:
                time.sleep(2)
                continue

            do_opt_scan = (now_ts - last_opt_scan) > 60
            if do_opt_scan:
                last_opt_scan = now_ts

            # Process symbols (parallel if many, sequential if few)
            if len(active_list) > CONFIG["parallel_batch_threshold"]:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
                    futures = {
                        executor.submit(
                            process_single_symbol, row, quotes, manager,
                            curr_bar_id, bar_changed, do_opt_scan, kite
                        ): row["symbol"]
                        for row in active_list
                    }
                    
                    for future in as_completed(futures):
                        sym = futures[future]
                        try:
                            stt, log_msg = future.result()
                            if stt:
                                if bar_changed:
                                    stt["bar_id"] = new_bar_id
                                else:
                                    stt["bar_id"] = curr_bar_id
                                manager.set_symbol_data(sym, stt)
                                if log_msg:
                                    manager.log(log_msg, "INFO")
                            elif log_msg:
                                manager.log(log_msg, "WARNING")
                        except Exception as e:
                            manager.log(f"{sym}: Future error - {e}", "ERROR")
            else:
                # Sequential processing
                for row in active_list:
                    sym = row["symbol"]
                    stt, log_msg = process_single_symbol(
                        row, quotes, manager, curr_bar_id, bar_changed, do_opt_scan, kite
                    )
                    if stt:
                        if bar_changed:
                            stt["bar_id"] = new_bar_id
                        else:
                            stt["bar_id"] = curr_bar_id
                        manager.set_symbol_data(sym, stt)
                        if log_msg:
                            manager.log(log_msg, "INFO")
                    elif log_msg:
                        manager.log(log_msg, "WARNING")

            # Update global bar tracking AFTER processing all symbols
            if bar_changed:
                manager.log(f"Bar closed: {curr_bar_start.strftime('%H:%M')} → {new_bar_start.strftime('%H:%M')}", "DEBUG")
                curr_bar_id = new_bar_id
                curr_bar_start = new_bar_start

            time.sleep(POLL_INTERVAL_SEC)

        except Exception as e:
            manager.log(f"Worker error: {e}", "ERROR")
            import traceback
            manager.log(traceback.format_exc()[:300], "DEBUG")
            time.sleep(2)

    manager.is_running = False
    manager.log("Worker stopped", "INFO")


# ====== SIDEBAR ====== #
st.sidebar.title("🔐 Kite Connect")

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
        st.sidebar.markdown(f"[🔗 Login to Zerodha]({temp_kite.login_url()})")
    except:
        pass

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

st.sidebar.markdown("---")

# Signal window selector
window_option = st.sidebar.selectbox(
    "📊 Signal Window",
    ["1 Hour (12 bars)", "90 Minutes (18 bars)", "2 Hours (24 bars)"],
    index=0
)
if "1 Hour" in window_option:
    CONFIG["agg_window_bars"] = 12
    CONFIG["signal_ttl_minutes"] = 60
elif "90 Minutes" in window_option:
    CONFIG["agg_window_bars"] = 18
    CONFIG["signal_ttl_minutes"] = 90
else:
    CONFIG["agg_window_bars"] = 24
    CONFIG["signal_ttl_minutes"] = 120

CONFIG["agg_bias_threshold"] = st.sidebar.slider("Bias Threshold", 20, 80, 40)

st.sidebar.markdown("---")

symbols_text = st.sidebar.text_area("Symbols", "M&M, PAYTM, GAIL, KEI, TVSMOTOR")

btn1, btn2 = st.sidebar.columns(2)

if btn1.button("▶ START"):
    if not manager.kite:
        st.error("Login first")
    elif manager.is_running:
        st.warning("Already running")
    else:
        syms = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
        valid = []
        
        manager.reset_all_state()
        
        with manager.lock:
            manager.active_symbols = []

        with st.spinner("Initializing..."):
            progress = st.progress(0)
            
            for idx, s in enumerate(syms):
                tkn = get_instrument_token(manager.kite, s)
                if not tkn:
                    manager.log(f"{s}: Not found", "WARNING")
                    continue
                
                avg_vol, bar_count = calc_avg_volume_750(manager.kite, tkn)
                manager.set_avg_volume(s, avg_vol, bar_count)
                manager.signal_aggregator.initialize_symbol(s)
                
                time_factor = get_intraday_volume_factor()
                min_vol = AdaptiveThresholds.get_min_volume(avg_vol)
                min_ticks = AdaptiveThresholds.get_min_ticks(avg_vol)
                
                manager.log(
                    f"{s}: AvgVol={avg_vol:,.0f} | TimeFactor={time_factor:.1f}x | "
                    f"MinVol={min_vol:,} | MinTicks={min_ticks}",
                    "INFO"
                )
                
                valid.append({"symbol": s, "token": tkn})
                progress.progress((idx + 1) / len(syms))
            
            progress.empty()

        if valid:
            with manager.lock:
                manager.active_symbols = valid
                manager.stop_event.clear()
            
            threading.Thread(target=sniper_worker, args=(manager.kite,), daemon=True).start()
            manager.is_running = True
            st.success(f"✅ Monitoring {len(valid)} stocks")

if btn2.button("⏹ STOP"):
    manager.stop_event.set()
    manager.is_running = False
    st.rerun()

# Debug info
with st.sidebar.expander("🔧 Debug"):
    st.markdown(f"**Market:** {'Open' if is_market_hours() else 'Closed'}")
    tradeable, reason = is_tradeable_time()
    st.markdown(f"**Tradeable:** {tradeable} ({reason})")
    st.markdown(f"**Vol Factor:** {get_intraday_volume_factor():.1f}x")
    st.markdown(f"**Time:** {now_ist().strftime('%H:%M:%S')}")

# ====== DASHBOARD ====== #
st.title("🎯 Institutional Sniper – Intraday v2")
st.caption(f"Window: {CONFIG['agg_window_bars']}×5m | TTL: {CONFIG['signal_ttl_minutes']}m | Threshold: {CONFIG['agg_bias_threshold']}")

# Status
last_beat = manager.last_beat
delta = time.time() - last_beat
hb_color = "#22c55e" if delta < 5 else "#ef4444"
hb_text = "LIVE" if delta < 5 else "STOPPED"

tradeable, trade_reason = is_tradeable_time()
trade_status = "🟢 Tradeable" if tradeable else f"⏸️ {trade_reason}"

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div style="display:flex;align-items:center;gap:8px;"><div style="width:12px;height:12px;border-radius:50%;background:{hb_color};"></div><span style="font-weight:bold;">{hb_text}</span></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f"**{now_ist().strftime('%H:%M:%S')}** | Vol×{get_intraday_volume_factor():.1f}")
with col3:
    st.markdown(trade_status)

st.markdown("---")

snapshot = manager.get_data_snapshot()

if snapshot:
    items = sorted(
        snapshot.items(),
        key=lambda x: (
            0 if x[1].get("intraday_signal", {}).get("signal") != "NEUTRAL" else 1,
            -x[1].get("intraday_signal", {}).get("remaining_ttl", 0)
        )
    )

    cols = st.columns(3)
    for i, (sym, d) in enumerate(items):
        with cols[i % 3]:
            intraday = d.get("intraday_signal", {})
            sig = intraday.get("signal", "NEUTRAL")
            ttl = intraday.get("remaining_ttl", 0)
            ttl_pct = intraday.get("ttl_pct", 0)
            ema_bias = intraday.get("ema_bias", 0)
            confirmations = intraday.get("confirmations", 0)
            is_strong = intraday.get("is_strong", False)
            entry_price = intraday.get("entry_price")
            sl_price = intraday.get("sl_price")
            tp_price = intraday.get("tp_price")
            
            trade_sig = d.get("trade_signal", "IDLE")
            rvol = d.get("rvol", 0)
            ltp = d.get("ltp", 0)
            absorption = d.get("absorption", {})

            if sig == "ACCUMULATION":
                card_class = "accum-glow signal-active"
                badge_style = "background:#064e3b;color:#6ee7b7;"
                ttl_color = "#22c55e"
            elif sig == "DISTRIBUTION":
                card_class = "dist-glow signal-active"
                badge_style = "background:#7f1d1d;color:#fca5a5;"
                ttl_color = "#ef4444"
            else:
                card_class = ""
                badge_style = "background:#374151;color:#9ca3af;"
                ttl_color = "#6b7280"

            pnl_pct = 0
            pnl_color = "#9ca3af"
            if entry_price and ltp and sig != "NEUTRAL":
                if sig == "ACCUMULATION":
                    pnl_pct = ((ltp - entry_price) / entry_price) * 100
                else:
                    pnl_pct = ((entry_price - ltp) / entry_price) * 100
                pnl_color = "#22c55e" if pnl_pct >= 0 else "#ef4444"

            if sig != "NEUTRAL":
                signal_html = f"""
                <div class="metric-box" style="margin-top:10px;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="color:{ttl_color};font-weight:bold;">⏱️ TTL: {ttl:.0f}m</span>
                        <span style="color:#9ca3af;">×{confirmations} {"💪" if is_strong else ""}</span>
                    </div>
                    <div class="ttl-bar"><div class="ttl-fill" style="width:{ttl_pct}%;background:{ttl_color};"></div></div>
                </div>
                <div style="margin-top:8px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;text-align:center;">
                    <div style="background:#1f2937;padding:6px;border-radius:4px;">
                        <div style="font-size:0.7em;color:#6b7280;">Entry</div>
                        <div style="color:white;">₹{entry_price:.2f}</div>
                    </div>
                    <div style="background:#1f2937;padding:6px;border-radius:4px;">
                        <div style="font-size:0.7em;color:#ef4444;">SL</div>
                        <div style="color:#ef4444;">₹{sl_price:.2f}</div>
                    </div>
                    <div style="background:#1f2937;padding:6px;border-radius:4px;">
                        <div style="font-size:0.7em;color:#22c55e;">TP</div>
                        <div style="color:#22c55e;">₹{tp_price:.2f}</div>
                    </div>
                </div>
                <div style="margin-top:8px;text-align:center;padding:6px;background:#0f172a;border-radius:4px;">
                    <span style="color:{pnl_color};font-weight:bold;font-size:1.1em;">{pnl_pct:+.2f}%</span>
                </div>
                """
            else:
                signal_html = f'<div class="metric-box" style="margin-top:10px;text-align:center;"><span style="color:#6b7280;">No signal | EMA: {ema_bias:+.0f}</span></div>'

            st.markdown(f"""
<div class="card-container {card_class}">
    <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:1.3em;font-weight:bold;color:white;">{sym}</span>
        <span style="padding:4px 10px;border-radius:4px;font-weight:bold;{badge_style}">{trade_sig}</span>
    </div>
    <div style="margin-top:8px;color:#9ca3af;">
        <b>₹{ltp:.2f}</b> | RVol {rvol:.1f}x | Vol {d.get('bar_vol', 0):,}
    </div>
    {signal_html}
    <div style="margin-top:8px;font-size:0.8em;color:#6b7280;">
        CE {d.get('opt_ce',0):.2f}M / PE {d.get('opt_pe',0):.2f}M | Bid ₹{d.get('bid_price',0):.2f} / Ask ₹{d.get('ask_price',0):.2f}
    </div>
</div>
""", unsafe_allow_html=True)
else:
    st.info("👆 Enter symbols and click START")

st.markdown("---")
logs = manager.get_logs_snapshot()
st.subheader("📝 Logs")
log_filter = st.selectbox("Filter", ["All", "INFO", "WARNING", "ERROR"], index=0, key="log_filter")
if log_filter != "All":
    logs = [l for l in logs if log_filter in l]
st.text_area("", "\n".join(logs) if logs else "No logs", height=200, disabled=True, key="log_area")

if manager.is_running:
    time.sleep(3)
    st.rerun()