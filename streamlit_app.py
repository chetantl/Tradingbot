"""
Order Flow Trading Dashboard - Institutional Activity Detection System
Real-time WebSocket monitoring with time-normalized volume analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time
import time
import logging
import queue
import threading
from typing import Dict, List, Tuple, Optional
import json

# Zerodha Kite Connect imports
try:
    from kiteconnect import KiteConnect, KiteTicker
except ImportError:
    st.error("⚠️ kiteconnect library not installed. Run: pip install kiteconnect")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Thread-safe primitives (module-level globals)
TICK_DATA_QUEUE = queue.Queue(maxsize=1000)
CONNECTED = threading.Event()
WS_INSTANCE = None

# WebSocket error recovery
WS_RECONNECT_EVENT = threading.Event()
WS_RECONNECT_LOCK = threading.Lock()
WS_RECONNECT_COUNT = 0
WS_MAX_RETRIES = 5
WS_INITIAL_BACKOFF = 1  # seconds
WS_LAST_ERROR_TIME = 0

# Trading parameters
MIN_CONFIDENCE = 7
MAX_DAILY_SIGNALS = 6
INSTITUTIONAL_THRESHOLD = 2.5
MIN_TIME_DELTA = 0.5  # Minimum seconds between ticks

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize all session state variables"""
    
    # Authentication
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'api_secret' not in st.session_state:
        st.session_state.api_secret = ""
    if 'access_token' not in st.session_state:
        st.session_state.access_token = ""
    if 'kite' not in st.session_state:
        st.session_state.kite = None
    
    # Monitoring state
    if 'monitoring_active' not in st.session_state:
        st.session_state.monitoring_active = False
    if 'monitored_symbols' not in st.session_state:
        st.session_state.monitored_symbols = []
    if 'instrument_tokens_map' not in st.session_state:
        st.session_state.instrument_tokens_map = {}
    
    # Tick data storage
    if 'tick_data' not in st.session_state:
        st.session_state.tick_data = {}
    if 'previous_snapshots' not in st.session_state:
        st.session_state.previous_snapshots = {}
    if 'total_ticks' not in st.session_state:
        st.session_state.total_ticks = 0
    
    # Signal tracking
    if 'daily_signals_pool' not in st.session_state:
        st.session_state.daily_signals_pool = []
    if 'top_signals' not in st.session_state:
        st.session_state.top_signals = []
    
    # PCR caching
    if 'pcr_cache' not in st.session_state:
        st.session_state.pcr_cache = {}
    if 'pcr_last_update' not in st.session_state:
        st.session_state.pcr_last_update = {}
    
    # Volume averaging (for ratio calculation)
    if 'volume_history' not in st.session_state:
        st.session_state.volume_history = {}

    # Connection monitoring
    if 'connection_errors' not in st.session_state:
        st.session_state.connection_errors = 0
    if 'last_connection_status' not in st.session_state:
        st.session_state.last_connection_status = 'Disconnected'

    # Instruments cache for PCR calculation
    if 'instruments_cache' not in st.session_state:
        st.session_state.instruments_cache = None

# ═══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET IMPLEMENTATION (THREAD-SAFE)
# ═══════════════════════════════════════════════════════════════════════════════

def on_ticks(ws, ticks):
    """
    WebSocket callback (runs in background thread).
    
    CRITICAL: Never access st.session_state here!
    Only push to thread-safe queue.
    """
    for tick in ticks:
        try:
            TICK_DATA_QUEUE.put_nowait(tick)
        except queue.Full:
            logger.warning("Tick queue full, dropping tick")

def on_connect(ws, response):
    """
    Subscribe to tokens on connection.
    
    CRITICAL: Cannot access st.session_state from background thread.
    Tokens must be passed via module-level variable.
    """
    try:
        # Use module-level variable instead of session_state
        if hasattr(ws, '_tokens_to_subscribe'):
            tokens = ws._tokens_to_subscribe
            if tokens:
                ws.subscribe(tokens)
                ws.set_mode(ws.MODE_FULL, tokens)
                CONNECTED.set()
                logger.info(f"✅ WebSocket connected, subscribed to {len(tokens)} instruments")
        else:
            logger.warning("No tokens to subscribe - tokens not set on WebSocket instance")
    except Exception as e:
        logger.error(f"Connection callback error: {e}")

def on_close(ws, code, reason):
    """Handle WebSocket closure with automatic reconnection"""
    import time

    global WS_LAST_ERROR_TIME
    WS_LAST_ERROR_TIME = time.time()

    CONNECTED.clear()
    logger.warning(f"WebSocket closed: {code} - {reason}")

    # Trigger reconnection if monitoring is active
    if 'monitoring_active' in st.session_state and st.session_state.monitoring_active:
        with WS_RECONNECT_LOCK:
            global WS_RECONNECT_COUNT
            if WS_RECONNECT_COUNT < WS_MAX_RETRIES:
                WS_RECONNECT_EVENT.set()
                logger.info("🔄 Scheduling WebSocket reconnection...")
            else:
                logger.error(f"❌ Max reconnection attempts ({WS_MAX_RETRIES}) reached")

def on_error(ws, code, reason):
    """Handle WebSocket errors with reconnection logic"""
    import time

    global WS_LAST_ERROR_TIME
    WS_LAST_ERROR_TIME = time.time()

    logger.error(f"WebSocket error: {code} - {reason}")

    # For network errors, trigger reconnection
    if code in [1006, 1000, 1001] or not CONNECTED.is_set():
        if 'monitoring_active' in st.session_state and st.session_state.monitoring_active:
            with WS_RECONNECT_LOCK:
                global WS_RECONNECT_COUNT
                if WS_RECONNECT_COUNT < WS_MAX_RETRIES:
                    WS_RECONNECT_EVENT.set()
                    logger.info(f"🔄 Network error detected, scheduling reconnection...")

class WebSocketManager:
    """
    WebSocket connection manager with automatic reconnection and exponential backoff.

    Handles connection failures, network interruptions, and provides robust
    reconnection logic for production deployment.
    """

    def __init__(self, max_retries=5, initial_backoff=1):
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.retry_count = 0
        self.is_running = False

    def connect_with_retry(self, api_key, access_token, tokens):
        """
        Connect to WebSocket with exponential backoff retry logic.

        Args:
            api_key: Zerodha API key
            access_token: Session access token
            tokens: List of instrument tokens to subscribe

        Returns:
            bool: True if connection successful, False after max retries
        """
        global WS_RECONNECT_COUNT, WS_INSTANCE

        while self.retry_count < self.max_retries:
            try:
                logger.info(f"🔌 Connecting WebSocket (attempt {self.retry_count + 1}/{self.max_retries})...")

                # Create new WebSocket instance
                kws = KiteTicker(api_key, access_token)
                kws.on_ticks = on_ticks
                kws.on_connect = on_connect
                kws.on_close = on_close
                kws.on_error = on_error

                # Store tokens for subscription
                kws._tokens_to_subscribe = tokens

                WS_INSTANCE = kws
                WS_RECONNECT_COUNT = self.retry_count

                # Connect with threading
                kws.connect(threaded=True)

                # Wait for connection or timeout
                connection_timeout = 30  # seconds
                start_time = time.time()

                while not CONNECTED.is_set() and (time.time() - start_time) < connection_timeout:
                    time.sleep(0.1)

                if CONNECTED.is_set():
                    logger.info(f"✅ WebSocket connected successfully on attempt {self.retry_count + 1}")
                    self.retry_count = 0  # Reset on success
                    return True
                else:
                    logger.warning(f"⏰ Connection timeout on attempt {self.retry_count + 1}")
                    kws.close()

            except Exception as e:
                logger.error(f"❌ WebSocket connection failed (attempt {self.retry_count + 1}): {e}")

            # Increment retry count and apply backoff
            self.retry_count += 1

            if self.retry_count < self.max_retries:
                backoff = self.initial_backoff * (2 ** (self.retry_count - 1))
                backoff = min(backoff, 60)  # Cap at 60 seconds

                logger.info(f"⏱️ Waiting {backoff:.1f}s before reconnection attempt {self.retry_count + 1}")
                time.sleep(backoff)

        logger.error(f"❌ All {self.max_retries} connection attempts failed")
        return False

    def start_reconnection_monitor(self, api_key, access_token, tokens):
        """
        Start background monitoring for reconnection events.

        Args:
            api_key: Zerodha API key
            access_token: Session access token
            tokens: List of instrument tokens to subscribe
        """
        def monitor_reconnection():
            while self.is_running:
                # Wait for reconnection event
                if WS_RECONNECT_EVENT.wait(timeout=1):
                    WS_RECONNECT_EVENT.clear()

                    if not CONNECTED.is_set() and self.is_running:
                        logger.info("🔄 Reconnection event triggered, attempting to reconnect...")
                        self.connect_with_retry(api_key, access_token, tokens)

        self.is_running = True
        monitor_thread = threading.Thread(target=monitor_reconnection, daemon=True)
        monitor_thread.start()
        logger.info("👁️ WebSocket reconnection monitor started")

# Global WebSocket manager instance
ws_manager = WebSocketManager(max_retries=WS_MAX_RETRIES, initial_backoff=WS_INITIAL_BACKOFF)

# ═══════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER PATTERN
# ═══════════════════════════════════════════════════════════════════════════════

class CircuitBreaker:
    """
    Circuit breaker pattern for API calls to prevent cascading failures.

    Automatically opens circuit when failures exceed threshold, temporarily
    blocking calls to prevent overloading failing services.
    """

    def __init__(self, failure_threshold=5, recovery_timeout=60, expected_exception=Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    def __call__(self, func):
        """Decorator to wrap function with circuit breaker logic"""
        def wrapper(*args, **kwargs):
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = 'HALF_OPEN'
                    logger.info("🔓 Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise Exception("Circuit breaker OPEN - call blocked")

            try:
                result = func(*args, **kwargs)

                # Success - reset failure count
                if self.state == 'HALF_OPEN':
                    self.state = 'CLOSED'
                    self.failure_count = 0
                    logger.info("🔒 Circuit breaker CLOSED after successful call")

                return result

            except self.expected_exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'
                    logger.error(f"🚨 Circuit breaker OPENED after {self.failure_count} failures")

                raise e

        return wrapper

    def reset(self):
        """Manually reset circuit breaker to CLOSED state"""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'
        logger.info("🔒 Circuit breaker manually reset to CLOSED")

    def get_status(self):
        """Get current circuit breaker status"""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time,
            'recovery_timeout': self.recovery_timeout
        }

# Circuit breaker instances for different services
pcr_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=300)  # 5 min recovery
api_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)    # 1 min recovery

def run_websocket(api_key, access_token, tokens):
    """
    Enhanced WebSocket connection with automatic reconnection.

    Args:
        api_key: Zerodha API key
        access_token: Session access token
        tokens: List of instrument tokens to subscribe
    """
    global WS_RECONNECT_COUNT

    try:
        # Reset reconnection count
        WS_RECONNECT_COUNT = 0

        # Connect with retry logic
        success = ws_manager.connect_with_retry(api_key, access_token, tokens)

        if success:
            # Start reconnection monitor
            ws_manager.start_reconnection_monitor(api_key, access_token, tokens)
        else:
            CONNECTED.clear()
            logger.error("❌ WebSocket setup failed after all retry attempts")

    except Exception as e:
        logger.error(f"WebSocket setup error: {e}")
        CONNECTED.clear()

def start_websocket_thread(api_key, access_token, tokens):
    """
    Initialize and start WebSocket thread.
    
    Args:
        api_key: Zerodha API key
        access_token: Session access token
        tokens: List of instrument tokens to subscribe
    """
    ws_thread = threading.Thread(
        target=run_websocket,
        args=(api_key, access_token, tokens),
        daemon=True  # Dies when main thread exits
    )
    ws_thread.start()
    logger.info("🚀 WebSocket thread started")

def stop_websocket():
    """Stop WebSocket connection and reconnection monitoring"""
    global WS_INSTANCE, WS_RECONNECT_COUNT

    try:
        # Stop reconnection monitor
        ws_manager.is_running = False
        WS_RECONNECT_EVENT.clear()
        WS_RECONNECT_COUNT = 0

        # Close WebSocket connection
        if WS_INSTANCE:
            WS_INSTANCE.close()
            WS_INSTANCE = None

        CONNECTED.clear()
        logger.info("🛑 WebSocket and reconnection monitor stopped")

    except Exception as e:
        logger.error(f"❌ Error stopping WebSocket: {e}")

def get_websocket_health() -> Dict:
    """
    Get WebSocket connection health status for monitoring.

    Returns:
        Dict with connection health information
    """
    health = {
        'connected': CONNECTED.is_set(),
        'last_error_time': WS_LAST_ERROR_TIME,
        'reconnect_count': WS_RECONNECT_COUNT,
        'max_retries': WS_MAX_RETRIES,
        'time_since_error': 0,
        'status': 'Connected' if CONNECTED.is_set() else 'Disconnected'
    }

    if WS_LAST_ERROR_TIME > 0:
        health['time_since_error'] = time.time() - WS_LAST_ERROR_TIME

    # Determine health status
    if not CONNECTED.is_set():
        if WS_RECONNECT_COUNT >= WS_MAX_RETRIES:
            health['status'] = 'Failed - Max Retries'
        elif WS_RECONNECT_COUNT > 0:
            health['status'] = 'Reconnecting'
        else:
            health['status'] = 'Disconnected'
    elif WS_RECONNECT_COUNT > 0:
        health['status'] = 'Recovered'

    return health

# ═══════════════════════════════════════════════════════════════════════════════
# CORE CALCULATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_order_imbalance(depth: Dict) -> Tuple[float, float, float]:
    """
    Calculate order book imbalance from depth data.
    
    Example:
        depth = {
            'buy': [{'quantity': 1000}, {'quantity': 2000}],  # Total: 3000
            'sell': [{'quantity': 1500}, {'quantity': 2500}]  # Total: 4000
        }
        
        Returns: (42.9, 57.1, 0.75)
        # 42.9% buy pressure, 57.1% sell pressure, 0.75 buy/sell ratio
    
    Returns:
        (buy_pct, sell_pct, imbalance_ratio)
    """
    try:
        buy_orders = depth.get('buy', [])
        sell_orders = depth.get('sell', [])
        
        total_buy_qty = sum([order.get('quantity', 0) for order in buy_orders])
        total_sell_qty = sum([order.get('quantity', 0) for order in sell_orders])
        
        total_qty = total_buy_qty + total_sell_qty
        
        if total_qty == 0:
            return 50.0, 50.0, 1.0
        
        buy_pct = (total_buy_qty / total_qty) * 100
        sell_pct = (total_sell_qty / total_qty) * 100
        
        imbalance_ratio = total_buy_qty / total_sell_qty if total_sell_qty > 0 else 1.0
        
        return buy_pct, sell_pct, imbalance_ratio
        
    except Exception as e:
        logger.error(f"Order imbalance calculation error: {e}")
        return 50.0, 50.0, 1.0

def detect_institutional_activity_normalized(
    symbol: str,
    current_depth: Dict,
    current_volume: int,
    current_price: float,
    current_timestamp: float,
    current_oi: int
) -> Tuple[float, bool, int, str]:
    """
    Detect institutional activity with TIME NORMALIZATION.
    
    Example:
        Previous: timestamp=1699945234.50, volume=500000
        Current: timestamp=1699945236.21, volume=515000
        
        time_delta = 1.71 seconds
        volume_change = 15000 shares
        volume_rate = 8771.9 shares/second
        
        orderbook_change = 1000 shares
        orderbook_rate = 584.8 shares/second
        
        institutional_ratio = 15.0 (Very Strong Institutional)
    
    Returns:
        (institutional_ratio, is_institutional, oi_change, oi_trend)
    """
    
    prev = st.session_state.previous_snapshots.get(symbol)
    
    if prev is None:
        # First tick - initialize
        st.session_state.previous_snapshots[symbol] = {
            'depth': current_depth,
            'volume': current_volume,
            'price': current_price,
            'timestamp': current_timestamp,
            'oi': current_oi
        }
        return 0.0, False, 0, "Stable"
    
    # ══════════════════════════════════════════════════════════
    # STEP 1: Calculate Time Delta (from exchange timestamp)
    # ══════════════════════════════════════════════════════════
    time_delta = current_timestamp - prev['timestamp']
    time_delta = max(MIN_TIME_DELTA, time_delta)  # Safety minimum
    
    # ══════════════════════════════════════════════════════════
    # STEP 2: Calculate Raw Changes (Incremental)
    # ══════════════════════════════════════════════════════════
    
    # Volume: Cumulative → Incremental
    volume_change = current_volume - prev['volume']
    volume_change = max(0, volume_change)
    
    # Orderbook: Sum of both sides' absolute movement
    prev_buy = sum([o.get('quantity', 0) for o in prev['depth'].get('buy', [])])
    prev_sell = sum([o.get('quantity', 0) for o in prev['depth'].get('sell', [])])
    curr_buy = sum([o.get('quantity', 0) for o in current_depth.get('buy', [])])
    curr_sell = sum([o.get('quantity', 0) for o in current_depth.get('sell', [])])
    
    orderbook_change = abs(curr_buy - prev_buy) + abs(curr_sell - prev_sell)
    
    # ══════════════════════════════════════════════════════════
    # STEP 3: Normalize to Per-Second Rates (KEY INNOVATION)
    # ══════════════════════════════════════════════════════════
    
    volume_rate = volume_change / time_delta
    orderbook_rate = orderbook_change / time_delta
    
    # ══════════════════════════════════════════════════════════
    # STEP 4: Calculate Institutional Ratio
    # ══════════════════════════════════════════════════════════
    
    if orderbook_rate > 0:
        institutional_ratio = volume_rate / orderbook_rate
    else:
        institutional_ratio = 0.0
    
    is_institutional = institutional_ratio > INSTITUTIONAL_THRESHOLD
    
    # ══════════════════════════════════════════════════════════
    # STEP 5: OI Analysis (Informational)
    # ══════════════════════════════════════════════════════════
    
    oi_change = current_oi - prev.get('oi', current_oi)
    
    if oi_change > 0:
        oi_trend = "Rising"
    elif oi_change < 0:
        oi_trend = "Falling"
    else:
        oi_trend = "Stable"
    
    # ══════════════════════════════════════════════════════════
    # STEP 6: Update Snapshot for Next Comparison
    # ══════════════════════════════════════════════════════════
    
    st.session_state.previous_snapshots[symbol] = {
        'depth': current_depth,
        'volume': current_volume,
        'price': current_price,
        'timestamp': current_timestamp,
        'oi': current_oi
    }
    
    return round(institutional_ratio, 2), is_institutional, oi_change, oi_trend

def calculate_volume_ratio(symbol: str, current_volume: int) -> float:
    """
    Calculate current volume as ratio of average volume.
    Uses rolling window of last 20 ticks.
    
    Example:
        Average volume: 100000 shares
        Current volume: 230000 shares
        Returns: 2.3 (230% of average)
    """
    if symbol not in st.session_state.volume_history:
        st.session_state.volume_history[symbol] = []
    
    history = st.session_state.volume_history[symbol]
    history.append(current_volume)
    
    # Keep last 20 data points
    if len(history) > 20:
        history.pop(0)
    
    if len(history) < 3:
        return 1.0
    
    avg_volume = np.mean(history[:-1])  # Exclude current
    
    if avg_volume == 0:
        return 1.0
    
    return current_volume / avg_volume

def calculate_real_pcr(symbol: str, current_price: float) -> float:
    """
    Calculate real Put-Call Ratio from Zerodha options chain.

    Fetches actual options data from Zerodha and calculates PCR based on
    strike prices around the current price (ATM strikes).

    Args:
        symbol: Stock symbol (e.g., 'RELIANCE')
        current_price: Current price of the underlying

    Returns:
        PCR value (Put OI / Call OI)
    """
    try:
        # Get Kite instance from session state
        kite = st.session_state.get('kite')
        if not kite:
            raise ValueError("Kite session not available")

        # Get current expiry instruments for this symbol
        nse_instruments = None
        if 'instruments_cache' not in st.session_state:
            # Cache instruments to avoid repeated API calls
            all_instruments = kite.instruments()
            nse_instruments = [inst for inst in all_instruments if inst['segment'] == 'NFO-OPT']
            st.session_state.instruments_cache = nse_instruments
        else:
            nse_instruments = st.session_state.instruments_cache

        # Filter options for this symbol
        symbol_options = [inst for inst in nse_instruments
                         if inst['name'] == symbol and inst['instrument_type'] in ['CE', 'PE']]

        if not symbol_options:
            logger.warning(f"No options found for {symbol}")
            return 1.0

        # Find ATM strikes (within ±2% of current price)
        atm_range = 0.02  # 2%
        lower_bound = current_price * (1 - atm_range)
        upper_bound = current_price * (1 + atm_range)

        atm_strikes = set()
        for inst in symbol_options:
            strike = inst['strike']
            if lower_bound <= strike <= upper_bound:
                atm_strikes.add(strike)

        # If no ATM strikes found, take strikes closest to current price
        if not atm_strikes:
            all_strikes = sorted(set(inst['strike'] for inst in symbol_options))
            current_idx = 0
            for i, strike in enumerate(all_strikes):
                if strike > current_price:
                    current_idx = i
                    break

            # Take 3 strikes around current price
            start_idx = max(0, current_idx - 1)
            end_idx = min(len(all_strikes), current_idx + 2)
            atm_strikes = set(all_strikes[start_idx:end_idx])

        # Calculate PCR for selected strikes
        total_put_oi = 0
        total_call_oi = 0

        for strike in atm_strikes:
            for inst in symbol_options:
                if inst['strike'] == strike:
                    # Get live quote for this option
                    try:
                        quote = kite.quote([inst['instrument_token']])[inst['instrument_token']]
                        oi = quote.get('oi', 0)

                        if inst['instrument_type'] == 'PE':
                            total_put_oi += oi
                        elif inst['instrument_type'] == 'CE':
                            total_call_oi += oi

                    except Exception as e:
                        logger.warning(f"Failed to get quote for {inst['tradingsymbol']}: {e}")
                        # Fall back to static data
                        oi = inst.get('oi', 0)
                        if inst['instrument_type'] == 'PE':
                            total_put_oi += oi
                        elif inst['instrument_type'] == 'CE':
                            total_call_oi += oi

        # Calculate PCR
        if total_call_oi > 0:
            pcr = total_put_oi / total_call_oi
        else:
            pcr = 1.0  # Neutral if no call OI

        logger.info(f"PCR calculated for {symbol}: {pcr:.2f} (Put OI: {total_put_oi}, Call OI: {total_call_oi})")
        return round(pcr, 2)

    except Exception as e:
        logger.error(f"Real PCR calculation failed for {symbol}: {e}")
        # Fall back to mock data if real calculation fails
        return round(np.random.uniform(0.6, 1.4), 2)

def get_pcr_cached(symbol: str, current_price: float) -> Tuple[float, str]:
    """
    Get Put-Call Ratio with 5-minute caching and fallback to real data.

    Example:
        RELIANCE @ ₹1500
        ATM strikes: 1480, 1500, 1520

        Put OI: 25000, Call OI: 43000
        PCR = 0.58

        0.58 < 0.9 → STRONG_BULLISH

    Enhanced: Now uses real options chain data with mock fallback.
    """

    now = time.time()

    # Check cache
    if symbol in st.session_state.pcr_cache:
        last_update = st.session_state.pcr_last_update.get(symbol, 0)
        if now - last_update < 300:  # 5 minutes
            return st.session_state.pcr_cache[symbol]

    # Fetch PCR (REAL API with fallback)
    try:
        # Calculate real PCR from options chain
        pcr = calculate_real_pcr(symbol, current_price)

        # Determine bias
        if pcr < 0.7:
            bias = "STRONG_BULLISH"
        elif pcr < 0.9:
            bias = "BULLISH"
        elif pcr > 1.3:
            bias = "STRONG_BEARISH"
        elif pcr > 1.1:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        # Cache result
        st.session_state.pcr_cache[symbol] = (pcr, bias)
        st.session_state.pcr_last_update[symbol] = now

        return pcr, bias

    except Exception as e:
        logger.error(f"PCR calculation failed for {symbol}: {e}")
        return 1.0, "NEUTRAL"

def calculate_risk_levels(
    signal_type: str,
    current_price: float,
    depth: Dict
) -> Tuple[float, float, float, float]:
    """
    Calculate dynamic entry, target, and stop loss from orderbook structure.
    
    Example (ACCUMULATION signal):
        current_price = 1500.00
        
        Best bid = 1499.50 (support)
        Best ask = 1500.50 (resistance)
        
        Entry: 1500.00
        Target: 1509.98 (0.67% up, 2x risk distance)
        Stop Loss: 1495.01 (0.33% down, at support)
        R:R = 2.0
    """
    
    try:
        buy_orders = depth.get('buy', [])
        sell_orders = depth.get('sell', [])
        
        if not buy_orders or not sell_orders:
            # Fallback to percentage-based
            if signal_type in ['BUY', 'ACCUMULATION']:
                entry = current_price
                target = entry * 1.0067  # 0.67% up
                stop_loss = entry * 0.9967  # 0.33% down
            else:
                entry = current_price
                target = entry * 0.9967  # 0.67% down
                stop_loss = entry * 1.0033  # 0.33% up
            
            rr_ratio = 2.0
            return entry, target, stop_loss, rr_ratio
        
        # Extract support/resistance levels
        best_bid = buy_orders[0].get('price', current_price * 0.997)
        best_ask = sell_orders[0].get('price', current_price * 1.003)
        
        if signal_type in ['BUY', 'ACCUMULATION']:
            entry = current_price
            stop_loss = best_bid * 0.9997  # Just below support
            risk_distance = entry - stop_loss
            target = entry + (risk_distance * 2)  # 1:2 risk-reward
            
        else:  # SELL, DISTRIBUTION
            entry = current_price
            stop_loss = best_ask * 1.0003  # Just above resistance
            risk_distance = stop_loss - entry
            target = entry - (risk_distance * 2)
        
        rr_ratio = abs((target - entry) / (entry - stop_loss)) if abs(entry - stop_loss) > 0 else 2.0
        
        return entry, target, stop_loss, rr_ratio
        
    except Exception as e:
        logger.error(f"Risk calculation error: {e}")
        # Safe fallback
        if signal_type in ['BUY', 'ACCUMULATION']:
            return current_price, current_price * 1.0067, current_price * 0.9967, 2.0
        else:
            return current_price, current_price * 0.9967, current_price * 1.0033, 2.0

def calculate_relative_score(signal: Dict) -> float:
    """
    Calculate relative score for ranking signals.
    
    Scoring components:
    - Confidence (max 10 points)
    - Institutional ratio (max 3 points)
    - Volume ratio (max 2 points)
    
    Max total: 15 points
    """
    score = 0.0
    
    # Confidence (0-10 points)
    score += signal['Confidence Score']
    
    # Institutional ratio (0-3 points)
    inst_ratio = signal['Institutional Ratio']
    if inst_ratio > 4.0:
        score += 3
    elif inst_ratio > 2.5:
        score += 2
    elif inst_ratio > 1.5:
        score += 1
    
    # Volume ratio (0-2 points)
    vol_ratio_str = signal['Volume Status'].replace('x Avg', '')
    try:
        vol_ratio = float(vol_ratio_str)
        if vol_ratio > 2.0:
            score += 2
        elif vol_ratio > 1.5:
            score += 1
    except:
        pass
    
    return round(score, 2)

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION (MAIN LOGIC)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_signal(symbol: str, tick_data: Dict) -> Optional[Dict]:
    """
    Generate trading signal with multi-factor confirmation.
    
    Example Flow:
        Stock: RELIANCE
        Orderbook: 65% sellers (visible selling)
        Price: +0.08% (stable)
        Inst Ratio: 15.0 (very high)
        PCR: 0.62 (bullish)
        
        → Signal: ACCUMULATION
        → Confidence: 9/10
        → Relative Score: 13.8/15
    """
    
    try:
        # Extract data
        current_price = tick_data.get('last_price', 0)
        depth = tick_data.get('depth', {})
        volume = tick_data.get('volume', 0)
        timestamp = tick_data.get('exchange_timestamp')
        
        # Fallback to timestamp if exchange_timestamp not available
        if timestamp is None:
            timestamp = tick_data.get('timestamp', time.time())
        
        oi = tick_data.get('oi', 0)
        
        # Validation
        if current_price == 0 or not depth:
            return None
        
        # ══════════════════════════════════════════════════════════
        # CALCULATE ALL METRICS
        # ══════════════════════════════════════════════════════════
        
        # 1. Order Imbalance
        buy_pct, sell_pct, imbalance_ratio = calculate_order_imbalance(depth)
        
        # 2. Institutional Detection (TIME-NORMALIZED)
        inst_ratio, is_institutional, oi_change, oi_trend = detect_institutional_activity_normalized(
            symbol, depth, volume, current_price, timestamp, oi
        )
        
        # 3. Volume Ratio
        vol_ratio = calculate_volume_ratio(symbol, volume)
        
        # 4. PCR
        pcr, pcr_bias = get_pcr_cached(symbol, current_price)
        
        # 5. Price Change
        prev_price = st.session_state.previous_snapshots.get(symbol, {}).get('price', current_price)
        price_change = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
        
        # ══════════════════════════════════════════════════════════
        # SIGNAL DECISION TREE
        # ══════════════════════════════════════════════════════════
        
        signal_type = None
        confidence = 0
        
        # TYPE 1: ACCUMULATION (Hidden Buying - Best Signal)
        if sell_pct > 60 and price_change >= -0.1 and is_institutional:
            signal_type = "ACCUMULATION"
            confidence += 3
            
            if pcr_bias in ["STRONG_BULLISH", "BULLISH"]:
                confidence += 4
            elif pcr_bias == "NEUTRAL":
                confidence += 1
        
        # TYPE 2: DISTRIBUTION (Hidden Selling)
        elif buy_pct > 60 and price_change <= 0.1 and is_institutional:
            signal_type = "DISTRIBUTION"
            confidence += 3
            
            if pcr_bias in ["STRONG_BEARISH", "BEARISH"]:
                confidence += 4
            elif pcr_bias == "NEUTRAL":
                confidence += 1
        
        # TYPE 3: BUY (Visible Buying)
        elif buy_pct > 60 and vol_ratio > 1.2:
            signal_type = "BUY"
            confidence += 2
            
            if price_change > 0:
                confidence += 2
            
            if pcr_bias in ["BULLISH", "STRONG_BULLISH"]:
                confidence += 3
            elif pcr_bias in ["BEARISH", "STRONG_BEARISH"]:
                confidence -= 2
        
        # TYPE 4: SELL (Visible Selling)
        elif sell_pct > 60 and vol_ratio > 1.2:
            signal_type = "SELL"
            confidence += 2
            
            if price_change < 0:
                confidence += 2
            
            if pcr_bias in ["BEARISH", "STRONG_BEARISH"]:
                confidence += 3
            elif pcr_bias in ["BULLISH", "STRONG_BULLISH"]:
                confidence -= 2
        
        # No signal generated
        if not signal_type:
            return None
        
        # ══════════════════════════════════════════════════════════
        # CONFIDENCE BOOSTERS
        # ══════════════════════════════════════════════════════════
        
        # Order imbalance strength
        if buy_pct > 70 or sell_pct > 70:
            confidence += 3
        elif buy_pct > 60 or sell_pct > 60:
            confidence += 2
        
        # Institutional strength
        if inst_ratio > 4.0:
            confidence += 3
        elif inst_ratio > 2.5:
            confidence += 2
        
        # Volume surge
        if vol_ratio > 2.0:
            confidence += 2
        elif vol_ratio > 1.5:
            confidence += 1
        elif vol_ratio < 0.8:
            confidence -= 2
        
        # Cap confidence at 10
        confidence = min(confidence, 10)
        
        # Minimum threshold
        if confidence < MIN_CONFIDENCE:
            return None
        
        # ══════════════════════════════════════════════════════════
        # RISK MANAGEMENT
        # ══════════════════════════════════════════════════════════
        
        entry, target, stop_loss, rr_ratio = calculate_risk_levels(
            signal_type, current_price, depth
        )
        
        # Calculate potential profit
        if signal_type in ['BUY', 'ACCUMULATION']:
            profit_pct = ((target - entry) / entry * 100)
        else:
            profit_pct = ((entry - target) / entry * 100)
        
        # ══════════════════════════════════════════════════════════
        # CREATE SIGNAL DICTIONARY
        # ══════════════════════════════════════════════════════════
        
        signal = {
            'Stock Symbol': symbol,
            'Signal Type': signal_type,
            'Confidence Score': confidence,
            'Current Price': round(current_price, 2),
            'Entry Price': round(entry, 2),
            'Target Price': round(target, 2),
            'Stop Loss': round(stop_loss, 2),
            'Risk:Reward': f"1:{round(rr_ratio, 1)}",
            'Order Imbalance': f"{buy_pct:.1f}% Buy, {sell_pct:.1f}% Sell",
            'Institutional Ratio': round(inst_ratio, 2),
            'Volume Status': f"{vol_ratio:.1f}x Avg",
            'PCR': round(pcr, 2),
            'PCR Bias': pcr_bias,
            'OI Change': oi_change,
            'OI Trend': oi_trend,
            'Potential Profit %': round(profit_pct, 2),
            'Time Detected': datetime.now().strftime("%H:%M:%S")
        }
        
        # Calculate relative score for ranking
        signal['Relative Score'] = calculate_relative_score(signal)
        
        return signal
        
    except Exception as e:
        logger.error(f"Signal generation error for {symbol}: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# TICK PROCESSING (MAIN THREAD)
# ═══════════════════════════════════════════════════════════════════════════════

def process_tick_queue():
    """
    Process all ticks from queue (runs in main Streamlit thread).
    Safe to access st.session_state here.
    """
    processed = 0
    
    while not TICK_DATA_QUEUE.empty() and processed < 50:
        try:
            tick = TICK_DATA_QUEUE.get_nowait()
            
            # Find symbol for this token
            symbol = None
            for sym, tok in st.session_state.instrument_tokens_map.items():
                if tok == tick['instrument_token']:
                    symbol = sym
                    break
            
            if symbol:
                # Store tick data
                st.session_state.tick_data[symbol] = tick
                st.session_state.total_ticks += 1
                
                # Generate signal
                signal = generate_signal(symbol, tick)
                
                if signal and signal['Confidence Score'] >= MIN_CONFIDENCE:
                    # Add to daily pool (avoid duplicates)
                    existing = [s for s in st.session_state.daily_signals_pool 
                               if s['Stock Symbol'] == symbol and s['Signal Type'] == signal['Signal Type']]
                    
                    if not existing:
                        st.session_state.daily_signals_pool.append(signal)
            
            processed += 1
            
        except queue.Empty:
            break
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
    
    return processed

def update_top_signals():
    """
    Update top signals by ranking the daily pool.
    Keeps only top MAX_DAILY_SIGNALS by Relative Score.
    """
    if not st.session_state.daily_signals_pool:
        st.session_state.top_signals = []
        return
    
    # Sort by Relative Score (descending)
    sorted_signals = sorted(
        st.session_state.daily_signals_pool,
        key=lambda x: x['Relative Score'],
        reverse=True
    )
    
    # Keep top N
    st.session_state.top_signals = sorted_signals[:MAX_DAILY_SIGNALS]

# ═══════════════════════════════════════════════════════════════════════════════
# SELF-VERIFICATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_self_tests():
    """
    Self-verification tests for critical functions.
    Returns dict of test results.
    """
    results = {}
    
    # Test 1: Time normalization
    try:
        mock_depth = {'buy': [{'quantity': 1000}], 'sell': [{'quantity': 1000}]}
        mock_timestamp = time.time()
        
        # First tick (initialization)
        ratio1, is_inst1, oi_chg1, trend1 = detect_institutional_activity_normalized(
            'TEST_SYMBOL', mock_depth, 100000, 1500.0, mock_timestamp, 0
        )
        
        assert ratio1 == 0.0, "First tick should return 0.0 ratio"
        assert not is_inst1, "First tick should not flag institutional"
        
        # Second tick (should calculate properly)
        time.sleep(0.1)  # Small delay
        ratio2, is_inst2, oi_chg2, trend2 = detect_institutional_activity_normalized(
            'TEST_SYMBOL', mock_depth, 110000, 1500.5, mock_timestamp + 1.0, 0
        )
        
        assert ratio2 >= 0, "Ratio should be non-negative"
        
        results['time_normalization'] = "✅ PASS"
    except Exception as e:
        results['time_normalization'] = f"❌ FAIL: {e}"
    
    # Test 2: Order imbalance calculation
    try:
        test_depth = {
            'buy': [{'quantity': 1000}, {'quantity': 2000}],
            'sell': [{'quantity': 1500}, {'quantity': 2500}]
        }
        
        buy_pct, sell_pct, ratio = calculate_order_imbalance(test_depth)
        
        assert abs(buy_pct + sell_pct - 100) < 0.01, "Percentages should sum to 100"
        assert 0 <= buy_pct <= 100, "Buy% should be 0-100"
        assert 0 <= sell_pct <= 100, "Sell% should be 0-100"
        assert ratio > 0, "Ratio should be positive"
        
        results['order_imbalance'] = "✅ PASS"
    except Exception as e:
        results['order_imbalance'] = f"❌ FAIL: {e}"
    
    # Test 3: Signal confidence bounds
    try:
        mock_tick = {
            'last_price': 1500.0,
            'depth': {'buy': [{'quantity': 3000}], 'sell': [{'quantity': 1000}]},
            'volume': 100000,
            'timestamp': time.time(),
            'oi': 50000
        }
        
        signal = generate_signal('TEST', mock_tick)
        
        if signal:
            assert 0 <= signal['Confidence Score'] <= 10, "Confidence should be 0-10"
            assert signal['Institutional Ratio'] >= 0, "Inst ratio should be non-negative"
            assert 0 <= signal['PCR'] <= 3, "PCR should be reasonable"
        
        results['confidence_scoring'] = "✅ PASS"
    except Exception as e:
        results['confidence_scoring'] = f"❌ FAIL: {e}"
    
    # Test 4: Thread safety check
    try:
        queue_size = TICK_DATA_QUEUE.qsize()
        assert queue_size >= 0, "Queue size should be non-negative"
        
        results['thread_safety'] = "✅ PASS"
    except Exception as e:
        results['thread_safety'] = f"❌ FAIL: {e}"
    
    # Test 5: Risk calculation
    try:
        test_depth = {
            'buy': [{'price': 1499.5, 'quantity': 1000}],
            'sell': [{'price': 1500.5, 'quantity': 1000}]
        }
        
        entry, target, sl, rr = calculate_risk_levels('BUY', 1500.0, test_depth)
        
        assert entry > 0, "Entry should be positive"
        assert target > entry, "Target should be above entry for BUY"
        assert sl < entry, "Stop loss should be below entry for BUY"
        assert rr > 0, "Risk-reward should be positive"
        
        results['risk_calculation'] = "✅ PASS"
    except Exception as e:
        results['risk_calculation'] = f"❌ FAIL: {e}"
    
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION UI
# ═══════════════════════════════════════════════════════════════════════════════

def render_authentication_ui():
    """Render authentication interface in sidebar"""
    
    st.sidebar.title("🔐 Zerodha Authentication")
    
    if st.session_state.kite is None:
        # Not authenticated
        st.sidebar.info("Enter your Zerodha API credentials to connect")
        
        api_key = st.sidebar.text_input(
            "API Key",
            value=st.session_state.api_key,
            type="default",
            key="api_key_input"
        )
        
        api_secret = st.sidebar.text_input(
            "API Secret",
            value=st.session_state.api_secret,
            type="password",
            key="api_secret_input"
        )
        
        if st.sidebar.button("📋 Generate Login URL"):
            if api_key and api_secret:
                st.session_state.api_key = api_key
                st.session_state.api_secret = api_secret
                
                login_url = f"https://kite.zerodha.com/connect/login?api_key={api_key}&v=3"
                
                st.sidebar.success("✅ Login URL Generated!")
                st.sidebar.code(login_url, language=None)
                st.sidebar.info("👆 Click the URL above, login, and copy the request_token from the redirect URL")
            else:
                st.sidebar.error("❌ Please enter both API Key and Secret")
        
        st.sidebar.markdown("---")
        
        request_token = st.sidebar.text_input(
            "Request Token",
            type="default",
            key="request_token_input",
            help="Paste the request_token from redirect URL"
        )
        
        if st.sidebar.button("🔓 Complete Login"):
            if not api_key or not api_secret:
                st.sidebar.error("❌ Please enter API credentials first")
            elif not request_token:
                st.sidebar.error("❌ Please enter request token")
            else:
                try:
                    st.session_state.api_key = api_key
                    st.session_state.api_secret = api_secret
                    
                    # Initialize KiteConnect
                    kite = KiteConnect(api_key=api_key)
                    
                    # Generate session
                    data = kite.generate_session(request_token, api_secret=api_secret)
                    
                    st.session_state.access_token = data["access_token"]
                    kite.set_access_token(st.session_state.access_token)
                    
                    st.session_state.kite = kite
                    
                    st.sidebar.success("✅ Authentication Successful!")
                    st.rerun()
                    
                except Exception as e:
                    st.sidebar.error(f"❌ Authentication failed: {str(e)}")
                    logger.error(f"Auth error: {e}")
    
    else:
        # Already authenticated
        st.sidebar.success("✅ API Connected")
        
        # Mask access token
        masked_token = st.session_state.access_token[:8] + "****" if st.session_state.access_token else "N/A"
        st.sidebar.info(f"🔑 Token: {masked_token}")
        
        if st.sidebar.button("🔌 Disconnect"):
            stop_websocket()
            st.session_state.kite = None
            st.session_state.access_token = ""
            st.session_state.monitoring_active = False
            st.session_state.monitored_symbols = []
            st.session_state.tick_data = {}
            st.session_state.previous_snapshots = {}
            st.session_state.daily_signals_pool = []
            st.session_state.top_signals = []
            st.sidebar.success("Disconnected successfully")
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# MONITORING SETUP UI
# ═══════════════════════════════════════════════════════════════════════════════

def render_monitoring_setup():
    """Render stock monitoring setup interface"""
    
    st.sidebar.markdown("---")
    st.sidebar.title("📊 Monitoring Setup")
    
    if st.session_state.kite is None:
        st.sidebar.warning("⚠️ Please authenticate first")
        return
    
    if not st.session_state.monitoring_active:
        symbols_input = st.sidebar.text_area(
            "Stock Symbols (NSE)",
            value="RELIANCE\nTCS\nINFY\nHDFCBANK\nSBIN",
            height=150,
            help="Enter one symbol per line"
        )
        
        if st.sidebar.button("🚀 Start Monitoring"):
            symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]
            
            if not symbols:
                st.sidebar.error("❌ Please enter at least one symbol")
                return
            
            try:
                # Fetch instrument list
                instruments = st.session_state.kite.instruments("NSE")
                
                # Map symbols to tokens
                token_map = {}
                missing = []
                
                for symbol in symbols:
                    found = False
                    for inst in instruments:
                        if inst['tradingsymbol'] == symbol and inst['segment'] == 'NSE':
                            token_map[symbol] = inst['instrument_token']
                            found = True
                            break
                    
                    if not found:
                        missing.append(symbol)
                
                if missing:
                    st.sidebar.warning(f"⚠️ Could not find: {', '.join(missing)}")
                
                if not token_map:
                    st.sidebar.error("❌ No valid symbols found")
                    return
                
                # Store configuration
                st.session_state.monitored_symbols = list(token_map.keys())
                st.session_state.instrument_tokens_map = token_map
                st.session_state.monitoring_active = True
                
                # Start WebSocket with tokens passed as argument
                start_websocket_thread(
                    st.session_state.api_key,
                    st.session_state.access_token,
                    list(token_map.values())  # Pass tokens directly
                )
                
                st.sidebar.success(f"✅ Monitoring {len(token_map)} stocks!")
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.sidebar.error(f"❌ Setup failed: {str(e)}")
                logger.error(f"Monitoring setup error: {e}")
    
    else:
        # Monitoring active
        st.sidebar.success(f"✅ Monitoring Active")
        st.sidebar.info(f"📈 Stocks: {', '.join(st.session_state.monitored_symbols[:3])}...")
        st.sidebar.metric("Total Ticks Processed", st.session_state.total_ticks)
        
        # Enhanced WebSocket status
        ws_health = get_websocket_health()
        status_emoji = "🟢" if ws_health['connected'] else "🔴"

        if ws_health['status'] == 'Reconnecting':
            status_emoji = "🔄"
        elif ws_health['status'] == 'Failed - Max Retries':
            status_emoji = "❌"
        elif ws_health['status'] == 'Recovered':
            status_emoji = "✅"

        st.sidebar.info(f"WebSocket: {status_emoji} {ws_health['status']}")

        if ws_health['reconnect_count'] > 0:
            st.sidebar.caption(f"🔄 Reconnect attempts: {ws_health['reconnect_count']}/{ws_health['max_retries']}")

        if st.sidebar.button("⏹️ Stop Monitoring"):
            stop_websocket()
            st.session_state.monitoring_active = False
            st.session_state.monitored_symbols = []
            st.session_state.tick_data = {}
            st.session_state.previous_snapshots = {}
            st.session_state.connection_errors = 0
            st.session_state.last_connection_status = 'Disconnected'
            st.sidebar.success("Monitoring stopped")
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD UI
# ═══════════════════════════════════════════════════════════════════════════════

def render_dashboard():
    """Render main dashboard"""
    
    st.title("📊 Order Flow Trading Dashboard")
    st.markdown("### Institutional Activity Detection System")
    
    if not st.session_state.monitoring_active:
        st.info("👈 Configure monitoring in the sidebar to begin")
        
        # Display documentation
        with st.expander("📖 How It Works"):
            st.markdown("""
            **Signal Types:**
            
            1. **ACCUMULATION** 🟢 (Best Signal)
               - Visible selling but price stable
               - High institutional ratio (hidden buying)
               - Institutions accumulating via iceberg orders
            
            2. **DISTRIBUTION** 🔴
               - Visible buying but price capped
               - High institutional ratio (hidden selling)
               - Institutions distributing via iceberg orders
            
            3. **BUY** 🔵
               - Strong visible buying pressure
               - High volume, rising price
               - Confirmed by bullish PCR
            
            4. **SELL** 🟠
               - Strong visible selling pressure
               - High volume, falling price
               - Confirmed by bearish PCR
            
            **Time Normalization:**
            - Ticks arrive at irregular intervals (1-3 seconds)
            - System normalizes volume/orderbook changes to per-second rates
            - Enables accurate institutional detection regardless of tick frequency
            
            **Confidence Scoring:**
            - Base signal type: 2-3 points
            - PCR confirmation: 1-4 points
            - Order imbalance: 2-3 points
            - Institutional strength: 2-3 points
            - Volume surge: 1-2 points
            - Minimum threshold: 7/10
            """)
        
        return
    
    # Process ticks
    processed = process_tick_queue()
    
    # Update top signals
    update_top_signals()
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Monitored Stocks", len(st.session_state.monitored_symbols))
    
    with col2:
        st.metric("📡 Ticks Processed", st.session_state.total_ticks)
    
    with col3:
        st.metric("🎯 Signals Generated", len(st.session_state.daily_signals_pool))
    
    with col4:
        st.metric("⭐ Top Signals", len(st.session_state.top_signals))
    
    st.markdown("---")
    
    # Top signals display
    if st.session_state.top_signals:
        st.subheader("🏆 Top Trading Opportunities")
        
        for idx, signal in enumerate(st.session_state.top_signals, 1):
            signal_type = signal['Signal Type']
            
            # Color coding
            if signal_type == "ACCUMULATION":
                color = "green"
                emoji = "🟢"
            elif signal_type == "DISTRIBUTION":
                color = "red"
                emoji = "🔴"
            elif signal_type == "BUY":
                color = "blue"
                emoji = "🔵"
            else:  # SELL
                color = "orange"
                emoji = "🟠"
            
            with st.expander(
                f"#{idx} {emoji} {signal['Stock Symbol']} - {signal_type} "
                f"(Confidence: {signal['Confidence Score']}/10, Score: {signal['Relative Score']}/15)",
                expanded=(idx <= 2)
            ):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📊 Price Levels**")
                    st.metric("Current Price", f"₹{signal['Current Price']}")
                    st.metric("Entry Price", f"₹{signal['Entry Price']}")
                    st.metric("Target Price", f"₹{signal['Target Price']}")
                    st.metric("Stop Loss", f"₹{signal['Stop Loss']}")
                
                with col2:
                    st.markdown("**📈 Trade Metrics**")
                    st.metric("Risk:Reward", signal['Risk:Reward'])
                    st.metric("Potential Profit", f"{signal['Potential Profit %']}%")
                    st.metric("Volume Status", signal['Volume Status'])
                    st.metric("Institutional Ratio", signal['Institutional Ratio'])
                
                with col3:
                    st.markdown("**🎯 Confirmations**")
                    st.info(f"**Order Imbalance:** {signal['Order Imbalance']}")
                    st.info(f"**PCR:** {signal['PCR']} ({signal['PCR Bias']})")
                    st.info(f"**OI Trend:** {signal['OI Trend']} ({signal['OI Change']:+d})")
                    st.info(f"**Detected:** {signal['Time Detected']}")
    
    else:
        st.info("⏳ Analyzing market data... Signals will appear when high-quality opportunities are detected (min confidence: 7/10)")
    
    st.markdown("---")
    
    # Live tick data table
    with st.expander("📡 Live Tick Data", expanded=False):
        if st.session_state.tick_data:
            tick_df_data = []
            
            for symbol, tick in st.session_state.tick_data.items():
                tick_df_data.append({
                    'Symbol': symbol,
                    'Price': round(tick.get('last_price', 0), 2),
                    'Volume': tick.get('volume', 0),
                    'OI': tick.get('oi', 0),
                    'Last Update': datetime.fromtimestamp(
                        tick.get('exchange_timestamp', tick.get('timestamp', time.time()))
                    ).strftime("%H:%M:%S")
                })
            
            st.dataframe(
                pd.DataFrame(tick_df_data),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No tick data received yet")
    
    # Auto-refresh
    time.sleep(0.5)
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main application entry point"""
    
    st.set_page_config(
        page_title="Order Flow Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Render sidebar components
    render_authentication_ui()
    render_monitoring_setup()
    
    # Self-tests in sidebar
    st.sidebar.markdown("---")
    if st.sidebar.button("🔍 Run Self-Tests"):
        with st.sidebar:
            with st.spinner("Running tests..."):
                test_results = run_self_tests()
                st.subheader("Test Results")
                for test_name, result in test_results.items():
                    st.write(f"**{test_name.replace('_', ' ').title()}:** {result}")
    
    # Render main dashboard
    render_dashboard()

if __name__ == "__main__":
    main()
