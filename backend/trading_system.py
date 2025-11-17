"""
Professional Trading System Core
Handles WebSocket connections, signal generation, and real-time data processing
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
import json
import queue
from collections import defaultdict, deque

from kiteconnect import KiteConnect, KiteTicker
from database import DatabaseManager
from monitoring import SystemMonitor
from config import get_config

logger = logging.getLogger(__name__)
config = get_config()

@dataclass
class Signal:
    """Trading signal data structure"""
    id: str
    symbol: str
    signal_type: str  # ACCUMULATION, DISTRIBUTION, BUY, SELL
    confidence_score: int
    current_price: float
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward: str
    order_imbalance: str
    institutional_ratio: float
    volume_status: str
    pcr: float
    pcr_bias: str
    oi_change: int
    oi_trend: str
    potential_profit_pct: float
    relative_score: float
    time_detected: str
    timestamp_created: float
    user_id: str
    broadcasted: bool = False

@dataclass
class WebSocketHealth:
    """WebSocket connection health status"""
    connected: bool
    last_error_time: Optional[float]
    reconnect_count: int
    max_retries: int
    uptime_percentage: float
    average_reconnect_time: float
    status: str

class CircuitBreaker:
    """Circuit breaker for API resilience"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker OPEN - call blocked")

        try:
            result = await func(*args, **kwargs)

            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker CLOSED after successful call")

            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker OPENED after {self.failure_count} failures")

            raise e

class WebSocketManager:
    """Enhanced WebSocket manager with reconnection logic"""

    def __init__(self):
        self.kws: Optional[KiteTicker] = None
        self.is_connected = False
        self.reconnect_count = 0
        self.last_error_time = 0
        self.reconnect_times: deque = deque(maxlen=10)
        self.monitoring_symbols: Set[str] = set()
        self.token_map: Dict[str, int] = {}
        self.tick_queue: queue.Queue = queue.Queue(maxsize=config.websocket.queue_max_size)
        self._stop_event = threading.Event()
        self._ws_thread: Optional[threading.Thread] = None

    async def connect(self, api_key: str, access_token: str):
        """Connect to WebSocket with retry logic"""
        max_retries = config.websocket.max_retries
        initial_backoff = config.websocket.initial_backoff

        for attempt in range(max_retries):
            try:
                logger.info(f"🔌 Connecting WebSocket (attempt {attempt + 1}/{max_retries})")

                self.kws = KiteTicker(api_key, access_token)

                # Set callbacks
                self.kws.on_ticks = self._on_ticks
                self.kws.on_connect = self._on_connect
                self.kws.on_close = self._on_close
                self.kws.on_error = self._on_error

                # Connect in background thread
                self._ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
                self._ws_thread.start()

                # Wait for connection
                for _ in range(30):  # 30 seconds timeout
                    if self.is_connected:
                        break
                    await asyncio.sleep(0.1)

                if self.is_connected:
                    self.reconnect_count = 0
                    logger.info("✅ WebSocket connected successfully")
                    return True
                else:
                    logger.warning("⏰ WebSocket connection timeout")
                    self.kws.close()

            except Exception as e:
                logger.error(f"❌ WebSocket connection failed (attempt {attempt + 1}): {e}")

            if attempt < max_retries - 1:
                backoff = initial_backoff * (2 ** attempt)
                backoff = min(backoff, 60)
                logger.info(f"⏱️ Waiting {backoff:.1f}s before retry...")
                await asyncio.sleep(backoff)

        logger.error(f"❌ All {max_retries} WebSocket connection attempts failed")
        return False

    def _run_websocket(self):
        """Run WebSocket in background thread"""
        try:
            if self.kws:
                self.kws.connect(threaded=True)
        except Exception as e:
            logger.error(f"WebSocket thread error: {e}")

    def _on_ticks(self, ws, ticks):
        """Handle incoming ticks"""
        for tick in ticks:
            try:
                self.tick_queue.put_nowait(tick)
            except queue.Full:
                logger.warning("Tick queue full, dropping tick")

    def _on_connect(self, ws, response):
        """Handle WebSocket connection"""
        self.is_connected = True
        logger.info("✅ WebSocket connected")

        # Subscribe to tokens if we have them
        if self.token_map:
            tokens = list(self.token_map.values())
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
            logger.info(f"📡 Subscribed to {len(tokens)} instruments")

    def _on_close(self, ws, code, reason):
        """Handle WebSocket closure"""
        reconnect_time = time.time()
        self.reconnect_times.append(reconnect_time)
        self.last_error_time = reconnect_time

        self.is_connected = False
        logger.warning(f"WebSocket closed: {code} - {reason}")

        # Schedule reconnection if not stopping
        if not self._stop_event.is_set():
            asyncio.create_task(self._schedule_reconnect())

    def _on_error(self, ws, code, reason):
        """Handle WebSocket errors"""
        self.last_error_time = time.time()
        logger.error(f"WebSocket error: {code} - {reason}")

    async def _schedule_reconnect(self):
        """Schedule reconnection attempt"""
        await asyncio.sleep(5)  # Wait 5 seconds before reconnecting

        if not self._stop_event.is_set() and not self.is_connected:
            logger.info("🔄 Attempting WebSocket reconnection...")
            # This would need the API credentials to reconnect
            # In practice, you'd store these securely

    async def disconnect(self):
        """Disconnect WebSocket"""
        self._stop_event.set()
        if self.kws:
            self.kws.close()
        self.is_connected = False
        logger.info("🛑 WebSocket disconnected")

    def get_health(self) -> WebSocketHealth:
        """Get WebSocket health status"""
        avg_reconnect_time = 0
        if self.reconnect_times:
            reconnect_intervals = [
                self.reconnect_times[i] - self.reconnect_times[i-1]
                for i in range(1, len(self.reconnect_times))
            ]
            avg_reconnect_time = sum(reconnect_intervals) / len(reconnect_intervals) if reconnect_intervals else 0

        status = "Connected" if self.is_connected else "Disconnected"
        if not self.is_connected and self.reconnect_count > 0:
            status = "Reconnecting"

        return WebSocketHealth(
            connected=self.is_connected,
            last_error_time=self.last_error_time,
            reconnect_count=self.reconnect_count,
            max_retries=config.websocket.max_retries,
            uptime_percentage=0.0,  # Would calculate from start time
            average_reconnect_time=avg_reconnect_time,
            status=status
        )

class SignalGenerator:
    """Professional signal generation with advanced algorithms"""

    def __init__(self):
        self.previous_snapshots: Dict[str, Dict] = {}
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self.pcr_cache: Dict[str, tuple] = {}
        self.pcr_last_update: Dict[str, float] = {}

    async def generate_signal(self, symbol: str, tick_data: Dict, user_id: str) -> Optional[Signal]:
        """Generate trading signal from tick data"""
        try:
            # Extract data
            current_price = tick_data.get("last_price", 0)
            depth = tick_data.get("depth", {})
            volume = tick_data.get("volume", 0)
            timestamp = tick_data.get("exchange_timestamp", time.time())
            oi = tick_data.get("oi", 0)

            if current_price == 0 or not depth:
                return None

            # Calculate metrics
            buy_pct, sell_pct, imbalance_ratio = self._calculate_order_imbalance(depth)
            inst_ratio, is_institutional, oi_change, oi_trend = self._detect_institutional_activity(
                symbol, depth, volume, current_price, timestamp, oi
            )
            vol_ratio = self._calculate_volume_ratio(symbol, volume)
            pcr, pcr_bias = await self._get_pcr(symbol, current_price)

            # Price change
            prev_price = self.previous_snapshots.get(symbol, {}).get("price", current_price)
            price_change = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0

            # Generate signal
            signal_type, confidence = self._generate_signal_type(
                buy_pct, sell_pct, price_change, is_institutional, pcr_bias, vol_ratio
            )

            if not signal_type or confidence < config.trading.min_confidence:
                return None

            # Calculate risk levels
            entry, target, stop_loss, rr_ratio = self._calculate_risk_levels(
                signal_type, current_price, depth
            )

            # Calculate potential profit
            if signal_type in ["BUY", "ACCUMULATION"]:
                profit_pct = ((target - entry) / entry * 100)
            else:
                profit_pct = ((entry - target) / entry * 100)

            # Create signal
            signal = Signal(
                id=f"{symbol}_{int(time.time())}_{hash(str(tick_data)) % 10000}",
                symbol=symbol,
                signal_type=signal_type,
                confidence_score=confidence,
                current_price=round(current_price, 2),
                entry_price=round(entry, 2),
                target_price=round(target, 2),
                stop_loss=round(stop_loss, 2),
                risk_reward=f"1:{round(rr_ratio, 1)}",
                order_imbalance=f"{buy_pct:.1f}% Buy, {sell_pct:.1f}% Sell",
                institutional_ratio=round(inst_ratio, 2),
                volume_status=f"{vol_ratio:.1f}x Avg",
                pcr=round(pcr, 2),
                pcr_bias=pcr_bias,
                oi_change=oi_change,
                oi_trend=oi_trend,
                potential_profit_pct=round(profit_pct, 2),
                relative_score=self._calculate_relative_score(
                    confidence, inst_ratio, vol_ratio
                ),
                time_detected=datetime.now().strftime("%H:%M:%S"),
                timestamp_created=time.time(),
                user_id=user_id
            )

            return signal

        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
            return None

    def _calculate_order_imbalance(self, depth: Dict) -> tuple:
        """Calculate order book imbalance"""
        try:
            buy_orders = depth.get("buy", [])
            sell_orders = depth.get("sell", [])

            total_buy_qty = sum(order.get("quantity", 0) for order in buy_orders)
            total_sell_qty = sum(order.get("quantity", 0) for order in sell_orders)

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

    def _detect_institutional_activity(self, symbol: str, depth: Dict, volume: int,
                                     current_price: float, timestamp: float, oi: int) -> tuple:
        """Detect institutional activity with time normalization"""
        prev = self.previous_snapshots.get(symbol)

        if prev is None:
            self.previous_snapshots[symbol] = {
                "depth": depth, "volume": volume, "price": current_price,
                "timestamp": timestamp, "oi": oi
            }
            return 0.0, False, 0, "Stable"

        # Time delta
        time_delta = max(config.trading.min_time_delta, timestamp - prev["timestamp"])

        # Volume change
        volume_change = max(0, volume - prev["volume"])
        volume_rate = volume_change / time_delta

        # Orderbook change
        prev_buy = sum(o.get("quantity", 0) for o in prev["depth"].get("buy", []))
        prev_sell = sum(o.get("quantity", 0) for o in prev["depth"].get("sell", []))
        curr_buy = sum(o.get("quantity", 0) for o in depth.get("buy", []))
        curr_sell = sum(o.get("quantity", 0) for o in depth.get("sell", []))

        orderbook_change = abs(curr_buy - prev_buy) + abs(curr_sell - prev_sell)
        orderbook_rate = orderbook_change / time_delta

        # Institutional ratio
        institutional_ratio = volume_rate / orderbook_rate if orderbook_rate > 0 else 0.0
        is_institutional = institutional_ratio > config.trading.institutional_threshold

        # OI analysis
        oi_change = oi - prev.get("oi", oi)
        oi_trend = "Rising" if oi_change > 0 else ("Falling" if oi_change < 0 else "Stable")

        # Update snapshot
        self.previous_snapshots[symbol] = {
            "depth": depth, "volume": volume, "price": current_price,
            "timestamp": timestamp, "oi": oi
        }

        return round(institutional_ratio, 2), is_institutional, oi_change, oi_trend

    def _calculate_volume_ratio(self, symbol: str, current_volume: int) -> float:
        """Calculate current volume as ratio of average"""
        history = self.volume_history[symbol]
        history.append(current_volume)

        if len(history) < 3:
            return 1.0

        avg_volume = sum(list(history)[:-1]) / (len(history) - 1)
        return current_volume / avg_volume if avg_volume > 0 else 1.0

    async def _get_pcr(self, symbol: str, current_price: float) -> tuple:
        """Get Put-Call Ratio with caching"""
        now = time.time()

        # Check cache
        if symbol in self.pcr_cache:
            last_update = self.pcr_last_update.get(symbol, 0)
            if now - last_update < 300:  # 5 minutes
                return self.pcr_cache[symbol]

        # Calculate PCR (simplified for demo)
        pcr = 0.8 + (hash(symbol) % 40) / 100  # Mock: 0.8 to 1.2

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
        self.pcr_cache[symbol] = (pcr, bias)
        self.pcr_last_update[symbol] = now

        return pcr, bias

    def _generate_signal_type(self, buy_pct: float, sell_pct: float, price_change: float,
                            is_institutional: bool, pcr_bias: str, vol_ratio: float) -> tuple:
        """Generate signal type and confidence"""
        signal_type = None
        confidence = 0

        # ACCUMULATION (Hidden Buying)
        if sell_pct > 60 and price_change >= -0.1 and is_institutional:
            signal_type = "ACCUMULATION"
            confidence += 3
            if pcr_bias in ["STRONG_BULLISH", "BULLISH"]:
                confidence += 4
            elif pcr_bias == "NEUTRAL":
                confidence += 1

        # DISTRIBUTION (Hidden Selling)
        elif buy_pct > 60 and price_change <= 0.1 and is_institutional:
            signal_type = "DISTRIBUTION"
            confidence += 3
            if pcr_bias in ["STRONG_BEARISH", "BEARISH"]:
                confidence += 4
            elif pcr_bias == "NEUTRAL":
                confidence += 1

        # BUY (Visible Buying)
        elif buy_pct > 60 and vol_ratio > 1.2:
            signal_type = "BUY"
            confidence += 2
            if price_change > 0:
                confidence += 2
            if pcr_bias in ["BULLISH", "STRONG_BULLISH"]:
                confidence += 3

        # SELL (Visible Selling)
        elif sell_pct > 60 and vol_ratio > 1.2:
            signal_type = "SELL"
            confidence += 2
            if price_change < 0:
                confidence += 2
            if pcr_bias in ["BEARISH", "STRONG_BEARISH"]:
                confidence += 3

        if not signal_type:
            return None, 0

        # Confidence boosters
        if buy_pct > 70 or sell_pct > 70:
            confidence += 3
        elif buy_pct > 60 or sell_pct > 60:
            confidence += 2

        if vol_ratio > 2.0:
            confidence += 2
        elif vol_ratio > 1.5:
            confidence += 1

        return signal_type, min(confidence, 10)

    def _calculate_risk_levels(self, signal_type: str, current_price: float, depth: Dict) -> tuple:
        """Calculate dynamic risk levels"""
        try:
            buy_orders = depth.get("buy", [])
            sell_orders = depth.get("sell", [])

            if not buy_orders or not sell_orders:
                # Fallback to percentage-based
                if signal_type in ["BUY", "ACCUMULATION"]:
                    return current_price, current_price * 1.0067, current_price * 0.9967, 2.0
                else:
                    return current_price, current_price * 0.9967, current_price * 1.0033, 2.0

            best_bid = buy_orders[0].get("price", current_price * 0.997)
            best_ask = sell_orders[0].get("price", current_price * 1.003)

            if signal_type in ["BUY", "ACCUMULATION"]:
                entry = current_price
                stop_loss = best_bid * 0.9997
                risk_distance = entry - stop_loss
                target = entry + (risk_distance * 2)
            else:
                entry = current_price
                stop_loss = best_ask * 1.0003
                risk_distance = stop_loss - entry
                target = entry - (risk_distance * 2)

            rr_ratio = abs((target - entry) / (entry - stop_loss)) if abs(entry - stop_loss) > 0 else 2.0
            return entry, target, stop_loss, rr_ratio

        except Exception as e:
            logger.error(f"Risk calculation error: {e}")
            # Safe fallback
            if signal_type in ["BUY", "ACCUMULATION"]:
                return current_price, current_price * 1.0067, current_price * 0.9967, 2.0
            else:
                return current_price, current_price * 0.9967, current_price * 1.0033, 2.0

    def _calculate_relative_score(self, confidence: int, inst_ratio: float, vol_ratio: float) -> float:
        """Calculate relative score for ranking"""
        score = confidence  # Base score

        # Institutional ratio bonus
        if inst_ratio > 4.0:
            score += 3
        elif inst_ratio > 2.5:
            score += 2
        elif inst_ratio > 1.5:
            score += 1

        # Volume ratio bonus
        if vol_ratio > 2.0:
            score += 2
        elif vol_ratio > 1.5:
            score += 1

        return round(score, 2)

class TradingSystem:
    """Main trading system orchestrator"""

    def __init__(self, database: DatabaseManager, monitor: SystemMonitor):
        self.database = database
        self.monitor = monitor
        self.websocket_manager = WebSocketManager()
        self.signal_generator = SignalGenerator()
        self.circuit_breaker = CircuitBreaker()

        self.user_sessions: Dict[str, Dict] = {}
        self.total_signals = 0
        self.last_signal_time = 0
        self.monitored_symbols: Set[str] = set()
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initialize the trading system"""
        logger.info("🚀 Initializing Trading System...")

        # Start processing task
        self._processing_task = asyncio.create_task(self._process_ticks())

        logger.info("✅ Trading System initialized")

    async def start_monitoring(self, user_id: str, symbols: List[str]) -> bool:
        """Start monitoring symbols for a user"""
        try:
            # Get user's API credentials
            user_creds = await self.database.get_user_credentials(user_id)
            if not user_creds:
                logger.error(f"No credentials found for user {user_id}")
                return False

            # Connect WebSocket if not connected
            if not self.websocket_manager.is_connected:
                success = await self.websocket_manager.connect(
                    user_creds["api_key"],
                    user_creds["access_token"]
                )
                if not success:
                    return False

            # Validate and map symbols to tokens
            token_map = await self._validate_and_map_symbols(symbols)
            if not token_map:
                return False

            # Store user session
            self.user_sessions[user_id] = {
                "symbols": symbols,
                "token_map": token_map,
                "start_time": time.time()
            }

            self.monitored_symbols.update(symbols)
            self.websocket_manager.monitoring_symbols.update(symbols)
            self.websocket_manager.token_map.update(token_map)

            # Subscribe to new tokens if needed
            await self._subscribe_to_tokens(list(token_map.values()))

            logger.info(f"✅ Started monitoring {len(symbols)} symbols for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to start monitoring for user {user_id}: {e}")
            return False

    async def stop_monitoring(self, user_id: str):
        """Stop monitoring for a user"""
        try:
            if user_id in self.user_sessions:
                session = self.user_sessions[user_id]

                # Remove user's symbols from monitoring
                for symbol in session["symbols"]:
                    self.monitored_symbols.discard(symbol)
                    self.websocket_manager.monitoring_symbols.discard(symbol)

                # Remove session
                del self.user_sessions[user_id]

                # Disconnect WebSocket if no users are monitoring
                if not self.user_sessions:
                    await self.websocket_manager.disconnect()

                logger.info(f"✅ Stopped monitoring for user {user_id}")

        except Exception as e:
            logger.error(f"Failed to stop monitoring for user {user_id}: {e}")

    async def validate_symbols(self, symbols: List[str]) -> List[str]:
        """Validate symbols against available instruments"""
        # Mock validation - in real implementation, check against Kite instruments
        valid_symbols = []
        common_symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN", "ICICIBANK", "KOTAKBANK"]

        for symbol in symbols:
            if symbol.upper() in common_symbols:
                valid_symbols.append(symbol.upper())

        return valid_symbols

    async def _validate_and_map_symbols(self, symbols: List[str]) -> Dict[str, int]:
        """Validate symbols and map to instrument tokens"""
        # Mock token mapping - in real implementation, fetch from Kite
        token_map = {}
        mock_tokens = {
            "RELIANCE": 738561,
            "TCS": 2953217,
            "INFY": 408065,
            "HDFCBANK": 340129,
            "SBIN": 304521,
            "ICICIBANK": 494031,
            "KOTAKBANK": 1271122
        }

        for symbol in symbols:
            if symbol in mock_tokens:
                token_map[symbol] = mock_tokens[symbol]

        return token_map

    async def _subscribe_to_tokens(self, tokens: List[int]):
        """Subscribe to WebSocket tokens"""
        if self.websocket_manager.kws and tokens:
            self.websocket_manager.kws.subscribe(tokens)
            self.websocket_manager.kws.set_mode(self.websocket_manager.kws.MODE_FULL, tokens)
            logger.info(f"📡 Subscribed to {len(tokens)} tokens")

    async def _process_ticks(self):
        """Process ticks from the queue"""
        while True:
            try:
                # Get tick from queue
                while not self.websocket_manager.tick_queue.empty():
                    tick = self.websocket_manager.tick_queue.get_nowait()
                    await self.processing_queue.put(tick)

                # Process ticks
                processed = 0
                while not self.processing_queue.empty() and processed < 50:
                    tick = await self.processing_queue.get()
                    await self._handle_tick(tick)
                    processed += 1

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Tick processing error: {e}")
                await asyncio.sleep(1)

    async def _handle_tick(self, tick: Dict):
        """Handle individual tick"""
        try:
            instrument_token = tick.get("instrument_token")
            if not instrument_token:
                return

            # Find symbol for this token
            symbol = None
            user_id = None

            for uid, session in self.user_sessions.items():
                for sym, token in session["token_map"].items():
                    if token == instrument_token:
                        symbol = sym
                        user_id = uid
                        break
                if symbol:
                    break

            if not symbol or not user_id:
                return

            # Generate signal
            signal = await self.signal_generator.generate_signal(symbol, tick, user_id)

            if signal and signal.confidence_score >= config.trading.min_confidence:
                # Save to database
                await self.database.save_signal(asdict(signal))

                self.total_signals += 1
                self.last_signal_time = signal.timestamp_created

                logger.info(f"🎯 Signal generated: {symbol} - {signal.signal_type} ({signal.confidence_score}/10)")

        except Exception as e:
            logger.error(f"Error handling tick: {e}")

    async def get_recent_signals(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get recent signals for a user"""
        try:
            signals = await self.database.get_signals(
                user_id=user_id,
                limit=limit,
                hours_back=24
            )
            return signals
        except Exception as e:
            logger.error(f"Failed to get recent signals: {e}")
            return []

    async def get_unbroadcasted_signals(self) -> List[Dict]:
        """Get signals that haven't been broadcasted yet"""
        try:
            signals = await self.database.get_unbroadcasted_signals(limit=100)
            return signals
        except Exception as e:
            logger.error(f"Failed to get unbroadcasted signals: {e}")
            return []

    async def mark_signal_broadcasted(self, signal_id: str):
        """Mark signal as broadcasted"""
        try:
            await self.database.mark_signal_broadcasted(signal_id)
        except Exception as e:
            logger.error(f"Failed to mark signal as broadcasted: {e}")

    async def get_user_status(self, user_id: str) -> Dict:
        """Get user's current trading status"""
        try:
            session = self.user_sessions.get(user_id)
            if not session:
                return {
                    "monitoring_active": False,
                    "symbols": [],
                    "total_signals": 0,
                    "last_signal_time": None
                }

            return {
                "monitoring_active": True,
                "symbols": session["symbols"],
                "total_signals": await self.database.get_user_signal_count(user_id),
                "last_signal_time": await self.database.get_user_last_signal_time(user_id),
                "websocket_health": asdict(self.websocket_manager.get_health())
            }

        except Exception as e:
            logger.error(f"Failed to get user status: {e}")
            return {"monitoring_active": False, "error": str(e)}

    def get_websocket_health(self) -> WebSocketHealth:
        """Get WebSocket health status"""
        return self.websocket_manager.get_health()

    async def shutdown(self):
        """Shutdown the trading system"""
        logger.info("🛑 Shutting down Trading System...")

        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        await self.websocket_manager.disconnect()

        logger.info("✅ Trading System shutdown complete")