import streamlit as st
from kiteconnect import KiteConnect
import threading
import time
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import numpy as np
import copy
from pytz import timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

# ====== TIMEZONE (INDIA) ====== #
INDIA_TZ = timezone("Asia/Kolkata")

# ====== PAGE CONFIG ====== #
st.set_page_config(
    page_title="Institutional Sniper v3 (Pro)",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== CUSTOM CSS ====== #
st.markdown("""
<style>
    /* Main Background */
    .stApp { background-color: #0e1117; font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
    
    /* Metric Cards */
    div[data-testid="stMetric"] { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #2e3245; }
    
    /* Signal Card - BUY CALL (Green) */
    .buy-card {
        background: linear-gradient(145deg, #0f2e1b, #0a1f12);
        border: 1px solid #00ff41;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 0 15px rgba(0, 255, 65, 0.2);
        margin-bottom: 15px;
        animation: pulse-green 2s infinite;
    }
    
    /* Signal Card - BUY PUT (Red) */
    .sell-card {
        background: linear-gradient(145deg, #381010, #260a0a);
        border: 1px solid #ff2b2b;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 0 15px rgba(255, 43, 43, 0.2);
        margin-bottom: 15px;
        animation: pulse-red 2s infinite;
    }
    
    /* Signal Card - NEUTRAL (Grey) */
    .neutral-card {
        background-color: #161924;
        border: 1px solid #30364d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        opacity: 0.7;
    }
    
    /* Typography */
    .signal-title { font-size: 24px; font-weight: 800; margin-bottom: 5px; }
    .signal-sub { font-size: 14px; color: #b0b3c5; letter-spacing: 1px; }
    .price-large { font-size: 32px; font-weight: 700; color: white; }
    .stat-label { font-size: 12px; color: #808495; }
    .stat-val { font-size: 16px; font-weight: 600; }
    
    /* Animations */
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(0, 255, 65, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(0, 255, 65, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 255, 65, 0); }
    }
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(255, 43, 43, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 43, 43, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 43, 43, 0); }
    }
</style>
""", unsafe_allow_html=True)


# ====== ENUMS & CONFIG ====== #
class SignalType(Enum):
    NEUTRAL = "NEUTRAL"
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"
    MOMENTUM_BUY = "BUY CALL"
    MOMENTUM_SELL = "BUY PUT"


@dataclass
class Config:
    spread_split_enabled: bool = True
    bucket_seconds: int = 5
    rolling_window_seconds: int = 300
    bar_seconds: int = 300
    
    # --- STRATEGY THRESHOLDS (Trend-Ride) ---
    rvol_threshold: float = 1.5
    trend_confirmation_bars: int = 3
    breakout_lookback_bars: int = 4
    pcr_min_bullish: float = 0.7
    pcr_max_bearish: float = 1.3
    
    # --- RVOL CONFIGURATION ---
    rvol_lookback_bars: int = 750  # 750 x 5min = 62.5 hours ≈ 10 trading days
    bars_to_keep: int = 20


CONFIG = Config()


# ====== DATA STRUCTURES ====== #
@dataclass
class ClassifiedTick:
    timestamp: datetime
    ltp: float
    bid: float
    ask: float
    volume: int
    aggr_buy: int = 0
    aggr_sell: int = 0
    neutral: int = 0


@dataclass
class Bucket5s:
    start_time: datetime
    end_time: datetime
    aggr_buy: int = 0
    aggr_sell: int = 0
    total_volume: int = 0
    vwap: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    tick_count: int = 0
    cumulative_value: float = 0.0


@dataclass
class Bar5m:
    start_time: datetime
    end_time: datetime
    aggr_buy_total: int = 0
    aggr_sell_total: int = 0
    agg_imbalance: float = 0.0
    total_volume: int = 0
    rvol: float = 0.0
    vwap: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    tick_count: int = 0
    price_range: float = 0.0
    cumulative_delta: int = 0
    price_change_pct: float = 0.0


@dataclass
class SignalResult:
    signal_type: SignalType = SignalType.NEUTRAL
    score: int = 0
    normalized_score: float = 0.0
    strength: str = "WEAK"
    diagnostics: Dict = field(default_factory=dict)
    score_breakdown: Dict = field(default_factory=dict)
    entry_suggestion: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    timestamp: datetime = None


# ====== HELPERS ====== #
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


def now_ist():
    return datetime.now(INDIA_TZ)


def get_expiry_date(expiry):
    if expiry is None:
        return None
    if hasattr(expiry, 'date'):
        return expiry.date()
    return expiry


def current_bucket_start(seconds=5):
    now = now_ist()
    bucket_num = now.second // seconds
    return now.replace(second=bucket_num * seconds, microsecond=0)


def current_bar_start(seconds=300):
    now = now_ist()
    minute_slot = (now.minute // (seconds // 60)) * (seconds // 60)
    return now.replace(minute=minute_slot, second=0, microsecond=0)


# ====== ROLLING VOLUME TRACKER (NEW) ====== #
class RollingVolumeTracker:
    """
    Maintains a rolling window of bar volumes for accurate RVol calculation.
    Seeded with historical data and updated in real-time.
    """
    def __init__(self, lookback_bars: int = 750):
        self.lookback_bars = lookback_bars
        self.volume_history: deque = deque(maxlen=lookback_bars)
        self.sum_volume: float = 0.0
        self.is_seeded: bool = False
        self.lock = threading.Lock()
    
    def seed_from_historical(self, volumes: List[int]):
        """Initialize with historical volume data"""
        with self.lock:
            self.volume_history.clear()
            self.sum_volume = 0.0
            
            for vol in volumes[-self.lookback_bars:]:
                self.volume_history.append(vol)
                self.sum_volume += vol
            
            self.is_seeded = len(self.volume_history) > 0
    
    def add_bar_volume(self, volume: int):
        """Add a new completed bar's volume"""
        with self.lock:
            if len(self.volume_history) >= self.lookback_bars:
                # Remove oldest volume from sum
                oldest = self.volume_history[0]
                self.sum_volume -= oldest
            
            self.volume_history.append(volume)
            self.sum_volume += volume
    
    def get_avg_volume(self) -> float:
        """Get current rolling average volume"""
        with self.lock:
            if len(self.volume_history) == 0:
                return 0.0
            return self.sum_volume / len(self.volume_history)
    
    def get_rvol(self, current_volume: int) -> float:
        """Calculate RVol for current bar"""
        avg = self.get_avg_volume()
        if avg <= 0:
            return 0.0
        return current_volume / avg
    
    def get_stats(self) -> Dict:
        """Get diagnostic stats"""
        with self.lock:
            return {
                'bars_in_history': len(self.volume_history),
                'total_lookback': self.lookback_bars,
                'avg_volume': self.get_avg_volume(),
                'is_seeded': self.is_seeded,
                'fill_pct': (len(self.volume_history) / self.lookback_bars) * 100
            }


# ====== TICK CLASSIFIER ====== #
class TickClassifier:
    def __init__(self):
        self.prev_ltp = {}
    
    def classify_tick(self, symbol: str, ltp: float, bid: float, ask: float, 
                      volume: int, timestamp: datetime) -> ClassifiedTick:
        result = ClassifiedTick(
            timestamp=timestamp, ltp=ltp, bid=bid, ask=ask, volume=volume
        )
        
        if volume <= 0:
            result.neutral = 0
            return result
        
        if bid > 0 and ask > 0 and ask >= bid:
            spread = ask - bid
            if ltp >= ask:
                result.aggr_buy = volume
            elif ltp <= bid:
                result.aggr_sell = volume
            elif spread > 0:
                buy_ratio = (ltp - bid) / spread
                result.aggr_buy = int(volume * buy_ratio)
                result.aggr_sell = int(volume * (1 - buy_ratio))
                result.neutral = volume - result.aggr_buy - result.aggr_sell
            else:
                result.aggr_buy = volume // 2
                result.aggr_sell = volume - result.aggr_buy
        else:
            prev = self.prev_ltp.get(symbol, ltp)
            if ltp > prev:
                result.aggr_buy = volume
            elif ltp < prev:
                result.aggr_sell = volume
            else:
                result.neutral = volume
        
        self.prev_ltp[symbol] = ltp
        return result


# ====== 5-SECOND BUCKET AGGREGATOR ====== #
class BucketAggregator:
    def __init__(self, bucket_seconds: int = 5, max_buckets: int = 60):
        self.bucket_seconds = bucket_seconds
        self.max_buckets = max_buckets
        self.buckets: deque = deque(maxlen=max_buckets)
        self.current_bucket: Optional[Bucket5s] = None
        self.lock = threading.Lock()
    
    def _get_bucket_start(self, ts: datetime) -> datetime:
        bucket_num = ts.second // self.bucket_seconds
        return ts.replace(second=bucket_num * self.bucket_seconds, microsecond=0)
    
    def append_tick(self, tick: ClassifiedTick) -> Optional[Bucket5s]:
        with self.lock:
            bucket_start = self._get_bucket_start(tick.timestamp)
            bucket_end = bucket_start + timedelta(seconds=self.bucket_seconds)
            
            completed = None
            
            if self.current_bucket is None or bucket_start != self.current_bucket.start_time:
                if self.current_bucket is not None:
                    if self.current_bucket.tick_count > 0 and self.current_bucket.total_volume > 0:
                        self.current_bucket.vwap = self.current_bucket.cumulative_value / self.current_bucket.total_volume
                    self.buckets.append(self.current_bucket)
                    completed = self.current_bucket
                
                self.current_bucket = Bucket5s(
                    start_time=bucket_start, end_time=bucket_end,
                    open=tick.ltp, high=tick.ltp, low=tick.ltp, close=tick.ltp
                )
            
            b = self.current_bucket
            b.aggr_buy += tick.aggr_buy
            b.aggr_sell += tick.aggr_sell
            b.total_volume += tick.volume
            b.cumulative_value += tick.ltp * tick.volume
            b.high = max(b.high, tick.ltp)
            b.low = min(b.low, tick.ltp)
            b.close = tick.ltp
            b.tick_count += 1
            
            return completed
    
    def get_rolling_stats(self, window_seconds: int = 300) -> Dict:
        with self.lock:
            num_buckets = window_seconds // self.bucket_seconds
            recent = list(self.buckets)[-num_buckets:] if self.buckets else []
            
            if not recent:
                return {}
            
            total_ab = sum(b.aggr_buy for b in recent)
            total_as = sum(b.aggr_sell for b in recent)
            total_vol = sum(b.total_volume for b in recent)
            
            return {
                'aggr_buy': total_ab,
                'aggr_sell': total_as,
                'total_volume': total_vol,
                'imbalance': (total_ab - total_as) / total_vol if total_vol > 0 else 0,
                'high': max(b.high for b in recent),
                'low': min(b.low for b in recent),
                'open': recent[0].open,
                'close': recent[-1].close,
                'bucket_count': len(recent)
            }
    
    def get_current_bucket(self) -> Optional[Bucket5s]:
        with self.lock:
            return copy.copy(self.current_bucket) if self.current_bucket else None


# ====== 5-MINUTE BAR AGGREGATOR (UPDATED) ====== #
class BarAggregator:
    def __init__(self, bar_seconds: int = 300, max_bars: int = 50):
        self.bar_seconds = bar_seconds
        self.max_bars = max_bars
        self.bars: deque = deque(maxlen=max_bars)
        self.current_bar: Optional[Bar5m] = None
        self.volume_tracker: RollingVolumeTracker = RollingVolumeTracker(CONFIG.rvol_lookback_bars)
        self.lock = threading.Lock()
    
    def _get_bar_start(self, ts: datetime) -> datetime:
        minute_slot = (ts.minute // (self.bar_seconds // 60)) * (self.bar_seconds // 60)
        return ts.replace(minute=minute_slot, second=0, microsecond=0)
    
    def seed_volume_history(self, volumes: List[int]):
        """Seed the volume tracker with historical data"""
        self.volume_tracker.seed_from_historical(volumes)
    
    def append_bucket(self, bucket: Bucket5s) -> Optional[Bar5m]:
        with self.lock:
            bar_start = self._get_bar_start(bucket.start_time)
            bar_end = bar_start + timedelta(seconds=self.bar_seconds)
            
            completed = None
            
            if self.current_bar is None or bar_start != self.current_bar.start_time:
                if self.current_bar is not None:
                    self._finalize_bar(self.current_bar)
                    self.bars.append(self.current_bar)
                    # Add completed bar's volume to rolling tracker
                    self.volume_tracker.add_bar_volume(self.current_bar.total_volume)
                    completed = self.current_bar
                
                self.current_bar = Bar5m(
                    start_time=bar_start, end_time=bar_end,
                    open=bucket.open, high=bucket.high, low=bucket.low, close=bucket.close
                )
            
            bar = self.current_bar
            bar.aggr_buy_total += bucket.aggr_buy
            bar.aggr_sell_total += bucket.aggr_sell
            bar.total_volume += bucket.total_volume
            bar.high = max(bar.high, bucket.high)
            bar.low = min(bar.low, bucket.low)
            bar.close = bucket.close
            bar.tick_count += bucket.tick_count
            bar.cumulative_delta = bar.aggr_buy_total - bar.aggr_sell_total
            bar.vwap = (bar.high + bar.low + bar.close) / 3
            
            return completed
    
    def _finalize_bar(self, bar: Bar5m):
        total = bar.aggr_buy_total + bar.aggr_sell_total
        if total > 0:
            bar.agg_imbalance = (bar.aggr_buy_total - bar.aggr_sell_total) / total
        
        # Calculate RVol using rolling tracker
        bar.rvol = self.volume_tracker.get_rvol(bar.total_volume)
        
        bar.price_range = bar.high - bar.low
        if bar.open > 0:
            bar.price_change_pct = (bar.close - bar.open) / bar.open
    
    def get_current_bar_with_rvol(self) -> Optional[Bar5m]:
        """Get current bar with live RVol calculation"""
        with self.lock:
            if self.current_bar:
                bar_copy = copy.copy(self.current_bar)
                # Calculate live RVol for current forming bar
                bar_copy.rvol = self.volume_tracker.get_rvol(bar_copy.total_volume)
                self._finalize_bar_partial(bar_copy)
                return bar_copy
            return None
    
    def _finalize_bar_partial(self, bar: Bar5m):
        """Finalize bar stats without adding to volume history"""
        total = bar.aggr_buy_total + bar.aggr_sell_total
        if total > 0:
            bar.agg_imbalance = (bar.aggr_buy_total - bar.aggr_sell_total) / total
        bar.price_range = bar.high - bar.low
        if bar.open > 0:
            bar.price_change_pct = (bar.close - bar.open) / bar.open
    
    def get_recent_bars(self, n: int = 3) -> List[Bar5m]:
        with self.lock:
            return list(self.bars)[-n:]
    
    def get_current_bar(self) -> Optional[Bar5m]:
        """Legacy method - use get_current_bar_with_rvol for accurate RVol"""
        return self.get_current_bar_with_rvol()
    
    def get_volume_tracker_stats(self) -> Dict:
        return self.volume_tracker.get_stats()


# ====== PATTERN DETECTOR ====== #
class PatternDetector:
    def __init__(self, config: Config = CONFIG):
        self.config = config
    
    def detect(self, bars: List[Bar5m], pcr: float, current_bar: Bar5m, current_vwap: float) -> SignalResult:
        result = SignalResult(timestamp=now_ist())
        
        if len(bars) < self.config.breakout_lookback_bars:
            return result
        
        lookback = self.config.breakout_lookback_bars
        trend_bars = self.config.trend_confirmation_bars
        
        # --- 1. TREND CONFIRMATION ---
        last_n_bars = bars[-trend_bars:]
        bullish_trend = all(b.close > current_vwap for b in last_n_bars)
        bearish_trend = all(b.close < current_vwap for b in last_n_bars)
        
        # --- 2. BREAKOUT TRIGGER ---
        consolidation_bars = bars[-lookback:]
        recent_high = max(b.high for b in consolidation_bars)
        recent_low = min(b.low for b in consolidation_bars)
        
        is_breaking_out_up = current_bar.close > recent_high
        is_breaking_out_down = current_bar.close < recent_low
        
        # --- 3. VOLUME FUEL ---
        is_volume_good = current_bar.rvol >= self.config.rvol_threshold
        
        # --- 4. PCR SAFETY ---
        is_pcr_bullish = pcr > self.config.pcr_min_bullish
        is_pcr_bearish = pcr < self.config.pcr_max_bearish
        
        diagnostics = {
            "Trend": "Bullish" if bullish_trend else "Bearish" if bearish_trend else "Choppy",
            "RVol": f"{current_bar.rvol:.2f}x",
            "PCR": f"{pcr:.2f}",
            "Breakout Lvl": f"{recent_high if bullish_trend else recent_low}",
            "Current": f"{current_bar.close}"
        }
        result.diagnostics = diagnostics
        
        # ====== DECISION MATRIX ====== #
        if bullish_trend and is_breaking_out_up and is_volume_good and is_pcr_bullish:
            result.signal_type = SignalType.MOMENTUM_BUY
            result.score = 13
            result.strength = "CONFIRMED"
            result.stop_loss = recent_low
            result.target = current_bar.close + (current_bar.close - recent_low) * 2
            result.entry_suggestion = current_bar.close
            
        elif bearish_trend and is_breaking_out_down and is_volume_good and is_pcr_bearish:
            result.signal_type = SignalType.MOMENTUM_SELL
            result.score = 13
            result.strength = "CONFIRMED"
            result.stop_loss = recent_high
            result.target = current_bar.close - (recent_high - current_bar.close) * 2
            result.entry_suggestion = current_bar.close
            
        else:
            result.signal_type = SignalType.NEUTRAL
            result.strength = "WAITING"
            
        return result


# ====== SCORER ====== #
class Scorer:
    def __init__(self, config: Config = CONFIG):
        self.config = config
    
    def score(self, bar: Bar5m, signal_result: SignalResult, pcr: float) -> SignalResult:
        return signal_result


# ====== PCR FETCHER ====== #
class PCRFetcher:
    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}
        self.lock = threading.Lock()
    
    def get_pcr(self, kite, symbol: str, spot_price: float) -> float:
        with self.lock:
            if symbol in self.cache:
                if now_ist() < self.cache_expiry.get(symbol, now_ist()):
                    return self.cache[symbol]
        
        try:
            all_opts = [
                i for i in kite.instruments("NFO")
                if i.get("segment") == "NFO-OPT" and i.get("name") == symbol
            ]
            
            if not all_opts:
                return 1.0
            
            today = now_ist().date()
            expiries = sorted({
                get_expiry_date(i["expiry"])
                for i in all_opts
                if get_expiry_date(i["expiry"]) is not None and get_expiry_date(i["expiry"]) >= today
            })
            
            if not expiries:
                return 1.0
            
            nearest = expiries[0]
            scoped = [i for i in all_opts if get_expiry_date(i["expiry"]) == nearest]
            
            ce_opts = {i['strike']: i for i in scoped if i["instrument_type"] == "CE"}
            pe_opts = {i['strike']: i for i in scoped if i["instrument_type"] == "PE"}
            
            relevant_strikes = [s for s in set(ce_opts.keys()) | set(pe_opts.keys())
                              if spot_price * 0.9 <= s <= spot_price * 1.1]
            
            if not relevant_strikes:
                return 1.0
            
            ce_syms = [f"NFO:{ce_opts[s]['tradingsymbol']}" for s in relevant_strikes if s in ce_opts]
            pe_syms = [f"NFO:{pe_opts[s]['tradingsymbol']}" for s in relevant_strikes if s in pe_opts]
            
            ce_oi_total = 0
            pe_oi_total = 0
            
            all_syms = ce_syms + pe_syms
            for i in range(0, len(all_syms), 50):
                batch = all_syms[i:i+50]
                try:
                    quotes = kite.quote(batch)
                    for k, q in quotes.items():
                        oi = safe_int(q.get('oi', 0))
                        if "CE" in k:
                            ce_oi_total += oi
                        if "PE" in k:
                            pe_oi_total += oi
                    time.sleep(0.1)
                except:
                    pass
            
            pcr = pe_oi_total / ce_oi_total if ce_oi_total > 0 else 1.0
            
            with self.lock:
                self.cache[symbol] = pcr
                self.cache_expiry[symbol] = now_ist() + timedelta(minutes=3)
            
            return round(pcr, 2)
            
        except:
            return self.cache.get(symbol, 1.0)


# ====== SYMBOL STATE (UPDATED) ====== #
@dataclass
class SymbolState:
    symbol: str
    token: int
    ltp: float = 0.0
    vwap: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume: int = 0
    
    tick_classifier: TickClassifier = field(default_factory=TickClassifier)
    bucket_aggregator: BucketAggregator = field(default_factory=BucketAggregator)
    bar_aggregator: BarAggregator = field(default_factory=BarAggregator)
    
    pcr: float = 1.0
    
    signal_result: SignalResult = field(default_factory=SignalResult)
    last_update: datetime = None


# ====== MANAGER ====== #
class SniperManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.logs = deque(maxlen=300)
        self.symbol_states: Dict[str, SymbolState] = {}
        self.active_symbols = []
        self.is_running = False
        self.stop_event = threading.Event()
        self.kite = None
        self.last_beat = 0.0
        
        self.pattern_detector = PatternDetector()
        self.scorer = Scorer()
        self.pcr_fetcher = PCRFetcher()

    def log(self, msg, level="INFO"):
        ts = now_ist().strftime("%H:%M:%S")
        with self.lock:
            self.logs.appendleft(f"[{ts}] {level}: {msg}")

    def get_logs_snapshot(self):
        with self.lock:
            return list(self.logs)
    
    def get_symbol_state(self, symbol: str) -> Optional[SymbolState]:
        with self.lock:
            return self.symbol_states.get(symbol)
    
    def get_all_states_snapshot(self) -> Dict[str, Dict]:
        with self.lock:
            result = {}
            for sym, state in self.symbol_states.items():
                current_bar = state.bar_aggregator.get_current_bar()
                vol_stats = state.bar_aggregator.get_volume_tracker_stats()
                result[sym] = {
                    'ltp': state.ltp,
                    'vwap': state.vwap,
                    'bid': state.bid,
                    'ask': state.ask,
                    'volume': state.volume,
                    'pcr': state.pcr,
                    'signal': state.signal_result,
                    'current_bar': current_bar,
                    'recent_bars': state.bar_aggregator.get_recent_bars(3),
                    'rolling_stats': state.bucket_aggregator.get_rolling_stats(),
                    'volume_tracker_stats': vol_stats,
                    'last_update': state.last_update
                }
            return result
    
    def get_ranked_signals(self) -> List[Tuple[str, SignalResult]]:
        with self.lock:
            signals = [
                (sym, state.signal_result)
                for sym, state in self.symbol_states.items()
                if state.signal_result.signal_type != SignalType.NEUTRAL
            ]
            return sorted(signals, key=lambda x: x[1].score, reverse=True)


def get_fresh_manager():
    return SniperManager()


if 'manager' not in st.session_state:
    st.session_state.manager = get_fresh_manager()

manager = st.session_state.manager


# ====== KITE HELPERS (UPDATED) ====== #
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


def fetch_historical_volumes(kite, token, lookback_bars: int = 750) -> List[int]:
    """
    Fetch historical 5-minute bar volumes for seeding the RVol tracker.
    750 bars = ~10 trading days of 5-min data
    """
    try:
        now = now_ist().replace(tzinfo=None)
        # Fetch extra days to account for market holidays
        from_date = now - timedelta(days=15)
        
        candles = kite.historical_data(token, from_date, now, "5minute")
        df = pd.DataFrame(candles)
        
        if df.empty:
            manager.log(f"No historical data for token {token}", "WARNING")
            return []
        
        volumes = df['volume'].tolist()
        
        # Return last N bars
        return volumes[-lookback_bars:] if len(volumes) >= lookback_bars else volumes
        
    except Exception as e:
        manager.log(f"Historical volume error: {e}", "ERROR")
        return []


# ====== WORKER THREAD (UPDATED) ====== #
def sniper_worker(kite, mgr: SniperManager):
    mgr.is_running = True
    mgr.log("Worker started - RVol using 750-bar rolling average", "INFO")
    
    last_pcr_update = 0
    last_bar_check = current_bar_start()
    
    while not mgr.stop_event.is_set():
        try:
            now_ts = time.time()
            mgr.last_beat = now_ts
            
            with mgr.lock:
                active_list = list(mgr.active_symbols)
            
            if not active_list:
                time.sleep(1)
                continue
            
            keys = [f"NSE:{row['symbol']}" for row in active_list]
            try:
                quotes = kite.quote(keys)
            except Exception as e:
                mgr.log(f"Quote error: {e}", "WARNING")
                time.sleep(2)
                continue
            
            update_pcr = (now_ts - last_pcr_update) > 180
            if update_pcr:
                last_pcr_update = now_ts
            
            for row in active_list:
                sym = row['symbol']
                key = f"NSE:{sym}"
                q = quotes.get(key)
                if not q:
                    continue
                
                state = mgr.get_symbol_state(sym)
                if not state:
                    continue
                
                ltp = safe_float(q.get('last_price'))
                depth = q.get('depth', {})
                buy_depth = depth.get('buy', [{}])
                sell_depth = depth.get('sell', [{}])
                
                bid = safe_float(buy_depth[0].get('price')) if buy_depth else 0
                ask = safe_float(sell_depth[0].get('price')) if sell_depth else 0
                volume = safe_int(q.get('volume'))
                vwap = safe_float(q.get('average_price'), ltp)
                
                vol_delta = max(0, volume - state.volume) if state.volume > 0 else 0
                
                state.ltp = ltp
                state.vwap = vwap
                state.bid = bid
                state.ask = ask
                state.volume = volume
                state.last_update = now_ist()
                
                if vol_delta > 0:
                    tick = state.tick_classifier.classify_tick(
                        sym, ltp, bid, ask, vol_delta, now_ist()
                    )
                    
                    completed_bucket = state.bucket_aggregator.append_tick(tick)
                    
                    if completed_bucket:
                        completed_bar = state.bar_aggregator.append_bucket(completed_bucket)
                        
                        if completed_bar:
                            vol_stats = state.bar_aggregator.get_volume_tracker_stats()
                            mgr.log(f"{sym}: Bar closed | Vol: {completed_bar.total_volume:,} | RVol: {completed_bar.rvol:.2f}x | Avg based on {vol_stats['bars_in_history']} bars", "DEBUG")
                
                if update_pcr:
                    state.pcr = mgr.pcr_fetcher.get_pcr(kite, sym, ltp)
                
                current_bar = state.bar_aggregator.get_current_bar()
                recent_bars = state.bar_aggregator.get_recent_bars(15)
                
                if current_bar:
                    signal = mgr.pattern_detector.detect(recent_bars, state.pcr, current_bar, vwap)
                    state.signal_result = signal
                    
                    if signal.score == 13:
                        mgr.log(f"🎯 {sym}: {signal.signal_type.value} @ {ltp} | RVol: {current_bar.rvol:.2f}x", "SIGNAL")
            
            new_bar_start = current_bar_start()
            if new_bar_start != last_bar_check:
                last_bar_check = new_bar_start
                mgr.log(f"=== NEW BAR: {new_bar_start.strftime('%H:%M')} ===", "INFO")
            
            time.sleep(1)
            
        except Exception as e:
            mgr.log(f"Error: {e}", "ERROR")
            time.sleep(2)
    
    mgr.is_running = False
    mgr.log("Worker stopped", "INFO")


# ====== RENDER MODERN CARD (UPDATED) ====== #
def render_modern_card(sym: str, data: Dict):
    signal = data.get('signal', SignalResult())
    current_bar = data.get('current_bar')
    ltp = data.get('ltp', 0.0)
    pcr = data.get('pcr', 1.0)
    vol_stats = data.get('volume_tracker_stats', {})
    
    if not current_bar:
        rvol = 0.0
    else:
        rvol = current_bar.rvol
    
    bars_in_history = vol_stats.get('bars_in_history', 0)
    fill_pct = vol_stats.get('fill_pct', 0)
    
    card_class = "neutral-card"
    text_color = "#b0b3c5"
    signal_text = "NEUTRAL"
    
    if signal.signal_type == SignalType.MOMENTUM_BUY:
        card_class = "buy-card"
        text_color = "#00ff41"
        signal_text = "BUY CALL 🚀"
    elif signal.signal_type == SignalType.MOMENTUM_SELL:
        card_class = "sell-card"
        text_color = "#ff2b2b"
        signal_text = "BUY PUT 📉"
    
    html_content = f"""
    <div class="{card_class}">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div class="signal-title" style="color: {text_color};">{sym}</div>
                <div class="signal-sub">{signal_text}</div>
            </div>
            <div class="price-large" style="color: white;">₹{ltp:.2f}</div>
        </div>
        <hr style="border-color: rgba(255,255,255,0.1); margin: 15px 0;">
        <div style="display:flex; justify-content:space-between;">
            <div>
                <div class="stat-label">RVol</div>
                <div class="stat-val" style="color: {'#00ff41' if rvol > 1.5 else '#fff'};">{rvol:.2f}x</div>
            </div>
            <div>
                <div class="stat-label">PCR</div>
                <div class="stat-val">{pcr:.2f}</div>
            </div>
            <div>
                <div class="stat-label">Entry</div>
                <div class="stat-val">{signal.entry_suggestion if signal.entry_suggestion > 0 else '-'}</div>
            </div>
            <div>
                <div class="stat-label">Stop Loss</div>
                <div class="stat-val">{signal.stop_loss if signal.stop_loss > 0 else '-'}</div>
            </div>
        </div>
        <div style="margin-top: 10px; font-size: 11px; color: #666;">
            Vol History: {bars_in_history}/750 bars ({fill_pct:.1f}% filled)
        </div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)


# ====== SIDEBAR ====== #
st.sidebar.title("🔐 1. Authenticate")

login_mode = st.sidebar.radio("Login Method", ["Request Token", "Access Token (Direct)"])

if "kite_connected" not in st.session_state:
    st.session_state.kite_connected = False

if not st.session_state.kite_connected:
    api_key = st.sidebar.text_input("API Key", value=st.session_state.get("api_key", ""))
    
    if login_mode == "Request Token":
        api_sec = st.sidebar.text_input("API Secret", value=st.session_state.get("api_sec", ""), type="password")
        if api_key and api_sec:
            try:
                temp_kite = KiteConnect(api_key=api_key)
                url = temp_kite.login_url()
                st.sidebar.markdown(f"""
                <a href="{url}" target="_blank" style="
                    display: block; text-align: center; background-color: #FF5722; color: white; padding: 10px;
                    text-decoration: none; border-radius: 5px; font-weight: bold; margin-bottom: 10px;">
                    👉 Click Here to Login & Get Token
                </a>
                """, unsafe_allow_html=True)
            except:
                pass
        req_token = st.sidebar.text_input("Paste Request Token", value=st.session_state.get("req_token", ""))
        
        if st.sidebar.button("🔌 Connect with Request Token", type="primary"):
            try:
                kite = KiteConnect(api_key=api_key)
                data = kite.generate_session(req_token, api_secret=api_sec)
                kite.set_access_token(data["access_token"])
                manager.kite = kite
                st.session_state.api_key = api_key
                st.session_state.api_sec = api_sec
                st.session_state.req_token = req_token
                st.session_state.kite_connected = True
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Login Failed: {e}")

    else:
        acc_token = st.sidebar.text_input("Paste Access Token")
        if st.sidebar.button("🔌 Connect with Access Token", type="primary"):
            try:
                kite = KiteConnect(api_key=api_key)
                kite.set_access_token(acc_token)
                manager.kite = kite
                st.session_state.api_key = api_key
                st.session_state.kite_connected = True
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Connection Failed: {e}")

else:
    st.sidebar.success("✅ Authenticated")
    st.sidebar.divider()
    
    st.sidebar.title("⚙️ 2. Strategy")
    default_syms = "RELIANCE, HDFCBANK, INFY, ICICIBANK, TCS, SBIN, TATASTEEL"
    sym_input = st.sidebar.text_area("Watchlist", default_syms, height=150)
    
    # RVol Configuration
    st.sidebar.subheader("📊 RVol Settings")
    st.sidebar.caption(f"Current: {CONFIG.rvol_lookback_bars} bars (~{CONFIG.rvol_lookback_bars * 5 / 60:.1f} hours)")
    
    if not manager.is_running:
        if st.sidebar.button("🚀 START SNIPER", type="primary"):
            with st.spinner("Initializing Strategy with 750-bar Volume History..."):
                manager.active_symbols = []
                valid = []
                syms = [s.strip().upper() for s in sym_input.split(",") if s.strip()]
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, s in enumerate(syms):
                    status_text.text(f"Loading {s}... ({idx+1}/{len(syms)})")
                    
                    tkn = get_instrument_token(manager.kite, s)
                    if tkn:
                        # Fetch 750 bars of historical volume
                        hist_volumes = fetch_historical_volumes(
                            manager.kite, tkn, CONFIG.rvol_lookback_bars
                        )
                        
                        state = SymbolState(symbol=s, token=tkn)
                        
                        # Seed volume tracker with historical data
                        if hist_volumes:
                            state.bar_aggregator.seed_volume_history(hist_volumes)
                            manager.log(f"{s}: Seeded with {len(hist_volumes)} historical bars", "INFO")
                        
                        manager.symbol_states[s] = state
                        valid.append({'symbol': s, 'token': tkn})
                    
                    progress_bar.progress((idx + 1) / len(syms))
                
                status_text.empty()
                progress_bar.empty()
                
                manager.active_symbols = valid
                manager.stop_event.clear()
                threading.Thread(target=sniper_worker, args=(manager.kite, manager), daemon=True).start()
                st.rerun()
    else:
        if st.sidebar.button("🛑 STOP SYSTEM"):
            manager.stop_event.set()
            time.sleep(1)
            st.rerun()
        
        # Show volume stats in sidebar
        st.sidebar.divider()
        st.sidebar.subheader("📈 Volume Stats")
        data = manager.get_all_states_snapshot()
        for sym, d in data.items():
            vol_stats = d.get('volume_tracker_stats', {})
            bars = vol_stats.get('bars_in_history', 0)
            avg = vol_stats.get('avg_volume', 0)
            st.sidebar.caption(f"{sym}: {bars}/750 bars | Avg: {avg:,.0f}")


# ====== MAIN DASHBOARD ====== #
st.title("Institutional Sniper Pro")
st.caption("Real-Time VWAP Trend + Breakout Detection | RVol: 750-bar Rolling Average")

if manager.is_running:
    data = manager.get_all_states_snapshot()
    
    signals = [d for k, d in data.items() if d['signal'].score == 13]
    others = [d for k, d in data.items() if d['signal'].score != 13]
    
    st.subheader(f"🔥 Active Signals ({len(signals)})")
    if signals:
        cols = st.columns(2)
        for i, row in enumerate(signals):
            with cols[i % 2]:
                sym_name = [k for k, v in data.items() if v == row][0]
                render_modern_card(sym_name, row)
    else:
        st.info("Scanning for Institutional Breakouts...")
        
    st.divider()
    
    st.subheader("👀 Watchlist")
    if others:
        cols = st.columns(3)
        for i, row in enumerate(others):
            with cols[i % 3]:
                sym_name = [k for k, v in data.items() if v == row][0]
                render_modern_card(sym_name, row)
    
    # Debug expander
    with st.expander("🔧 Debug Logs"):
        logs = manager.get_logs_snapshot()
        for log in logs[:50]:
            st.text(log)
    
    time.sleep(1)
    st.rerun()

else:
    st.info("👈 Please Login in Sidebar to Start System")