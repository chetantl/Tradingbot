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
    page_title="Institutional Sniper v2",
    page_icon="🎯",
    layout="wide",
)


# ====== ENUMS & CONFIG ====== #
class SignalType(Enum):
    NEUTRAL = "NEUTRAL"
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"
    MOMENTUM_BUY = "MOMENTUM_BUY"
    MOMENTUM_SELL = "MOMENTUM_SELL"


@dataclass
class Config:
    spread_split_enabled: bool = True
    bucket_seconds: int = 5
    rolling_window_seconds: int = 300
    bar_seconds: int = 300
    rvol_threshold_accum: float = 2.0
    rvol_threshold_momentum: float = 1.5
    price_change_eps: float = 0.002
    pcr_bullish_threshold: float = 0.9
    pcr_bearish_threshold: float = 1.1
    imbalance_momentum_threshold: float = 0.40
    consecutive_bars_momentum: int = 2
    weight_aggression: int = 3
    weight_volume: int = 3
    weight_pcr: int = 2
    weight_price_structure: int = 2
    weight_signal_type: int = 3
    avg_volume_lookback_bars: int = 20
    bars_to_keep: int = 12


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


# ====== 5-MINUTE BAR AGGREGATOR ====== #
class BarAggregator:
    def __init__(self, bar_seconds: int = 300, max_bars: int = 12):
        self.bar_seconds = bar_seconds
        self.max_bars = max_bars
        self.bars: deque = deque(maxlen=max_bars)
        self.current_bar: Optional[Bar5m] = None
        self.avg_volume: float = 0.0
        self.lock = threading.Lock()
    
    def _get_bar_start(self, ts: datetime) -> datetime:
        minute_slot = (ts.minute // (self.bar_seconds // 60)) * (self.bar_seconds // 60)
        return ts.replace(minute=minute_slot, second=0, microsecond=0)
    
    def set_avg_volume(self, avg_vol: float):
        with self.lock:
            self.avg_volume = avg_vol
    
    def append_bucket(self, bucket: Bucket5s) -> Optional[Bar5m]:
        with self.lock:
            bar_start = self._get_bar_start(bucket.start_time)
            bar_end = bar_start + timedelta(seconds=self.bar_seconds)
            
            completed = None
            
            if self.current_bar is None or bar_start != self.current_bar.start_time:
                if self.current_bar is not None:
                    self._finalize_bar(self.current_bar)
                    self.bars.append(self.current_bar)
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
            
            return completed
    
    def _finalize_bar(self, bar: Bar5m):
        total = bar.aggr_buy_total + bar.aggr_sell_total
        if total > 0:
            bar.agg_imbalance = (bar.aggr_buy_total - bar.aggr_sell_total) / total
        if self.avg_volume > 0:
            bar.rvol = bar.total_volume / self.avg_volume
        bar.price_range = bar.high - bar.low
        if bar.open > 0:
            bar.price_change_pct = (bar.close - bar.open) / bar.open
    
    def get_recent_bars(self, n: int = 3) -> List[Bar5m]:
        with self.lock:
            return list(self.bars)[-n:]
    
    def get_current_bar(self) -> Optional[Bar5m]:
        with self.lock:
            if self.current_bar:
                bar_copy = copy.copy(self.current_bar)
                self._finalize_bar(bar_copy)
                return bar_copy
            return None


# ====== PATTERN DETECTOR ====== #
class PatternDetector:
    def __init__(self, config: Config = CONFIG):
        self.config = config
    
    def detect(self, bars: List[Bar5m], pcr: float, current_bar: Bar5m) -> SignalResult:
        result = SignalResult(timestamp=now_ist())
        
        if not current_bar:
            return result
        
        diagnostics = {}
        
        ab = current_bar.aggr_buy_total
        as_ = current_bar.aggr_sell_total
        imbalance = current_bar.agg_imbalance
        rvol = current_bar.rvol
        price_change = current_bar.price_change_pct
        delta = current_bar.cumulative_delta
        
        diagnostics['AB'] = ab
        diagnostics['AS'] = as_
        diagnostics['Imbalance'] = f"{imbalance:.2%}"
        diagnostics['RVol'] = f"{rvol:.2f}x"
        diagnostics['PriceChg'] = f"{price_change:.2%}"
        diagnostics['Delta'] = delta
        diagnostics['PCR'] = f"{pcr:.2f}"
        
        # Accumulation conditions
        is_selling_dominant = as_ > ab
        is_price_flat_or_up = price_change >= -self.config.price_change_eps
        is_high_rvol_accum = rvol >= self.config.rvol_threshold_accum
        is_pcr_bullish = pcr < self.config.pcr_bullish_threshold
        
        delta_rising = True
        if len(bars) >= 1:
            prev_delta = bars[-1].cumulative_delta if bars else 0
            delta_rising = delta > prev_delta
        
        accum_score = sum([is_selling_dominant, is_price_flat_or_up, is_high_rvol_accum, is_pcr_bullish, delta_rising])
        
        # Distribution conditions
        is_buying_dominant = ab > as_
        is_price_flat_or_down = price_change <= self.config.price_change_eps
        is_pcr_bearish = pcr > self.config.pcr_bearish_threshold
        delta_falling = not delta_rising
        
        dist_score = sum([is_buying_dominant, is_price_flat_or_down, is_high_rvol_accum, is_pcr_bearish, delta_falling])
        
        # Momentum Buy conditions
        is_high_positive_imbalance = imbalance >= self.config.imbalance_momentum_threshold
        is_rvol_momentum = rvol >= self.config.rvol_threshold_momentum
        recent_high = max(b.high for b in bars) if bars else current_bar.high
        is_breakout_up = current_bar.close > recent_high
        
        consecutive_bullish = sum(1 for b in bars[-3:] if b.agg_imbalance >= self.config.imbalance_momentum_threshold)
        
        mom_buy_score = sum([
            is_high_positive_imbalance or consecutive_bullish >= self.config.consecutive_bars_momentum,
            is_rvol_momentum,
            is_breakout_up
        ])
        
        # Momentum Sell conditions
        is_high_negative_imbalance = imbalance <= -self.config.imbalance_momentum_threshold
        recent_low = min(b.low for b in bars) if bars else current_bar.low
        is_breakout_down = current_bar.close < recent_low
        
        consecutive_bearish = sum(1 for b in bars[-3:] if b.agg_imbalance <= -self.config.imbalance_momentum_threshold)
        
        mom_sell_score = sum([
            is_high_negative_imbalance or consecutive_bearish >= self.config.consecutive_bars_momentum,
            is_rvol_momentum,
            is_breakout_down
        ])
        
        diagnostics['AccumCond'] = f"{accum_score}/5"
        diagnostics['DistCond'] = f"{dist_score}/5"
        diagnostics['MomBuyCond'] = f"{mom_buy_score}/3"
        diagnostics['MomSellCond'] = f"{mom_sell_score}/3"
        
        # Determine signal
        if mom_buy_score >= 2:
            result.signal_type = SignalType.MOMENTUM_BUY
            result.strength = "STRONG" if mom_buy_score == 3 else "MODERATE"
        elif mom_sell_score >= 2:
            result.signal_type = SignalType.MOMENTUM_SELL
            result.strength = "STRONG" if mom_sell_score == 3 else "MODERATE"
        elif accum_score >= 4:
            result.signal_type = SignalType.ACCUMULATION
            result.strength = "STRONG" if accum_score == 5 else "MODERATE"
        elif dist_score >= 4:
            result.signal_type = SignalType.DISTRIBUTION
            result.strength = "STRONG" if dist_score == 5 else "MODERATE"
        elif accum_score >= 3:
            result.signal_type = SignalType.ACCUMULATION
            result.strength = "WEAK"
        elif dist_score >= 3:
            result.signal_type = SignalType.DISTRIBUTION
            result.strength = "WEAK"
        else:
            result.signal_type = SignalType.NEUTRAL
            result.strength = "NONE"
        
        result.diagnostics = diagnostics
        
        # Entry/SL/TP
        if result.signal_type in [SignalType.ACCUMULATION, SignalType.MOMENTUM_BUY]:
            result.entry_suggestion = current_bar.close
            result.stop_loss = current_bar.low - (current_bar.price_range * 0.5)
            result.target = current_bar.close + (current_bar.price_range * 2)
        elif result.signal_type in [SignalType.DISTRIBUTION, SignalType.MOMENTUM_SELL]:
            result.entry_suggestion = current_bar.close
            result.stop_loss = current_bar.high + (current_bar.price_range * 0.5)
            result.target = current_bar.close - (current_bar.price_range * 2)
        
        return result


# ====== SCORER ====== #
class Scorer:
    def __init__(self, config: Config = CONFIG):
        self.config = config
    
    def score(self, bar: Bar5m, signal_result: SignalResult, pcr: float) -> SignalResult:
        breakdown = {}
        total = 0
        
        # Aggression (0-3)
        imb_abs = abs(bar.agg_imbalance)
        if imb_abs >= 0.5:
            aggr_score = 3
        elif imb_abs >= 0.35:
            aggr_score = 2
        elif imb_abs >= 0.2:
            aggr_score = 1
        else:
            aggr_score = 0
        breakdown['Aggr'] = aggr_score
        total += aggr_score
        
        # Volume (0-3)
        if bar.rvol >= 3.0:
            vol_score = 3
        elif bar.rvol >= 2.0:
            vol_score = 2
        elif bar.rvol >= 1.5:
            vol_score = 1
        else:
            vol_score = 0
        breakdown['Vol'] = vol_score
        total += vol_score
        
        # PCR (0-2)
        if signal_result.signal_type in [SignalType.ACCUMULATION, SignalType.MOMENTUM_BUY]:
            if pcr < 0.7:
                pcr_score = 2
            elif pcr < 0.9:
                pcr_score = 1
            else:
                pcr_score = 0
        elif signal_result.signal_type in [SignalType.DISTRIBUTION, SignalType.MOMENTUM_SELL]:
            if pcr > 1.3:
                pcr_score = 2
            elif pcr > 1.1:
                pcr_score = 1
            else:
                pcr_score = 0
        else:
            pcr_score = 0
        breakdown['PCR'] = pcr_score
        total += pcr_score
        
        # Price Structure (0-2)
        price_struct_score = 0
        if signal_result.signal_type == SignalType.MOMENTUM_BUY and bar.price_change_pct > 0.003:
            price_struct_score = 2
        elif signal_result.signal_type == SignalType.MOMENTUM_SELL and bar.price_change_pct < -0.003:
            price_struct_score = 2
        elif signal_result.signal_type == SignalType.ACCUMULATION and bar.price_change_pct >= 0:
            price_struct_score = 1
        elif signal_result.signal_type == SignalType.DISTRIBUTION and bar.price_change_pct <= 0:
            price_struct_score = 1
        breakdown['Price'] = price_struct_score
        total += price_struct_score
        
        # Signal Type (0-3)
        if signal_result.strength == "STRONG":
            sig_score = 3
        elif signal_result.strength == "MODERATE":
            sig_score = 2
        elif signal_result.strength == "WEAK":
            sig_score = 1
        else:
            sig_score = 0
        breakdown['Signal'] = sig_score
        total += sig_score
        
        signal_result.score = total
        signal_result.normalized_score = (total / 13) * 100
        signal_result.score_breakdown = breakdown
        
        if total >= 10:
            signal_result.strength = "STRONG"
        elif total >= 7:
            signal_result.strength = "CONFIRMED"
        elif total >= 4:
            signal_result.strength = "MODERATE"
        else:
            signal_result.strength = "WEAK"
        
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
            
            for i in range(0, len(ce_syms), 40):
                batch = ce_syms[i:i+40]
                try:
                    quotes = kite.quote(batch)
                    ce_oi_total += sum(safe_int(q.get('oi', 0)) for q in quotes.values())
                    time.sleep(0.1)
                except:
                    pass
            
            for i in range(0, len(pe_syms), 40):
                batch = pe_syms[i:i+40]
                try:
                    quotes = kite.quote(batch)
                    pe_oi_total += sum(safe_int(q.get('oi', 0)) for q in quotes.values())
                    time.sleep(0.1)
                except:
                    pass
            
            pcr = pe_oi_total / ce_oi_total if ce_oi_total > 0 else 1.0
            
            with self.lock:
                self.cache[symbol] = pcr
                self.cache_expiry[symbol] = now_ist() + timedelta(minutes=1)
            
            return round(pcr, 2)
            
        except:
            return self.cache.get(symbol, 1.0)


# ====== SYMBOL STATE ====== #
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
    avg_volume: float = 0.0
    
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


def calc_avg_volume(kite, token, lookback_bars=20):
    try:
        now = now_ist().replace(tzinfo=None)
        from_date = now - timedelta(days=5)
        candles = kite.historical_data(token, from_date, now, "5minute")
        df = pd.DataFrame(candles)
        if df.empty or len(df) < lookback_bars:
            return 0.0
        return float(df['volume'].tail(lookback_bars).mean())
    except Exception as e:
        manager.log(f"Avg volume error: {e}", "ERROR")
        return 0.0


# ====== WORKER THREAD ====== #
def sniper_worker(kite, mgr: SniperManager):
    mgr.is_running = True
    mgr.log("Worker started - Strategy v2", "INFO")
    
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
            
            update_pcr = (now_ts - last_pcr_update) > 120
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
                            mgr.log(f"{sym} Bar: AB={completed_bar.aggr_buy_total:,} "
                                   f"AS={completed_bar.aggr_sell_total:,} "
                                   f"Imb={completed_bar.agg_imbalance:.2%}", "INFO")
                
                if update_pcr:
                    state.pcr = mgr.pcr_fetcher.get_pcr(kite, sym, ltp)
                
                current_bar = state.bar_aggregator.get_current_bar()
                recent_bars = state.bar_aggregator.get_recent_bars(3)
                
                if current_bar:
                    signal = mgr.pattern_detector.detect(recent_bars, state.pcr, current_bar)
                    signal = mgr.scorer.score(current_bar, signal, state.pcr)
                    state.signal_result = signal
                    
                    if signal.signal_type != SignalType.NEUTRAL and signal.score >= 6:
                        mgr.log(f"🎯 {sym}: {signal.signal_type.value} Score={signal.score}/13", "SIGNAL")
            
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


# ====== RENDER CARD USING NATIVE STREAMLIT ====== #
def render_signal_card(sym: str, data: Dict):
    signal = data.get('signal', SignalResult())
    current_bar = data.get('current_bar')
    
    if not current_bar:
        st.warning(f"{sym}: No bar data yet")
        return
    
    sig_type = signal.signal_type
    score = signal.score
    strength = signal.strength
    breakdown = signal.score_breakdown
    diagnostics = signal.diagnostics
    
    ltp = data.get('ltp', 0)
    vwap = data.get('vwap', 0)
    pcr = data.get('pcr', 1.0)
    
    # Determine signal info
    if sig_type == SignalType.ACCUMULATION:
        sig_emoji = "🟢"
        sig_label = "ACCUMULATION"
    elif sig_type == SignalType.DISTRIBUTION:
        sig_emoji = "🔴"
        sig_label = "DISTRIBUTION"
    elif sig_type == SignalType.MOMENTUM_BUY:
        sig_emoji = "🚀"
        sig_label = "MOMENTUM BUY"
    elif sig_type == SignalType.MOMENTUM_SELL:
        sig_emoji = "📉"
        sig_label = "MOMENTUM SELL"
    else:
        sig_emoji = "⚪"
        sig_label = "NEUTRAL"
    
    # Card container
    with st.container(border=True):
        # Header row
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.subheader(f"{sig_emoji} {sym}")
        with col2:
            st.caption(f"**{sig_label}** • {strength}")
        with col3:
            if score >= 10:
                st.success(f"**{score}/13**")
            elif score >= 6:
                st.warning(f"**{score}/13**")
            else:
                st.info(f"**{score}/13**")
        
        # Metrics row
        m1, m2, m3, m4, m5 = st.columns(5)
        
        with m1:
            price_delta = ltp - vwap
            st.metric("LTP", f"₹{ltp:.2f}", f"{price_delta:+.2f} vs VWAP")
        
        with m2:
            imb_pct = current_bar.agg_imbalance * 100
            st.metric("Imbalance", f"{imb_pct:+.1f}%", f"AB:{current_bar.aggr_buy_total:,}")
        
        with m3:
            st.metric("RVol", f"{current_bar.rvol:.2f}x", f"Vol:{current_bar.total_volume:,}")
        
        with m4:
            st.metric("PCR", f"{pcr:.2f}", f"Δ:{current_bar.cumulative_delta:,}")
        
        with m5:
            if signal.entry_suggestion > 0:
                st.metric("Entry", f"₹{signal.entry_suggestion:.2f}", f"SL:{signal.stop_loss:.2f}")
        
        # Score breakdown
        if breakdown:
            st.caption("**Score Breakdown:** " + " | ".join([f"{k}:{v}" for k, v in breakdown.items()]))
        
        # Diagnostics expander
        with st.expander("Diagnostics"):
            if diagnostics:
                diag_cols = st.columns(4)
                items = list(diagnostics.items())
                for i, (k, v) in enumerate(items):
                    with diag_cols[i % 4]:
                        st.caption(f"**{k}:** {v}")


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
        st.sidebar.markdown(f"[🔗 Login to Zerodha]({temp_kite.login_url()})")
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

# Config expander
with st.sidebar.expander("⚙️ Strategy Config"):
    st.number_input("RVol Threshold (Accum)", value=CONFIG.rvol_threshold_accum, key="cfg_rvol_accum")
    st.number_input("RVol Threshold (Momentum)", value=CONFIG.rvol_threshold_momentum, key="cfg_rvol_mom")
    st.number_input("PCR Bullish (<)", value=CONFIG.pcr_bullish_threshold, key="cfg_pcr_bull")
    st.number_input("PCR Bearish (>)", value=CONFIG.pcr_bearish_threshold, key="cfg_pcr_bear")
    st.number_input("Imbalance Threshold", value=CONFIG.imbalance_momentum_threshold, key="cfg_imb")

symbols_text = st.sidebar.text_area("Symbols (comma separated)", "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK", key="symbols_input")

col1, col2 = st.sidebar.columns(2)

if col1.button("▶️ START", key="start_btn"):
    if not manager.kite:
        st.error("Connect to Kite first")
    elif manager.is_running:
        st.warning("Already running")
    else:
        syms = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
        
        with manager.lock:
            manager.active_symbols = []
            manager.symbol_states = {}
        
        with st.spinner("Initializing symbols..."):
            valid = []
            for s in syms:
                tkn = get_instrument_token(manager.kite, s)
                if not tkn:
                    manager.log(f"Token not found: {s}", "WARNING")
                    continue
                
                avg_vol = calc_avg_volume(manager.kite, tkn, CONFIG.avg_volume_lookback_bars)
                
                state = SymbolState(symbol=s, token=tkn, avg_volume=avg_vol)
                state.bar_aggregator.set_avg_volume(avg_vol)
                
                try:
                    q = manager.kite.quote(f"NSE:{s}")[f"NSE:{s}"]
                    state.ltp = safe_float(q.get('last_price'))
                    state.vwap = safe_float(q.get('average_price'), state.ltp)
                    state.volume = safe_int(q.get('volume'))
                except:
                    pass
                
                state.pcr = manager.pcr_fetcher.get_pcr(manager.kite, s, state.ltp)
                
                manager.symbol_states[s] = state
                valid.append({"symbol": s, "token": tkn})
                manager.log(f"Init {s}: AvgVol={avg_vol:.0f} PCR={state.pcr:.2f}", "INFO")
        
        if valid:
            with manager.lock:
                manager.active_symbols = valid
                manager.stop_event.clear()
            t = threading.Thread(target=sniper_worker, args=(manager.kite, manager), daemon=True)
            t.start()
            st.success(f"Started monitoring {len(valid)} symbols")

if col2.button("⏹️ STOP", key="stop_btn"):
    manager.stop_event.set()
    manager.is_running = False
    time.sleep(1)
    st.rerun()

if st.sidebar.button("🔄 Reset App", key="reset_btn"):
    st.session_state.manager = get_fresh_manager()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"Status: {'🟢 Running' if manager.is_running else '🔴 Stopped'}")


# ====== MAIN DASHBOARD ====== #
st.title("🎯 Institutional Sniper v2")
st.caption("Tick Classification → 5s/5m Aggregation → Pattern Detection → Scoring")

# Status row
status_cols = st.columns(5)

last_beat = manager.last_beat
delta = time.time() - last_beat
is_live = delta < 5

bar_start = current_bar_start()
bar_end = bar_start + timedelta(minutes=5)
time_to_close = max(0, (bar_end - now_ist()).total_seconds())
beat_time = datetime.fromtimestamp(last_beat, INDIA_TZ).strftime("%H:%M:%S") if last_beat > 0 else "-"

ranked = manager.get_ranked_signals()
active_signals = len([r for r in ranked if r[1].score >= 6])

with status_cols[0]:
    if is_live:
        st.success("🟢 LIVE")
    else:
        st.error("🔴 STOPPED")

with status_cols[1]:
    st.info(f"📊 Bar: {bar_start.strftime('%H:%M')}-{bar_end.strftime('%H:%M')}")

with status_cols[2]:
    st.warning(f"⏱️ Closes: {int(time_to_close)}s")

with status_cols[3]:
    st.success(f"🎯 Signals: {active_signals}")

with status_cols[4]:
    st.caption(f"Beat: {beat_time}")

st.markdown("---")

# Get all states
all_states = manager.get_all_states_snapshot()

if all_states:
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Ranked Signals", "📈 All Symbols", "📖 Guide"])
    
    with tab1:
        st.subheader("Top Signals (Score ≥ 6)")
        
        ranked_data = [
            (sym, all_states[sym])
            for sym, sig in ranked
            if sym in all_states and sig.score >= 6
        ]
        
        if ranked_data:
            for sym, data in ranked_data:
                render_signal_card(sym, data)
        else:
            st.info("🔍 No high-confidence signals detected. Waiting for patterns...")
    
    with tab2:
        st.subheader(f"All {len(all_states)} Symbols")
        
        sorted_states = sorted(
            all_states.items(),
            key=lambda x: x[1].get('signal', SignalResult()).score,
            reverse=True
        )
        
        col_left, col_right = st.columns(2)
        
        for i, (sym, data) in enumerate(sorted_states):
            with col_left if i % 2 == 0 else col_right:
                render_signal_card(sym, data)
    
    with tab3:
        st.subheader("Strategy Guide")
        
        st.markdown("### Signal Types")
        
        signal_df = pd.DataFrame({
            "Signal": ["ACCUMULATION", "DISTRIBUTION", "MOMENTUM BUY", "MOMENTUM SELL"],
            "Description": [
                "Smart money buying while appearing to sell",
                "Smart money selling while appearing to buy",
                "Strong directional buying pressure",
                "Strong directional selling pressure"
            ],
            "Key Conditions": [
                "AS > AB, Price flat/up, RVol ≥ 2.0, Delta rising, PCR < 0.9",
                "AB > AS, Price flat, RVol ≥ 2.0, Delta falling, PCR > 1.1",
                "Imbalance ≥ 40%, RVol ≥ 1.5, Breakout above recent high",
                "Imbalance ≤ -40%, RVol ≥ 1.5, Breakdown below recent low"
            ]
        })
        st.dataframe(signal_df, use_container_width=True, hide_index=True)
        
        st.markdown("### Scoring (0-13 points)")
        
        scoring_df = pd.DataFrame({
            "Component": ["Aggression", "Volume", "PCR", "Price Structure", "Signal Type"],
            "Max Score": [3, 3, 2, 2, 3],
            "Criteria": [
                "Imbalance magnitude (≥50%=3, ≥35%=2, ≥20%=1)",
                "RVol (≥3x=3, ≥2x=2, ≥1.5x=1)",
                "Options confirmation matching signal direction",
                "Price breakout/trend alignment",
                "Pattern strength (STRONG=3, MODERATE=2, WEAK=1)"
            ]
        })
        st.dataframe(scoring_df, use_container_width=True, hide_index=True)
        
        st.markdown("### Tick Classification")
        st.markdown("""
        - **Aggressive Buy**: Trade executed at or above Ask price
        - **Aggressive Sell**: Trade executed at or below Bid price
        - **Neutral**: Trade inside spread (split proportionally)
        """)

else:
    st.info("👆 Enter symbols in the sidebar and click START to begin monitoring")

st.markdown("---")

# Logs
with st.expander("📝 System Logs", expanded=False):
    logs_list = manager.get_logs_snapshot()
    if logs_list:
        log_text = "\n".join(logs_list[:100])
        st.code(log_text, language=None)
    else:
        st.caption("No logs yet...")

# Auto-refresh
if manager.is_running:
    time.sleep(2)
    st.rerun()
