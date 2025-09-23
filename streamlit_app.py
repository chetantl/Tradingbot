import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime
from kiteconnect import KiteConnect
from urllib.parse import urlparse, parse_qs
import logging

# Page config
st.set_page_config(
    page_title="🚀 TradeFinder R-Factor Scanner (Strict Performance)",
    page_icon="🎯",
    layout="wide"
)

st.title("🚀 TradeFinder R-Factor Scanner - Strict Performance Formula")
st.markdown("> _Only Strong Trending Stocks Get High R-Factors | Auto-Refresh Every 5 Minutes_")

# =========================
# SESSION STATE INIT
# =========================
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'api_secret' not in st.session_state:
    st.session_state.api_secret = ""
if 'kite' not in st.session_state:
    st.session_state.kite = None
if 'auth_step' not in st.session_state:
    st.session_state.auth_step = 1
if 'instruments' not in st.session_state:
    st.session_state.instruments = {}
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'market_regime' not in st.session_state:
    st.session_state.market_regime = "NEUTRAL"
if 'regime_multiplier' not in st.session_state:
    st.session_state.regime_multiplier = 1.0
if 'nifty_20d_return' not in st.session_state:
    st.session_state.nifty_20d_return = 0
if 'nifty_hist_data' not in st.session_state:
    st.session_state.nifty_hist_data = []
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'last_scan_results' not in st.session_state:
    st.session_state.last_scan_results = None
if 'symbols_to_scan' not in st.session_state:
    st.session_state.symbols_to_scan = "ADANIENT, KALYANKJIL, ADANIGREEN, IREDA, HDFCBANK, AXISBANK, UNIONBANK, SAIL, DLF, EXIDEIND, ADANIENSOL, RELIANCE, TCS, INFY"
if 'last_refresh_time' not in st.session_state:
    st.session_state.last_refresh_time = None

# =========================
# AUTO-REFRESH FUNCTIONS
# =========================
def get_next_5min_candle_time():
    """Get the next 5-minute candle time aligned with market"""
    now = datetime.datetime.now()
    # Round down to nearest 5 minutes
    minutes = now.minute
    rounded_minutes = (minutes // 5) * 5
    next_candle = now.replace(minute=rounded_minutes, second=0, microsecond=0)
    next_candle += datetime.timedelta(minutes=5)
    return next_candle

def get_seconds_to_next_candle():
    """Get seconds remaining to next 5-minute candle"""
    now = datetime.datetime.now()
    next_candle = get_next_5min_candle_time()
    return (next_candle - now).total_seconds()

def is_market_hours():
    """Check if current time is within market hours"""
    now = datetime.datetime.now()
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    # Check if it's a weekday (Monday=0, Sunday=6)
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    
    return market_open <= now <= market_close

def should_refresh():
    """Check if we should refresh based on 5-minute candle alignment"""
    if not st.session_state.auto_refresh:
        return False
    
    if not is_market_hours():
        return False
    
    now = datetime.datetime.now()
    
    # If never refreshed, refresh immediately
    if st.session_state.last_refresh_time is None:
        return True
    
    # Check if 5 minutes have passed since last refresh
    time_since_refresh = (now - st.session_state.last_refresh_time).total_seconds()
    
    # Refresh if we're within 2 seconds of a 5-minute mark and haven't refreshed recently
    seconds_to_next = get_seconds_to_next_candle()
    if seconds_to_next > 298 and time_since_refresh > 290:  # Within 2 seconds of 5-min mark
        return True
    
    return False

# =========================
# HELPER FUNCTIONS
# =========================
def get_quote_data(kite, symbol, instrument_token):
    """Get quote data with proper format handling"""
    try:
        quote_key = f"NSE:{symbol}"
        quote_response = kite.quote([quote_key])
        
        if quote_key in quote_response:
            return quote_response[quote_key]
        
        for key in quote_response:
            if symbol in key or str(instrument_token) in str(key):
                return quote_response[key]
                
        return None
    except Exception as e:
        return None

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    if len(prices) < period + 1:
        return 50
    
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    
    if down == 0:
        return 100
    
    rs = up / down
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# =========================
# STRICT 20-DAY RELATIVE STRENGTH
# =========================
def calculate_20day_relative_strength(kite, symbols):
    """Calculate 20-day relative strength with strict criteria"""
    relative_strengths = {}
    
    nifty_hist_data = []
    try:
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=60)
        
        nifty_token = 256265
        nifty_hist_data = kite.historical_data(nifty_token, start_date, end_date, "day")
        st.session_state.nifty_hist_data = nifty_hist_data
        
        if len(nifty_hist_data) >= 20:
            nifty_20d_return = ((nifty_hist_data[-1]['close'] - nifty_hist_data[-20]['close']) / nifty_hist_data[-20]['close']) * 100
            st.session_state.nifty_20d_return = nifty_20d_return
        else:
            nifty_20d_return = 0
            
    except:
        nifty_20d_return = st.session_state.nifty_20d_return or 0
        nifty_hist_data = st.session_state.nifty_hist_data
    
    for symbol in symbols:
        try:
            if symbol not in st.session_state.instruments:
                continue
                
            token = st.session_state.instruments[symbol]
            hist_data = kite.historical_data(token, start_date, end_date, "day")
            
            if len(hist_data) >= 20:
                # Calculate various performance metrics
                stock_20d_return = ((hist_data[-1]['close'] - hist_data[-20]['close']) / hist_data[-20]['close']) * 100
                stock_10d_return = ((hist_data[-1]['close'] - hist_data[-10]['close']) / hist_data[-10]['close']) * 100
                stock_5d_return = ((hist_data[-1]['close'] - hist_data[-5]['close']) / hist_data[-5]['close']) * 100
                
                # Relative strength
                if nifty_20d_return != 0:
                    relative_strength = stock_20d_return / nifty_20d_return
                else:
                    relative_strength = stock_20d_return / 5
                
                relative_strengths[symbol] = {
                    'relative_strength': relative_strength,
                    'stock_20d_return': stock_20d_return,
                    'stock_10d_return': stock_10d_return,
                    'stock_5d_return': stock_5d_return,
                    'hist_data': hist_data
                }
            else:
                relative_strengths[symbol] = {
                    'relative_strength': 0,
                    'stock_20d_return': 0,
                    'stock_10d_return': 0,
                    'stock_5d_return': 0,
                    'hist_data': []
                }
                
        except Exception as e:
            relative_strengths[symbol] = {
                'relative_strength': 0,
                'stock_20d_return': 0,
                'stock_10d_return': 0,
                'stock_5d_return': 0,
                'hist_data': []
            }
    
    return relative_strengths

def assign_strict_tiers(relative_strengths):
    """Assign tiers with STRICT performance criteria"""
    # Sort by 20-day return (actual performance)
    sorted_symbols = sorted(relative_strengths.items(), 
                          key=lambda x: x[1]['stock_20d_return'], 
                          reverse=True)
    
    tier_assignments = {}
    
    for idx, (symbol, data) in enumerate(sorted_symbols):
        stock_20d = data['stock_20d_return']
        stock_10d = data['stock_10d_return']
        stock_5d = data['stock_5d_return']
        
        # STRICT tier assignment based on actual returns
        if stock_20d > 25 and stock_10d > 12 and stock_5d > 5:
            tier = "TIER_1"
            base_score = 2.0  # Reduced from 2.5
        elif stock_20d > 15 and stock_10d > 7 and stock_5d > 2:
            tier = "TIER_2"
            base_score = 1.7  # Reduced from 2.2
        elif stock_20d > 10 and stock_10d > 5:
            tier = "TIER_3"
            base_score = 1.4  # Reduced from 2.0
        elif stock_20d > 5:
            tier = "TIER_4"
            base_score = 1.2  # Reduced from 1.8
        elif stock_20d > 0:
            tier = "TIER_5"
            base_score = 1.0  # Reduced from 1.6
        else:
            tier = "TIER_6"
            base_score = 0.8  # Reduced from 1.4
        
        # Additional penalty for sideways stocks
        if abs(stock_20d) < 3:  # Sideways/flat movement
            base_score *= 0.7  # 30% penalty
        
        tier_assignments[symbol] = {
            'tier': tier,
            'base_score': base_score,
            'relative_strength': data['relative_strength'],
            'stock_20d_return': stock_20d,
            'stock_10d_return': stock_10d,
            'stock_5d_return': stock_5d,
            'rank': idx + 1
        }
    
    return tier_assignments

# =========================
# STRICT VOLUME WEIGHT
# =========================
def calculate_strict_volume_weight(kite, symbol, instrument_token, hist_data):
    """Strict volume weight - only reward significant volume with price movement"""
    try:
        quote = get_quote_data(kite, symbol, instrument_token)
        current_volume = 0
        
        if quote:
            current_volume = float(quote.get('volume', 0))
            change_percent = float(quote.get('change_percent', 0))
        else:
            change_percent = 0
        
        if current_volume == 0 and hist_data:
            current_volume = float(hist_data[-1].get('volume', 0))
        
        if len(hist_data) >= 20:
            volumes = [float(d.get('volume', 0)) for d in hist_data[-21:-1]]
            avg_volume_20d = np.mean(volumes) if volumes else 1
        else:
            avg_volume_20d = current_volume if current_volume > 0 else 1
        
        if avg_volume_20d == 0:
            avg_volume_20d = 1
            
        volume_ratio = current_volume / avg_volume_20d if current_volume > 0 else 1.0
        
        # STRICT: Only reward volume if accompanied by price movement
        if abs(change_percent) < 1:  # Flat day
            volume_weight = 1.0  # No bonus for volume on flat days
        elif volume_ratio > 2.0 and abs(change_percent) > 2:
            volume_weight = 1.2  # Only 20% bonus for high volume with movement
        elif volume_ratio > 1.5 and abs(change_percent) > 1:
            volume_weight = 1.1
        elif volume_ratio < 0.5:
            volume_weight = 0.9  # Penalty for low volume
        else:
            volume_weight = 1.0
        
        return volume_weight, volume_ratio
        
    except Exception as e:
        return 1.0, 1.0

# =========================
# STRICT MOMENTUM FACTOR
# =========================
def calculate_strict_momentum_factor(kite, symbol, instrument_token, hist_data):
    """Strict momentum - only reward consistent upward movement"""
    try:
        if len(hist_data) < 15:
            return 0.8, {}  # Penalty for insufficient data
        
        quote = get_quote_data(kite, symbol, instrument_token)
        
        current_price = float(hist_data[-1]['close'])
        open_price = float(hist_data[-1]['open'])
        
        if quote:
            current_price = float(quote.get('last_price', current_price))
            ohlc = quote.get('ohlc', {})
            if ohlc:
                open_price = float(ohlc.get('open', open_price))
        
        # 1. Intraday momentum - STRICT
        if open_price > 0:
            intraday_change = ((current_price - open_price) / open_price) * 100
        else:
            intraday_change = 0
        
        # Only reward significant positive movement
        if intraday_change > 3:
            intraday_momentum = 1.1
        elif intraday_change > 1.5:
            intraday_momentum = 1.05
        elif intraday_change > 0:
            intraday_momentum = 1.0
        elif intraday_change > -1:
            intraday_momentum = 0.95
        else:
            intraday_momentum = 0.9
        
        # 2. Consistency check - count positive days
        positive_days = 0
        for i in range(1, min(10, len(hist_data))):
            if hist_data[-i]['close'] > hist_data[-i-1]['close']:
                positive_days += 1
        
        consistency_factor = 1.0
        if positive_days >= 7:
            consistency_factor = 1.1  # Consistent uptrend
        elif positive_days >= 5:
            consistency_factor = 1.0
        elif positive_days <= 3:
            consistency_factor = 0.9  # Too many down days
        
        # 3. Trend strength
        if len(hist_data) >= 20:
            # Check if making higher highs
            recent_highs = [d['high'] for d in hist_data[-10:]]
            older_highs = [d['high'] for d in hist_data[-20:-10]]
            
            if max(recent_highs) > max(older_highs) * 1.05:
                trend_factor = 1.1
            elif max(recent_highs) > max(older_highs):
                trend_factor = 1.0
            else:
                trend_factor = 0.9  # Not making new highs
        else:
            trend_factor = 1.0
        
        # Combine factors
        composite_momentum = intraday_momentum * consistency_factor * trend_factor
        
        # STRICT bounds
        composite_momentum = max(0.7, min(1.3, composite_momentum))
        
        momentum_breakdown = {
            "intraday": round(intraday_momentum, 3),
            "consistency": round(consistency_factor, 3),
            "trend": round(trend_factor, 3),
            "intraday_change": round(intraday_change, 2),
            "positive_days": positive_days
        }
        
        return composite_momentum, momentum_breakdown
        
    except Exception as e:
        return 0.8, {}

# =========================
# STRICT MARKET REGIME
# =========================
def detect_strict_market_regime(kite, relative_strengths):
    """Strict market regime with minimal multipliers"""
    try:
        nifty_20d = st.session_state.nifty_20d_return
        
        positive_stocks = sum(1 for s, d in relative_strengths.items() if d['stock_20d_return'] > 5)  # Only count significant gainers
        breadth = (positive_stocks / len(relative_strengths)) * 100 if relative_strengths else 50
        
        # Very conservative multipliers
        if nifty_20d > 10 and breadth > 70:
            regime = "STRONG_BULL"
            multiplier = 1.2  # Max 20% boost
        elif nifty_20d > 5 and breadth > 50:
            regime = "MODERATE_BULL"
            multiplier = 1.1  # 10% boost
        elif nifty_20d > -2:
            regime = "NEUTRAL"
            multiplier = 1.0  # No boost
        else:
            regime = "WEAK_BEAR"
            multiplier = 0.9  # 10% penalty
        
        return regime, multiplier, breadth
        
    except Exception as e:
        return "NEUTRAL", 1.0, 50

# =========================
# STRICT R-FACTOR CALCULATION
# =========================
def calculate_strict_rfactor(symbol, kite, instrument_token, tier_data, regime_multiplier, all_stocks_data):
    """
    Strict R-Factor - only trending stocks get high scores
    """
    try:
        quote = get_quote_data(kite, symbol, instrument_token)
        hist_data = all_stocks_data.get(symbol, {}).get('hist_data', [])
        
        if not quote and not hist_data:
            return 0, {}
        
        ltp = float(quote.get('last_price', 0)) if quote else hist_data[-1]['close']
        volume = float(quote.get('volume', 0)) if quote else hist_data[-1]['volume']
        
        change_percent = 0
        if quote and 'ohlc' in quote:
            ohlc = quote['ohlc']
            prev_close = float(ohlc.get('close', 0))
            if prev_close > 0:
                change_percent = ((ltp - prev_close) / prev_close) * 100
        
        # Get performance metrics
        stock_20d = tier_data['stock_20d_return']
        stock_10d = tier_data['stock_10d_return']
        stock_5d = tier_data['stock_5d_return']
        
        # 1. Base RS Score (already reduced in tier assignment)
        rs_score = tier_data['base_score']
        
        # 2. Performance multiplier (STRICT)
        performance_multiplier = 1.0
        
        # Only boost for strong performers
        if stock_20d > 30 and stock_10d > 15 and stock_5d > 7:
            performance_multiplier = 1.5
        elif stock_20d > 20 and stock_10d > 10 and stock_5d > 5:
            performance_multiplier = 1.3
        elif stock_20d > 15 and stock_10d > 7:
            performance_multiplier = 1.2
        elif stock_20d > 10 and stock_10d > 5:
            performance_multiplier = 1.1
        elif stock_20d > 5:
            performance_multiplier = 1.0
        elif stock_20d > 0:
            performance_multiplier = 0.9
        else:
            performance_multiplier = 0.7  # Penalty for negative returns
        
        # 3. Market Regime (minimal impact)
        market_regime = regime_multiplier
        
        # 4. Volume Weight (strict)
        volume_weight, volume_ratio = calculate_strict_volume_weight(
            kite, symbol, instrument_token, hist_data
        )
        
        # 5. Momentum Factor (strict)
        momentum_factor, momentum_breakdown = calculate_strict_momentum_factor(
            kite, symbol, instrument_token, hist_data
        )
        
        # 6. Sideways penalty
        sideways_penalty = 1.0
        if abs(stock_20d) < 5 and abs(stock_10d) < 3:  # Sideways movement
            sideways_penalty = 0.6  # 40% penalty
        elif abs(stock_20d) < 10 and abs(stock_10d) < 5:
            sideways_penalty = 0.8  # 20% penalty
        
        # 7. Base R-Factor Calculation (multiplicative)
        base_rfactor = (
            rs_score * 
            performance_multiplier *
            market_regime * 
            volume_weight * 
            momentum_factor * 
            sideways_penalty
        )
        
        # 8. No technical adjustments for sideways stocks
        technical_adjustment = 0
        
        # Only add bonus for clear breakouts with volume
        if stock_5d > 7 and volume_ratio > 1.5 and change_percent > 2:
            technical_adjustment = 0.3  # Breakout bonus
        elif stock_5d < -7 and volume_ratio > 1.5:
            technical_adjustment = -0.3  # Breakdown penalty
        
        # Final R-Factor
        enhanced_rfactor = base_rfactor + technical_adjustment
        
        # STRICT caps based on performance
        if stock_20d < 5:
            max_rfactor = 2.0  # Low performers capped at 2.0
        elif stock_20d < 10:
            max_rfactor = 3.0
        elif stock_20d < 20:
            max_rfactor = 4.0
        elif stock_20d < 30:
            max_rfactor = 5.0
        else:
            max_rfactor = 6.0  # Only exceptional performers can reach 6.0
        
        # Minimum based on performance
        if stock_20d < 0:
            min_rfactor = 0.5
        else:
            min_rfactor = 1.0
        
        enhanced_rfactor = max(min_rfactor, min(max_rfactor, enhanced_rfactor))
        enhanced_rfactor = round(enhanced_rfactor, 2)
        
        components = {
            'ltp': ltp,
            'change_percent': round(change_percent, 2),
            'volume': volume,
            'volume_ratio': volume_ratio,
            'tier': tier_data['tier'],
            'rs_score': rs_score,
            'performance_multiplier': performance_multiplier,
            'sideways_penalty': sideways_penalty,
            'relative_strength': tier_data['relative_strength'],
            'stock_20d_return': stock_20d,
            'stock_10d_return': stock_10d,
            'stock_5d_return': stock_5d,
            'rank': tier_data['rank'],
            'market_regime': market_regime,
            'volume_weight': volume_weight,
            'momentum_factor': momentum_factor,
            'momentum_breakdown': momentum_breakdown,
            'base_rfactor': base_rfactor,
            'max_allowed': max_rfactor
        }
        
        return enhanced_rfactor, components
        
    except Exception as e:
        st.error(f"Error calculating R-Factor for {symbol}: {str(e)}")
        return 0, {}

# =========================
# AUTHENTICATION
# =========================
def authenticate_kite(api_key, api_secret, request_token):
    """Authenticate with Kite and return kite object"""
    try:
        kite = KiteConnect(api_key=api_key)
        data = kite.generate_session(request_token, api_secret=api_secret)
        kite.set_access_token(data["access_token"])
        return kite, None
    except Exception as e:
        return None, str(e)

def load_instruments(kite):
    """Load NSE instruments"""
    try:
        instruments = kite.instruments("NSE")
        inst_map = {}
        for inst in instruments:
            if inst['exchange'] == 'NSE' and inst['segment'] == 'NSE':
                inst_map[inst['tradingsymbol']] = inst['instrument_token']
        return inst_map
    except Exception as e:
        st.error(f"Failed to load instruments: {str(e)}")
        return {}

# =========================
# SCANNING FUNCTION
# =========================
def perform_scan(symbols_input, show_progress=True):
    """Perform the actual scanning"""
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    
    valid_symbols = []
    invalid_symbols = []
    
    for symbol in symbols:
        if symbol in st.session_state.instruments:
            valid_symbols.append(symbol)
        else:
            invalid_symbols.append(symbol)
    
    if invalid_symbols and show_progress:
        st.warning(f"Invalid symbols removed: {', '.join(invalid_symbols)}")
    
    if not valid_symbols:
        if show_progress:
            st.error("No valid symbols to scan")
        return None
    
    # Calculate relative strengths
    if show_progress:
        with st.spinner("Calculating strict performance rankings..."):
            relative_strengths = calculate_20day_relative_strength(st.session_state.kite, valid_symbols)
            tier_assignments = assign_strict_tiers(relative_strengths)
            
            regime, multiplier, breadth = detect_strict_market_regime(
                st.session_state.kite, 
                relative_strengths
            )
            st.session_state.market_regime = regime
            st.session_state.regime_multiplier = multiplier
    else:
        relative_strengths = calculate_20day_relative_strength(st.session_state.kite, valid_symbols)
        tier_assignments = assign_strict_tiers(relative_strengths)
        
        regime, multiplier, breadth = detect_strict_market_regime(
            st.session_state.kite, 
            relative_strengths
        )
        st.session_state.market_regime = regime
        st.session_state.regime_multiplier = multiplier
    
    # Calculate R-Factors
    results = []
    
    if show_progress:
        progress = st.progress(0)
        status = st.empty()
    
    for idx, symbol in enumerate(valid_symbols):
        if show_progress:
            status.text(f"Calculating Strict R-Factor for {symbol}... ({idx+1}/{len(valid_symbols)})")
        
        if symbol in tier_assignments:
            instrument_token = st.session_state.instruments[symbol]
            tier_data = tier_assignments[symbol]
            
            enhanced_rfactor, components = calculate_strict_rfactor(
                symbol,
                st.session_state.kite,
                instrument_token,
                tier_data,
                multiplier,
                relative_strengths
            )
            
            if enhanced_rfactor > 0:
                results.append({
                    'Symbol': symbol,
                    'R-Factor': enhanced_rfactor,
                    '20D%': round(tier_data['stock_20d_return'], 2),
                    '10D%': round(tier_data['stock_10d_return'], 2),
                    '5D%': round(tier_data['stock_5d_return'], 2),
                    'Tier': tier_data['tier'],
                    'Momentum': round(components.get('momentum_factor', 1.0), 2),
                    'Volume': round(components.get('volume_weight', 1.0), 2),
                    'LTP': components.get('ltp', 0),
                    'Change%': components.get('change_percent', 0),
                    'Vol Ratio': round(components.get('volume_ratio', 1.0), 1),
                    'Max Cap': components.get('max_allowed', 0),
                    '_components': components
                })
        
        if show_progress:
            progress.progress((idx + 1) / len(valid_symbols))
        time.sleep(0.05)
    
    if show_progress:
        progress.empty()
        status.empty()
    
    # Update last refresh time
    st.session_state.last_refresh_time = datetime.datetime.now()
    st.session_state.last_update = datetime.datetime.now()
    
    return {
        'results': results,
        'regime': regime,
        'multiplier': multiplier,
        'breadth': breadth,
        'valid_symbols': valid_symbols,
        'invalid_symbols': invalid_symbols
    }

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("🔐 Zerodha Authentication")
    
    if not st.session_state.authenticated:
        if st.session_state.auth_step == 1:
            st.markdown("### Step 1: API Credentials")
            api_key = st.text_input("API Key", value=st.session_state.api_key)
            api_secret = st.text_input("API Secret", value=st.session_state.api_secret, type="password")
            
            if st.button("Generate Login URL", type="primary"):
                if api_key and api_secret:
                    st.session_state.api_key = api_key
                    st.session_state.api_secret = api_secret
                    kite = KiteConnect(api_key=api_key)
                    login_url = kite.login_url()
                    st.session_state.auth_step = 2
                    st.success("Login URL generated!")
                    st.markdown(f"[Click here to login]({login_url})")
                    st.rerun()
                else:
                    st.error("Enter both API Key and Secret")
        
        elif st.session_state.auth_step == 2:
            st.markdown("### Step 2: Complete Login")
            kite = KiteConnect(api_key=st.session_state.api_key)
            login_url = kite.login_url()
            st.markdown(f"[Login to Zerodha]({login_url})")
            
            redirect_url = st.text_input("Paste redirect URL after login:")
            
            if st.button("Complete Authentication", type="primary"):
                if redirect_url:
                    try:
                        parsed = urlparse(redirect_url)
                        params = parse_qs(parsed.query)
                        request_token = params.get('request_token', [None])[0]
                        
                        if request_token:
                            kite, error = authenticate_kite(
                                st.session_state.api_key,
                                st.session_state.api_secret,
                                request_token
                            )
                            
                            if kite:
                                st.session_state.kite = kite
                                st.session_state.authenticated = True
                                st.session_state.instruments = load_instruments(kite)
                                st.success(f"✅ Authenticated! Loaded {len(st.session_state.instruments)} stocks")
                                st.rerun()
                            else:
                                st.error(f"Authentication failed: {error}")
                        else:
                            st.error("No request token found in URL")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            if st.button("Back"):
                st.session_state.auth_step = 1
                st.rerun()
    else:
        st.success("✅ Connected to Zerodha")
        
        st.markdown("---")
        st.markdown("### 📊 Market Regime")
        
        regime_emoji = {
            "STRONG_BULL": "🚀",
            "MODERATE_BULL": "📈",
            "NEUTRAL": "➡️",
            "WEAK_BEAR": "📉"
        }
        
        st.metric(
            "Current Regime",
            f"{regime_emoji.get(st.session_state.market_regime, '⚪')} {st.session_state.market_regime.replace('_', ' ')}",
            f"Multiplier: {st.session_state.regime_multiplier:.2f}x"
        )
        
        if st.session_state.nifty_20d_return:
            st.metric("NIFTY 20-Day", f"{st.session_state.nifty_20d_return:+.2f}%")
        
        st.markdown("---")
        st.markdown("### ⏰ Auto-Refresh Settings")
        
        # Auto-refresh toggle
        auto_refresh = st.toggle("Enable Auto-Refresh", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh
        
        if auto_refresh:
            # Market hours check
            if is_market_hours():
                st.success("📈 Market is OPEN")
                
                # Next refresh countdown
                seconds_to_next = get_seconds_to_next_candle()
                minutes_to_next = int(seconds_to_next // 60)
                seconds_remaining = int(seconds_to_next % 60)
                
                st.info(f"⏱️ Next refresh in: {minutes_to_next}:{seconds_remaining:02d}")
                
                # Show next candle time
                next_candle = get_next_5min_candle_time()
                st.caption(f"Next 5-min candle: {next_candle.strftime('%H:%M:%S')}")
            else:
                st.warning("💤 Market is CLOSED")
                st.caption("Auto-refresh works only during market hours (9:15 AM - 3:30 PM)")
        
        if st.session_state.last_refresh_time:
            st.caption(f"Last refresh: {st.session_state.last_refresh_time.strftime('%H:%M:%S')}")
        
        st.markdown("---")
        
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.kite = None
            st.session_state.instruments = {}
            st.session_state.auth_step = 1
            st.session_state.auto_refresh = False
            st.rerun()
    
    st.markdown("---")
    st.markdown("""
    ### STRICT Performance Formula
    
    **Only trending stocks get high R-Factors!**
    
    #### Strict Criteria:
    - **20D < 5%**: Max R-Factor = 2.0
    - **20D < 10%**: Max R-Factor = 3.0
    - **20D < 20%**: Max R-Factor = 4.0
    - **20D > 30%**: Can reach 6.0
    
    #### Auto-Refresh:
    - **Aligned with 5-min candles**
    - **Market hours only**
    - **Updates every 5 minutes**
    """)

# =========================
# MAIN SCANNER
# =========================
if not st.session_state.authenticated:
    st.info("👈 Please authenticate with Zerodha to start scanning")
else:
    # Auto-refresh check
    if should_refresh() and st.session_state.symbols_to_scan:
        scan_data = perform_scan(st.session_state.symbols_to_scan, show_progress=False)
        if scan_data:
            st.session_state.last_scan_results = scan_data
            st.rerun()
    
    # Top bar with refresh status
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
    with col1:
        symbols_input = st.text_input(
            "Enter symbols to scan (comma-separated):",
            value=st.session_state.symbols_to_scan,
            help="Enter NSE symbols without .NS suffix"
        )
        st.session_state.symbols_to_scan = symbols_input
    
    with col2:
        st.write("")
        scan_btn = st.button("🔍 SCAN NOW", type="primary", use_container_width=True)
    
    with col3:
        st.write("")
        if st.button("🔄 Force Refresh", use_container_width=True):
            scan_data = perform_scan(st.session_state.symbols_to_scan)
            if scan_data:
                st.session_state.last_scan_results = scan_data
            st.rerun()
    
    with col4:
        st.write("")
        # Show auto-refresh status
        if st.session_state.auto_refresh:
            if is_market_hours():
                seconds_to_next = get_seconds_to_next_candle()
                st.metric("Next Refresh", f"{int(seconds_to_next//60)}:{int(seconds_to_next%60):02d}", "AUTO ON ✅")
            else:
                st.metric("Auto-Refresh", "Market Closed", "PAUSED ⏸️")
        else:
            st.metric("Auto-Refresh", "Disabled", "OFF ❌")
    
    # Perform scan if button clicked or display last results
    if scan_btn:
        scan_data = perform_scan(symbols_input)
        if scan_data:
            st.session_state.last_scan_results = scan_data
    
    # Display results
    if st.session_state.last_scan_results:
        scan_data = st.session_state.last_scan_results
        results = scan_data['results']
        
        # Market overview
        st.markdown("### 🌍 Market Overview")
        overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
        
        with overview_col1:
            st.metric("Market Regime", scan_data['regime'].replace("_", " "), f"×{scan_data['multiplier']:.2f}")
        with overview_col2:
            st.metric("NIFTY 20-Day", f"{st.session_state.nifty_20d_return:+.2f}%")
        with overview_col3:
            st.metric("Strong Stocks", f"{scan_data['breadth']:.1f}%", ">5% gainers")
        with overview_col4:
            if st.session_state.last_refresh_time:
                st.metric("Last Update", st.session_state.last_refresh_time.strftime("%H:%M:%S"))
            else:
                st.metric("Scan Time", datetime.datetime.now().strftime("%H:%M:%S"))
        
        if results:
            df = pd.DataFrame(results)
            df = df.sort_values('R-Factor', ascending=False)
            
            # Display results
            st.markdown("### 📊 Strict Performance R-Factor Results")
            
            # Categories
            col1, col2, col3, col4 = st.columns(4)
            
            trending_up = len(df[df['20D%'] > 15])
            moderate_up = len(df[(df['20D%'] > 5) & (df['20D%'] <= 15)])
            sideways = len(df[(df['20D%'] > -5) & (df['20D%'] <= 5)])
            trending_down = len(df[df['20D%'] < -5])
            
            col1.metric("🚀 Trending Up", trending_up, ">15% in 20D")
            col2.metric("📈 Moderate Up", moderate_up, "5-15% in 20D")
            col3.metric("➡️ Sideways", sideways, "-5% to +5%")
            col4.metric("📉 Down", trending_down, "<-5%")
            
            # Main table
            st.markdown("### 📋 Detailed Results")
            
            display_df = df[['Symbol', 'R-Factor', '20D%', '10D%', '5D%', 'Change%', 
                           'Momentum', 'Volume', 'Tier', 'LTP', 'Vol Ratio', 'Max Cap']].copy()
            
            def style_rfactor(val):
                if val >= 5.0:
                    return 'background-color: #0d5016; color: white; font-weight: bold'
                elif val >= 4.0:
                    return 'background-color: #1f7a1f; color: white; font-weight: bold'
                elif val >= 3.0:
                    return 'background-color: #28a745; color: white'
                elif val >= 2.0:
                    return 'background-color: #f0ad4e'
                else:
                    return 'background-color: #dc3545; color: white'
            
            def style_returns(val):
                if val > 20:
                    return 'color: #0d5016; font-weight: bold'
                elif val > 10:
                    return 'color: #1f7a1f; font-weight: bold'
                elif val > 5:
                    return 'color: #28a745'
                elif val > 0:
                    return 'color: green'
                else:
                    return 'color: #dc3545'
            
            styled_df = display_df.style.applymap(
                style_rfactor, subset=['R-Factor']
            ).applymap(
                style_returns, subset=['20D%', '10D%', '5D%', 'Change%']
            ).format({
                'R-Factor': '{:.2f}',
                '20D%': '{:+.2f}%',
                '10D%': '{:+.2f}%',
                '5D%': '{:+.2f}%',
                'Change%': '{:+.2f}%',
                'Momentum': '{:.2f}x',
                'Volume': '{:.2f}x',
                'Vol Ratio': '{:.1f}x',
                'LTP': '₹{:.2f}',
                'Max Cap': '{:.1f}'
            })
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Insights
            st.markdown("### 🎯 Key Insights")
            
            if not df[df['20D%'] > 15].empty:
                st.success(f"**Strong Performers:** {', '.join(df[df['20D%'] > 15]['Symbol'].head(5).tolist())}")
            
            if not df[(df['20D%'] > -5) & (df['20D%'] <= 5)].empty:
                sideways_stocks = df[(df['20D%'] > -5) & (df['20D%'] <= 5)]
                st.warning(f"**Sideways Stocks (Low R-Factor):** {', '.join(sideways_stocks['Symbol'].tolist())}")
                st.caption(f"Average R-Factor for sideways stocks: {sideways_stocks['R-Factor'].mean():.2f} (capped)")

st.markdown("---")
st.caption("Strict Performance Scanner with Auto-Refresh: Aligned with 5-minute market candles for real-time updates during market hours.")
