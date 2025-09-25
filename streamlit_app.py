import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime
from kiteconnect import KiteConnect
from urllib.parse import urlparse, parse_qs
import threading
import asyncio
from collections import defaultdict
import json

# Page config for real-time dashboard
st.set_page_config(
    page_title="⚡ Enhanced R-Factor Dashboard - All Markets",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# REAL-TIME SESSION STATE
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
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["ADANIENT", "ADANIGREEN", "IREDA", "HDFCBANK", "RELIANCE", "TCS", "INFY", "AXISBANK", "SAIL", "DLF"]
if 'live_data' not in st.session_state:
    st.session_state.live_data = {}
if 'base_calculations' not in st.session_state:
    st.session_state.base_calculations = {}
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'update_interval' not in st.session_state:
    st.session_state.update_interval = 10
if 'positions' not in st.session_state:
    st.session_state.positions = {}
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'rfactor_history' not in st.session_state:
    st.session_state.rfactor_history = defaultdict(list)
if 'performance_stats' not in st.session_state:
    st.session_state.performance_stats = {}
if 'market_regime' not in st.session_state:
    st.session_state.market_regime = "NEUTRAL"
if 'trading_strategy' not in st.session_state:
    st.session_state.trading_strategy = "WAIT_AND_WATCH"
if 'nifty_20d_return' not in st.session_state:
    st.session_state.nifty_20d_return = 0
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'position_tracker' not in st.session_state:
    st.session_state.position_tracker = {}

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

def get_live_quotes(kite, symbols):
    """Get live quotes for all symbols in one API call"""
    try:
        quote_keys = [f"NSE:{symbol}" for symbol in symbols if symbol in st.session_state.instruments]
        if not quote_keys:
            return {}
        
        quotes = kite.quote(quote_keys)
        return quotes
    except Exception as e:
        st.error(f"Error fetching quotes: {str(e)}")
        return {}

def get_market_time_status():
    """Check if market is open with IST timezone handling"""
    # Get current UTC time and convert to IST (UTC + 5:30)
    utc_now = datetime.datetime.utcnow()
    ist_now = utc_now + datetime.timedelta(hours=5, minutes=30)
    current_time = ist_now.time()
    
    # Check if it's a weekday (Monday=0, Sunday=6)
    weekday = ist_now.weekday()
    
    # Check if it's weekend
    if weekday >= 5:  # Saturday=5, Sunday=6
        return "CLOSED", "🌙 Weekend - Market Closed", "#6c757d"
    
    # Debug: Show current IST time
    current_time_str = ist_now.strftime("%H:%M:%S IST")
    
    # Market hours: 9:15 AM to 3:30 PM IST
    if datetime.time(9, 15) <= current_time <= datetime.time(15, 30):
        if datetime.time(9, 15) <= current_time < datetime.time(10, 0):
            return "OPENING", f"🚀 Opening Hour ({current_time_str})", "#28a745"
        elif datetime.time(10, 0) <= current_time < datetime.time(11, 30):
            return "TRENDING", f"📈 Trending Hour ({current_time_str})", "#20c997" 
        elif datetime.time(11, 30) <= current_time < datetime.time(14, 0):
            return "LUNCH", f"😴 Lunch Time ({current_time_str})", "#ffc107"
        elif datetime.time(14, 0) <= current_time < datetime.time(15, 0):
            return "POWER", f"⚡ Power Hour ({current_time_str})", "#fd7e14"
        else:
            return "CLOSING", f"🔔 Closing ({current_time_str})", "#dc3545"
    elif datetime.time(9, 0) <= current_time < datetime.time(9, 15):
        return "PRE_OPEN", f"🌅 Pre-Market ({current_time_str})", "#17a2b8"
    else:
        return "CLOSED", f"🌙 Market Closed ({current_time_str})", "#6c757d"

# =========================
# ENHANCED MARKET REGIME DETECTION
# =========================
def detect_enhanced_market_regime(relative_strengths):
    """Enhanced market regime detection for bull, bear, and sideways markets"""
    try:
        nifty_20d = st.session_state.nifty_20d_return
        
        # Count outperformers vs underperformers (more nuanced)
        outperformers = sum(1 for s, d in relative_strengths.items() if d['relative_strength'] > 1.2)
        underperformers = sum(1 for s, d in relative_strengths.items() if d['relative_strength'] < 0.8)
        total_stocks = len(relative_strengths) if relative_strengths else 1
        
        outperform_ratio = (outperformers / total_stocks) * 100
        underperform_ratio = (underperformers / total_stocks) * 100
        
        # Enhanced regime classification
        if nifty_20d > 8 and outperform_ratio > 60:
            regime = "STRONG_BULL"
            multiplier = 1.3
            strategy = "AGGRESSIVE_LONG"
        elif nifty_20d > 3 and outperform_ratio > 40:
            regime = "MODERATE_BULL" 
            multiplier = 1.15
            strategy = "SELECTIVE_LONG"
        elif nifty_20d > -3 and abs(outperform_ratio - underperform_ratio) < 20:
            regime = "SIDEWAYS"
            multiplier = 1.0
            strategy = "RANGE_TRADING"
        elif nifty_20d < -8 and underperform_ratio > 60:
            regime = "STRONG_BEAR"
            multiplier = 1.2  # HIGH multiplier for bear market leaders!
            strategy = "DEFENSIVE_LONG_SHORT"
        elif nifty_20d < -3:
            regime = "MODERATE_BEAR"
            multiplier = 1.1
            strategy = "RELATIVE_STRENGTH"
        else:
            regime = "TRANSITIONAL"
            multiplier = 1.0
            strategy = "WAIT_AND_WATCH"
        
        return regime, multiplier, strategy, outperform_ratio, underperform_ratio
        
    except Exception as e:
        return "NEUTRAL", 1.0, "WAIT_AND_WATCH", 50, 50

# =========================
# BASE DATA CALCULATION (ONE TIME)
# =========================
def calculate_base_data_once():
    """Calculate heavy computations once at startup/refresh"""
    if not st.session_state.authenticated:
        return False
    
    try:
        # Get historical data for watchlist (do this once)
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=60)
        
        base_data = {}
        
        # Get NIFTY data once
        nifty_token = 256265
        nifty_hist = st.session_state.kite.historical_data(nifty_token, start_date, end_date, "day")
        
        if len(nifty_hist) >= 20:
            nifty_20d_return = ((nifty_hist[-1]['close'] - nifty_hist[-20]['close']) / nifty_hist[-20]['close']) * 100
        else:
            nifty_20d_return = 0
        
        base_data['nifty_20d_return'] = nifty_20d_return
        base_data['nifty_data'] = nifty_hist
        st.session_state.nifty_20d_return = nifty_20d_return
        
        # Calculate stock historical data once
        for symbol in st.session_state.watchlist:
            if symbol not in st.session_state.instruments:
                continue
                
            try:
                token = st.session_state.instruments[symbol]
                hist_data = st.session_state.kite.historical_data(token, start_date, end_date, "day")
                
                if len(hist_data) >= 20:
                    stock_20d = ((hist_data[-1]['close'] - hist_data[-20]['close']) / hist_data[-20]['close']) * 100
                    stock_10d = ((hist_data[-1]['close'] - hist_data[-10]['close']) / hist_data[-10]['close']) * 100
                    stock_5d = ((hist_data[-1]['close'] - hist_data[-5]['close']) / hist_data[-5]['close']) * 100
                    
                    # Calculate relative strength
                    rel_strength = stock_20d / nifty_20d_return if nifty_20d_return != 0 else stock_20d / 5
                    
                    # Calculate average volume
                    volumes = [d['volume'] for d in hist_data[-20:]]
                    avg_volume = np.mean(volumes) if volumes else 1
                    
                    base_data[symbol] = {
                        'hist_data': hist_data,
                        'stock_20d_return': stock_20d,
                        'stock_10d_return': stock_10d,
                        'stock_5d_return': stock_5d,
                        'relative_strength': rel_strength,
                        'avg_volume_20d': avg_volume
                    }
                else:
                    base_data[symbol] = None
            except Exception as e:
                base_data[symbol] = None
                
        st.session_state.base_calculations = base_data
        return True
        
    except Exception as e:
        st.error(f"Error in base calculations: {str(e)}")
        return False

# =========================
# ENHANCED TIER ASSIGNMENT
# =========================
def calculate_bear_market_tiers(relative_strengths, market_regime):
    """Assign tiers that work in ALL market conditions"""
    
    # Sort by RELATIVE STRENGTH (not absolute returns)
    sorted_symbols = sorted(relative_strengths.items(), 
                          key=lambda x: x[1]['relative_strength'], 
                          reverse=True)
    
    tier_assignments = {}
    nifty_20d = st.session_state.nifty_20d_return
    
    for idx, (symbol, data) in enumerate(sorted_symbols):
        stock_20d = data['stock_20d_return']
        stock_10d = data['stock_10d_return'] 
        stock_5d = data['stock_5d_return']
        rel_strength = data['relative_strength']
        
        # BEAR MARKET LOGIC
        if market_regime in ["STRONG_BEAR", "MODERATE_BEAR"]:
            
            # Tier 1: Defensive leaders (losing less or gaining in bear market)
            if rel_strength > 1.5 or (stock_20d > 0 and nifty_20d < -5):
                tier = "BEAR_TIER_1_DEFENSIVE"
                base_score = 2.8  # HIGHER score for bear market outperformers
            
            # Tier 2: Relative outperformers  
            elif rel_strength > 1.2 or (stock_20d > nifty_20d + 5):
                tier = "BEAR_TIER_2_RELATIVE"
                base_score = 2.4
            
            # Tier 3: Market performers
            elif rel_strength > 0.8:
                tier = "BEAR_TIER_3_MARKET"
                base_score = 2.0
            
            # Tier 4: Underperformers (SHORT candidates)
            elif rel_strength < 0.6 and stock_20d < nifty_20d - 10:
                tier = "BEAR_TIER_4_SHORT"
                base_score = 2.2  # Good for shorting!
                
            else:
                tier = "BEAR_AVOID"
                base_score = 1.0
        
        # BULL MARKET LOGIC
        elif market_regime in ["STRONG_BULL", "MODERATE_BULL"]:
            if stock_20d > 25 and stock_10d > 12:
                tier = "BULL_TIER_1"
                base_score = 2.5
            elif stock_20d > 15 and stock_10d > 7:
                tier = "BULL_TIER_2"  
                base_score = 2.2
            elif stock_20d > 10:
                tier = "BULL_TIER_3"
                base_score = 1.8
            else:
                tier = "BULL_TIER_4"
                base_score = 1.4
        
        # SIDEWAYS MARKET LOGIC
        else:  # SIDEWAYS/TRANSITIONAL
            if abs(stock_20d) > 15:  # High volatility stocks
                tier = "SIDEWAYS_TIER_1_MOMENTUM"
                base_score = 2.0
            elif rel_strength > 1.3:  # Strong relative performers
                tier = "SIDEWAYS_TIER_2_RELATIVE"
                base_score = 1.8
            elif abs(stock_20d) < 5:  # Range-bound stocks
                tier = "SIDEWAYS_TIER_3_RANGE"
                base_score = 1.5
            else:
                tier = "SIDEWAYS_AVOID"
                base_score = 1.2
        
        # Recent momentum adjustment (works in all markets)
        momentum_adj = 1.0
        if stock_5d > 3:
            momentum_adj = 1.15
        elif stock_5d < -3:
            momentum_adj = 0.9 if market_regime.startswith("BULL") else 1.1  # Bear market: down move can be good for shorts
        
        final_score = base_score * momentum_adj
        
        tier_assignments[symbol] = {
            'tier': tier,
            'base_score': final_score,
            'relative_strength': rel_strength,
            'stock_20d_return': stock_20d,
            'stock_10d_return': stock_10d,
            'stock_5d_return': stock_5d,
            'rank': idx + 1,
            'momentum_adj': momentum_adj
        }
    
    return tier_assignments

# =========================
# ENHANCED R-FACTOR CALCULATION
# =========================
def calculate_enhanced_rfactor(symbol, kite, instrument_token, tier_data, regime_multiplier, market_regime, strategy):
    """Enhanced R-Factor that adapts to market conditions"""
    try:
        quote = get_quote_data(kite, symbol, instrument_token)
        if not quote:
            return 0, {}
        
        # Get live data
        ltp = float(quote.get('last_price', 0))
        ohlc = quote.get('ohlc', {})
        volume = float(quote.get('volume', 0))
        
        if not ohlc:
            return 0, {}
            
        open_price = float(ohlc.get('open', 0))
        high = float(ohlc.get('high', 0))
        low = float(ohlc.get('low', 0))
        prev_close = float(ohlc.get('close', 0))
        
        if prev_close == 0 or open_price == 0:
            return 0, {}
        
        change_percent = ((ltp - prev_close) / prev_close) * 100
        intraday_change = ((ltp - open_price) / open_price) * 100
        gap_percent = ((open_price - prev_close) / prev_close) * 100
        
        # Range position
        if high > low:
            range_position = ((ltp - low) / (high - low)) * 100
        else:
            range_position = 50
        
        # Base R-Factor from tier
        base_rfactor = tier_data['base_score'] * regime_multiplier
        
        # STRATEGY-SPECIFIC ADJUSTMENTS
        strategy_multiplier = 1.0
        
        if strategy == "AGGRESSIVE_LONG":
            # Bull market: reward upward moves
            if intraday_change > 2:
                strategy_multiplier = 1.3
            elif intraday_change > 1:
                strategy_multiplier = 1.15
            elif intraday_change < -1:
                strategy_multiplier = 0.85
                
        elif strategy == "DEFENSIVE_LONG_SHORT":
            # Bear market: reward defensive characteristics OR strong downward moves (for shorts)
            if tier_data['tier'].startswith("BEAR_TIER_1"):
                # Defensive stocks: reward stability
                if abs(intraday_change) < 1:  # Stability in bear market is good
                    strategy_multiplier = 1.2
                elif intraday_change > 1:  # Going up in bear market = excellent
                    strategy_multiplier = 1.4
            elif tier_data['tier'] == "BEAR_TIER_4_SHORT":
                # Short candidates: reward downward moves
                if intraday_change < -2:  # Falling fast = good for shorts
                    strategy_multiplier = 1.3
                elif intraday_change < -1:
                    strategy_multiplier = 1.15
                    
        elif strategy == "RELATIVE_STRENGTH":
            # Focus on relative performance vs absolute
            rel_strength = tier_data['relative_strength']
            if rel_strength > 1.5:
                strategy_multiplier = 1.25
            elif rel_strength > 1.2:
                strategy_multiplier = 1.1
            elif rel_strength < 0.8:
                strategy_multiplier = 0.9
                
        elif strategy == "RANGE_TRADING":
            # Sideways market: reward range breakouts
            if abs(intraday_change) > 2:  # Breakout from range
                strategy_multiplier = 1.2
            elif abs(intraday_change) < 0.5:  # Stable in range
                strategy_multiplier = 1.0
        
        # Volume confirmation (universal)
        volume_data = st.session_state.base_calculations.get(symbol, {})
        if volume_data and 'avg_volume_20d' in volume_data:
            volume_ratio = volume / volume_data['avg_volume_20d'] if volume_data['avg_volume_20d'] > 0 else 1
            
            if volume_ratio > 2 and abs(change_percent) > 1:
                volume_multiplier = 1.15
            elif volume_ratio > 1.5:
                volume_multiplier = 1.08
            elif volume_ratio < 0.5:
                volume_multiplier = 0.92
            else:
                volume_multiplier = 1.0
        else:
            volume_multiplier = 1.0
            volume_ratio = 1.0
        
        # Final R-Factor calculation
        final_rfactor = base_rfactor * strategy_multiplier * volume_multiplier
        
        # ADAPTIVE CAPS based on market regime
        if market_regime in ["STRONG_BEAR", "MODERATE_BEAR"]:
            # In bear market: allow higher R-Factors for defensive leaders
            if tier_data['tier'].startswith("BEAR_TIER_1"):
                max_rf = 6.0  # Defensive leaders can reach high scores
            elif tier_data['tier'] == "BEAR_TIER_4_SHORT":
                max_rf = 5.0  # Short candidates can be strong signals
            else:
                max_rf = 4.0
        elif market_regime in ["STRONG_BULL", "MODERATE_BULL"]:
            # Bull market: existing logic
            if tier_data['stock_20d_return'] > 20:
                max_rf = 6.0
            elif tier_data['stock_20d_return'] > 10:
                max_rf = 5.0
            else:
                max_rf = 4.0
        else:
            # Sideways: moderate caps
            max_rf = 4.5
        
        final_rfactor = max(0.5, min(max_rf, final_rfactor))
        
        components = {
            'ltp': ltp,
            'change_percent': round(change_percent, 2),
            'intraday_change': round(intraday_change, 2),
            'gap_percent': round(gap_percent, 2),
            'range_position': round(range_position, 1),
            'volume_ratio': round(volume_ratio, 2),
            'tier': tier_data['tier'],
            'relative_strength': round(tier_data['relative_strength'], 2),
            'stock_20d_return': tier_data['stock_20d_return'],
            'stock_10d_return': tier_data['stock_10d_return'],
            'stock_5d_return': tier_data['stock_5d_return'],
            'base_rfactor': round(base_rfactor, 2),
            'strategy_multiplier': round(strategy_multiplier, 3),
            'volume_multiplier': round(volume_multiplier, 3),
            'market_regime': market_regime,
            'strategy': strategy,
            'max_allowed': max_rf,
            'volume': volume,
            'open': open_price,
            'high': high,
            'low': low
        }
        
        return round(final_rfactor, 2), components
        
    except Exception as e:
        return 0, {}

# =========================
# ENHANCED ALERT GENERATION
# =========================
def generate_enhanced_alerts(live_data, market_regime, strategy):
    """Generate real-time alerts based on market conditions"""
    alerts = []
    current_time = datetime.datetime.now()
    
    # Sort by R-Factor
    sorted_data = sorted(live_data.items(), key=lambda x: x[1].get('rfactor', 0), reverse=True)
    
    for symbol, data in sorted_data[:8]:  # Top 8 stocks
        rfactor = data.get('rfactor', 0)
        components = data.get('components', {})
        
        if rfactor < 3.0:
            continue
            
        tier = components.get('tier', '')
        intraday_change = components.get('intraday_change', 0)
        volume_ratio = components.get('volume_ratio', 1)
        range_pos = components.get('range_position', 50)
        gap = components.get('gap_percent', 0)
        rel_strength = components.get('relative_strength', 1.0)
        
        # Store in history for trend detection
        history = st.session_state.rfactor_history[symbol]
        history.append({'time': current_time, 'rfactor': rfactor, 'price': components.get('ltp', 0)})
        
        # Keep last 10 readings
        if len(history) > 10:
            st.session_state.rfactor_history[symbol] = history[-10:]
        
        # Trend detection
        if len(history) >= 3:
            recent_rf = [h['rfactor'] for h in history[-3:]]
            if recent_rf[-1] > recent_rf[0] + 0.3:  # Rising trend
                trend = "🔥 RISING"
            elif recent_rf[-1] < recent_rf[0] - 0.3:  # Falling trend
                trend = "❄️ FALLING"
            else:
                trend = "➡️ STABLE"
        else:
            trend = "📊 NEW"
        
        # MARKET REGIME SPECIFIC ALERTS
        if market_regime in ["STRONG_BEAR", "MODERATE_BEAR"]:
            # Bear market alerts
            if tier.startswith("BEAR_TIER_1") and rfactor >= 4.5:
                alert_type = "🛡️ DEFENSIVE_LEADER"
                message = f"{symbol}: R-Factor {rfactor} | DefensiveStr: {rel_strength:.1f} | {trend}"
            elif tier == "BEAR_TIER_4_SHORT" and rfactor >= 4.0 and intraday_change < -2:
                alert_type = "🩸 SHORT_OPPORTUNITY"
                message = f"{symbol}: R-Factor {rfactor} | Falling: {intraday_change:.1f}% | {trend}"
            elif tier.startswith("BEAR_TIER_2") and intraday_change > 1:
                alert_type = "📈 RELATIVE_STRENGTH"
                message = f"{symbol}: R-Factor {rfactor} | RelStr: {rel_strength:.1f} | Rising in bear market"
            else:
                continue
                
        elif market_regime in ["STRONG_BULL", "MODERATE_BULL"]:
            # Bull market alerts
            if rfactor >= 4.5 and intraday_change > 2 and volume_ratio > 2:
                alert_type = "🚀 MOMENTUM_BREAKOUT"
                message = f"{symbol}: R-Factor {rfactor} | Move: {intraday_change:+.1f}% | Vol: {volume_ratio:.1f}x"
            elif rfactor >= 4.0 and gap > 1.5 and intraday_change > 0:
                alert_type = "📈 GAP_CONTINUATION"
                message = f"{symbol}: R-Factor {rfactor} | Gap: {gap:+.1f}% | Continuation"
            elif rfactor >= 4.0 and intraday_change < -1 and components.get('stock_20d_return', 0) > 15:
                alert_type = "💎 PULLBACK_ENTRY"
                message = f"{symbol}: R-Factor {rfactor} | Dip: {intraday_change:.1f}% | Strong base"
            else:
                continue
                
        else:
            # Sideways market alerts
            if rfactor >= 4.0 and abs(intraday_change) > 2:
                alert_type = "⚡ RANGE_BREAKOUT"
                message = f"{symbol}: R-Factor {rfactor} | Break: {intraday_change:+.1f}% | Volume: {volume_ratio:.1f}x"
            elif rfactor >= 3.8 and tier.endswith("RELATIVE"):
                alert_type = "🎯 RELATIVE_LEADER"
                message = f"{symbol}: R-Factor {rfactor} | RelStr: {rel_strength:.1f} | {trend}"
            else:
                continue
        
        alerts.append({
            'type': alert_type,
            'message': message,
            'symbol': symbol,
            'rfactor': rfactor,
            'trend': trend,
            'time': current_time,
            'priority': 1 if rfactor >= 5.0 else 2 if rfactor >= 4.0 else 3
        })
    
    # Sort by priority and R-Factor
    alerts.sort(key=lambda x: (x['priority'], -x['rfactor']))
    return alerts

# =========================
# ENHANCED TRADING SIGNALS
# =========================
def generate_enhanced_trading_signals(results_df, market_regime, strategy):
    """Generate trading signals adapted to market conditions"""
    signals = []
    
    for _, row in results_df.iterrows():
        symbol = row['Symbol']
        rfactor = row['R-Factor']
        components = row.get('_components', {})
        
        tier = components.get('tier', '')
        intraday_change = components.get('intraday_change', 0)
        rel_strength = components.get('relative_strength', 1.0)
        volume_ratio = components.get('volume_ratio', 1.0)
        
        # BEAR MARKET SIGNALS
        if market_regime in ["STRONG_BEAR", "MODERATE_BEAR"]:
            
            if tier == "BEAR_TIER_1_DEFENSIVE" and rfactor >= 4.5:
                signal = "🛡️ DEFENSIVE_BUY"
                reason = f"Defensive leader in bear market (RelStr: {rel_strength:.1f})"
                confidence = 90
                
            elif tier == "BEAR_TIER_2_RELATIVE" and rfactor >= 4.0 and intraday_change > 1:
                signal = "📈 RELATIVE_BUY"
                reason = f"Outperforming market (RelStr: {rel_strength:.1f}, Up: {intraday_change:+.1f}%)"
                confidence = 80
                
            elif tier == "BEAR_TIER_4_SHORT" and rfactor >= 4.0 and intraday_change < -2:
                signal = "🩸 SHORT_CANDIDATE"  
                reason = f"Weak stock breaking down (Down: {intraday_change:.1f}%)"
                confidence = 85
                
            elif rfactor < 2.5:
                signal = "❌ AVOID"
                reason = "Underperforming in bear market"
                confidence = 70
            else:
                signal = "⏸️ WAIT"
                reason = "No clear bear market setup"
                confidence = 50
        
        # BULL MARKET SIGNALS
        elif market_regime in ["STRONG_BULL", "MODERATE_BULL"]:
            
            if rfactor >= 5.0 and intraday_change > 2:
                signal = "🚀 STRONG_BUY"
                reason = f"Explosive momentum (R-Factor: {rfactor}, Move: {intraday_change:+.1f}%)"
                confidence = 95
                
            elif rfactor >= 4.0 and volume_ratio > 1.5:
                signal = "📈 BUY"
                reason = f"Strong setup with volume (R-Factor: {rfactor})"
                confidence = 85
                
            elif rfactor >= 3.5:
                signal = "👀 MONITOR"
                reason = "Good strength, watch for entry"
                confidence = 65
            else:
                signal = "❌ AVOID"
                reason = "Weak in strong market"
                confidence = 70
        
        # SIDEWAYS MARKET SIGNALS
        else:  # SIDEWAYS/TRANSITIONAL
            
            if rfactor >= 4.5 and abs(intraday_change) > 2:
                signal = "⚡ BREAKOUT"
                reason = f"Range breakout (R-Factor: {rfactor}, Move: {intraday_change:+.1f}%)"
                confidence = 80
                
            elif rfactor >= 4.0 and tier.endswith("RELATIVE"):
                signal = "🎯 SELECTIVE_BUY"
                reason = f"Relative strength leader (RelStr: {rel_strength:.1f})"
                confidence = 75
                
            elif rfactor >= 3.0:
                signal = "⏸️ WAIT"
                reason = "Decent but wait for clear direction"
                confidence = 50
            else:
                signal = "❌ AVOID"
                reason = "Weak in choppy market"
                confidence = 60
        
        signals.append({
            'Symbol': symbol,
            'Signal': signal,
            'Reason': reason,
            'Confidence': confidence,
            'R-Factor': rfactor,
            'Strategy': strategy,
            'Tier': tier,
            'RelativeStrength': rel_strength,
            'IntradayChange': intraday_change
        })
    
    return signals

# =========================
# UPDATE LIVE DATA
# =========================
def update_live_data():
    """Update live data for all watchlist stocks"""
    if not st.session_state.authenticated or not st.session_state.base_calculations:
        return False
    
    try:
        # Get live quotes for all symbols
        quotes = get_live_quotes(st.session_state.kite, st.session_state.watchlist)
        
        # Get relative strengths from base calculations
        relative_strengths = {}
        base_data = st.session_state.base_calculations
        
        for symbol in st.session_state.watchlist:
            if symbol in base_data and base_data[symbol]:
                relative_strengths[symbol] = base_data[symbol]
        
        # Detect enhanced market regime
        regime, multiplier, strategy, outperform_ratio, underperform_ratio = detect_enhanced_market_regime(relative_strengths)
        st.session_state.market_regime = regime
        st.session_state.trading_strategy = strategy
        
        # Calculate tier assignments
        tier_assignments = calculate_bear_market_tiers(relative_strengths, regime)
        
        live_data = {}
        
        for symbol in st.session_state.watchlist:
            quote_key = f"NSE:{symbol}"
            if quote_key in quotes and symbol in tier_assignments:
                quote = quotes[quote_key]
                tier_data = tier_assignments[symbol]
                
                rfactor, components = calculate_enhanced_rfactor(
                    symbol, 
                    st.session_state.kite, 
                    st.session_state.instruments[symbol], 
                    tier_data, 
                    multiplier, 
                    regime, 
                    strategy
                )
                
                live_data[symbol] = {
                    'rfactor': rfactor,
                    'components': components,
                    'last_updated': datetime.datetime.now()
                }
        
        st.session_state.live_data = live_data
        st.session_state.last_update = datetime.datetime.now()
        
        # Generate alerts
        alerts = generate_enhanced_alerts(live_data, regime, strategy)
        st.session_state.alerts = alerts
        
        return True
        
    except Exception as e:
        st.error(f"Error updating live data: {str(e)}")
        return False

# =========================
# DISPLAY FUNCTIONS
# =========================
def display_enhanced_market_overview(regime, strategy, outperform_ratio, underperform_ratio):
    """Display market overview adapted to current conditions"""
    
    st.markdown("### 🌍 Enhanced Market Analysis")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        regime_colors = {
            "STRONG_BULL": "#28a745",
            "MODERATE_BULL": "#20c997", 
            "SIDEWAYS": "#ffc107",
            "MODERATE_BEAR": "#fd7e14",
            "STRONG_BEAR": "#dc3545",
            "TRANSITIONAL": "#6c757d"
        }
        color = regime_colors.get(regime, "#6c757d")
        st.markdown(f'<div style="background-color: {color}; color: white; padding: 10px; border-radius: 5px; text-align: center;"><b>{regime.replace("_", " ")}</b></div>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Strategy", strategy.replace("_", " "))
    
    with col3:
        st.metric("NIFTY 20D", f"{st.session_state.nifty_20d_return:+.1f}%")
    
    with col4:
        st.metric("Outperformers", f"{outperform_ratio:.0f}%")
    
    with col5:
        st.metric("Underperformers", f"{underperform_ratio:.0f}%")
    
    # Strategy explanation
    strategy_explanations = {
        "AGGRESSIVE_LONG": "🚀 Go aggressive on strong momentum stocks",
        "SELECTIVE_LONG": "🎯 Pick only the best setups",
        "DEFENSIVE_LONG_SHORT": "🛡️ Focus on defensive leaders + short weak stocks",
        "RELATIVE_STRENGTH": "📊 Trade stocks outperforming the market",
        "RANGE_TRADING": "↔️ Look for breakouts from consolidation",
        "WAIT_AND_WATCH": "⏸️ Preserve capital, wait for clarity"
    }
    
    explanation = strategy_explanations.get(strategy, "Monitor market conditions")
    st.info(f"**Strategy Focus:** {explanation}")

# =========================
# AUTHENTICATION
# =========================
def authenticate_kite(api_key, api_secret, request_token):
    try:
        kite = KiteConnect(api_key=api_key)
        data = kite.generate_session(request_token, api_secret=api_secret)
        kite.set_access_token(data["access_token"])
        return kite, None
    except Exception as e:
        return None, str(e)

def load_instruments(kite):
    try:
        instruments = kite.instruments("NSE")
        inst_map = {}
        for inst in instruments:
            if inst['exchange'] == 'NSE' and inst['segment'] == 'NSE':
                inst_map[inst['tradingsymbol']] = inst['instrument_token']
        return inst_map
    except Exception as e:
        return {}

# =========================
# MAIN DASHBOARD
# =========================
def main():
    st.title("⚡ Enhanced R-Factor Dashboard - All Market Conditions")
    
    # Authentication check
    if not st.session_state.authenticated:
        with st.expander("🔐 Quick Authentication", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                api_key = st.text_input("API Key", value=st.session_state.api_key)
                api_secret = st.text_input("API Secret", value=st.session_state.api_secret, type="password")
                
                if st.button("Generate Login URL", type="primary"):
                    if api_key and api_secret:
                        st.session_state.api_key = api_key
                        st.session_state.api_secret = api_secret
                        st.session_state.auth_step = 2
                        kite = KiteConnect(api_key=api_key)
                        login_url = kite.login_url()
                        st.success("Login URL generated!")
                        st.markdown(f"[Click here to login]({login_url})")
                        st.rerun()
                    else:
                        st.error("Enter both API Key and Secret")
            
            with col2:
                if st.session_state.auth_step == 2:
                    kite = KiteConnect(api_key=st.session_state.api_key)
                    login_url = kite.login_url()
                    st.markdown(f"[🔗 Login to Zerodha]({login_url})")
                    
                    redirect_url = st.text_input("Paste redirect URL after login:")
                    if st.button("🚀 Connect") and redirect_url:
                        try:
                            parsed = urlparse(redirect_url)
                            params = parse_qs(parsed.query)
                            request_token = params.get('request_token', [None])[0]
                            
                            if request_token:
                                kite_obj, error = authenticate_kite(st.session_state.api_key, st.session_state.api_secret, request_token)
                                if kite_obj:
                                    st.session_state.kite = kite_obj
                                    st.session_state.authenticated = True
                                    st.session_state.instruments = load_instruments(kite_obj)
                                    st.success("✅ Connected!")
                                    st.rerun()
                                else:
                                    st.error(f"Failed: {error}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        return
    
    # Market status
    market_status, market_label, market_color = get_market_time_status()
    
    # Dashboard header
    header_col1, header_col2, header_col3, header_col4, header_col5 = st.columns([2, 2, 2, 2, 2])
    
    with header_col1:
        st.markdown(f'<div style="background-color: {market_color}; color: white; padding: 10px; border-radius: 5px; text-align: center;"><b>{market_label}</b></div>', unsafe_allow_html=True)
    
    with header_col2:
        if st.session_state.last_update:
            st.metric("Last Update", st.session_state.last_update.strftime("%H:%M:%S"))
        else:
            st.metric("Last Update", "Never")
    
    with header_col3:
        active_stocks = len([s for s, d in st.session_state.live_data.items() if d.get('rfactor', 0) > 3.5])
        st.metric("Active Signals", active_stocks, f"R-Factor > 3.5")
    
    with header_col4:
        if st.session_state.live_data:
            avg_rf = np.mean([d.get('rfactor', 0) for d in st.session_state.live_data.values()])
            st.metric("Avg R-Factor", f"{avg_rf:.2f}")
        else:
            st.metric("Avg R-Factor", "0.00")
    
    with header_col5:
        total_alerts = len(st.session_state.alerts)
        st.metric("Live Alerts", total_alerts)
    
    # Control panel
    st.markdown("---")
    control_col1, control_col2, control_col3, control_col4, control_col5 = st.columns(5)
    
    with control_col1:
        if st.button("📊 Initialize Base Data", type="primary"):
            with st.spinner("Calculating base data..."):
                if calculate_base_data_once():
                    st.success("✅ Base data ready!")
                    st.rerun()
                else:
                    st.error("❌ Failed to initialize")
    
    with control_col2:
        if st.button("🔄 Update Live Data"):
            with st.spinner("Updating..."):
                if update_live_data():
                    st.success("✅ Updated!")
                    st.rerun()
    
    with control_col3:
        auto_update = st.checkbox("🤖 Auto Update", value=st.session_state.is_running)
        if auto_update != st.session_state.is_running:
            st.session_state.is_running = auto_update
    
    with control_col4:
        update_interval = st.selectbox("Update Every", [5, 10, 15, 30], index=1, format_func=lambda x: f"{x}s")
        st.session_state.update_interval = update_interval
    
    with control_col5:
        watchlist_str = st.text_input("Watchlist", value=",".join(st.session_state.watchlist), help="Comma-separated symbols")
        if watchlist_str:
            new_watchlist = [s.strip().upper() for s in watchlist_str.split(",")]
            if new_watchlist != st.session_state.watchlist:
                st.session_state.watchlist = new_watchlist
                st.session_state.base_calculations = {}

    # Main dashboard content
    if not st.session_state.base_calculations:
        st.info("👆 Click 'Initialize Base Data' to start the enhanced dashboard")
        return
    
    if not st.session_state.live_data:
        st.info("👆 Click 'Update Live Data' to see real-time enhanced R-Factors")
        return
    
    # Enhanced market overview
    regime = st.session_state.market_regime
    strategy = st.session_state.trading_strategy
    
    # Calculate ratios for display
    base_data = st.session_state.base_calculations
    relative_strengths = {s: data for s, data in base_data.items() if s != 'nifty_20d_return' and s != 'nifty_data' and data}
    
    if relative_strengths:
        outperformers = sum(1 for s, d in relative_strengths.items() if d.get('relative_strength', 1) > 1.2)
        underperformers = sum(1 for s, d in relative_strengths.items() if d.get('relative_strength', 1) < 0.8)
        total = len(relative_strengths)
        outperform_ratio = (outperformers / total) * 100 if total > 0 else 0
        underperform_ratio = (underperformers / total) * 100 if total > 0 else 0
    else:
        outperform_ratio = underperform_ratio = 0
    
    display_enhanced_market_overview(regime, strategy, outperform_ratio, underperform_ratio)
    
    # Live alerts section
    if st.session_state.alerts:
        st.markdown("### 🚨 LIVE ENHANCED ALERTS")
        alert_container = st.container()
        with alert_container:
            for alert in st.session_state.alerts[:6]:  # Top 6 alerts
                alert_type = alert['type']
                message = alert['message']
                
                if "DEFENSIVE" in alert_type or "RELATIVE_STRENGTH" in alert_type:
                    st.success(f"{alert_type}: {message}")
                elif "SHORT" in alert_type:
                    st.error(f"{alert_type}: {message}")
                elif "BREAKOUT" in alert_type or "MOMENTUM" in alert_type:
                    st.info(f"{alert_type}: {message}")
                elif "PULLBACK" in alert_type:
                    st.warning(f"{alert_type}: {message}")
                else:
                    st.info(f"{alert_type}: {message}")
    
    # Live Enhanced R-Factor table
    st.markdown("### 📊 Live Enhanced R-Factor Dashboard")
    
    # Prepare data for display
    display_data = []
    for symbol, data in st.session_state.live_data.items():
        if not data:
            continue
            
        rfactor = data.get('rfactor', 0)
        components = data.get('components', {})
        
        # Get trend
        history = st.session_state.rfactor_history.get(symbol, [])
        if len(history) >= 2:
            prev_rf = history[-2]['rfactor']
            rf_change = rfactor - prev_rf
            trend_emoji = "🔥" if rf_change > 0.1 else "❄️" if rf_change < -0.1 else "➡️"
        else:
            rf_change = 0
            trend_emoji = "📊"
        
        # Determine signal based on regime and rfactor
        tier = components.get('tier', '')
        intraday_change = components.get('intraday_change', 0)
        
        if regime in ["STRONG_BEAR", "MODERATE_BEAR"]:
            if tier.startswith("BEAR_TIER_1") and rfactor >= 4.0:
                signal = "🛡️ DEF_BUY"
            elif tier == "BEAR_TIER_4_SHORT" and rfactor >= 4.0:
                signal = "🩸 SHORT"
            elif rfactor >= 3.5:
                signal = "📈 REL_BUY"
            else:
                signal = "❌ AVOID"
        elif regime in ["STRONG_BULL", "MODERATE_BULL"]:
            if rfactor >= 4.5:
                signal = "🚀 STR_BUY"
            elif rfactor >= 4.0:
                signal = "📈 BUY"
            elif rfactor >= 3.5:
                signal = "👀 MONITOR"
            else:
                signal = "❌ AVOID"
        else:
            if rfactor >= 4.0:
                signal = "⚡ BREAKOUT"
            elif rfactor >= 3.5:
                signal = "🎯 SEL_BUY"
            else:
                signal = "⏸️ WAIT"
        
        display_data.append({
            'Symbol': symbol,
            'R-Factor': rfactor,
            'Signal': signal,
            'Trend': trend_emoji,
            'Change': f"{rf_change:+.2f}",
            'LTP': f"₹{components.get('ltp', 0):.1f}",
            'Day%': f"{components.get('intraday_change', 0):+.1f}%",
            'Gap%': f"{components.get('gap_percent', 0):+.1f}%",
            'Vol': f"{components.get('volume_ratio', 1):.1f}x",
            'RelStr': f"{components.get('relative_strength', 1):.1f}",
            '20D%': f"{components.get('stock_20d_return', 0):+.1f}%",
            'Tier': components.get('tier', 'N/A').replace('BEAR_TIER_', 'B').replace('BULL_TIER_', 'U').replace('SIDEWAYS_TIER_', 'S').replace('_DEFENSIVE', 'D').replace('_RELATIVE', 'R').replace('_SHORT', '$')
        })
    
    if display_data:
        df = pd.DataFrame(display_data)
        df = df.sort_values('R-Factor', ascending=False)
        
        # Style the dataframe
        def highlight_enhanced_rfactor(val):
            val_num = float(val)
            if val_num >= 5.5:
                return 'background-color: #0d5016; color: white; font-weight: bold'
            elif val_num >= 4.5:
                return 'background-color: #28a745; color: white; font-weight: bold'
            elif val_num >= 4.0:
                return 'background-color: #20c997; color: white; font-weight: bold'
            elif val_num >= 3.5:
                return 'background-color: #17a2b8; color: white'
            elif val_num >= 3.0:
                return 'background-color: #ffc107'
            else:
                return 'background-color: #dc3545; color: white'
        
        def highlight_signal(val):
            if "STR_BUY" in val or "DEF_BUY" in val:
                return 'color: #0d5016; font-weight: bold'
            elif "BUY" in val or "BREAKOUT" in val:
                return 'color: #28a745; font-weight: bold'
            elif "SHORT" in val:
                return 'color: #dc3545; font-weight: bold'
            elif "MONITOR" in val or "SEL_BUY" in val:
                return 'color: #17a2b8; font-weight: bold'
            elif "AVOID" in val:
                return 'color: #6c757d'
            else:
                return ''
        
        def highlight_trend(val):
            if "🔥" in val:
                return 'color: green; font-weight: bold'
            elif "❄️" in val:
                return 'color: red; font-weight: bold'
            else:
                return ''
        
        styled_df = df.style.applymap(highlight_enhanced_rfactor, subset=['R-Factor']).applymap(highlight_signal, subset=['Signal']).applymap(highlight_trend, subset=['Trend'])
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Enhanced Trading Signals
        st.markdown("### ⚡ Enhanced Trading Signals")
        
        signals_data = []
        for _, row in df.iterrows():
            if row['R-Factor'] >= 3.5:  # Only show actionable signals
                signals_data.append(row)
        
        if signals_data:
            signal_cols = st.columns(min(3, len(signals_data)))
            
            for idx, row in enumerate(signals_data[:3]):  # Top 3 signals
                with signal_cols[idx]:
                    st.markdown(f"**#{idx+1}: {row['Symbol']}**")
                    st.write(f"R-Factor: {row['R-Factor']} {row['Trend']}")
                    st.write(f"Signal: {row['Signal']}")
                    st.write(f"Price: {row['LTP']} ({row['Day%']})")
                    st.write(f"RelStr: {row['RelStr']} | 20D: {row['20D%']}")
                    
                    # Action recommendation
                    if row['R-Factor'] >= 4.5:
                        st.success("🎯 HIGH PRIORITY")
                    elif row['R-Factor'] >= 4.0:
                        st.info("📈 GOOD SETUP")
                    else:
                        st.warning("👀 MONITOR")
        
        # Performance summary by market regime
        st.markdown("### 📈 Market Regime Performance")
        
        regime_col1, regime_col2, regime_col3 = st.columns(3)
        
        with regime_col1:
            strong_signals = len([r for r in signals_data if r['R-Factor'] >= 4.5])
            st.metric("Strong Signals", strong_signals, "R-Factor ≥ 4.5")
        
        with regime_col2:
            avg_rel_str = np.mean([float(r['RelStr']) for r in signals_data]) if signals_data else 1.0
            st.metric("Avg Rel Strength", f"{avg_rel_str:.1f}", "vs Market")
        
        with regime_col3:
            if regime.startswith("BEAR"):
                defensive_count = len([r for r in signals_data if "DEF" in r['Signal'] or "REL" in r['Signal']])
                st.metric("Defensive Plays", defensive_count, "Bear market leaders")
            elif regime.startswith("BULL"):
                momentum_count = len([r for r in signals_data if "BUY" in r['Signal']])
                st.metric("Momentum Plays", momentum_count, "Bull market leaders")
            else:
                breakout_count = len([r for r in signals_data if "BREAKOUT" in r['Signal']])
                st.metric("Breakout Plays", breakout_count, "Range breakouts")
    
    # Auto-update logic
    if st.session_state.is_running:
        # Update during market hours or pre-market
        if market_status in ["OPENING", "TRENDING", "POWER", "PRE_OPEN"]:
            # Show countdown timer
            countdown_placeholder = st.empty()
            for i in range(st.session_state.update_interval, 0, -1):
                countdown_placeholder.info(f"⏰ Next update in {i} seconds...")
                time.sleep(1)
            countdown_placeholder.empty()
            
            # Update data
            update_live_data()
            st.rerun()
        elif market_status == "LUNCH":
            # Slower updates during lunch
            st.info("😴 Lunch time - Updates every 30 seconds")
            time.sleep(30)
            update_live_data()
            st.rerun()
        else:
            # Market closed - stop auto updates but keep dashboard alive
            st.info("🌙 Auto-update paused - Market is closed")
            time.sleep(60)  # Check every minute if market reopens
            st.rerun()

if __name__ == "__main__":
    main()
