import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime
from kiteconnect import KiteConnect
from urllib.parse import urlparse, parse_qs
import logging
import threading

# Page config
st.set_page_config(
    page_title="⚡ Intraday R-Factor Scanner",
    page_icon="🎯",
    layout="wide"
)

st.title("⚡ Intraday R-Factor Scanner - Real-Time Stock Selection")
st.markdown("> _Pre-select strong stocks, then trade intraday setups_")

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
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 5  # minutes
if 'intraday_alerts' not in st.session_state:
    st.session_state.intraday_alerts = []
if 'focus_list' not in st.session_state:
    st.session_state.focus_list = []

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

def get_current_time_category():
    """Determine current market time category for intraday strategies"""
    now = datetime.datetime.now().time()
    
    if datetime.time(9, 15) <= now < datetime.time(10, 0):
        return "OPENING_HOUR"
    elif datetime.time(10, 0) <= now < datetime.time(11, 30):
        return "TRENDING_HOUR"
    elif datetime.time(11, 30) <= now < datetime.time(14, 0):
        return "LUNCH_LULL"
    elif datetime.time(14, 0) <= now < datetime.time(15, 0):
        return "POWER_HOUR"
    elif datetime.time(15, 0) <= now <= datetime.time(15, 15):
        return "CLOSING_HOUR"
    else:
        return "PRE_MARKET"

# =========================
# INTRADAY STRENGTH CALCULATION
# =========================
def calculate_intraday_strength(kite, symbol, instrument_token):
    """Calculate intraday-specific strength factors"""
    try:
        quote = get_quote_data(kite, symbol, instrument_token)
        if not quote:
            return 0, {}
        
        ltp = float(quote.get('last_price', 0))
        ohlc = quote.get('ohlc', {})
        open_price = float(ohlc.get('open', 0))
        high = float(ohlc.get('high', 0))
        low = float(ohlc.get('low', 0))
        prev_close = float(ohlc.get('close', 0))
        volume = float(quote.get('volume', 0))
        
        # Get historical data for VWAP calculation
        try:
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=5)
            hist_data = kite.historical_data(instrument_token, start_date, end_date, "15minute")
            
            # Calculate VWAP for today
            today_data = [d for d in hist_data if d['date'].date() == datetime.date.today()]
            if today_data:
                vwap = sum([(d['high'] + d['low'] + d['close']) / 3 * d['volume'] for d in today_data]) / sum([d['volume'] for d in today_data])
            else:
                vwap = ltp
        except:
            vwap = ltp
        
        # 1. Gap Strength
        gap_percent = ((open_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
        
        # 2. Intraday Performance
        intraday_change = ((ltp - open_price) / open_price) * 100 if open_price > 0 else 0
        
        # 3. Range Analysis
        if high > low:
            range_position = ((ltp - low) / (high - low)) * 100  # Where price is in day's range
        else:
            range_position = 50
            
        # 4. VWAP Position
        vwap_position = "ABOVE" if ltp > vwap else "BELOW"
        vwap_distance = ((ltp - vwap) / vwap) * 100 if vwap > 0 else 0
        
        # 5. Volume Analysis
        avg_volume_multiplier = volume / (volume / 6.5) if volume > 0 else 1  # Rough intraday volume estimation
        
        # 6. Momentum Score
        momentum_score = 0
        if gap_percent > 2 and intraday_change > 0:
            momentum_score += 2  # Strong gap up continuation
        elif gap_percent > 0.5 and intraday_change > gap_percent:
            momentum_score += 1.5  # Gap up with acceleration
        elif gap_percent < 0 and intraday_change > 1:
            momentum_score += 1.5  # Recovery from gap down
        elif intraday_change > 2:
            momentum_score += 1  # Strong intraday move
            
        if vwap_position == "ABOVE" and vwap_distance > 0.5:
            momentum_score += 0.5
            
        if range_position > 70:  # Trading near highs
            momentum_score += 0.5
        elif range_position < 30:  # Trading near lows
            momentum_score -= 0.5
        
        # Intraday strength composite (0-5 scale)
        intraday_strength = min(5.0, max(0, momentum_score))
        
        intraday_data = {
            'gap_percent': round(gap_percent, 2),
            'intraday_change': round(intraday_change, 2),
            'range_position': round(range_position, 1),
            'vwap_position': vwap_position,
            'vwap_distance': round(vwap_distance, 2),
            'volume_multiplier': round(avg_volume_multiplier, 1),
            'momentum_score': round(momentum_score, 2),
            'intraday_strength': round(intraday_strength, 2),
            'current_price': ltp,
            'open_price': open_price,
            'high': high,
            'low': low,
            'vwap': round(vwap, 2)
        }
        
        return intraday_strength, intraday_data
        
    except Exception as e:
        return 0, {}

# =========================
# BASE R-FACTOR CALCULATION (Kept from original)
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
                stock_20d_return = ((hist_data[-1]['close'] - hist_data[-20]['close']) / hist_data[-20]['close']) * 100
                stock_10d_return = ((hist_data[-1]['close'] - hist_data[-10]['close']) / hist_data[-10]['close']) * 100
                stock_5d_return = ((hist_data[-1]['close'] - hist_data[-5]['close']) / hist_data[-5]['close']) * 100
                
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

def assign_intraday_tiers(relative_strengths):
    """Assign tiers optimized for intraday trading"""
    sorted_symbols = sorted(relative_strengths.items(), 
                          key=lambda x: x[1]['stock_20d_return'], 
                          reverse=True)
    
    tier_assignments = {}
    
    for idx, (symbol, data) in enumerate(sorted_symbols):
        stock_20d = data['stock_20d_return']
        stock_10d = data['stock_10d_return']
        stock_5d = data['stock_5d_return']
        
        # Intraday-focused tier assignment
        if stock_20d > 20 and stock_10d > 10 and stock_5d > 0:
            tier = "INTRADAY_TIER_1"  # Best for intraday
            base_score = 2.5
        elif stock_20d > 15 and stock_10d > 7 and stock_5d > -2:
            tier = "INTRADAY_TIER_2"  # Good for intraday
            base_score = 2.2
        elif stock_20d > 10 and stock_10d > 5:
            tier = "INTRADAY_TIER_3"  # Acceptable for intraday
            base_score = 1.8
        elif stock_20d > 5:
            tier = "INTRADAY_TIER_4"  # Watch list only
            base_score = 1.5
        else:
            tier = "AVOID_INTRADAY"  # Avoid for intraday
            base_score = 1.0
        
        # Boost for recent momentum (important for intraday)
        if stock_5d > 3:
            base_score *= 1.2
        elif stock_5d < -3:
            base_score *= 0.8
        
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
# INTRADAY R-FACTOR CALCULATION
# =========================
def calculate_intraday_rfactor(symbol, kite, instrument_token, tier_data, regime_multiplier, all_stocks_data):
    """
    Enhanced R-Factor for intraday trading
    Combines base R-Factor with real-time intraday strength
    """
    try:
        quote = get_quote_data(kite, symbol, instrument_token)
        hist_data = all_stocks_data.get(symbol, {}).get('hist_data', [])
        
        if not quote and not hist_data:
            return 0, {}
        
        # Base R-Factor from swing analysis
        base_rfactor = tier_data['base_score'] * regime_multiplier
        
        # Get intraday strength
        intraday_strength, intraday_data = calculate_intraday_strength(kite, symbol, instrument_token)
        
        # Time-based adjustments
        time_category = get_current_time_category()
        time_multiplier = {
            "PRE_MARKET": 0.8,
            "OPENING_HOUR": 1.2,  # High volatility, good for trading
            "TRENDING_HOUR": 1.3,  # Best time for intraday
            "LUNCH_LULL": 0.7,  # Avoid trading
            "POWER_HOUR": 1.1,
            "CLOSING_HOUR": 0.6  # Risk management
        }.get(time_category, 1.0)
        
        # Combine base R-Factor with intraday strength
        # 60% base strength, 40% intraday performance
        combined_rfactor = (base_rfactor * 0.6) + (intraday_strength * 0.8)
        
        # Apply time multiplier
        final_rfactor = combined_rfactor * time_multiplier
        
        # Intraday-specific caps
        stock_20d = tier_data['stock_20d_return']
        if stock_20d < 5:
            max_rfactor = 2.5  # Conservative cap for weak trends
        elif stock_20d < 15:
            max_rfactor = 4.0
        else:
            max_rfactor = 6.0  # Strong trends can reach higher
        
        # Minimum based on tier
        if tier_data['tier'] == "AVOID_INTRADAY":
            min_rfactor = 0.5
        else:
            min_rfactor = 1.0
        
        final_rfactor = max(min_rfactor, min(max_rfactor, final_rfactor))
        final_rfactor = round(final_rfactor, 2)
        
        # Get basic quote data
        ltp = float(quote.get('last_price', 0)) if quote else hist_data[-1]['close']
        volume = float(quote.get('volume', 0)) if quote else hist_data[-1]['volume']
        change_percent = 0
        
        if quote and 'ohlc' in quote:
            ohlc = quote['ohlc']
            prev_close = float(ohlc.get('close', 0))
            if prev_close > 0:
                change_percent = ((ltp - prev_close) / prev_close) * 100
        
        components = {
            'ltp': ltp,
            'change_percent': round(change_percent, 2),
            'volume': volume,
            'tier': tier_data['tier'],
            'base_rfactor': round(base_rfactor, 2),
            'intraday_strength': intraday_strength,
            'time_category': time_category,
            'time_multiplier': time_multiplier,
            'combined_rfactor': round(combined_rfactor, 2),
            'stock_20d_return': tier_data['stock_20d_return'],
            'stock_10d_return': tier_data['stock_10d_return'],
            'stock_5d_return': tier_data['stock_5d_return'],
            'rank': tier_data['rank'],
            'intraday_data': intraday_data
        }
        
        return final_rfactor, components
        
    except Exception as e:
        st.error(f"Error calculating Intraday R-Factor for {symbol}: {str(e)}")
        return 0, {}

# =========================
# INTRADAY ALERTS SYSTEM
# =========================
def generate_intraday_alerts(results_df):
    """Generate real-time trading alerts for intraday"""
    alerts = []
    time_category = get_current_time_category()
    
    if time_category in ["LUNCH_LULL", "CLOSING_HOUR"]:
        return ["⚠️ Low activity period - Avoid new positions"]
    
    for _, row in results_df.iterrows():
        if row['R-Factor'] < 3.0:  # Skip weak stocks
            continue
            
        components = row.get('_components', {})
        intraday_data = components.get('intraday_data', {})
        
        symbol = row['Symbol']
        gap = intraday_data.get('gap_percent', 0)
        intraday_change = intraday_data.get('intraday_change', 0)
        range_pos = intraday_data.get('range_position', 50)
        vwap_pos = intraday_data.get('vwap_position', 'NEUTRAL')
        
        # Gap-up continuation setup
        if (gap > 1.5 and intraday_change > 0 and 
            range_pos > 60 and vwap_pos == "ABOVE"):
            alerts.append(f"🚀 GAP-UP: {symbol} | R-Factor: {row['R-Factor']} | Gap: {gap:+.1f}% | Price: ₹{components.get('ltp', 0):.1f}")
        
        # Pullback entry setup
        elif (row['R-Factor'] > 4.0 and intraday_change < 0 and 
              intraday_change > -2 and row['20D%'] > 10):
            alerts.append(f"📈 PULLBACK: {symbol} | R-Factor: {row['R-Factor']} | Dip: {intraday_change:.1f}% | Strong 20D: {row['20D%']:+.1f}%")
        
        # Breakout setup
        elif (intraday_change > 2.5 and range_pos > 80 and 
              vwap_pos == "ABOVE"):
            alerts.append(f"💥 BREAKOUT: {symbol} | R-Factor: {row['R-Factor']} | Move: {intraday_change:+.1f}% | Near highs")
        
        # Volume spike
        elif intraday_data.get('volume_multiplier', 1) > 2 and intraday_change > 1:
            alerts.append(f"📊 VOLUME: {symbol} | R-Factor: {row['R-Factor']} | Vol: {intraday_data.get('volume_multiplier', 1):.1f}x | Move: {intraday_change:+.1f}%")
    
    return alerts[:10]  # Limit to top 10 alerts

def detect_market_regime(kite, relative_strengths):
    """Detect market regime for intraday adjustments"""
    try:
        nifty_20d = st.session_state.nifty_20d_return
        
        # Count strong performers
        strong_performers = sum(1 for s, d in relative_strengths.items() if d['stock_20d_return'] > 10)
        breadth = (strong_performers / len(relative_strengths)) * 100 if relative_strengths else 50
        
        # Intraday-focused regime detection
        if nifty_20d > 8 and breadth > 60:
            regime = "INTRADAY_BULL"
            multiplier = 1.3  # More aggressive for intraday
        elif nifty_20d > 3 and breadth > 40:
            regime = "INTRADAY_MODERATE"
            multiplier = 1.1
        elif nifty_20d > -3:
            regime = "INTRADAY_NEUTRAL"
            multiplier = 1.0
        else:
            regime = "INTRADAY_WEAK"
            multiplier = 0.8  # More conservative
        
        return regime, multiplier, breadth
        
    except Exception as e:
        return "INTRADAY_NEUTRAL", 1.0, 50

# =========================
# AUTHENTICATION (Kept same)
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
        st.markdown("### ⚡ Intraday Settings")
        
        # Time category display
        time_cat = get_current_time_category()
        time_emoji = {
            "PRE_MARKET": "🌅",
            "OPENING_HOUR": "🚀",
            "TRENDING_HOUR": "📈",
            "LUNCH_LULL": "😴",
            "POWER_HOUR": "⚡",
            "CLOSING_HOUR": "🔔"
        }
        
        st.metric("Market Session", f"{time_emoji.get(time_cat, '⏰')} {time_cat.replace('_', ' ')}")
        
        # Auto-refresh settings
        st.markdown("#### Auto-Refresh")
        auto_refresh = st.checkbox("Enable Auto-Refresh", value=st.session_state.auto_refresh)
        if auto_refresh:
            refresh_interval = st.selectbox("Refresh Interval", [2, 5, 10, 15], index=1)
            st.session_state.refresh_interval = refresh_interval
            st.session_state.auto_refresh = True
        else:
            st.session_state.auto_refresh = False
        
        st.markdown("---")
        st.markdown("### 📊 Market Regime")
        
        regime_emoji = {
            "INTRADAY_BULL": "🚀",
            "INTRADAY_MODERATE": "📈",
            "INTRADAY_NEUTRAL": "➡️",
            "INTRADAY_WEAK": "📉"
        }
        
        st.metric(
            "Regime",
            f"{regime_emoji.get(st.session_state.market_regime, '⚪')} {st.session_state.market_regime.replace('_', ' ')}",
            f"Multiplier: {st.session_state.regime_multiplier:.2f}x"
        )
        
        if st.session_state.nifty_20d_return:
            st.metric("NIFTY 20-Day", f"{st.session_state.nifty_20d_return:+.2f}%")
        
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.kite = None
            st.session_state.instruments = {}
            st.session_state.auth_step = 1
            st.rerun()
    
    st.markdown("---")
    st.markdown("""
    ### ⚡ Intraday Strategy
    
    **Two-Stage Process:**
    1. **Pre-select** strongest stocks (R-Factor 3.5+)
    2. **Trade setups** on selected stocks only
    
    #### Best Time Slots:
    - **🚀 Opening (9:15-10:00)**: Gap trades
    - **📈 Trending (10:00-11:30)**: Breakouts
    - **⚡ Power (14:00-15:00)**: Final moves
    
    #### Avoid:
    - **😴 Lunch (11:30-14:00)**: Low volume
    - **🔔 Closing (15:00+)**: Exit only
    
    #### Entry Criteria:
    - R-Factor > 3.5
    - Volume > 1.5x avg
    - Price above VWAP
    - Clear setup pattern
    """)

# =========================
# MAIN SCANNER
# =========================
if not st.session_state.authenticated:
    st.info("👈 Please authenticate with Zerodha to start scanning")
else:
    # Header controls
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
    with col1:
        symbols_input = st.text_input(
            "Enter symbols to scan (comma-separated):",
            value="ADANIENT, KALYANKJIL, ADANIGREEN, IREDA, HDFCBANK, AXISBANK, UNIONBANK, SAIL, DLF, EXIDEIND, ADANIENSOL, RELIANCE, TCS, INFY",
            help="Enter NSE symbols without .NS suffix"
        )
    
    with col2:
        st.write("")
        scan_btn = st.button("⚡ SCAN", type="primary", use_container_width=True)
    
    with col3:
        st.write("")
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    with col4:
        st.write("")
        if st.button("📋 Focus List", use_container_width=True):
            st.session_state.show_focus_list = not st.session_state.get('show_focus_list', False)
    
    # Auto-refresh logic
    if st.session_state.auto_refresh and st.session_state.authenticated:
        placeholder = st.empty()
        with placeholder.container():
            st.info(f"🔄 Auto-refresh enabled - Next scan in {st.session_state.refresh_interval} minutes")
        
        time.sleep(st.session_state.refresh_interval * 60)
        st.rerun()
    
    # Current time and market session info
    current_time = datetime.datetime.now()
    time_category = get_current_time_category()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Time", current_time.strftime("%H:%M:%S"))
    with col2:
        st.metric("Market Session", time_category.replace('_', ' '))
    with col3:
        if st.session_state.last_update:
            st.metric("Last Scan", st.session_state.last_update.strftime("%H:%M:%S"))
    with col4:
        if time_category in ["LUNCH_LULL", "CLOSING_HOUR"]:
            st.warning("⚠️ Low Activity Period")
        elif time_category in ["TRENDING_HOUR", "OPENING_HOUR"]:
            st.success("✅ Active Trading Time")
    
    if scan_btn or (st.session_state.auto_refresh and time_category in ["OPENING_HOUR", "TRENDING_HOUR", "POWER_HOUR"]):
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        
        valid_symbols = []
        invalid_symbols = []
        
        for symbol in symbols:
            if symbol in st.session_state.instruments:
                valid_symbols.append(symbol)
            else:
                invalid_symbols.append(symbol)
        
        if invalid_symbols:
            st.warning(f"Invalid symbols removed: {', '.join(invalid_symbols)}")
        
        if not valid_symbols:
            st.error("No valid symbols to scan")
        else:
            with st.spinner("🔍 Analyzing stocks for intraday opportunities..."):
                # Calculate base relative strengths
                relative_strengths = calculate_20day_relative_strength(st.session_state.kite, valid_symbols)
                tier_assignments = assign_intraday_tiers(relative_strengths)
                
                # Detect market regime
                regime, multiplier, breadth = detect_market_regime(
                    st.session_state.kite, 
                    relative_strengths
                )
                st.session_state.market_regime = regime
                st.session_state.regime_multiplier = multiplier
                st.session_state.last_update = current_time
            
            # Market overview
            st.markdown("### 🌍 Market Overview")
            overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
            
            with overview_col1:
                st.metric("Market Regime", regime.replace("_", " "), f"×{multiplier:.2f}")
            with overview_col2:
                st.metric("NIFTY 20-Day", f"{st.session_state.nifty_20d_return:+.2f}%")
            with overview_col3:
                st.metric("Strong Breadth", f"{breadth:.1f}%", ">10% gainers")
            with overview_col4:
                st.metric("Scan Time", current_time.strftime("%H:%M:%S"))
            
            # Calculate Intraday R-Factors
            results = []
            progress = st.progress(0)
            status = st.empty()
            
            for idx, symbol in enumerate(valid_symbols):
                status.text(f"⚡ Calculating Intraday R-Factor for {symbol}... ({idx+1}/{len(valid_symbols)})")
                
                if symbol in tier_assignments:
                    instrument_token = st.session_state.instruments[symbol]
                    tier_data = tier_assignments[symbol]
                    
                    enhanced_rfactor, components = calculate_intraday_rfactor(
                        symbol,
                        st.session_state.kite,
                        instrument_token,
                        tier_data,
                        multiplier,
                        relative_strengths
                    )
                    
                    if enhanced_rfactor > 0:
                        intraday_data = components.get('intraday_data', {})
                        results.append({
                            'Symbol': symbol,
                            'R-Factor': enhanced_rfactor,
                            '20D%': round(tier_data['stock_20d_return'], 2),
                            '10D%': round(tier_data['stock_10d_return'], 2),
                            '5D%': round(tier_data['stock_5d_return'], 2),
                            'Today%': round(intraday_data.get('intraday_change', 0), 2),
                            'Gap%': round(intraday_data.get('gap_percent', 0), 2),
                            'VWAP': intraday_data.get('vwap_position', 'N/A'),
                            'Range%': round(intraday_data.get('range_position', 50), 1),
                            'Vol': round(intraday_data.get('volume_multiplier', 1), 1),
                            'Tier': tier_data['tier'].replace('INTRADAY_', ''),
                            'LTP': components.get('ltp', 0),
                            'Intraday_Strength': round(components.get('intraday_strength', 0), 2),
                            'Time_Cat': components.get('time_category', 'N/A'),
                            '_components': components
                        })
                
                progress.progress((idx + 1) / len(valid_symbols))
                time.sleep(0.05)
            
            progress.empty()
            status.empty()
            
            if results:
                df = pd.DataFrame(results)
                df = df.sort_values('R-Factor', ascending=False)
                
                # Generate intraday alerts
                alerts = generate_intraday_alerts(df)
                st.session_state.intraday_alerts = alerts
                
                # Create focus list (R-Factor > 3.5)
                focus_df = df[df['R-Factor'] > 3.5].copy()
                st.session_state.focus_list = focus_df['Symbol'].tolist()
                
                # Display alerts
                if alerts:
                    st.markdown("### 🚨 Real-Time Intraday Alerts")
                    alert_container = st.container()
                    with alert_container:
                        for alert in alerts[:5]:  # Show top 5 alerts
                            if "GAP-UP" in alert:
                                st.success(alert)
                            elif "BREAKOUT" in alert:
                                st.info(alert)
                            elif "PULLBACK" in alert:
                                st.warning(alert)
                            elif "VOLUME" in alert:
                                st.info(alert)
                            else:
                                st.error(alert)
                
                # Focus List Section
                if focus_df.empty:
                    st.warning("⚠️ No stocks meet intraday criteria (R-Factor > 3.5)")
                else:
                    st.markdown(f"### 🎯 Intraday Focus List ({len(focus_df)} stocks)")
                    st.success(f"**Trade Only These Stocks Today:** {', '.join(focus_df['Symbol'].head(8).tolist())}")
                    
                    # Focus list metrics
                    focus_col1, focus_col2, focus_col3, focus_col4 = st.columns(4)
                    
                    with focus_col1:
                        avg_rfactor = focus_df['R-Factor'].mean()
                        st.metric("Avg R-Factor", f"{avg_rfactor:.2f}")
                    
                    with focus_col2:
                        gap_up_count = len(focus_df[focus_df['Gap%'] > 1])
                        st.metric("Gap-Up Stocks", gap_up_count)
                    
                    with focus_col3:
                        above_vwap = len(focus_df[focus_df['VWAP'] == 'ABOVE'])
                        st.metric("Above VWAP", above_vwap)
                    
                    with focus_col4:
                        strong_today = len(focus_df[focus_df['Today%'] > 1])
                        st.metric("Strong Today", strong_today)
                
                # Main results table
                st.markdown("### 📊 Complete Intraday Analysis")
                
                # Category breakdown
                tier1_count = len(df[df['Tier'] == 'TIER_1'])
                tier2_count = len(df[df['Tier'] == 'TIER_2'])
                tier3_count = len(df[df['Tier'] == 'TIER_3'])
                avoid_count = len(df[df['Tier'] == 'AVOID'])
                
                category_col1, category_col2, category_col3, category_col4 = st.columns(4)
                category_col1.metric("🚀 Tier 1", tier1_count, "Best for intraday")
                category_col2.metric("📈 Tier 2", tier2_count, "Good for intraday")
                category_col3.metric("⚠️ Tier 3", tier3_count, "Watch only")
                category_col4.metric("❌ Avoid", avoid_count, "Skip these")
                
                # Display table
                display_df = df[['Symbol', 'R-Factor', '20D%', '5D%', 'Today%', 'Gap%', 
                               'VWAP', 'Range%', 'Vol', 'Tier', 'LTP', 'Intraday_Strength']].copy()
                
                def style_intraday_rfactor(val):
                    if val >= 5.0:
                        return 'background-color: #0d5016; color: white; font-weight: bold'
                    elif val >= 4.0:
                        return 'background-color: #1f7a1f; color: white; font-weight: bold'
                    elif val >= 3.5:
                        return 'background-color: #28a745; color: white'
                    elif val >= 2.5:
                        return 'background-color: #f0ad4e'
                    else:
                        return 'background-color: #dc3545; color: white'
                
                def style_intraday_performance(val):
                    if val > 3:
                        return 'color: #0d5016; font-weight: bold'
                    elif val > 1:
                        return 'color: #1f7a1f; font-weight: bold'
                    elif val > 0:
                        return 'color: #28a745'
                    elif val > -1:
                        return 'color: orange'
                    else:
                        return 'color: #dc3545'
                
                def style_vwap(val):
                    if val == 'ABOVE':
                        return 'color: green; font-weight: bold'
                    elif val == 'BELOW':
                        return 'color: red'
                    else:
                        return 'color: gray'
                
                styled_df = display_df.style.applymap(
                    style_intraday_rfactor, subset=['R-Factor']
                ).applymap(
                    style_intraday_performance, subset=['20D%', '5D%', 'Today%', 'Gap%']
                ).applymap(
                    style_vwap, subset=['VWAP']
                ).format({
                    'R-Factor': '{:.2f}',
                    '20D%': '{:+.1f}%',
                    '5D%': '{:+.1f}%',
                    'Today%': '{:+.1f}%',
                    'Gap%': '{:+.1f}%',
                    'Range%': '{:.0f}%',
                    'Vol': '{:.1f}x',
                    'LTP': '₹{:.1f}',
                    'Intraday_Strength': '{:.1f}'
                })
                
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # Detailed analysis for top stocks
                if not focus_df.empty:
                    st.markdown("### 📈 Top 3 Intraday Setups - Detailed Analysis")
                    
                    top_3 = focus_df.head(3)
                    for idx, (_, row) in enumerate(top_3.iterrows()):
                        components = row['_components']
                        intraday_data = components.get('intraday_data', {})
                        
                        with st.expander(f"🎯 #{idx+1}: {row['Symbol']} - R-Factor: {row['R-Factor']}", expanded=(idx==0)):
                            detail_col1, detail_col2, detail_col3 = st.columns(3)
                            
                            with detail_col1:
                                st.markdown("**📊 Performance**")
                                st.write(f"20-Day: {row['20D%']:+.1f}%")
                                st.write(f"5-Day: {row['5D%']:+.1f}%")
                                st.write(f"Today: {row['Today%']:+.1f}%")
                                st.write(f"Gap: {row['Gap%']:+.1f}%")
                            
                            with detail_col2:
                                st.markdown("**📈 Technical**")
                                st.write(f"VWAP: {row['VWAP']}")
                                st.write(f"Range Position: {row['Range%']:.0f}%")
                                st.write(f"Volume: {row['Vol']:.1f}x avg")
                                st.write(f"Current: ₹{row['LTP']:.1f}")
                            
                            with detail_col3:
                                st.markdown("**⚡ Intraday Setup**")
                                
                                # Determine setup type
                                if row['Gap%'] > 1.5 and row['Today%'] > 0:
                                    setup_type = "🚀 Gap-Up Continuation"
                                    strategy = "Enter on break of opening range high"
                                elif row['Today%'] < 0 and row['20D%'] > 10:
                                    setup_type = "📈 Pullback Entry"
                                    strategy = "Enter on bounce with volume"
                                elif row['Today%'] > 2 and row['Range%'] > 70:
                                    setup_type = "💥 Breakout"
                                    strategy = "Already breaking out - trail stops"
                                else:
                                    setup_type = "👀 Monitor"
                                    strategy = "Wait for clear signal"
                                
                                st.write(f"**Setup:** {setup_type}")
                                st.write(f"**Strategy:** {strategy}")
                                st.write(f"**Stop Loss:** ₹{row['LTP'] * 0.98:.1f} (-2%)")
                                st.write(f"**Target:** ₹{row['LTP'] * 1.04:.1f} (+4%)")
                
                # Trading guidelines based on time
                st.markdown("### ⏰ Time-Based Trading Guidelines")
                
                if time_category == "OPENING_HOUR":
                    st.info("🚀 **Opening Hour Strategy**: Focus on gap-up continuations. Wait for 9:30-9:35 range break. Use 1% stop loss.")
                elif time_category == "TRENDING_HOUR":
                    st.success("📈 **Best Trading Time**: Look for breakouts and pullbacks. Full position sizes allowed. Target 2-4% moves.")
                elif time_category == "LUNCH_LULL":
                    st.warning("😴 **Lunch Time**: Avoid new positions. Manage existing trades only. Low volume period.")
                elif time_category == "POWER_HOUR":
                    st.info("⚡ **Power Hour**: Final momentum moves. Quick scalps preferred. Start exiting by 2:45 PM.")
                elif time_category == "CLOSING_HOUR":
                    st.error("🔔 **Closing Time**: EXIT ALL POSITIONS. No new entries. Risk management only.")
                else:
                    st.info("🌅 **Pre-Market**: Plan your trades. Review focus list. Check overnight gaps.")
                
                # Summary insights
                st.markdown("### 💡 Key Insights")
                
                insights = []
                
                if tier1_count > 0:
                    tier1_stocks = df[df['Tier'] == 'TIER_1']['Symbol'].head(3).tolist()
                    insights.append(f"🎯 **Top Tier Stocks**: {', '.join(tier1_stocks)} - Focus your trades here")
                
                gap_stocks = df[df['Gap%'] > 2]['Symbol'].head(3).tolist()
                if gap_stocks:
                    insights.append(f"🚀 **Strong Gaps**: {', '.join(gap_stocks)} - Watch for continuation")
                
                weak_stocks = df[df['Today%'] < -2]['Symbol'].tolist()
                if weak_stocks:
                    insights.append(f"⚠️ **Avoid Today**: {', '.join(weak_stocks[:3])} - Showing weakness")
                
                volume_leaders = df[df['Vol'] > 2]['Symbol'].head(3).tolist()
                if volume_leaders:
                    insights.append(f"📊 **Volume Leaders**: {', '.join(volume_leaders)} - Institutional activity")
                
                for insight in insights:
                    st.success(insight)
                
                if not insights:
                    st.warning("⚠️ Mixed market conditions. Be selective with trades.")

# Auto-refresh implementation
if st.session_state.auto_refresh and st.session_state.authenticated:
    refresh_placeholder = st.empty()
    
    # Show countdown timer
    for i in range(st.session_state.refresh_interval * 60, 0, -1):
        mins, secs = divmod(i, 60)
        refresh_placeholder.metric("⏰ Next Auto-Scan", f"{mins:02d}:{secs:02d}")
        time.sleep(1)
    
    refresh_placeholder.empty()
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
<p>⚡ Intraday R-Factor Scanner - Trade Smart, Trade Strong Stocks Only</p>
<p>📊 Focus List → Entry Setups → Risk Management → Profit Booking</p>
</div>
""", unsafe_allow_html=True)
