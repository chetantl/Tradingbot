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
    page_title="⚡ Real-Time R-Factor Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# REAL-TIME SESSION STATE
# =========================
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'kite' not in st.session_state:
    st.session_state.kite = None
if 'instruments' not in st.session_state:
    st.session_state.instruments = {}
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["ADANIENT", "ADANIGREEN", "IREDA", "HDFCBANK", "RELIANCE", "TCS", "INFY", "AXISBANK"]
if 'live_data' not in st.session_state:
    st.session_state.live_data = {}
if 'base_calculations' not in st.session_state:
    st.session_state.base_calculations = {}  # Store heavy calculations once
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'update_interval' not in st.session_state:
    st.session_state.update_interval = 10  # seconds
if 'positions' not in st.session_state:
    st.session_state.positions = {}
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'rfactor_history' not in st.session_state:
    st.session_state.rfactor_history = defaultdict(list)
if 'performance_stats' not in st.session_state:
    st.session_state.performance_stats = {}

# =========================
# LIGHTWEIGHT HELPER FUNCTIONS
# =========================
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

def calculate_base_data_once():
    """Calculate heavy computations once at startup/refresh"""
    if not st.session_state.authenticated:
        return
    
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
                    
                    # Assign base tier (simplified)
                    if stock_20d > 20 and stock_10d > 10:
                        tier = "TIER_1"
                        base_score = 2.5
                    elif stock_20d > 15 and stock_10d > 7:
                        tier = "TIER_2"
                        base_score = 2.2
                    elif stock_20d > 10:
                        tier = "TIER_3"
                        base_score = 1.8
                    elif stock_20d > 5:
                        tier = "TIER_4"
                        base_score = 1.5
                    else:
                        tier = "TIER_5"
                        base_score = 1.0
                    
                    # Calculate average volume
                    volumes = [d['volume'] for d in hist_data[-20:]]
                    avg_volume = np.mean(volumes) if volumes else 1
                    
                    base_data[symbol] = {
                        'hist_data': hist_data,
                        'stock_20d_return': stock_20d,
                        'stock_10d_return': stock_10d,
                        'stock_5d_return': stock_5d,
                        'relative_strength': rel_strength,
                        'tier': tier,
                        'base_score': base_score,
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

def calculate_live_rfactor(symbol, quote_data, base_data):
    """Lightweight R-Factor calculation using pre-computed base data"""
    try:
        if not base_data or symbol not in base_data or base_data[symbol] is None:
            return 0, {}
        
        stock_base = base_data[symbol]
        
        # Extract live quote data
        ltp = float(quote_data.get('last_price', 0))
        volume = float(quote_data.get('volume', 0))
        ohlc = quote_data.get('ohlc', {})
        
        if not ohlc:
            return 0, {}
            
        open_price = float(ohlc.get('open', 0))
        high = float(ohlc.get('high', 0))
        low = float(ohlc.get('low', 0))
        prev_close = float(ohlc.get('close', 0))
        
        if prev_close == 0 or open_price == 0:
            return 0, {}
        
        # Quick calculations
        change_percent = ((ltp - prev_close) / prev_close) * 100
        intraday_change = ((ltp - open_price) / open_price) * 100
        gap_percent = ((open_price - prev_close) / prev_close) * 100
        
        # Volume ratio
        volume_ratio = volume / stock_base['avg_volume_20d'] if stock_base['avg_volume_20d'] > 0 else 1
        
        # Range position
        if high > low:
            range_position = ((ltp - low) / (high - low)) * 100
        else:
            range_position = 50
        
        # Base R-Factor from pre-computed data
        base_rfactor = stock_base['base_score']
        
        # Live adjustments (lightweight)
        intraday_multiplier = 1.0
        
        # Positive momentum boost
        if intraday_change > 2 and volume_ratio > 1.5:
            intraday_multiplier = 1.3
        elif intraday_change > 1 and volume_ratio > 1.2:
            intraday_multiplier = 1.2
        elif intraday_change > 0.5:
            intraday_multiplier = 1.1
        elif intraday_change < -2:
            intraday_multiplier = 0.8
        elif intraday_change < -1:
            intraday_multiplier = 0.9
        
        # Volume boost
        if volume_ratio > 2 and abs(change_percent) > 1:
            intraday_multiplier *= 1.1
        elif volume_ratio < 0.5:
            intraday_multiplier *= 0.9
        
        # Range position boost
        if range_position > 80 and intraday_change > 1:
            intraday_multiplier *= 1.05
        elif range_position < 20:
            intraday_multiplier *= 0.95
        
        # Final R-Factor
        live_rfactor = base_rfactor * intraday_multiplier
        
        # Cap based on base performance
        if stock_base['stock_20d_return'] < 5:
            max_rf = 2.5
        elif stock_base['stock_20d_return'] < 15:
            max_rf = 4.0
        else:
            max_rf = 6.0
            
        live_rfactor = max(0.5, min(max_rf, live_rfactor))
        
        components = {
            'ltp': ltp,
            'change_percent': round(change_percent, 2),
            'intraday_change': round(intraday_change, 2),
            'gap_percent': round(gap_percent, 2),
            'volume_ratio': round(volume_ratio, 2),
            'range_position': round(range_position, 1),
            'tier': stock_base['tier'],
            'base_rfactor': round(base_rfactor, 2),
            'intraday_multiplier': round(intraday_multiplier, 3),
            'stock_20d_return': stock_base['stock_20d_return'],
            'stock_10d_return': stock_base['stock_10d_return'],
            'stock_5d_return': stock_base['stock_5d_return'],
            'volume': volume,
            'open': open_price,
            'high': high,
            'low': low
        }
        
        return round(live_rfactor, 2), components
        
    except Exception as e:
        return 0, {}

def generate_live_alerts(live_data):
    """Generate real-time alerts based on live data changes"""
    alerts = []
    current_time = datetime.datetime.now()
    
    # Sort by R-Factor
    sorted_data = sorted(live_data.items(), key=lambda x: x[1].get('rfactor', 0), reverse=True)
    
    for symbol, data in sorted_data[:5]:  # Top 5 only
        rfactor = data.get('rfactor', 0)
        components = data.get('components', {})
        
        if rfactor < 3.0:
            continue
            
        intraday_change = components.get('intraday_change', 0)
        volume_ratio = components.get('volume_ratio', 1)
        range_pos = components.get('range_position', 50)
        gap = components.get('gap_percent', 0)
        
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
        
        # Generate alerts based on patterns
        if rfactor >= 4.5 and intraday_change > 2 and volume_ratio > 2:
            alert_type = "🚀 BREAKOUT"
            message = f"{symbol}: R-Factor {rfactor} | Move: {intraday_change:+.1f}% | Vol: {volume_ratio:.1f}x"
        elif rfactor >= 4.0 and gap > 1.5 and intraday_change > 0:
            alert_type = "📈 GAP-UP"
            message = f"{symbol}: R-Factor {rfactor} | Gap: {gap:+.1f}% | Continuation"
        elif rfactor >= 4.0 and intraday_change < -1 and components.get('stock_20d_return', 0) > 15:
            alert_type = "💎 PULLBACK"
            message = f"{symbol}: R-Factor {rfactor} | Dip: {intraday_change:.1f}% | Strong base"
        elif rfactor < 2.5 and trend == "❄️ FALLING":
            alert_type = "⚠️ WEAK"
            message = f"{symbol}: R-Factor collapsed to {rfactor} | {trend}"
        else:
            continue
        
        alerts.append({
            'type': alert_type,
            'message': message,
            'symbol': symbol,
            'rfactor': rfactor,
            'trend': trend,
            'time': current_time
        })
    
    return alerts

def update_live_data():
    """Update live data for all watchlist stocks"""
    if not st.session_state.authenticated or not st.session_state.base_calculations:
        return False
    
    try:
        # Get live quotes for all symbols
        quotes = get_live_quotes(st.session_state.kite, st.session_state.watchlist)
        
        live_data = {}
        base_data = st.session_state.base_calculations
        
        for symbol in st.session_state.watchlist:
            quote_key = f"NSE:{symbol}"
            if quote_key in quotes:
                quote = quotes[quote_key]
                rfactor, components = calculate_live_rfactor(symbol, quote, base_data)
                
                live_data[symbol] = {
                    'rfactor': rfactor,
                    'components': components,
                    'last_updated': datetime.datetime.now()
                }
        
        st.session_state.live_data = live_data
        st.session_state.last_update = datetime.datetime.now()
        
        # Generate alerts
        alerts = generate_live_alerts(live_data)
        st.session_state.alerts = alerts
        
        return True
        
    except Exception as e:
        st.error(f"Error updating live data: {str(e)}")
        return False

def get_market_time_status():
    """Check if market is open"""
    now = datetime.datetime.now().time()
    
    if datetime.time(9, 15) <= now <= datetime.time(15, 30):
        if datetime.time(9, 15) <= now < datetime.time(10, 0):
            return "OPENING", "🚀 Opening Hour", "#28a745"
        elif datetime.time(10, 0) <= now < datetime.time(11, 30):
            return "TRENDING", "📈 Trending Hour", "#20c997" 
        elif datetime.time(11, 30) <= now < datetime.time(14, 0):
            return "LUNCH", "😴 Lunch Time", "#ffc107"
        elif datetime.time(14, 0) <= now < datetime.time(15, 0):
            return "POWER", "⚡ Power Hour", "#fd7e14"
        else:
            return "CLOSING", "🔔 Closing", "#dc3545"
    else:
        return "CLOSED", "🌙 Market Closed", "#6c757d"

# =========================
# AUTHENTICATION (Simplified)
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
    st.title("⚡ Real-Time R-Factor Trading Dashboard")
    
    # Authentication check
    if not st.session_state.authenticated:
        with st.expander("🔐 Quick Authentication", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                api_key = st.text_input("API Key")
                api_secret = st.text_input("API Secret", type="password")
            with col2:
                if api_key and api_secret:
                    kite = KiteConnect(api_key=api_key)
                    login_url = kite.login_url()
                    st.markdown(f"[🔗 Login to Zerodha]({login_url})")
                    
                    redirect_url = st.text_input("Paste redirect URL:")
                    if st.button("🚀 Connect") and redirect_url:
                        try:
                            parsed = urlparse(redirect_url)
                            params = parse_qs(parsed.query)
                            request_token = params.get('request_token', [None])[0]
                            
                            if request_token:
                                kite_obj, error = authenticate_kite(api_key, api_secret, request_token)
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
                st.session_state.base_calculations = {}  # Reset base data
    
    # Main dashboard content
    if not st.session_state.base_calculations:
        st.info("👆 Click 'Initialize Base Data' to start the dashboard")
        return
    
    if not st.session_state.live_data:
        st.info("👆 Click 'Update Live Data' to see real-time R-Factors")
        return
    
    # Live alerts section
    if st.session_state.alerts:
        st.markdown("### 🚨 LIVE ALERTS")
        alert_container = st.container()
        with alert_container:
            for alert in st.session_state.alerts[:5]:
                alert_type = alert['type']
                message = alert['message']
                
                if "BREAKOUT" in alert_type:
                    st.success(f"{alert_type}: {message}")
                elif "GAP-UP" in alert_type:
                    st.info(f"{alert_type}: {message}")
                elif "PULLBACK" in alert_type:
                    st.warning(f"{alert_type}: {message}")
                elif "WEAK" in alert_type:
                    st.error(f"{alert_type}: {message}")
                else:
                    st.info(f"{alert_type}: {message}")
    
    # Live R-Factor table
    st.markdown("### 📊 Live R-Factor Dashboard")
    
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
        
        display_data.append({
            'Symbol': symbol,
            'R-Factor': rfactor,
            'Trend': trend_emoji,
            'Change': f"{rf_change:+.2f}",
            'LTP': f"₹{components.get('ltp', 0):.1f}",
            'Day%': f"{components.get('intraday_change', 0):+.1f}%",
            'Gap%': f"{components.get('gap_percent', 0):+.1f}%",
            'Vol': f"{components.get('volume_ratio', 1):.1f}x",
            'Range': f"{components.get('range_position', 50):.0f}%",
            '20D%': f"{components.get('stock_20d_return', 0):+.1f}%",
            'Tier': components.get('tier', 'N/A')
        })
    
    if display_data:
        df = pd.DataFrame(display_data)
        df = df.sort_values('R-Factor', ascending=False)
        
        # Style the dataframe
        def highlight_rfactor(val):
            val_num = float(val)
            if val_num >= 5.0:
                return 'background-color: #28a745; color: white; font-weight: bold'
            elif val_num >= 4.0:
                return 'background-color: #20c997; color: white; font-weight: bold'
            elif val_num >= 3.5:
                return 'background-color: #17a2b8; color: white'
            elif val_num >= 3.0:
                return 'background-color: #ffc107'
            else:
                return 'background-color: #dc3545; color: white'
        
        def highlight_trend(val):
            if "🔥" in val:
                return 'color: green; font-weight: bold'
            elif "❄️" in val:
                return 'color: red; font-weight: bold'
            else:
                return ''
        
        styled_df = df.style.applymap(highlight_rfactor, subset=['R-Factor']).applymap(highlight_trend, subset=['Trend'])
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Quick action buttons for top 3
        st.markdown("### ⚡ Quick Actions")
        top_3 = df.head(3)
        
        action_cols = st.columns(3)
        for idx, (_, row) in enumerate(top_3.iterrows()):
            with action_cols[idx]:
                st.markdown(f"**#{idx+1}: {row['Symbol']}**")
                st.write(f"R-Factor: {row['R-Factor']} {row['Trend']}")
                st.write(f"Price: {row['LTP']} ({row['Day%']})")
                
                if float(row['R-Factor']) >= 4.0 and row['Day%'].replace('%', '').replace('+', '') and float(row['Day%'].replace('%', '').replace('+', '')) > 1:
                    st.success("🚀 BUY SIGNAL")
                elif float(row['R-Factor']) < 3.0:
                    st.error("⚠️ AVOID")
                else:
                    st.info("👀 MONITOR")
    
    # Auto-update logic
    if st.session_state.is_running and market_status in ["OPENING", "TRENDING", "POWER"]:
        time.sleep(st.session_state.update_interval)
        update_live_data()
        st.rerun()

if __name__ == "__main__":
    main()
