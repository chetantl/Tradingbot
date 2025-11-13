"""
Real-Time Order Flow Trading Strategy Dashboard
Zerodha KiteConnect Integration with Advanced Signal Generation
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time, timedelta
from kiteconnect import KiteConnect, KiteTicker
import threading
import queue
import time
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
MARKET_OPEN = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 30)
SIGNAL_COOLDOWN_MINUTES = 15
SIGNAL_EXPIRY_MINUTES = 30
UPDATE_INTERVAL = 2  # seconds

# ==================== INITIALIZE SESSION STATE ====================
def init_session_state():
    """Initialize all session state variables"""
    if 'kite' not in st.session_state:
        st.session_state.kite = None
    if 'kite_ticker' not in st.session_state:
        st.session_state.kite_ticker = None
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None
    if 'connected' not in st.session_state:
        st.session_state.connected = False
    if 'monitoring' not in st.session_state:
        st.session_state.monitoring = False
    if 'tick_data' not in st.session_state:
        st.session_state.tick_data = {}
    if 'signals_df' not in st.session_state:
        st.session_state.signals_df = pd.DataFrame()
    if 'previous_snapshots' not in st.session_state:
        st.session_state.previous_snapshots = {}
    if 'signal_history' not in st.session_state:
        st.session_state.signal_history = {}
    if 'instrument_tokens' not in st.session_state:
        st.session_state.instrument_tokens = {}
    if 'ticks_per_second' not in st.session_state:
        st.session_state.ticks_per_second = 0
    if 'last_tick_count' not in st.session_state:
        st.session_state.last_tick_count = 0
    if 'total_ticks' not in st.session_state:
        st.session_state.total_ticks = 0

# ==================== HELPER FUNCTIONS ====================

def is_market_open() -> bool:
    """Check if market is currently open"""
    now = datetime.now().time()
    return MARKET_OPEN <= now <= MARKET_CLOSE

def get_login_url(api_key: str) -> str:
    """Generate Zerodha login URL"""
    return f"https://kite.zerodha.com/connect/login?api_key={api_key}&v=3"

def establish_connection(api_key: str, api_secret: str, request_token: str) -> Tuple[bool, str]:
    """Establish connection with Zerodha and generate access token"""
    try:
        kite = KiteConnect(api_key=api_key)
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        kite.set_access_token(access_token)
        
        st.session_state.kite = kite
        st.session_state.access_token = access_token
        return True, "Connection established successfully!"
    except Exception as e:
        logger.error(f"Connection error: {e}")
        return False, f"Connection failed: {str(e)}"

def get_instrument_tokens(symbols: List[str]) -> Dict[str, int]:
    """Get instrument tokens for given symbols"""
    try:
        instruments = st.session_state.kite.instruments("NSE")
        token_map = {}
        
        for symbol in symbols:
            symbol = symbol.strip().upper()
            for inst in instruments:
                if inst['tradingsymbol'] == symbol and inst['segment'] == 'NSE':
                    token_map[symbol] = inst['instrument_token']
                    break
        
        return token_map
    except Exception as e:
        logger.error(f"Error fetching instruments: {e}")
        return {}

# ==================== ORDER FLOW CALCULATIONS ====================

def calculate_order_imbalance(depth: Dict) -> Tuple[float, float, float]:
    """
    Calculate order book imbalance from market depth
    Returns: (buy_percentage, sell_percentage, imbalance_ratio)
    """
    try:
        buy_depth = depth.get('buy', [])
        sell_depth = depth.get('sell', [])
        
        total_buy_qty = sum([order['quantity'] for order in buy_depth])
        total_sell_qty = sum([order['quantity'] for order in sell_depth])
        
        total = total_buy_qty + total_sell_qty
        if total == 0:
            return 50.0, 50.0, 1.0
        
        buy_pct = (total_buy_qty / total) * 100
        sell_pct = (total_sell_qty / total) * 100
        
        imbalance_ratio = total_buy_qty / total_sell_qty if total_sell_qty > 0 else 1.0
        
        return buy_pct, sell_pct, imbalance_ratio
    except Exception as e:
        logger.error(f"Error calculating imbalance: {e}")
        return 50.0, 50.0, 1.0

def detect_institutional_activity(symbol: str, current_depth: Dict, current_volume: int) -> Tuple[float, bool]:
    """
    Detect institutional activity by analyzing order book changes vs volume changes
    Returns: (institutional_ratio, is_institutional)
    """
    try:
        prev = st.session_state.previous_snapshots.get(symbol, {})
        
        if not prev:
            # First snapshot, store and return
            st.session_state.previous_snapshots[symbol] = {
                'depth': current_depth,
                'volume': current_volume,
                'timestamp': datetime.now()
            }
            return 0.0, False
        
        # Calculate order book change
        prev_buy = sum([o['quantity'] for o in prev['depth'].get('buy', [])])
        prev_sell = sum([o['quantity'] for o in prev['depth'].get('sell', [])])
        curr_buy = sum([o['quantity'] for o in current_depth.get('buy', [])])
        curr_sell = sum([o['quantity'] for o in current_depth.get('sell', [])])
        
        orderbook_change = abs(curr_buy - prev_buy) + abs(curr_sell - prev_sell)
        volume_change = max(0, current_volume - prev['volume'])
        
        # Calculate institutional ratio
        if orderbook_change > 0:
            institutional_ratio = volume_change / orderbook_change
        else:
            institutional_ratio = 0.0
        
        # Update snapshot
        st.session_state.previous_snapshots[symbol] = {
            'depth': current_depth,
            'volume': current_volume,
            'timestamp': datetime.now()
        }
        
        is_institutional = institutional_ratio > 1.5
        
        return institutional_ratio, is_institutional
    
    except Exception as e:
        logger.error(f"Error detecting institutional activity: {e}")
        return 0.0, False

def calculate_volume_ratio(current_volume: int, avg_volume: int) -> float:
    """Calculate current volume vs average volume ratio"""
    if avg_volume > 0:
        return current_volume / avg_volume
    return 1.0

def calculate_targets_and_stops(signal_type: str, current_price: float, depth: Dict) -> Tuple[float, float, float]:
    """
    Calculate entry, target, and stop loss prices
    Returns: (entry_price, target_price, stop_loss)
    """
    try:
        if signal_type in ['BUY', 'ACCUMULATION']:
            # Buy signal
            entry = depth['buy'][0]['price'] if depth.get('buy') else current_price
            target = current_price * 1.005  # 0.5% target
            stop_loss = current_price * 0.997  # 0.3% stop
            
            # Check for resistance in order book
            sell_orders = depth.get('sell', [])
            if sell_orders:
                heavy_sell = max(sell_orders, key=lambda x: x['quantity'])
                if heavy_sell['quantity'] > 10000:  # Heavy resistance
                    target = min(target, heavy_sell['price'])
        
        else:  # SELL or DISTRIBUTION
            entry = depth['sell'][0]['price'] if depth.get('sell') else current_price
            target = current_price * 0.995  # 0.5% target
            stop_loss = current_price * 1.003  # 0.3% stop
            
            # Check for support in order book
            buy_orders = depth.get('buy', [])
            if buy_orders:
                heavy_buy = max(buy_orders, key=lambda x: x['quantity'])
                if heavy_buy['quantity'] > 10000:  # Heavy support
                    target = max(target, heavy_buy['price'])
        
        return entry, target, stop_loss
    
    except Exception as e:
        logger.error(f"Error calculating targets: {e}")
        return current_price, current_price * 1.005, current_price * 0.997

def generate_signal(symbol: str, tick_data: Dict) -> Dict:
    """
    Generate trading signal based on order flow analysis
    Returns signal dictionary or None
    """
    try:
        current_price = tick_data.get('last_price', 0)
        depth = tick_data.get('depth', {})
        volume = tick_data.get('volume', 0)
        avg_volume = tick_data.get('average_price', volume)  # Fallback to current if avg not available
        
        if current_price == 0 or not depth:
            return None
        
        # Check signal cooldown
        signal_key = f"{symbol}"
        last_signal_time = st.session_state.signal_history.get(signal_key)
        if last_signal_time:
            time_diff = (datetime.now() - last_signal_time).total_seconds() / 60
            if time_diff < SIGNAL_COOLDOWN_MINUTES:
                return None
        
        # Calculate order imbalance
        buy_pct, sell_pct, imbalance_ratio = calculate_order_imbalance(depth)
        
        # Detect institutional activity
        inst_ratio, is_institutional = detect_institutional_activity(symbol, depth, volume)
        
        # Calculate volume ratio
        vol_ratio = calculate_volume_ratio(volume, avg_volume)
        
        # Get previous price for trend detection
        prev_price = st.session_state.previous_snapshots.get(symbol, {}).get('price', current_price)
        price_change = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
        
        # Signal generation logic
        signal_type = None
        confidence = 0
        
        # Standard BUY signal
        if buy_pct > 60 and vol_ratio > 1.2:
            signal_type = "BUY"
            confidence += 2
            if price_change > 0:
                confidence += 2  # Price confirms direction
        
        # Standard SELL signal
        elif sell_pct > 60 and vol_ratio > 1.2:
            signal_type = "SELL"
            confidence += 2
            if price_change < 0:
                confidence += 2
        
        # Institutional ACCUMULATION (selling pressure but price stable/rising)
        if sell_pct > 60 and price_change >= -0.1 and is_institutional:
            signal_type = "ACCUMULATION"
            confidence += 3
        
        # Institutional DISTRIBUTION (buying pressure but price stable/falling)
        if buy_pct > 60 and price_change <= 0.1 and is_institutional:
            signal_type = "DISTRIBUTION"
            confidence += 3
        
        # Confidence scoring
        if buy_pct > 70 or sell_pct > 70:
            confidence += 3
        elif buy_pct > 60 or sell_pct > 60:
            confidence += 2
        
        if inst_ratio > 4.0:
            confidence += 3
        elif inst_ratio > 2.5:
            confidence += 2
        elif inst_ratio > 1.5:
            confidence += 1
        
        if vol_ratio > 2.0:
            confidence += 2
        elif vol_ratio > 1.5:
            confidence += 1
        
        # Only generate signal if confidence >= 4
        if signal_type and confidence >= 4:
            confidence = min(confidence, 10)  # Cap at 10
            
            # Calculate targets and stops
            entry, target, stop_loss = calculate_targets_and_stops(signal_type, current_price, depth)
            
            # Calculate potential profit
            if signal_type in ['BUY', 'ACCUMULATION']:
                profit_pct = ((target - entry) / entry) * 100
            else:
                profit_pct = ((entry - target) / entry) * 100
            
            # Update signal history
            st.session_state.signal_history[signal_key] = datetime.now()
            
            return {
                'Stock Symbol': symbol,
                'Signal Type': signal_type,
                'Confidence Score': confidence,
                'Current Price': round(current_price, 2),
                'Entry Price': round(entry, 2),
                'Target Price': round(target, 2),
                'Stop Loss': round(stop_loss, 2),
                'Order Imbalance': f"{buy_pct:.1f}% Buy, {sell_pct:.1f}% Sell",
                'Institutional Ratio': round(inst_ratio, 2),
                'Volume Status': f"{vol_ratio:.1f}x Avg",
                'Potential Profit %': round(profit_pct, 2),
                'Time Detected': datetime.now().strftime("%H:%M:%S")
            }
        
        return None
    
    except Exception as e:
        logger.error(f"Error generating signal for {symbol}: {e}")
        return None

# ==================== WEBSOCKET HANDLERS ====================

def on_ticks(ws, ticks):
    """Callback for receiving ticks"""
    try:
        st.session_state.total_ticks += len(ticks)
        
        for tick in ticks:
            token = tick['instrument_token']
            # Find symbol for this token
            symbol = None
            for sym, tok in st.session_state.instrument_tokens.items():
                if tok == token:
                    symbol = sym
                    break
            
            if symbol:
                st.session_state.tick_data[symbol] = tick
                
                # Generate signal
                signal = generate_signal(symbol, tick)
                if signal:
                    # Add or update signal in dataframe
                    if st.session_state.signals_df.empty:
                        st.session_state.signals_df = pd.DataFrame([signal])
                    else:
                        # Remove old signal for same stock if exists
                        st.session_state.signals_df = st.session_state.signals_df[
                            st.session_state.signals_df['Stock Symbol'] != symbol
                        ]
                        # Add new signal
                        st.session_state.signals_df = pd.concat([
                            st.session_state.signals_df,
                            pd.DataFrame([signal])
                        ], ignore_index=True)
                        
                        # Sort by confidence
                        st.session_state.signals_df = st.session_state.signals_df.sort_values(
                            'Confidence Score', ascending=False
                        ).reset_index(drop=True)
        
        # Clean expired signals
        clean_expired_signals()
    
    except Exception as e:
        logger.error(f"Error processing ticks: {e}")

def on_connect(ws, response):
    """Callback on successful connection"""
    logger.info("WebSocket connected")
    st.session_state.connected = True
    
    # Subscribe to instruments
    tokens = list(st.session_state.instrument_tokens.values())
    if tokens:
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)  # Full mode for depth data

def on_close(ws, code, reason):
    """Callback on connection close"""
    logger.info(f"WebSocket closed: {code} - {reason}")
    st.session_state.connected = False

def on_error(ws, code, reason):
    """Callback on error"""
    logger.error(f"WebSocket error: {code} - {reason}")

def clean_expired_signals():
    """Remove expired signals from the dataframe"""
    if not st.session_state.signals_df.empty:
        current_time = datetime.now()
        
        def is_valid(row):
            try:
                signal_time = datetime.strptime(row['Time Detected'], "%H:%M:%S")
                signal_time = current_time.replace(
                    hour=signal_time.hour,
                    minute=signal_time.minute,
                    second=signal_time.second
                )
                time_diff = (current_time - signal_time).total_seconds() / 60
                return time_diff < SIGNAL_EXPIRY_MINUTES
            except:
                return False
        
        st.session_state.signals_df = st.session_state.signals_df[
            st.session_state.signals_df.apply(is_valid, axis=1)
        ].reset_index(drop=True)

def start_websocket():
    """Start WebSocket connection in background thread"""
    try:
        if st.session_state.access_token and st.session_state.instrument_tokens:
            api_key = st.session_state.get('api_key', '')
            
            kws = KiteTicker(api_key, st.session_state.access_token)
            kws.on_ticks = on_ticks
            kws.on_connect = on_connect
            kws.on_close = on_close
            kws.on_error = on_error
            
            st.session_state.kite_ticker = kws
            
            # Start in background thread
            ws_thread = threading.Thread(target=kws.connect, daemon=True)
            ws_thread.start()
            
            return True
        return False
    except Exception as e:
        logger.error(f"Error starting WebSocket: {e}")
        return False

# ==================== STREAMLIT UI ====================

def main():
    st.set_page_config(
        page_title="Order Flow Trading Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    init_session_state()
    
    # Custom CSS
    st.markdown("""
        <style>
        .big-font {
            font-size:20px !important;
            font-weight: bold;
        }
        .status-connected {
            color: #00ff00;
            font-weight: bold;
        }
        .status-disconnected {
            color: #ff0000;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📊 Real-Time Order Flow Trading Dashboard")
    st.markdown("---")
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Market status
        if is_market_open():
            st.success("🟢 Market is OPEN")
        else:
            st.warning("🔴 Market is CLOSED")
        
        st.markdown("### 🔐 Zerodha Authentication")
        
        api_key = st.text_input("API Key", type="password", key="api_key")
        api_secret = st.text_input("API Secret", type="password", key="api_secret")
        
        if api_key and st.button("🔗 Generate Login URL"):
            login_url = get_login_url(api_key)
            st.markdown(f"[Click here to login]({login_url})")
            st.info("After login, copy the request token from URL")
        
        request_token = st.text_input("Request Token", key="request_token")
        
        if st.button("🔌 Connect to Zerodha"):
            if api_key and api_secret and request_token:
                with st.spinner("Connecting..."):
                    success, message = establish_connection(api_key, api_secret, request_token)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            else:
                st.error("Please fill all fields")
        
        # Connection status
        st.markdown("### 📡 Connection Status")
        if st.session_state.connected:
            st.markdown('<p class="status-connected">🟢 CONNECTED</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-disconnected">🔴 DISCONNECTED</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Stock universe configuration
        st.markdown("### 📋 Stock Universe")
        
        stock_input = st.text_area(
            "Enter stock symbols (one per line or comma-separated)",
            height=150,
            placeholder="RELIANCE\nTCS\nINFY\nHDFC"
        )
        
        if stock_input:
            symbols = [s.strip().upper() for s in stock_input.replace(',', '\n').split('\n') if s.strip()]
            st.info(f"📊 {len(symbols)} stocks configured")
            
            if st.button("🚀 Start Monitoring") and st.session_state.kite:
                with st.spinner("Fetching instruments..."):
                    token_map = get_instrument_tokens(symbols)
                    st.session_state.instrument_tokens = token_map
                    
                    if token_map:
                        st.success(f"Found {len(token_map)} instruments")
                        if start_websocket():
                            st.session_state.monitoring = True
                            st.success("Monitoring started!")
                        else:
                            st.error("Failed to start WebSocket")
                    else:
                        st.error("No valid instruments found")
            
            if st.session_state.monitoring and st.button("⏹️ Stop Monitoring"):
                st.session_state.monitoring = False
                if st.session_state.kite_ticker:
                    st.session_state.kite_ticker.close()
                st.info("Monitoring stopped")
        
        st.markdown("---")
        
        # Filters
        st.markdown("### 🎯 Filters")
        min_confidence = st.slider("Minimum Confidence Score", 1, 10, 4)
        signal_filter = st.multiselect(
            "Signal Types",
            ["BUY", "SELL", "ACCUMULATION", "DISTRIBUTION"],
            default=["BUY", "SELL", "ACCUMULATION", "DISTRIBUTION"]
        )
    
    # ==================== MAIN AREA ====================
    
    if st.session_state.monitoring:
        # Summary statistics
        st.markdown("### 📈 Summary Statistics")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        total_stocks = len(st.session_state.instrument_tokens)
        active_signals = len(st.session_state.signals_df)
        
        # Safe column access with empty DataFrame check
        if not st.session_state.signals_df.empty and 'Signal Type' in st.session_state.signals_df.columns:
            buy_signals = len(st.session_state.signals_df[st.session_state.signals_df['Signal Type'] == 'BUY'])
            sell_signals = len(st.session_state.signals_df[st.session_state.signals_df['Signal Type'] == 'SELL'])
            accum_signals = len(st.session_state.signals_df[st.session_state.signals_df['Signal Type'] == 'ACCUMULATION'])
            avg_confidence = st.session_state.signals_df['Confidence Score'].mean()
        else:
            buy_signals = 0
            sell_signals = 0
            accum_signals = 0
            avg_confidence = 0
        
        col1.metric("Total Stocks", total_stocks)
        col2.metric("Active Signals", active_signals)
        col3.metric("🟢 BUY", buy_signals)
        col4.metric("🔴 SELL", sell_signals)
        col5.metric("🔵 ACCUMULATION", accum_signals)
        col6.metric("Avg Confidence", f"{avg_confidence:.1f}")
        
        st.markdown("---")
        
        # Main signals table
        st.markdown("### 🎯 Live Trading Opportunities")
        
        # Filter dataframe
        filtered_df = st.session_state.signals_df.copy()
        if not filtered_df.empty:
            filtered_df = filtered_df[
                (filtered_df['Confidence Score'] >= min_confidence) &
                (filtered_df['Signal Type'].isin(signal_filter))
            ]
        
        # Display options
        col_a, col_b, col_c = st.columns([2, 2, 1])
        with col_c:
            if st.button("📥 Export to CSV") and not filtered_df.empty:
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "signals.csv",
                    "text/csv"
                )
        
        # Display table
        if not filtered_df.empty:
            # Color code by signal type
            def highlight_signal(row):
                if row['Signal Type'] == 'BUY':
                    return ['background-color: #90EE90'] * len(row)
                elif row['Signal Type'] == 'SELL':
                    return ['background-color: #FFB6C6'] * len(row)
                elif row['Signal Type'] == 'ACCUMULATION':
                    return ['background-color: #ADD8E6'] * len(row)
                elif row['Signal Type'] == 'DISTRIBUTION':
                    return ['background-color: #FFD580'] * len(row)
                return [''] * len(row)
            
            styled_df = filtered_df.style.apply(highlight_signal, axis=1)
            st.dataframe(styled_df, height=400, use_container_width=True)
            
            st.caption(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")
        else:
            st.info("No signals detected yet. Monitoring in progress...")
        
        # Connection health
        tps = st.session_state.total_ticks / max(1, time.time() - st.session_state.get('start_time', time.time()))
        st.caption(f"📶 Connection Health: {st.session_state.total_ticks} total ticks received")
        
        # Auto-refresh
        time.sleep(UPDATE_INTERVAL)
        st.rerun()
    
    else:
        st.info("👈 Configure authentication and stocks in the sidebar to start monitoring")
        
        st.markdown("""
        ### 📖 How to Use:
        1. **Authentication**: Enter your Zerodha API credentials in the sidebar
        2. **Generate Login URL**: Click to get the login link
        3. **Get Request Token**: Login and copy the request token from the URL
        4. **Connect**: Paste the request token and click Connect
        5. **Add Stocks**: Enter stock symbols (NSE) you want to monitor
        6. **Start Monitoring**: Click Start Monitoring to begin real-time analysis
        
        ### 🎯 Signal Types:
        - **🟢 BUY**: Strong buying pressure with volume increase
        - **🔴 SELL**: Strong selling pressure with volume increase
        - **🔵 ACCUMULATION**: Institutional buying (selling pressure but price stable/rising)
        - **🟠 DISTRIBUTION**: Institutional selling (buying pressure but price stable/falling)
        
        ### ⚡ Features:
        - Real-time order flow analysis
        - Institutional activity detection
        - Confidence scoring (1-10)
        - Automatic target & stop loss calculation
        - Signal cooldown to avoid duplicates
        - Auto-refresh every 2 seconds
        """)

if __name__ == "__main__":
    main()
