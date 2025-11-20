import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time, timedelta
import requests
import json
import time
import hashlib
from collections import defaultdict, deque
import warnings
from urllib.parse import quote
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Institutional Order Flow Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .signal-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .buy-signal {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .sell-signal {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .accumulation-signal {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    .distribution-signal {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
if 'monitoring_stocks' not in st.session_state:
    st.session_state.monitoring_stocks = []
if 'signals_history' not in st.session_state:
    st.session_state.signals_history = []
if 'orderbook_history' not in st.session_state:
    st.session_state.orderbook_history = defaultdict(lambda: deque(maxlen=50))
if 'volume_history' not in st.session_state:
    st.session_state.volume_history = defaultdict(lambda: deque(maxlen=50))
if 'price_history' not in st.session_state:
    st.session_state.price_history = defaultdict(lambda: deque(maxlen=50))
if 'instrument_map' not in st.session_state:
    st.session_state.instrument_map = {}
if 'expiry_cache' not in st.session_state:
    st.session_state.expiry_cache = {}

class UpstoxAPI:
    def __init__(self, access_token):
        self.access_token = access_token
        self.base_url = "https://api.upstox.com/v2"
        self.instrument_cache = {}
        
        self.symbol_corrections = {
            'MARUTI': 'MARUTI',
            'M&M': 'M&M',
            'HDFCBANK': 'HDFCBANK',
            'ICICIBANK': 'ICICIBANK',
            'RELIANCE': 'RELIANCE',
            'TCS': 'TCS',
            'INFY': 'INFY',
            'SBIN': 'SBIN',
            'BHARTIARTL': 'BHARTIARTL',
            'ITC': 'ITC',
            'KOTAKBANK': 'KOTAKBANK',
            'LT': 'LT',
            'TATAMOTORS': 'TATAMOTORS',
            'AXISBANK': 'AXISBANK',
            'WIPRO': 'WIPRO',
            'SUNPHARMA': 'SUNPHARMA',
            'ONGC': 'ONGC',
            'NTPC': 'NTPC',
            'POWERGRID': 'POWERGRID',
            'ULTRACEMCO': 'ULTRACEMCO',
            'DRREDDY': 'DRREDDY',
            'ALKEM': 'ALKEM',
            'BIOCON': 'BIOCON',
            'COFORGE': 'COFORGE'
        }
    
    def download_instruments(self):
        """Download and cache instrument master file"""
        try:
            url = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
            
            import gzip
            import io
            
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
                    instruments_data = json.loads(f.read().decode('utf-8'))
                
                for instrument in instruments_
                    if instrument.get('segment') == 'NSE_EQ' and instrument.get('instrument_type') == 'EQ':
                        trading_symbol = instrument.get('trading_symbol', '')
                        instrument_key = instrument.get('instrument_key', '')
                        if trading_symbol and instrument_key:
                            self.instrument_cache[trading_symbol.upper()] = instrument_key
                
                print(f"✓ Loaded {len(self.instrument_cache)} NSE_EQ instruments")
                return True
            else:
                print(f"✗ Failed to download instruments: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Error downloading instruments: {str(e)}")
            return False
    
    def get_instrument_key(self, symbol):
        """Get instrument key for a symbol"""
        if not self.instrument_cache:
            print("Downloading instrument master file...")
            if not self.download_instruments():
                return None
        
        corrected_symbol = self.symbol_corrections.get(symbol.upper(), symbol.upper())
        instrument_key = self.instrument_cache.get(corrected_symbol)
        
        if instrument_key:
            print(f"✓ Found instrument key for {symbol}: {instrument_key}")
            return instrument_key
        else:
            print(f"✗ Symbol {symbol} not found in instrument cache")
            return self.search_instrument(symbol)
    
    def search_instrument(self, symbol):
        """Search for instrument via API"""
        url = f"{self.base_url}/search/instruments"
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        params = {'query': symbol}
        
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    for instrument in data['data']:
                        if instrument.get('segment') == 'NSE_EQ':
                            instrument_key = instrument.get('instrument_key')
                            print(f"✓ Found via search: {instrument_key}")
                            return instrument_key
            return None
        except Exception as e:
            print(f"Search error: {str(e)}")
            return None
    
    def get_market_quote(self, symbol, exchange="NSE_EQ"):
        """Get full market quote including depth for a symbol"""
        if not self.access_token:
            return None
        
        instrument_key = self.get_instrument_key(symbol)
        
        if not instrument_key:
            print(f"✗ Could not find instrument key for {symbol}")
            return None
        
        url = f"{self.base_url}/market-quote/quotes"
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        params = {'instrument_key': instrument_key}
        
        try:
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and data['data']:
                    return data
                else:
                    print(f"✗ No data in response for {symbol}")
                    return None
            else:
                print(f"✗ API Error {response.status_code} for {symbol}")
                return None
                
        except Exception as e:
            print(f"✗ Exception for {symbol}: {str(e)}")
            return None
    
    def get_nearest_expiry(self, symbol):
        """Get the nearest expiry date for a stock's options"""
        # Check cache first
        if symbol in st.session_state.expiry_cache:
            return st.session_state.expiry_cache[symbol]
        
        instrument_key = self.get_instrument_key(symbol)
        
        if not instrument_key:
            return None
        
        url = f"{self.base_url}/option/contract"
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        params = {'instrument_key': instrument_key}
        
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    # Get unique expiry dates and sort
                    expiry_dates = sorted(list(set([contract['expiry'] for contract in data['data']])))
                    if expiry_dates:
                        nearest_expiry = expiry_dates[0]
                        # Cache it
                        st.session_state.expiry_cache[symbol] = nearest_expiry
                        return nearest_expiry
            return None
        except Exception as e:
            print(f"Error getting expiry: {str(e)}")
            return None
    
    def get_option_chain(self, symbol):
        """Get option chain data for PCR calculation"""
        if not self.access_token:
            return None
        
        instrument_key = self.get_instrument_key(symbol)
        
        if not instrument_key:
            return None
        
        expiry_date = self.get_nearest_expiry(symbol)
        
        if not expiry_date:
            print(f"✗ No expiry date found for {symbol}")
            return None
        
        url = f"{self.base_url}/option/chain"
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        params = {
            'instrument_key': instrument_key,
            'expiry_date': expiry_date
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Option chain fetched for {symbol} (Expiry: {expiry_date})")
                return data
            else:
                print(f"✗ Option chain failed for {symbol}: {response.status_code}")
                return None
        except Exception as e:
            print(f"✗ Option chain error for {symbol}: {str(e)}")
            return None

class TradingSignalEngine:
    def __init__(self):
        self.min_confidence = 7
        self.min_institutional_ratio = 2.5
        self.strong_institutional_ratio = 4.0
        self.orderbook_imbalance_threshold = 0.60
        self.volume_multiplier_threshold = 1.5
        
    def calculate_orderbook_imbalance(self, depth):
        """Calculate buy/sell imbalance from order book"""
        if not depth or 'buy' not in depth or 'sell' not in depth:
            return 0.5, 0, 0
        
        buy_orders = depth.get('buy', [])
        sell_orders = depth.get('sell', [])
        
        total_buy_qty = sum([order.get('quantity', 0) for order in buy_orders])
        total_sell_qty = sum([order.get('quantity', 0) for order in sell_orders])
        
        total_qty = total_buy_qty + total_sell_qty
        
        if total_qty == 0:
            return 0.5, 0, 0
        
        buy_ratio = total_buy_qty / total_qty
        
        return buy_ratio, total_buy_qty, total_sell_qty
    
    def detect_hidden_orders(self, symbol, current_volume, prev_volume, 
                            current_orderbook_qty, prev_orderbook_qty):
        """Detect institutional hidden orders"""
        if prev_volume == 0 or prev_orderbook_qty == 0:
            return 1.0
        
        volume_change = abs(current_volume - prev_volume)
        orderbook_change = abs(current_orderbook_qty - prev_orderbook_qty)
        
        if orderbook_change == 0:
            orderbook_change = 1
        
        institutional_ratio = volume_change / orderbook_change
        
        return institutional_ratio
    
    def calculate_pcr(self, option_data):
        """Calculate Put-Call Ratio"""
        if not option_data or 'data' not in option_
            return 1.0
        
        try:
            data = option_data['data']
            put_oi = 0
            call_oi = 0
            
            # New format: data is a list of strike prices with call/put options
            for strike_data in 
                if isinstance(strike_data, dict):
                    # Get call and put OI
                    call_options = strike_data.get('call_options', {})
                    put_options = strike_data.get('put_options', {})
                    
                    call_market_data = call_options.get('market_data', {})
                    put_market_data = put_options.get('market_data', {})
                    
                    call_oi += call_market_data.get('oi', 0)
                    put_oi += put_market_data.get('oi', 0)
            
            if call_oi == 0:
                return 1.0
            
            pcr = put_oi / call_oi
            return pcr
        except Exception as e:
            print(f"PCR calculation error: {str(e)}")
            return 1.0
    
    def generate_signal(self, symbol, quote_data, option_data, historical_data):
        """Generate trading signal based on all factors"""
        if not quote_
            return None
        
        try:
            data_keys = list(quote_data['data'].keys())
            if not data_keys:
                return None
            
            instrument_key = data_keys[0]
            market_data = quote_data['data'][instrument_key]
            
            ohlc = market_data.get('ohlc', {})
            depth = market_data.get('depth', {})
            
            current_price = ohlc.get('close', 0)
            if current_price == 0:
                current_price = market_data.get('last_price', 0)
            
            current_volume = market_data.get('volume', 0)
            
            buy_ratio, total_buy, total_sell = self.calculate_orderbook_imbalance(depth)
            current_orderbook_qty = total_buy + total_sell
            
            hist = historical_data.get(symbol, {})
            prev_volume = hist.get('volume', current_volume)
            prev_orderbook_qty = hist.get('orderbook_qty', current_orderbook_qty)
            prev_price = hist.get('price', current_price)
            avg_volume = hist.get('avg_volume', current_volume)
            
            institutional_ratio = self.detect_hidden_orders(
                symbol, current_volume, prev_volume,
                current_orderbook_qty, prev_orderbook_qty
            )
            
            pcr = self.calculate_pcr(option_data)
            
            volume_multiplier = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            price_change = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
            
            signal_type = None
            confidence = 0
            
            buyers_dominating = buy_ratio >= 0.65
            sellers_dominating = buy_ratio <= 0.35
            
            price_rising = price_change > 0.1
            price_falling = price_change < -0.1
            price_stable = abs(price_change) <= 0.1
            
            has_hidden_orders = institutional_ratio >= self.min_institutional_ratio
            strong_hidden_orders = institutional_ratio >= self.strong_institutional_ratio
            
            pcr_bullish = pcr < 0.9
            pcr_bearish = pcr > 1.1
            pcr_very_bullish = pcr < 0.7
            pcr_very_bearish = pcr > 1.3
            
            if buyers_dominating and price_rising and volume_multiplier >= self.volume_multiplier_threshold:
                signal_type = "BUY"
                confidence += 3
                
            elif sellers_dominating and price_falling and volume_multiplier >= self.volume_multiplier_threshold:
                signal_type = "SELL"
                confidence += 3
                
            elif sellers_dominating and (price_stable or price_rising) and has_hidden_orders and pcr_bullish:
                signal_type = "ACCUMULATION"
                confidence += 4
                
            elif buyers_dominating and (price_stable or price_falling) and has_hidden_orders and pcr_bearish:
                signal_type = "DISTRIBUTION"
                confidence += 4
            
            if signal_type is None:
                return None
            
            if buy_ratio >= 0.70 or buy_ratio <= 0.30:
                confidence += 3
            
            if strong_hidden_orders:
                confidence += 3
            elif has_hidden_orders:
                confidence += 2
            
            if (signal_type in ["BUY", "ACCUMULATION"] and pcr_very_bullish) or \
               (signal_type in ["SELL", "DISTRIBUTION"] and pcr_very_bearish):
                confidence += 4
            elif (signal_type in ["BUY", "ACCUMULATION"] and pcr_bullish) or \
                 (signal_type in ["SELL", "DISTRIBUTION"] and pcr_bearish):
                confidence += 2
            
            if volume_multiplier >= 2.0:
                confidence += 2
            elif volume_multiplier >= 1.5:
                confidence += 1
            
            if signal_type in ["ACCUMULATION", "DISTRIBUTION"]:
                confidence += 1
            
            confidence = min(confidence, 10)
            
            pcr_score = 0
            if signal_type in ["BUY", "ACCUMULATION"]:
                pcr_score = max(0, (1.0 - pcr) * 5)
            else:
                pcr_score = max(0, (pcr - 1.0) * 5)
            
            institutional_score = min(institutional_ratio / 2, 5)
            
            relative_score = confidence + pcr_score + institutional_score
            relative_score = min(relative_score, 15)
            
            if confidence < self.min_confidence:
                return None
            
            if signal_type in ["BUY", "ACCUMULATION"]:
                entry = current_price
                target = entry * 1.01
                stop_loss = entry * 0.996
            else:
                entry = current_price
                target = entry * 0.99
                stop_loss = entry * 1.004
            
            risk_reward = abs(target - entry) / abs(entry - stop_loss)
            
            signal = {
                'symbol': symbol,
                'signal_type': signal_type,
                'confidence': confidence,
                'relative_score': round(relative_score, 2),
                'timestamp': datetime.now(),
                'entry_price': round(entry, 2),
                'target_price': round(target, 2),
                'stop_loss': round(stop_loss, 2),
                'risk_reward': round(risk_reward, 2),
                'buy_ratio': round(buy_ratio * 100, 1),
                'institutional_ratio': round(institutional_ratio, 2),
                'pcr': round(pcr, 2),
                'volume_multiplier': round(volume_multiplier, 2),
                'price_change': round(price_change, 2),
                'current_price': round(current_price, 2)
            }
            
            return signal
            
        except Exception as e:
            st.error(f"Signal generation error for {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def render_authentication():
    """Render authentication section"""
    st.markdown("<h1 class='main-header'>🎯 Institutional Order Flow Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    with st.container():
        st.subheader("🔐 Upstox Authentication")
        
        tab1, tab2 = st.tabs(["📱 Use Access Token", "🔑 Generate New Token"])
        
        with tab1:
            st.info("If you already have a valid access token, paste it here:")
            access_token = st.text_input(
                "Enter your Upstox Access Token",
                type="password",
                key="access_token_input",
                help="Paste your Upstox API access token here"
            )
            
            if st.button("✅ Connect with Token", use_container_width=True, type="primary"):
                if access_token:
                    with st.spinner("Testing connection..."):
                        upstox = UpstoxAPI(access_token)
                        test_result = upstox.get_market_quote("RELIANCE")
                        
                        if test_result and 'data' in test_result:
                            st.session_state.authenticated = True
                            st.session_state.access_token = access_token
                            st.session_state.upstox_client = upstox
                            st.success("✅ Connected successfully!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Token validation failed.")
                            st.info("💡 Try generating a new token")
                else:
                    st.error("Please enter your access token")
        
        with tab2:
            st.info("""
            **Steps to generate a new Upstox Access Token:**
            
            1. Enter your API Key and API Secret
            2. Click "Generate Login URL"
            3. **Copy the URL and paste it in your browser**
            4. Login and authorize
            5. Copy the **code** from redirect URL
            6. Paste it below and click "Get Access Token"
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                api_key = st.text_input("API Key", type="password", key="api_key_gen")
            with col2:
                api_secret = st.text_input("API Secret", type="password", key="api_secret_gen")
            
            redirect_uri = st.text_input(
                "Redirect URI", 
                value="https://127.0.0.1"
            )
            
            if st.button("🔗 Generate Login URL", use_container_width=True):
                if api_key and api_secret:
                    login_url = f"https://api.upstox.com/v2/login/authorization/dialog?client_id={api_key}&redirect_uri={redirect_uri}&response_type=code"
                    st.session_state.temp_api_key = api_key
                    st.session_state.temp_api_secret = api_secret
                    st.session_state.temp_redirect_uri = redirect_uri
                    
                    st.success("✅ Login URL generated!")
                    st.code(login_url, language=None)
                    st.warning("⚠️ Copy and paste this in your browser")
                else:
                    st.error("Please enter both API Key and Secret")
            
            st.markdown("---")
            
            auth_code = st.text_input(
                "Authorization Code",
                key="auth_code_gen"
            )
            
            if st.button("🎫 Get Access Token", use_container_width=True, type="primary"):
                if auth_code and hasattr(st.session_state, 'temp_api_key'):
                    with st.spinner("Generating access token..."):
                        url = "https://api.upstox.com/v2/login/authorization/token"
                        headers = {
                            'accept': 'application/json',
                            'Content-Type': 'application/x-www-form-urlencoded',
                        }
                        data = {
                            'code': auth_code,
                            'client_id': st.session_state.temp_api_key,
                            'client_secret': st.session_state.temp_api_secret,
                            'redirect_uri': st.session_state.temp_redirect_uri,
                            'grant_type': 'authorization_code',
                        }
                        
                        try:
                            response = requests.post(url, headers=headers, data=data)
                            
                            if response.status_code == 200:
                                result = response.json()
                                new_token = result.get('access_token')
                                
                                if new_token:
                                    st.success("✅ Access token generated!")
                                    st.code(new_token, language=None)
                                    
                                    upstox = UpstoxAPI(new_token)
                                    st.session_state.authenticated = True
                                    st.session_state.access_token = new_token
                                    st.session_state.upstox_client = upstox
                                    
                                    st.info("💾 Save this token!")
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    st.error("Failed to extract token")
                            else:
                                st.error(f"Token generation failed: {response.text}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.error("Please generate login URL first")

def render_dashboard(upstox_client, signal_engine):
    """Render main trading dashboard"""
    st.markdown("<h1 class='main-header'>📊 Live Trading Dashboard</h1>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("📈 Stock Selection")
        
        with st.expander("💡 Popular Stock Examples"):
            st.code("""RELIANCE
TCS
INFY
HDFCBANK
ICICIBANK
SBIN
BHARTIARTL
ITC
KOTAKBANK
LT
MARUTI
TATAMOTORS
AXISBANK
DRREDDY
ALKEM
BIOCON
COFORGE""")
        
        stock_input = st.text_area(
            "Enter stocks (one per line, max 10)",
            value="\n".join(st.session_state.monitoring_stocks),
            height=200
        )
        
        if st.button("💾 Update Stocks", use_container_width=True):
            stocks = [s.strip().upper() for s in stock_input.split('\n') if s.strip()]
            stocks = stocks[:10]
            st.session_state.monitoring_stocks = stocks
            st.success(f"Monitoring {len(stocks)} stocks")
        
        st.markdown("---")
        
        st.subheader("🎚️ Signal Parameters")
        min_confidence = st.slider("Minimum Confidence", 5, 10, 7)
        signal_engine.min_confidence = min_confidence
        
        st.subheader("⏱️ Refresh Settings")
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        
        if auto_refresh:
            refresh_interval = st.select_slider(
                "Refresh Interval",
                options=[2, 5, 10, 15, 30, 60],
                value=5,
                format_func=lambda x: f"{x} seconds"
            )
        else:
            refresh_interval = 30
        
        if st.button("🔄 Manual Refresh", use_container_width=True):
            st.rerun()
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.signals_history = []
            st.success("History cleared")
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.access_token = None
            st.rerun()
    
    if not st.session_state.monitoring_stocks:
        st.warning("⚠️ Please add stocks to monitor in the sidebar")
        return
    
    now = datetime.now().time()
    market_open = dt_time(9, 15)
    market_close = dt_time(15, 30)
    is_market_hours = market_open <= now <= market_close
    
    if not is_market_hours:
        st.warning(f"⏰ Market is closed. Market hours: 9:15 AM - 3:30 PM (Current: {now.strftime('%H:%M')})")
    
    all_signals = []
    historical_data = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, symbol in enumerate(st.session_state.monitoring_stocks):
        status_text.text(f"Scanning {symbol}... ({idx+1}/{len(st.session_state.monitoring_stocks)})")
        progress_bar.progress((idx + 1) / len(st.session_state.monitoring_stocks))
        
        print(f"\n{'='*50}")
        print(f"Processing: {symbol}")
        print(f"{'='*50}")
        
        quote_data = upstox_client.get_market_quote(symbol)
        option_data = upstox_client.get_option_chain(symbol)
        
        print(f"Quote Data: {'✓ Available' if quote_data else '✗ Failed'}")
        print(f"Option Data: {'✓ Available' if option_data else '✗ Failed'}")
        
        if quote_
            try:
                data_keys = list(quote_data['data'].keys())
                if not data_keys:
                    continue
                
                instrument_key = data_keys[0]
                market_data = quote_data['data'][instrument_key]
                
                print(f"\nInstrument Key: {instrument_key}")
                
                ohlc = market_data.get('ohlc', {})
                depth = market_data.get('depth', {})
                current_volume = market_data.get('volume', 0)
                current_price = ohlc.get('close', market_data.get('last_price', 0))
                
                print(f"Current Price: ₹{current_price}")
                print(f"Current Volume: {current_volume}")
                
                if depth and 'buy' in depth and 'sell' in depth:
                    print(f"Depth Data: ✓ Available")
                    
                    buy_ratio, total_buy, total_sell = signal_engine.calculate_orderbook_imbalance(depth)
                    current_orderbook_qty = total_buy + total_sell
                    
                    print(f"Buy Ratio: {buy_ratio*100:.1f}%")
                else:
                    current_orderbook_qty = 0
                
                st.session_state.volume_history[symbol].append(current_volume)
                st.session_state.price_history[symbol].append(current_price)
                st.session_state.orderbook_history[symbol].append(current_orderbook_qty)
                
                avg_volume = np.mean(list(st.session_state.volume_history[symbol])) if len(st.session_state.volume_history[symbol]) > 0 else current_volume
                
                historical_data[symbol] = {
                    'volume': list(st.session_state.volume_history[symbol])[-2] if len(st.session_state.volume_history[symbol]) > 1 else current_volume,
                    'price': list(st.session_state.price_history[symbol])[-2] if len(st.session_state.price_history[symbol]) > 1 else current_price,
                    'orderbook_qty': list(st.session_state.orderbook_history[symbol])[-2] if len(st.session_state.orderbook_history[symbol]) > 1 else current_orderbook_qty,
                    'avg_volume': avg_volume
                }
                
                signal = signal_engine.generate_signal(
                    symbol, quote_data, option_data, historical_data
                )
                
                if signal:
                    print(f"\n🎯 SIGNAL: {signal['signal_type']} - Confidence: {signal['confidence']}/10")
                    all_signals.append(signal)
                else:
                    print(f"\n⚠️ No signal")
                    
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
        
        time.sleep(0.5)
    
    progress_bar.empty()
    status_text.empty()
    
    all_signals.sort(key=lambda x: x['relative_score'], reverse=True)
    
    if all_signals:
        st.session_state.signals_history.extend(all_signals)
        st.session_state.signals_history = st.session_state.signals_history[-100:]
    
    st.header("🏆 Top Trading Opportunities")
    
    if not all_signals:
        st.info("🔍 No signals detected yet.")
    else:
        for rank, signal in enumerate(all_signals[:6], 1):
            display_signal_card(signal, rank)
    
    st.markdown("---")
    st.header("📊 Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Signals", len(st.session_state.signals_history))
    with col2:
        if all_signals:
            st.metric("Avg Confidence", f"{np.mean([s['confidence'] for s in all_signals]):.1f}/10")
        else:
            st.metric("Avg Confidence", "N/A")
    with col3:
        if all_signals:
            st.metric("Accumulation", sum(1 for s in all_signals if s['signal_type'] == 'ACCUMULATION'))
        else:
            st.metric("Accumulation", "0")
    with col4:
        st.metric("Stocks Monitored", len(st.session_state.monitoring_stocks))
    
    if st.session_state.signals_history:
        st.subheader("📜 Recent Signals")
        recent_df = pd.DataFrame(st.session_state.signals_history[-20:])
        recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%H:%M:%S')
        st.dataframe(recent_df[['timestamp', 'symbol', 'signal_type', 'confidence', 'relative_score', 'entry_price']], use_container_width=True, hide_index=True)
    
    if auto_refresh and is_market_hours:
        time.sleep(refresh_interval)
        st.rerun()

def display_signal_card(signal, rank):
    """Display individual signal card"""
    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
    signal_class = {
        'BUY': 'buy-signal',
        'SELL': 'sell-signal',
        'ACCUMULATION': 'accumulation-signal',
        'DISTRIBUTION': 'distribution-signal'
    }.get(signal['signal_type'], '')
    stars = "⭐" * min(int(signal['confidence']), 5)
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown(f"""<div class='signal-card {signal_class}'>
            <h2>{medal} {signal['symbol']} - {signal['signal_type']}</h2>
            <p><strong>Confidence:</strong> {signal['confidence']}/10 {stars}</p>
            <p><strong>Score:</strong> {signal['relative_score']}/15</p>
            <p><strong>Time:</strong> {signal['timestamp'].strftime('%H:%M:%S')}</p>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <p><strong>Entry:</strong> ₹{signal['entry_price']}</p>
            <p><strong>Target:</strong> ₹{signal['target_price']}</p>
            <p><strong>SL:</strong> ₹{signal['stop_loss']}</p>
            <p><strong>R:R:</strong> 1:{signal['risk_reward']}</p>
        </div>""", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <p><strong>Buy:</strong> {signal['buy_ratio']}%</p>
            <p><strong>Inst:</strong> {signal['institutional_ratio']}</p>
            <p><strong>PCR:</strong> {signal['pcr']}</p>
            <p><strong>Vol:</strong> {signal['volume_multiplier']}x</p>
        </div>""", unsafe_allow_html=True)
    
    with st.expander(f"📖 Why {signal['symbol']} is ranked #{rank}"):
        st.write(f"""
        **Signal Analysis:**
        - **Order Book:** {signal['buy_ratio']}% buy pressure
        - **Hidden Orders:** Institutional ratio of {signal['institutional_ratio']} 
          {'(STRONG institutional activity)' if signal['institutional_ratio'] >= 4.0 else '(institutional activity detected)'}
        - **Options Market:** PCR of {signal['pcr']} 
          {'(Very Bullish)' if signal['pcr'] < 0.7 else '(Bullish)' if signal['pcr'] < 0.9 else '(Very Bearish)' if signal['pcr'] > 1.3 else '(Bearish)' if signal['pcr'] > 1.1 else '(Neutral)'}
        - **Volume:** {signal['volume_multiplier']}x average volume
        - **Price Action:** {signal['price_change']:+.2f}% change
        
        **Why this signal?**
        """)
        
        if signal['signal_type'] == 'ACCUMULATION':
            st.success("""
            🔵 **ACCUMULATION detected**: Retail selling (low buy %), but institutions are absorbing 
            with hidden orders. Price staying stable despite selling pressure. Once retail panic ends, 
            expect sharp upward move.
            """)
        elif signal['signal_type'] == 'DISTRIBUTION':
            st.warning("""
            🟠 **DISTRIBUTION detected**: Retail buying (high buy %), but institutions are dumping 
            with hidden orders. Price not rising despite buying pressure. Once retail demand exhausts, 
            expect sharp downward move.
            """)
        elif signal['signal_type'] == 'BUY':
            st.info("""
            🟢 **BUY signal**: Clear buying pressure visible in order book, high volume, and price 
            rising. Market consensus is bullish. Momentum trade opportunity.
            """)
        else:
            st.info("""
            🔴 **SELL signal**: Clear selling pressure visible in order book, high volume, and price 
            falling. Market consensus is bearish. Momentum short trade opportunity.
            """)

def main():
    """Main application logic"""
    if not st.session_state.authenticated:
        render_authentication()
    else:
        if not hasattr(st.session_state, 'upstox_client'):
            st.session_state.upstox_client = UpstoxAPI(st.session_state.access_token)
        
        signal_engine = TradingSignalEngine()
        render_dashboard(st.session_state.upstox_client, signal_engine)

if __name__ == "__main__":
    main()
