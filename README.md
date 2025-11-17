# Professional Order Flow Trading Dashboard

A sophisticated, production-ready institutional order flow detection system built with FastAPI backend and React frontend. This dashboard provides real-time trading signals with confidence scoring, PCR analysis, and professional UI for monitoring institutional trading activity.

## 🚀 Features

### Core Trading System
- **Real-time WebSocket streaming** with Zerodha Kite Connect integration
- **Time-normalized institutional detection** identifying iceberg orders
- **Multi-factor signal generation** (ACCUMULATION, DISTRIBUTION, BUY, SELL)
- **Confidence scoring** (0-10 scale) with relative ranking (0-15 points)
- **Put-Call Ratio (PCR) analysis** with bias classification
- **Dynamic risk management** with entry/target/stop-loss calculations

### Professional Dashboard
- **Modern React frontend** with Material-UI components
- **Real-time signal display** with professional dark theme
- **WebSocket integration** with automatic reconnection
- **JWT authentication** with secure session management
- **Responsive design** optimized for desktop and tablet
- **Comprehensive error handling** with user-friendly messages

### Production Features
- **FastAPI backend** with automatic API documentation
- **Circuit breaker patterns** for API resilience
- **System monitoring** with health checks
- **Docker deployment** ready
- **Environment-based configuration**
- **Comprehensive logging** and error tracking

## 📊 How It Works

### Institutional Detection Algorithm

The system detects hidden institutional orders using time-normalized analysis:

```
Normal Retail Activity:
- Orderbook shows: 10,000 shares
- Volume trades: 10,000 shares
- Ratio: 10,000/10,000 = 1.0

Institutional Activity (Iceberg):
- Orderbook shows: 1,000 shares
- Volume trades: 15,000 shares
- Ratio: 15,000/1,000 = 15.0 ← Detected!
```

### Signal Types

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

### Confidence Scoring

- Base signal type: 2-3 points
- PCR confirmation: 1-4 points
- Order imbalance: 2-3 points
- Institutional strength: 2-3 points
- Volume surge: 1-2 points
- **Minimum threshold: 7/10**

## 📋 Prerequisites

- Node.js 16+ and npm
- Python 3.8+
- Zerodha Kite Connect API credentials
- Git

## 🛠️ Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Tradingbot
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env with your Zerodha API credentials
```

### 3. Frontend Setup

```bash
# Navigate to frontend directory (from project root)
cd frontend

# Install dependencies
npm install

# Create environment file
cp .env.example .env.local
# Edit .env.local with your API configuration
```

### 4. Start the Applications

#### Option A: Development Mode (Recommended for first-time setup)

```bash
# Terminal 1: Start backend
cd backend
python main.py

# Terminal 2: Start frontend (new terminal)
cd frontend
npm start
```

#### Option B: Docker Deployment

```bash
# From project root
docker-compose up -d
```

### 5. Access the Dashboard

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🔧 Configuration

### Environment Variables

#### Backend (`.env`)
```env
# Zerodha Kite Connect API
KITE_API_KEY=your_api_key_here
KITE_API_SECRET=your_api_secret_here

# Database (optional)
DATABASE_URL=sqlite:///./trading_dashboard.db

# Application Settings
ENV=development
LOG_LEVEL=INFO
SECRET_KEY=your_secret_key_here
CORS_ORIGINS=["http://localhost:3000"]

# WebSocket Settings
WS_HEARTBEAT_INTERVAL=30
WS_MAX_CONNECTIONS=100
```

#### Frontend (`.env.local`)
```env
# API Configuration
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000

# Application Settings
REACT_APP_ENV=development
REACT_APP_LOG_LEVEL=INFO
```

## 📊 Usage Guide

### 1. Initial Authentication

1. Open http://localhost:3000 in your browser
2. Click "Connect to Zerodha"
3. Enter your Kite Connect API Key and API Secret
4. Complete the OAuth flow on Zerodha's website
5. Return to dashboard - you'll be authenticated automatically

### 2. Setting Up Monitoring

1. **Select Instruments**: Choose stocks/indices to monitor
2. **Configure Parameters**:
   - Institutional detection threshold (default: 2.5)
   - Signal confidence minimum (default: 6/10)
   - Risk-reward ratio (default: 1:2)
3. **Start Monitoring**: Click "Start Real-time Monitoring"

### 3. Interpreting Signals

The dashboard generates four types of signals:

- **🟢 ACCUMULATION**: Institutional buying detected
- **🔴 DISTRIBUTION**: Institutional selling detected
- **📈 BUY**: Strong bullish momentum with confidence
- **📉 SELL**: Strong bearish momentum with confidence

Each signal includes:
- **Confidence Score**: 0-10 scale (higher = more reliable)
- **Entry Price**: Suggested entry level
- **Target Price**: Profit target (1:2 risk-reward)
- **Stop Loss**: Risk management level
- **PCR Bias**: Market sentiment confirmation

### 4. Real-time Monitoring

- **Live Ticks**: Real-time price and volume data
- **Signal Pool**: Active signals with countdown timers
- **Market Metrics**: PCR trends, institutional activity levels
- **Performance Stats**: Win rate, profit/loss tracking

## 🏗️ Architecture

### Backend (FastAPI)
```
backend/
├── main.py              # FastAPI application and WebSocket endpoints
├── trading_system.py    # Core trading logic and signal generation
├── database.py          # Database operations and persistence
├── auth.py             # JWT authentication and Kite Connect integration
├── monitoring.py        # System health monitoring
└── requirements.txt     # Python dependencies
```

### Frontend (React)
```
frontend/
├── src/
│   ├── App.js                    # Main application with routing
│   ├── contexts/                 # React contexts (Auth, WebSocket, Notifications)
│   ├── pages/                    # Page components (Dashboard, Analytics, Settings)
│   ├── components/               # Reusable UI components
│   └── utils/                    # Utility functions and helpers
├── package.json                  # Node.js dependencies
└── public/                       # Static assets
```

## 🔧 Development

### Project Structure

```
Tradingbot/
├── streamlit_app.py          # Main application
├── config.py                 # Configuration management
├── monitoring.py             # System health monitoring
├── database.py               # Signal persistence
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Multi-container setup
├── .env.example             # Environment template
├── deploy.sh                # Deployment script
└── README.md                # This file
```

### Key Components

1. **WebSocket Manager**: Handles real-time data streaming with reconnection
2. **Circuit Breaker**: Prevents API overload during failures
3. **Signal Generator**: Core trading logic with multi-factor analysis
4. **Health Monitor**: System resource monitoring and alerting
5. **Persistence Layer**: Optional database for historical analysis

### Running Tests

The application includes built-in self-tests:

1. Click "🔍 Run Self-Tests" in the sidebar
2. Tests cover:
   - Time normalization algorithms
   - Order imbalance calculations
   - Signal confidence bounds
   - Thread safety
   - Risk calculation accuracy

## 🚨 Production Deployment

### System Requirements

- **Minimum**: 1 CPU core, 512MB RAM, 1GB storage
- **Recommended**: 2 CPU cores, 1GB RAM, 5GB storage
- **Network**: Stable internet connection for WebSocket streaming

### Security Considerations

- API credentials stored in environment variables
- Non-root Docker execution
- Input validation and sanitization
- HTTPS recommended for production

### Monitoring

The system provides comprehensive monitoring:

- **Health Endpoint**: `/health` returns system status
- **Resource Monitoring**: CPU, memory, disk usage
- **WebSocket Health**: Connection status and reconnection attempts
- **Circuit Breaker Status**: API failure tracking

## 🆘 Troubleshooting

### Common Issues

**WebSocket Connection Issues**
- Check API credentials are correct
- Verify market hours (9:15 AM - 3:30 PM IST)
- Check internet connectivity
- Review logs for specific error messages

**No Signals Generated**
- Lower confidence threshold in configuration
- Check if symbols are valid and liquid
- Verify market is open
- Review signal generation logic in logs

**High CPU/Memory Usage**
- Reduce number of monitored symbols
- Lower tick processing batch size
- Enable data retention limits
- Check for memory leaks in logs

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
streamlit run streamlit_app.py
```

## 📚 API Reference

### Core Functions

- `generate_signal()`: Main signal generation logic
- `detect_institutional_activity_normalized()`: Time-normalized institutional detection
- `calculate_real_pcr()`: Real Put-Call Ratio calculation
- `health_check()`: System health status

### WebSocket Events

- `on_ticks()`: Process incoming tick data
- `on_connect()`: Handle WebSocket connection
- `on_close()`: Handle connection closure with reconnection
- `on_error()`: Handle errors with recovery logic

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**IMPORTANT**: This is a trading tool for educational and research purposes. Trading in financial markets involves substantial risk. Past performance is not indicative of future results.

- Use at your own risk
- Start with paper trading
- Never risk more than you can afford to lose
- Consult with financial advisors before trading

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section
2. Review logs for specific error messages
3. Create GitHub issue with detailed description
4. Include system logs and configuration details

---

**Built with ❤️ for the trading community**