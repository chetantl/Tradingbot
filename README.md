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

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Zerodha Kite Connect API credentials
- Docker (for production deployment)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Tradingbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API credentials
   ```

4. **Run the dashboard**
   ```bash
   streamlit run streamlit_app.py --server.enableCORS false --server.enableXsrfProtection false
   ```

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   # Copy environment file
   cp .env.example .env

   # Edit configuration
   nano .env

   # Deploy
   chmod +x deploy.sh
   ./deploy.sh
   ```

2. **Access the dashboard**
   - Open http://localhost:8501 in your browser

## ⚙️ Configuration

### Environment Variables

```bash
# Environment
ENV=production
LOG_LEVEL=INFO
TZ=Asia/Kolkata

# Zerodha Kite Connect
KITE_API_KEY=your_api_key_here
KITE_API_SECRET=your_api_secret_here

# Trading Parameters
MIN_CONFIDENCE=7
MAX_DAILY_SIGNALS=6
INSTITUTIONAL_THRESHOLD=2.5

# Features
ENABLE_REAL_PCR=true
ENABLE_SIGNAL_PERSISTENCE=false
ENABLE_PERFORMANCE_TRACKING=true
```

### Key Configuration Options

- `MIN_CONFIDENCE`: Minimum confidence score for signals (1-10)
- `MAX_DAILY_SIGNALS`: Maximum signals to show per day
- `INSTITUTIONAL_THRESHOLD`: Threshold for institutional detection
- `ENABLE_SIGNAL_PERSISTENCE`: Save signals to database for analysis
- `ENABLE_REAL_PCR`: Use real options chain data vs mock data

## 📱 Getting Started

### 1. Authentication

1. Open the dashboard in your browser
2. Enter your Zerodha API credentials in the sidebar
3. Click "Generate Login URL"
4. Authorize on Zerodha website
5. Copy the `request_token` from redirect URL
6. Paste it in the dashboard and complete login

### 2. Monitoring Setup

1. Enter stock symbols (one per line):
   ```
   RELIANCE
   TCS
   INFY
   HDFCBANK
   SBIN
   ```

2. Click "Start Monitoring"
3. System will begin real-time analysis

### 3. Interpreting Signals

**High-Conviction Signals (9-10/10):**
- ACCUMULATION with bullish PCR
- DISTRIBUTION with bearish PCR
- Strong institutional ratio (>4.0)
- High volume surge (>2x average)

**Medium-Conviction Signals (7-8/10):**
- Good institutional detection
- Moderate volume surge
- Reasonable PCR confirmation

## 📈 Analytics & Performance

When `ENABLE_SIGNAL_PERSISTENCE=true`, the system provides:

### Statistics Dashboard
- Total signals generated
- Win rate analysis
- Average profit/loss
- Signal type distribution

### Historical Analysis
- Filter signals by symbol, type, confidence
- View performance over time
- Track effectiveness of different strategies

### Performance Tracking
- Manually record trade results
- Calculate win rates and profitability
- Identify best performing signal types

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