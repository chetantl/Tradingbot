# 🚀 Quick Start Guide

## Prerequisites
- Python 3.8+
- Node.js 16+
- Zerodha Kite Connect API credentials

## Step 1: Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd Tradingbot

# Copy environment files
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local
```

## Step 2: Configure API Credentials

Edit `backend/.env` and add your Zerodha credentials:
```env
KITE_API_KEY=your_api_key_here
KITE_API_SECRET=your_api_secret_here
```

## Step 3: Start Development

### Option A: Automatic Setup (Recommended)
```bash
# Make script executable and run
chmod +x start.sh
./start.sh
```

### Option B: Manual Setup

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**Frontend:**
```bash
# New terminal
cd frontend
npm install
npm start
```

## Step 4: Access Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Authentication

1. Open http://localhost:3000
2. Click "Connect to Zerodha"
3. Enter your API credentials
4. Complete OAuth flow on Zerodha website
5. Return to dashboard - you're authenticated!

## Start Trading

1. Add stock symbols (RELIANCE, TCS, INFY, etc.)
2. Click "Start Real-time Monitoring"
3. Watch for trading signals with confidence scores

## Troubleshooting

**WebSocket issues?** Check API credentials and market hours (9:15 AM - 3:30 PM IST)

**No signals?** Lower confidence threshold or add liquid stocks

**High memory?** Reduce monitored symbols or restart application

## Need Help?

- Check the full README.md for detailed instructions
- Review logs for error messages
- Run health checks: http://localhost:8000/health

---

🎉 **You're ready to start professional trading!**