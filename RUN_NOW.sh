#!/bin/bash

# One-click setup and run script for Professional Trading Dashboard
# This script will set up everything and start the application

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         Professional Trading Dashboard - Quick Setup          ║"
echo "║                                                              ║"
echo "║  Institutional Order Flow Detection System                   ║"
echo "║  Real-time Trading Signals with Confidence Scoring           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking system requirements..."

    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_status "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.8+ is required but not installed!"
        echo "Please install Python from https://python.org"
        exit 1
    fi

    # Check Node.js
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_status "Node.js $NODE_VERSION found"
    else
        print_error "Node.js 16+ is required but not installed!"
        echo "Please install Node.js from https://nodejs.org"
        exit 1
    fi

    # Check npm
    if command -v npm &> /dev/null; then
        NPM_VERSION=$(npm --version)
        print_status "npm $NPM_VERSION found"
    else
        print_error "npm is required but not installed!"
        exit 1
    fi

    print_status "All prerequisites met ✓"
}

# Setup backend
setup_backend() {
    print_info "Setting up backend environment..."

    cd backend

    # Create virtual environment
    if [ ! -d "venv" ]; then
        print_info "Creating Python virtual environment..."
        python3 -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Install dependencies
    print_info "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt

    # Setup environment file
    if [ ! -f ".env" ]; then
        print_info "Creating environment configuration..."
        cp .env.example .env
        print_warning "Please edit backend/.env with your Zerodha API credentials!"
        echo ""
        echo "Required credentials:"
        echo "  • KITE_API_KEY=your_api_key_here"
        echo "  • KITE_API_SECRET=your_api_secret_here"
        echo ""
        read -p "Press Enter after you've configured your API credentials..."
    fi

    cd ..
    print_status "Backend setup completed ✓"
}

# Setup frontend
setup_frontend() {
    print_info "Setting up frontend environment..."

    cd frontend

    # Install dependencies
    print_info "Installing Node.js dependencies..."
    npm install

    # Setup environment file
    if [ ! -f ".env.local" ]; then
        print_info "Creating frontend environment configuration..."
        cp .env.example .env.local
    fi

    cd ..
    print_status "Frontend setup completed ✓"
}

# Start services
start_services() {
    print_info "Starting the Professional Trading Dashboard..."
    echo ""

    # Function to cleanup
    cleanup() {
        echo ""
        print_info "Shutting down services..."
        jobs -p | xargs -r kill 2>/dev/null || true
        exit 0
    }

    # Set up signal handlers
    trap cleanup SIGINT SIGTERM

    # Start backend
    print_info "Starting backend server..."
    cd backend
    source venv/bin/activate
    python main.py &
    BACKEND_PID=$!
    cd ..

    # Wait for backend
    sleep 5

    # Start frontend
    print_info "Starting frontend application..."
    cd frontend
    npm start &
    FRONTEND_PID=$!
    cd ..

    # Wait a moment for services to start
    sleep 10

    # Display access information
    echo ""
    echo -e "${GREEN}🎉 Professional Trading Dashboard is running!${NC}"
    echo ""
    echo -e "${BLUE}📱 Access Points:${NC}"
    echo "  🌐 Frontend Application:  http://localhost:3000"
    echo "  🔧 Backend API:          http://localhost:8000"
    echo "  📚 API Documentation:    http://localhost:8000/docs"
    echo "  ❤️  Health Check:         http://localhost:8000/health"
    echo ""
    echo -e "${BLUE}🚀 Getting Started:${NC}"
    echo "  1. Open http://localhost:3000 in your browser"
    echo "  2. Click 'Connect to Zerodha'"
    echo "  3. Enter your API credentials"
    echo "  4. Complete the OAuth flow"
    echo "  5. Add stock symbols (RELIANCE, TCS, INFY, etc.)"
    echo "  6. Click 'Start Real-time Monitoring'"
    echo "  7. Watch for trading signals!"
    echo ""
    echo -e "${YELLOW}⚠️  Important Notes:${NC}"
    echo "  • Market hours: 9:15 AM - 3:30 PM IST"
    echo "  • Keep your API credentials secure"
    echo "  • Start with paper trading"
    echo "  • Monitor signals, not automated trading"
    echo ""
    echo -e "${GREEN}Press Ctrl+C to stop all services${NC}"
    echo ""

    # Wait for background processes
    wait
}

# Main execution
main() {
    echo ""
    print_info "Starting Professional Trading Dashboard setup..."
    echo ""

    check_prerequisites
    echo ""

    setup_backend
    echo ""

    setup_frontend
    echo ""

    start_services
}

# Run main function
main "$@"