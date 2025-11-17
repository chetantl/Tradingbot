#!/bin/bash

# Professional Trading Dashboard - Development Startup Script
# This script starts both backend and frontend services for development

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  Trading Dashboard Development${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Check if required directories exist
check_directories() {
    print_status "Checking project structure..."

    if [ ! -d "backend" ]; then
        print_error "Backend directory not found!"
        exit 1
    fi

    if [ ! -d "frontend" ]; then
        print_error "Frontend directory not found!"
        exit 1
    fi

    print_status "Project structure validated ✓"
}

# Setup backend environment
setup_backend() {
    print_status "Setting up backend environment..."

    cd backend

    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Check if requirements are installed
    if ! python -c "import fastapi" 2>/dev/null; then
        print_status "Installing Python dependencies..."
        pip install -r requirements.txt
    fi

    # Check if .env file exists
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating from template..."
        cp .env.example .env
        print_warning "Please edit backend/.env with your Zerodha API credentials!"
        echo ""
        print_status "Required environment variables:"
        echo "  - KITE_API_KEY=your_api_key_here"
        echo "  - KITE_API_SECRET=your_api_secret_here"
        echo ""
        read -p "Press Enter to continue after editing .env file..."
    fi

    cd ..
    print_status "Backend setup completed ✓"
}

# Setup frontend environment
setup_frontend() {
    print_status "Setting up frontend environment..."

    cd frontend

    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        print_status "Installing Node.js dependencies..."
        npm install
    fi

    # Check if .env.local exists
    if [ ! -f ".env.local" ]; then
        print_warning ".env.local file not found. Creating from template..."
        cp .env.example .env.local
        print_status "Frontend environment file created ✓"
    fi

    cd ..
    print_status "Frontend setup completed ✓"
}

# Start services
start_services() {
    print_status "Starting development services..."
    echo ""

    # Function to cleanup background processes
    cleanup() {
        print_status "Shutting down services..."
        jobs -p | xargs -r kill
        exit 0
    }

    # Set up signal handlers
    trap cleanup SIGINT SIGTERM

    # Start backend
    print_status "Starting backend server..."
    cd backend
    source venv/bin/activate
    python main.py &
    BACKEND_PID=$!
    cd ..

    # Wait a moment for backend to start
    sleep 3

    # Check if backend is running
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_status "Backend server started successfully ✓"
    else
        print_warning "Backend server may still be starting..."
    fi

    # Start frontend
    print_status "Starting frontend development server..."
    cd frontend
    npm start &
    FRONTEND_PID=$!
    cd ..

    print_status "Development servers started!"
    echo ""
    print_status "Services are running at:"
    echo "  🚀 Frontend: http://localhost:3000"
    echo "  🔧 Backend API: http://localhost:8000"
    echo "  📚 API Docs: http://localhost:8000/docs"
    echo ""
    print_status "Press Ctrl+C to stop all services"

    # Wait for background processes
    wait
}

# Main execution
main() {
    print_header
    echo ""

    # Check dependencies
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed!"
        exit 1
    fi

    if ! command -v npm &> /dev/null; then
        print_error "Node.js and npm are required but not installed!"
        exit 1
    fi

    print_status "Dependencies verified ✓"
    echo ""

    # Run setup functions
    check_directories
    setup_backend
    setup_frontend
    echo ""

    # Start services
    start_services
}

# Run main function
main "$@"