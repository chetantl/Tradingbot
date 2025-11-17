#!/bin/bash

# Professional Trading Dashboard - Production Deployment Script
# This script handles production deployment using Docker Compose

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="production"
COMPOSE_FILE="docker-compose.prod.yml"
WITH_MONITORING=false

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
    echo -e "${BLUE}  Trading Dashboard Deployment${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi

    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi

    # Check if .env file exists
    if [ ! -f .env ]; then
        warning ".env file not found. Creating from template..."
        cp .env.example .env
        warning "Please edit .env file with your configuration before running again."
        exit 1
    fi

    # Check if required files exist
    required_files=("streamlit_app.py" "requirements.txt" "Dockerfile" "docker-compose.yml")
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            error "Required file $file not found."
        fi
    done

    log "Prerequisites check completed."
}

# Backup current deployment
backup_current() {
    if docker-compose ps | grep -q "Up"; then
        log "Creating backup of current deployment..."
        BACKUP_DIR="backup/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$BACKUP_DIR"

        # Backup logs and data
        if [ -d "logs" ]; then
            cp -r logs "$BACKUP_DIR/"
        fi
        if [ -d "data" ]; then
            cp -r data "$BACKUP_DIR/"
        fi

        log "Backup created at $BACKUP_DIR"
    fi
}

# Build and deploy
deploy() {
    log "Starting deployment..."

    # Stop existing services
    info "Stopping existing services..."
    docker-compose down || true

    # Build new image
    info "Building Docker image..."
    docker-compose build --no-cache

    # Start services
    info "Starting services..."
    docker-compose up -d

    # Wait for health check
    info "Waiting for service to be healthy..."
    for i in {1..30}; do
        if curl -f http://localhost:8501/health &> /dev/null; then
            log "Service is healthy!"
            break
        fi
        if [ $i -eq 30 ]; then
            error "Service failed to become healthy within 30 seconds."
        fi
        sleep 2
    done

    log "Deployment completed successfully!"
}

# Rollback function
rollback() {
    error "Rollback not implemented yet. Manual rollback required."
}

# Show status
show_status() {
    log "Deployment Status:"
    docker-compose ps

    log "Recent logs:"
    docker-compose logs --tail=20 trading-dashboard
}

# Cleanup old images and containers
cleanup() {
    info "Cleaning up old Docker resources..."
    docker image prune -f
    docker container prune -f
    docker volume prune -f
}

# Main deployment flow
main() {
    log "Starting Order Flow Trading Dashboard deployment..."

    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            backup_current
            deploy
            show_status
            ;;
        "status")
            show_status
            ;;
        "stop")
            info "Stopping services..."
            docker-compose down
            ;;
        "restart")
            info "Restarting services..."
            docker-compose restart
            ;;
        "logs")
            docker-compose logs -f trading-dashboard
            ;;
        "cleanup")
            cleanup
            ;;
        "backup")
            backup_current
            ;;
        *)
            echo "Usage: $0 {deploy|status|stop|restart|logs|cleanup|backup}"
            echo ""
            echo "Commands:"
            echo "  deploy   - Full deployment (default)"
            echo "  status   - Show deployment status"
            echo "  stop     - Stop services"
            echo "  restart  - Restart services"
            echo "  logs     - Show logs"
            echo "  cleanup  - Cleanup Docker resources"
            echo "  backup   - Backup current deployment"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"