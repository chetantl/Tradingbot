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

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --env ENVIRONMENT    Set environment (development|production) [default: production]"
    echo "  -m, --monitoring        Enable monitoring stack (Prometheus, Grafana)"
    echo "  -f, --file FILE         Use specific docker-compose file"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                      # Deploy production version"
    echo "  $0 -e development       # Deploy development version"
    echo "  $0 -m                   # Deploy with monitoring enabled"
    echo "  $0 -f docker-compose.yml  # Use custom compose file"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -m|--monitoring)
                WITH_MONITORING=true
                shift
                ;;
            -f|--file)
                COMPOSE_FILE="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is required but not installed!"
        exit 1
    fi

    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is required but not installed!"
        exit 1
    fi

    # Check if compose file exists
    if [ ! -f "$COMPOSE_FILE" ]; then
        print_error "Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi

    # Check if .env files exist
    if [ ! -f "backend/.env.prod" ]; then
        print_error "Production environment file not found: backend/.env.prod"
        print_status "Please create it from backend/.env.prod.example"
        exit 1
    fi

    if [ ! -f "frontend/.env.production" ]; then
        print_error "Frontend environment file not found: frontend/.env.production"
        print_status "Please create it from frontend/.env.example"
        exit 1
    fi

    print_status "Prerequisites verified ✓"
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

# Validate environment files
validate_environment() {
    print_status "Validating environment configuration..."

    # Check for required backend environment variables
    local backend_env="backend/.env.prod"
    local required_vars=("KITE_API_KEY" "KITE_API_SECRET" "SECRET_KEY")

    for var in "${required_vars[@]}"; do
        if ! grep -q "^${var}=" "$backend_env" || grep -q "${var}=your_\|${var}=change_this" "$backend_env"; then
            print_warning "Environment variable $var may not be properly configured in $backend_env"
        fi
    done

    print_status "Environment validation completed ✓"
}

# Build and deploy services
deploy_services() {
    print_status "Deploying services with environment: $ENVIRONMENT"

    # Set compose file based on environment
    if [ "$ENVIRONMENT" = "development" ]; then
        COMPOSE_FILE="docker-compose.yml"
    fi

    # Build Docker images
    print_status "Building Docker images..."
    docker-compose -f "$COMPOSE_FILE" build --no-cache

    # Start services
    if [ "$WITH_MONITORING" = true ]; then
        print_status "Starting services with monitoring..."
        docker-compose -f "$COMPOSE_FILE" --profile monitoring up -d
    else
        print_status "Starting services..."
        docker-compose -f "$COMPOSE_FILE" up -d
    fi

    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 30

    print_status "Deployment completed ✓"
}

# Health check
health_check() {
    print_status "Performing health checks..."

    local services=()
    if [ "$ENVIRONMENT" = "production" ]; then
        services=("trading-backend-prod" "trading-frontend-prod" "trading-nginx-prod")
        if [ "$WITH_MONITORING" = true ]; then
            services+=("trading-prometheus" "trading-grafana")
        fi
    else
        services=("trading-backend" "trading-frontend" "trading-nginx")
    fi

    local healthy_services=0
    local total_services=${#services[@]}

    for service in "${services[@]}"; do
        if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up (healthy)"; then
            print_status "$service is healthy ✓"
            ((healthy_services++))
        else
            print_warning "$service may not be ready yet"
        fi
    done

    if [ $healthy_services -eq $total_services ]; then
        print_status "All services are healthy! ✓"
    else
        print_warning "$healthy_services/$total_services services are healthy"
        print_status "Check service logs for more information"
    fi
}

# Display service URLs
display_urls() {
    print_status "Service URLs:"
    echo ""

    if [ "$ENVIRONMENT" = "production" ]; then
        echo "  🌐 Main Application: https://yourdomain.com"
        echo "  🔧 Backend API: https://api.yourdomain.com"
        echo "  📊 API Documentation: https://api.yourdomain.com/docs"

        if [ "$WITH_MONITORING" = true ]; then
            echo "  📈 Prometheus: http://yourdomain.com:9090"
            echo "  📊 Grafana: http://yourdomain.com:3001"
        fi
    else
        echo "  🌐 Frontend: http://localhost:3000"
        echo "  🔧 Backend API: http://localhost:8000"
        echo "  📊 API Documentation: http://localhost:8000/docs"
        echo "  🌐 Nginx Proxy: http://localhost:80"
    fi

    echo ""
    print_status "Use 'docker-compose -f $COMPOSE_FILE logs -f' to view logs"
    print_status "Use 'docker-compose -f $COMPOSE_FILE ps' to check service status"
}

# Main deployment function
main() {
    print_header
    echo ""

    # Parse command line arguments
    parse_args "$@"

    # Run deployment steps
    check_prerequisites
    validate_environment
    echo ""

    deploy_services
    echo ""

    health_check
    echo ""

    display_urls

    print_status "Deployment completed successfully! 🚀"
}

# Handle script interruption
trap 'print_status "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"