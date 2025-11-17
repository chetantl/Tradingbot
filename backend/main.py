"""
Professional Trading Dashboard Backend
FastAPI application with WebSocket support for real-time institutional order flow detection
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import os
from dataclasses import dataclass, asdict
import time

# Import trading components
from trading_system import TradingSystem, Signal, WebSocketHealth
from database import DatabaseManager
from auth import AuthManager
from monitoring import SystemMonitor
from config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
trading_system: Optional[TradingSystem] = None
database: Optional[DatabaseManager] = None
auth_manager: Optional[AuthManager] = None
monitor: Optional[SystemMonitor] = None
config = get_config()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global trading_system, database, auth_manager, monitor

    logger.info("🚀 Starting Professional Trading Dashboard Backend...")

    try:
        # Initialize components
        auth_manager = AuthManager()
        database = DatabaseManager(config.database.database_url)
        await database.initialize()

        monitor = SystemMonitor()
        await monitor.start_monitoring()

        trading_system = TradingSystem(database, monitor)
        await trading_system.initialize()

        logger.info("✅ Backend initialized successfully")
        yield

    except Exception as e:
        logger.error(f"❌ Failed to initialize backend: {e}")
        raise

    finally:
        logger.info("🛑 Shutting down backend...")
        if trading_system:
            await trading_system.shutdown()
        if monitor:
            await monitor.stop_monitoring()
        if database:
            await database.close()

# Initialize FastAPI app
app = FastAPI(
    title="Professional Trading Dashboard API",
    description="Production-ready institutional order flow detection system",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],  # Production domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time data streaming"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.authenticated_connections: Dict[WebSocket, str] = {}

    async def connect(self, websocket: WebSocket, token: str):
        await websocket.accept()

        # Validate token
        if not auth_manager.verify_token(token):
            await websocket.close(code=1008, reason="Invalid token")
            return False

        self.active_connections.append(websocket)
        user_id = auth_manager.get_user_id_from_token(token)
        self.authenticated_connections[websocket] = user_id

        logger.info(f"✅ WebSocket connected: User {user_id}")
        return True

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        if websocket in self.authenticated_connections:
            user_id = self.authenticated_connections[websocket]
            del self.authenticated_connections[websocket]
            logger.info(f"❌ WebSocket disconnected: User {user_id}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Broadcast failed: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    def get_connection_count(self) -> int:
        return len(self.active_connections)

# Global connection manager
manager = ConnectionManager()

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Professional Trading Dashboard API",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - start_time,
            "version": "2.0.0",
            "components": {}
        }

        # Check trading system
        if trading_system:
            ws_health = trading_system.get_websocket_health()
            health_status["components"]["trading_system"] = {
                "status": "healthy" if ws_health["connected"] else "degraded",
                "websocket_connected": ws_health["connected"],
                "monitored_symbols": len(trading_system.monitored_symbols),
                "total_signals": trading_system.total_signals,
                "last_signal_time": trading_system.last_signal_time
            }

        # Check database
        if database:
            db_health = await database.health_check()
            health_status["components"]["database"] = db_health

        # Check system monitor
        if monitor:
            system_health = await monitor.get_system_health()
            health_status["components"]["monitoring"] = system_health

        # Check WebSocket connections
        health_status["components"]["websocket_server"] = {
            "active_connections": manager.get_connection_count(),
            "status": "healthy"
        }

        # Determine overall status
        component_statuses = [comp.get("status", "unknown") for comp in health_status["components"].values()]
        if any(status == "unhealthy" for status in component_statuses):
            health_status["status"] = "unhealthy"
        elif any(status == "degraded" for status in component_statuses):
            health_status["status"] = "degraded"

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.post("/api/auth/login")
async def login(credentials: dict):
    """Authenticate user and return JWT token"""
    try:
        api_key = credentials.get("api_key")
        api_secret = credentials.get("api_secret")
        request_token = credentials.get("request_token")

        if not all([api_key, api_secret, request_token]):
            raise HTTPException(
                status_code=400,
                detail="Missing required credentials"
            )

        # Authenticate with Zerodha
        kite_session = await auth_manager.authenticate_with_kite(
            api_key, api_secret, request_token
        )

        if not kite_session:
            raise HTTPException(
                status_code=401,
                detail="Authentication failed"
            )

        # Generate JWT token
        token = auth_manager.create_token(kite_session)

        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": 3600,
            "user_info": {
                "api_key": api_key[:8] + "****",  # Masked
                "session_active": True
            }
        }

    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=401, detail=str(e))

@app.post("/api/auth/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Logout user and invalidate token"""
    try:
        auth_manager.invalidate_token(credentials.credentials)
        return {"message": "Logged out successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/trading/signals")
async def get_signals(
    limit: int = 50,
    min_confidence: int = 7,
    symbol: Optional[str] = None,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get trading signals with filtering"""
    try:
        user_id = auth_manager.get_user_id_from_token(credentials.credentials)
        signals = await database.get_signals(
            user_id=user_id,
            limit=limit,
            min_confidence=min_confidence,
            symbol=symbol
        )

        return {
            "signals": signals,
            "count": len(signals),
            "filters": {
                "min_confidence": min_confidence,
                "symbol": symbol
            }
        }

    except Exception as e:
        logger.error(f"Failed to get signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trading/statistics")
async def get_statistics(
    days_back: int = 30,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get trading statistics and performance metrics"""
    try:
        user_id = auth_manager.get_user_id_from_token(credentials.credentials)
        stats = await database.get_statistics(user_id, days_back)

        return stats

    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/trading/start-monitoring")
async def start_monitoring(
    symbols: List[str],
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Start monitoring specified symbols"""
    try:
        user_id = auth_manager.get_user_id_from_token(credentials.credentials)

        if not trading_system:
            raise HTTPException(status_code=503, detail="Trading system not available")

        # Validate symbols
        valid_symbols = await trading_system.validate_symbols(symbols)
        if not valid_symbols:
            raise HTTPException(status_code=400, detail="No valid symbols provided")

        # Start monitoring
        success = await trading_system.start_monitoring(user_id, valid_symbols)

        if success:
            return {
                "message": "Monitoring started successfully",
                "symbols": valid_symbols,
                "count": len(valid_symbols)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to start monitoring")

    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/trading/stop-monitoring")
async def stop_monitoring(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Stop monitoring"""
    try:
        user_id = auth_manager.get_user_id_from_token(credentials.credentials)

        if trading_system:
            await trading_system.stop_monitoring(user_id)

        return {"message": "Monitoring stopped"}

    except Exception as e:
        logger.error(f"Failed to stop monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trading/status")
async def get_trading_status(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current trading status"""
    try:
        user_id = auth_manager.get_user_id_from_token(credentials.credentials)

        if not trading_system:
            return {"monitoring_active": False}

        status = await trading_system.get_user_status(user_id)
        return status

    except Exception as e:
        logger.error(f"Failed to get trading status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = None):
    """WebSocket endpoint for real-time data streaming"""
    try:
        # Extract token from query parameter
        if not token:
            # Try to get from headers
            token = websocket.query_params.get("token")

        if not token:
            await websocket.close(code=1008, reason="Token required")
            return

        # Authenticate connection
        if not await manager.connect(websocket, token):
            return

        user_id = manager.authenticated_connections.get(websocket)

        # Send initial status
        await manager.send_personal_message({
            "type": "connection_established",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }, websocket)

        # Keep connection alive and handle messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle different message types
                if message.get("type") == "ping":
                    await manager.send_personal_message({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }, websocket)

                elif message.get("type") == "subscribe_signals":
                    # Send recent signals
                    if trading_system:
                        recent_signals = await trading_system.get_recent_signals(user_id, limit=10)
                        await manager.send_personal_message({
                            "type": "recent_signals",
                            "signals": recent_signals,
                            "timestamp": datetime.now().isoformat()
                        }, websocket)

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break

    finally:
        manager.disconnect(websocket)

# Background task for broadcasting signals
async def broadcast_signals():
    """Broadcast new signals to connected clients"""
    while True:
        try:
            if trading_system and manager.active_connections:
                new_signals = await trading_system.get_unbroadcasted_signals()

                for signal in new_signals:
                    await manager.broadcast({
                        "type": "new_signal",
                        "signal": signal,
                        "timestamp": datetime.now().isoformat()
                    })

                    # Mark as broadcasted
                    await trading_system.mark_signal_broadcasted(signal["id"])

            await asyncio.sleep(1)  # Check every second

        except Exception as e:
            logger.error(f"Broadcast task error: {e}")
            await asyncio.sleep(5)

# Global start time
start_time = time.time()

# Startup event
@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    # Start broadcasting task
    asyncio.create_task(broadcast_signals())
    logger.info("🚀 Background tasks started")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=config.environment == "development",
        log_level="info"
    )