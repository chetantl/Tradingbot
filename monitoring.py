"""
System Monitoring and Health Checks for Order Flow Trading Dashboard

Provides comprehensive monitoring of WebSocket connections, API health,
system performance, and trading signal quality.
"""

import time
import psutil
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class HealthMetrics:
    """System health metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_percent: float
    uptime_seconds: float

@dataclass
class WebSocketMetrics:
    """WebSocket connection metrics"""
    status: str  # 'Connected', 'Disconnected', 'Reconnecting', 'Failed'
    last_error_time: Optional[float]
    reconnect_count: int
    max_retries: int
    uptime_percentage: float
    average_reconnect_time: float

@dataclass
class TradingMetrics:
    """Trading system metrics"""
    total_ticks_processed: int
    signals_generated: int
    high_confidence_signals: int
    average_confidence: float
    processing_latency_ms: float
    queue_size: int

@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker status metrics"""
    pcr_breaker_state: str
    pcr_failure_count: int
    pcr_last_failure: Optional[float]
    api_breaker_state: str
    api_failure_count: int
    api_last_failure: Optional[float]

@dataclass
class SystemAlert:
    """System alert"""
    level: str  # 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    message: str
    timestamp: float
    component: str
    resolved: bool = False

class HealthMonitor:
    """Comprehensive system health monitoring"""

    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.start_time = time.time()
        self.is_running = False

        # Metrics storage
        self.health_metrics: List[HealthMetrics] = []
        self.alerts: List[SystemAlert] = []
        self.max_metrics_history = 1440  # 24 hours at 30-second intervals

        # Performance tracking
        self.tick_processing_times: List[float] = []
        self.signal_generation_times: List[float] = []
        self.max_performance_history = 1000

        # Threading
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start_monitoring(self):
        """Start background health monitoring"""
        if self.is_running:
            logger.warning("Health monitoring already running")
            return

        self.is_running = True
        self._stop_event.clear()

        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        logger.info("Health monitoring started")

    def stop_monitoring(self):
        """Stop health monitoring"""
        if not self.is_running:
            return

        self.is_running = False
        self._stop_event.set()

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)

        logger.info("Health monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running and not self._stop_event.wait(self.check_interval):
            try:
                self._collect_health_metrics()
                self._check_system_alerts()
                self._cleanup_old_data()
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")

    def _collect_health_metrics(self):
        """Collect system health metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            uptime = time.time() - self.start_time

            metrics = HealthMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_mb=memory.available / 1024 / 1024,
                disk_percent=disk.percent,
                uptime_seconds=uptime
            )

            self.health_metrics.append(metrics)

            # Maintain history size
            if len(self.health_metrics) > self.max_metrics_history:
                self.health_metrics.pop(0)

        except Exception as e:
            logger.error(f"Error collecting health metrics: {e}")

    def _check_system_alerts(self):
        """Check for system alerts"""
        if not self.health_metrics:
            return

        latest = self.health_metrics[-1]

        # CPU alerts
        if latest.cpu_percent > 90:
            self._create_alert('CRITICAL', f'High CPU usage: {latest.cpu_percent:.1f}%', 'System')
        elif latest.cpu_percent > 75:
            self._create_alert('WARNING', f'Elevated CPU usage: {latest.cpu_percent:.1f}%', 'System')

        # Memory alerts
        if latest.memory_percent > 90:
            self._create_alert('CRITICAL', f'High memory usage: {latest.memory_percent:.1f}%', 'System')
        elif latest.memory_percent > 75:
            self._create_alert('WARNING', f'Elevated memory usage: {latest.memory_percent:.1f}%', 'System')

        # Disk alerts
        if latest.disk_percent > 95:
            self._create_alert('CRITICAL', f'Low disk space: {latest.disk_percent:.1f}% used', 'System')
        elif latest.disk_percent > 85:
            self._create_alert('WARNING', f'Elevated disk usage: {latest.disk_percent:.1f}% used', 'System')

    def _cleanup_old_data(self):
        """Clean up old metrics data"""
        cutoff_time = time.time() - (24 * 60 * 60)  # 24 hours

        # Clean health metrics
        self.health_metrics = [m for m in self.health_metrics if m.timestamp > cutoff_time]

        # Clean performance data
        if len(self.tick_processing_times) > self.max_performance_history:
            self.tick_processing_times = self.tick_processing_times[-self.max_performance_history:]

        if len(self.signal_generation_times) > self.max_performance_history:
            self.signal_generation_times = self.signal_generation_times[-self.max_performance_history:]

        # Clean old alerts (older than 7 days)
        alert_cutoff = time.time() - (7 * 24 * 60 * 60)
        self.alerts = [a for a in self.alerts if a.timestamp > alert_cutoff]

    def _create_alert(self, level: str, message: str, component: str):
        """Create a system alert"""
        # Check for duplicate alerts
        recent_time = time.time() - 300  # 5 minutes
        recent_alerts = [a for a in self.alerts
                        if a.component == component and
                           a.message == message and
                           a.timestamp > recent_time and
                           not a.resolved]

        if recent_alerts:
            return  # Avoid duplicate alerts

        alert = SystemAlert(
            level=level,
            message=message,
            timestamp=time.time(),
            component=component
        )

        self.alerts.append(alert)
        logger.warning(f"ALERT [{level}] {component}: {message}")

    def record_tick_processing_time(self, duration_ms: float):
        """Record tick processing performance"""
        self.tick_processing_times.append(duration_ms)

    def record_signal_generation_time(self, duration_ms: float):
        """Record signal generation performance"""
        self.signal_generation_times.append(duration_ms)

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        if not self.health_metrics:
            return {'status': 'Unknown', 'message': 'No health data available'}

        latest = self.health_metrics[-1]

        # Determine overall health
        if latest.cpu_percent > 90 or latest.memory_percent > 90 or latest.disk_percent > 95:
            status = 'Critical'
            message = 'System resources critically low'
        elif latest.cpu_percent > 75 or latest.memory_percent > 75 or latest.disk_percent > 85:
            status = 'Warning'
            message = 'System resources elevated'
        else:
            status = 'Healthy'
            message = 'All systems normal'

        return {
            'status': status,
            'message': message,
            'cpu_percent': latest.cpu_percent,
            'memory_percent': latest.memory_percent,
            'disk_percent': latest.disk_percent,
            'uptime_seconds': latest.uptime_seconds,
            'last_check': latest.timestamp
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'avg_tick_processing_ms': sum(self.tick_processing_times[-100:]) / len(self.tick_processing_times[-100:]) if self.tick_processing_times else 0,
            'avg_signal_generation_ms': sum(self.signal_generation_times[-100:]) / len(self.signal_generation_times[-100:]) if self.signal_generation_times else 0,
            'total_tick_measurements': len(self.tick_processing_times),
            'total_signal_measurements': len(self.signal_generation_times)
        }

    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        cutoff_time = time.time() - (hours * 60 * 60)
        recent_alerts = [a for a in self.alerts if a.timestamp > cutoff_time]

        return [
            {
                'level': alert.level,
                'message': alert.message,
                'component': alert.component,
                'timestamp': alert.timestamp,
                'time': datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                'resolved': alert.resolved
            }
            for alert in recent_alerts
        ]

# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None

def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor

def initialize_health_monitoring(check_interval: int = 30):
    """Initialize health monitoring"""
    monitor = get_health_monitor()
    monitor.start_monitoring()
    return monitor

def shutdown_health_monitoring():
    """Shutdown health monitoring"""
    monitor = get_health_monitor()
    monitor.stop_monitoring()