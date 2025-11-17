"""
Professional System Monitoring
Comprehensive monitoring for production trading dashboard
"""

import asyncio
import psutil
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_percent: float
    disk_free_gb: float
    network_io: Dict[str, float]
    process_count: int
    load_average: Optional[List[float]] = None

@dataclass
class ApplicationMetrics:
    """Application-specific metrics"""
    active_websocket_connections: int
    total_signals_generated: int
    signals_last_hour: int
    api_requests_per_minute: float
    average_response_time_ms: float
    error_rate_percent: float
    active_users: int
    monitored_symbols: int

@dataclass
class Alert:
    """System alert"""
    id: str
    level: str  # INFO, WARNING, ERROR, CRITICAL
    message: str
    component: str
    timestamp: float
    resolved: bool = False
    metadata: Dict[str, Any] = None

class SystemMonitor:
    """Professional system monitoring with alerts and metrics"""

    def __init__(self):
        self.start_time = time.time()
        self.is_running = False

        # Metrics storage
        self.system_metrics_history: List[SystemMetrics] = []
        self.application_metrics_history: List[ApplicationMetrics] = []
        self.alerts: List[Alert] = []
        self.max_history_size = 1440  # 24 hours at 1-minute intervals

        # Alert thresholds
        self.thresholds = {
            "cpu_critical": 90.0,
            "cpu_warning": 75.0,
            "memory_critical": 90.0,
            "memory_warning": 75.0,
            "disk_critical": 95.0,
            "disk_warning": 85.0,
            "response_time_critical": 5000.0,  # ms
            "response_time_warning": 2000.0,
            "error_rate_critical": 10.0,  # %
            "error_rate_warning": 5.0,
        }

        # Performance tracking
        self.api_request_times: List[float] = []
        self.api_request_count = 0
        self.error_count = 0
        self.max_performance_history = 1000

        # Monitoring task
        self._monitor_task: Optional[asyncio.Task] = None

    async def start_monitoring(self):
        """Start system monitoring"""
        if self.is_running:
            logger.warning("System monitoring already running")
            return

        self.is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("🔍 System monitoring started")

    async def stop_monitoring(self):
        """Stop system monitoring"""
        if not self.is_running:
            return

        self.is_running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("🛑 System monitoring stopped")

    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)

                # Collect application metrics
                app_metrics = self._collect_application_metrics()
                self.application_metrics_history.append(app_metrics)

                # Check for alerts
                await self._check_alerts(system_metrics, app_metrics)

                # Cleanup old data
                self._cleanup_old_data()

                # Sleep for next collection
                await asyncio.sleep(60)  # Collect every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_available_mb = memory.available / 1024 / 1024

            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_free_gb = disk.free / 1024 / 1024 / 1024

            # Network I/O
            net_io = psutil.net_io_counters()
            network_io = {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }

            # Process count
            process_count = len(psutil.pids())

            # Load average (Unix-like systems)
            load_average = None
            try:
                load_average = list(psutil.getloadavg())
            except (AttributeError, OSError):
                # Not available on Windows
                pass

            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_mb=memory_available_mb,
                disk_percent=disk.percent,
                disk_free_gb=disk_free_gb,
                network_io=network_io,
                process_count=process_count,
                load_average=load_average
            )

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            # Return zeroed metrics on error
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_mb=0.0,
                disk_percent=0.0,
                disk_free_gb=0.0,
                network_io={},
                process_count=0
            )

    def _collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-specific metrics"""
        try:
            # Calculate API metrics
            avg_response_time = 0.0
            if self.api_request_times:
                avg_response_time = sum(self.api_request_times[-100:]) / len(self.api_request_times[-100:])

            error_rate = 0.0
            if self.api_request_count > 0:
                error_rate = (self.error_count / self.api_request_count) * 100

            # API requests per minute (last 5 minutes)
            recent_requests = len([t for t in self.api_request_times if time.time() - t < 300])
            api_requests_per_minute = recent_requests / 5.0

            # Signals in last hour
            signals_last_hour = len([
                m for m in self.application_metrics_history[-60:]
                if m.total_signals_generated > 0
            ])

            # These would be populated by the actual application
            active_websocket_connections = 0  # Would get from WebSocket manager
            total_signals_generated = 0  # Would get from trading system
            active_users = 0  # Would get from session manager
            monitored_symbols = 0  # Would get from trading system

            return ApplicationMetrics(
                timestamp=time.time(),
                active_websocket_connections=active_websocket_connections,
                total_signals_generated=total_signals_generated,
                signals_last_hour=signals_last_hour,
                api_requests_per_minute=api_requests_per_minute,
                average_response_time_ms=avg_response_time,
                error_rate_percent=error_rate,
                active_users=active_users,
                monitored_symbols=monitored_symbols
            )

        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")
            return ApplicationMetrics(
                timestamp=time.time(),
                active_websocket_connections=0,
                total_signals_generated=0,
                signals_last_hour=0,
                api_requests_per_minute=0.0,
                average_response_time_ms=0.0,
                error_rate_percent=0.0,
                active_users=0,
                monitored_symbols=0
            )

    async def _check_alerts(self, system_metrics: SystemMetrics, app_metrics: ApplicationMetrics):
        """Check for alert conditions"""
        alerts_created = []

        # CPU alerts
        if system_metrics.cpu_percent >= self.thresholds["cpu_critical"]:
            alerts_created.append(self._create_alert(
                "CRITICAL",
                f"Critical CPU usage: {system_metrics.cpu_percent:.1f}%",
                "System",
                {"cpu_percent": system_metrics.cpu_percent}
            ))
        elif system_metrics.cpu_percent >= self.thresholds["cpu_warning"]:
            alerts_created.append(self._create_alert(
                "WARNING",
                f"High CPU usage: {system_metrics.cpu_percent:.1f}%",
                "System",
                {"cpu_percent": system_metrics.cpu_percent}
            ))

        # Memory alerts
        if system_metrics.memory_percent >= self.thresholds["memory_critical"]:
            alerts_created.append(self._create_alert(
                "CRITICAL",
                f"Critical memory usage: {system_metrics.memory_percent:.1f}%",
                "System",
                {"memory_percent": system_metrics.memory_percent}
            ))
        elif system_metrics.memory_percent >= self.thresholds["memory_warning"]:
            alerts_created.append(self._create_alert(
                "WARNING",
                f"High memory usage: {system_metrics.memory_percent:.1f}%",
                "System",
                {"memory_percent": system_metrics.memory_percent}
            ))

        # Disk alerts
        if system_metrics.disk_percent >= self.thresholds["disk_critical"]:
            alerts_created.append(self._create_alert(
                "CRITICAL",
                f"Critical disk usage: {system_metrics.disk_percent:.1f}%",
                "System",
                {"disk_percent": system_metrics.disk_percent}
            ))
        elif system_metrics.disk_percent >= self.thresholds["disk_warning"]:
            alerts_created.append(self._create_alert(
                "WARNING",
                f"High disk usage: {system_metrics.disk_percent:.1f}%",
                "System",
                {"disk_percent": system_metrics.disk_percent}
            ))

        # Application performance alerts
        if app_metrics.average_response_time_ms >= self.thresholds["response_time_critical"]:
            alerts_created.append(self._create_alert(
                "CRITICAL",
                f"Critical response time: {app_metrics.average_response_time_ms:.0f}ms",
                "Application",
                {"response_time_ms": app_metrics.average_response_time_ms}
            ))
        elif app_metrics.average_response_time_ms >= self.thresholds["response_time_warning"]:
            alerts_created.append(self._create_alert(
                "WARNING",
                f"High response time: {app_metrics.average_response_time_ms:.0f}ms",
                "Application",
                {"response_time_ms": app_metrics.average_response_time_ms}
            ))

        if app_metrics.error_rate_percent >= self.thresholds["error_rate_critical"]:
            alerts_created.append(self._create_alert(
                "CRITICAL",
                f"Critical error rate: {app_metrics.error_rate_percent:.1f}%",
                "Application",
                {"error_rate_percent": app_metrics.error_rate_percent}
            ))
        elif app_metrics.error_rate_percent >= self.thresholds["error_rate_warning"]:
            alerts_created.append(self._create_alert(
                "WARNING",
                f"High error rate: {app_metrics.error_rate_percent:.1f}%",
                "Application",
                {"error_rate_percent": app_metrics.error_rate_percent}
            ))

        # Add alerts to the list
        self.alerts.extend(alerts_created)

        # Log new alerts
        for alert in alerts_created:
            logger.warning(f"ALERT [{alert.level}] {alert.component}: {alert.message}")

    def _create_alert(self, level: str, message: str, component: str, metadata: Dict = None) -> Alert:
        """Create a new alert"""
        # Check for duplicate alerts in last 5 minutes
        recent_time = time.time() - 300
        recent_alerts = [
            alert for alert in self.alerts
            if alert.component == component and
               alert.message == message and
               alert.timestamp > recent_time and
               not alert.resolved
        ]

        if recent_alerts:
            # Return existing alert instead of creating duplicate
            return recent_alerts[0]

        return Alert(
            id=f"alert_{int(time.time())}_{hash(message) % 10000}",
            level=level,
            message=message,
            component=component,
            timestamp=time.time(),
            metadata=metadata or {}
        )

    def _cleanup_old_data(self):
        """Clean up old metrics and alerts"""
        cutoff_time = time.time() - (24 * 60 * 60)  # 24 hours ago

        # Clean system metrics
        self.system_metrics_history = [
            m for m in self.system_metrics_history if m.timestamp > cutoff_time
        ]

        # Clean application metrics
        self.application_metrics_history = [
            m for m in self.application_metrics_history if m.timestamp > cutoff_time
        ]

        # Clean old alerts (older than 7 days)
        alert_cutoff = time.time() - (7 * 24 * 60 * 60)
        self.alerts = [alert for alert in self.alerts if alert.timestamp > alert_cutoff]

        # Clean performance history
        if len(self.api_request_times) > self.max_performance_history:
            self.api_request_times = self.api_request_times[-self.max_performance_history:]

    def record_api_request(self, response_time_ms: float, is_error: bool = False):
        """Record API request for performance tracking"""
        self.api_request_times.append(response_time_ms)
        self.api_request_count += 1

        if is_error:
            self.error_count += 1

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        if not self.system_metrics_history:
            return {
                "status": "Unknown",
                "message": "No monitoring data available",
                "uptime": time.time() - self.start_time
            }

        latest = self.system_metrics_history[-1]

        # Determine overall health
        status = "Healthy"
        issues = []

        if latest.cpu_percent >= self.thresholds["cpu_critical"]:
            status = "Critical"
            issues.append("High CPU usage")
        elif latest.memory_percent >= self.thresholds["memory_critical"]:
            status = "Critical"
            issues.append("High memory usage")
        elif latest.disk_percent >= self.thresholds["disk_critical"]:
            status = "Critical"
            issues.append("Low disk space")
        elif (latest.cpu_percent >= self.thresholds["cpu_warning"] or
              latest.memory_percent >= self.thresholds["memory_warning"] or
              latest.disk_percent >= self.thresholds["disk_warning"]):
            status = "Warning"
            if latest.cpu_percent >= self.thresholds["cpu_warning"]:
                issues.append("Elevated CPU usage")
            if latest.memory_percent >= self.thresholds["memory_warning"]:
                issues.append("Elevated memory usage")
            if latest.disk_percent >= self.thresholds["disk_warning"]:
                issues.append("Elevated disk usage")

        return {
            "status": status,
            "message": "; ".join(issues) if issues else "All systems normal",
            "uptime": time.time() - self.start_time,
            "timestamp": latest.timestamp,
            "metrics": asdict(latest)
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        if not self.application_metrics_history:
            return {
                "avg_response_time_ms": 0.0,
                "api_requests_per_minute": 0.0,
                "error_rate_percent": 0.0,
                "total_requests": self.api_request_count,
                "total_errors": self.error_count
            }

        latest = self.application_metrics_history[-1]

        return {
            "avg_response_time_ms": latest.average_response_time_ms,
            "api_requests_per_minute": latest.api_requests_per_minute,
            "error_rate_percent": latest.error_rate_percent,
            "total_requests": self.api_request_count,
            "total_errors": self.error_count,
            "active_websocket_connections": latest.active_websocket_connections,
            "active_users": latest.active_users,
            "monitored_symbols": latest.monitored_symbols
        }

    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts"""
        cutoff_time = time.time() - (hours * 60 * 60)
        recent_alerts = [
            alert for alert in self.alerts if alert.timestamp > cutoff_time
        ]

        return [
            {
                "id": alert.id,
                "level": alert.level,
                "message": alert.message,
                "component": alert.component,
                "timestamp": alert.timestamp,
                "time": datetime.fromtimestamp(alert.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                "resolved": alert.resolved,
                "metadata": alert.metadata or {}
            }
            for alert in recent_alerts
        ]

    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for specified time period"""
        cutoff_time = time.time() - (hours * 60 * 60)

        # Filter metrics by time period
        system_metrics = [m for m in self.system_metrics_history if m.timestamp > cutoff_time]
        app_metrics = [m for m in self.application_metrics_history if m.timestamp > cutoff_time]

        if not system_metrics:
            return {"error": "No metrics available for specified period"}

        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in system_metrics) / len(system_metrics)
        avg_memory = sum(m.memory_percent for m in system_metrics) / len(system_metrics)
        avg_disk = sum(m.disk_percent for m in system_metrics) / len(system_metrics)

        # Find max values
        max_cpu = max(m.cpu_percent for m in system_metrics)
        max_memory = max(m.memory_percent for m in system_metrics)
        max_response_time = max(m.average_response_time_ms for m in app_metrics) if app_metrics else 0

        return {
            "period_hours": hours,
            "data_points": len(system_metrics),
            "system": {
                "avg_cpu_percent": round(avg_cpu, 2),
                "max_cpu_percent": round(max_cpu, 2),
                "avg_memory_percent": round(avg_memory, 2),
                "max_memory_percent": round(max_memory, 2),
                "avg_disk_percent": round(avg_disk, 2),
            },
            "application": {
                "max_response_time_ms": round(max_response_time, 2),
                "total_signals": sum(m.total_signals_generated for m in app_metrics),
                "avg_active_connections": sum(m.active_websocket_connections for m in app_metrics) / len(app_metrics) if app_metrics else 0
            },
            "alerts": {
                "total": len([a for a in self.alerts if a.timestamp > cutoff_time]),
                "critical": len([a for a in self.alerts if a.timestamp > cutoff_time and a.level == "CRITICAL"]),
                "warning": len([a for a in self.alerts if a.timestamp > cutoff_time and a.level == "WARNING"]),
            }
        }

# Global monitor instance
_system_monitor: Optional[SystemMonitor] = None

def get_system_monitor() -> SystemMonitor:
    """Get global system monitor instance"""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor()
    return _system_monitor