"""
Configuration Management for Order Flow Trading Dashboard

Centralized configuration with environment variable support,
validation, and default values for production deployment.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class WebSocketConfig:
    """WebSocket connection configuration"""
    max_retries: int = 5
    initial_backoff: int = 1  # seconds
    connection_timeout: int = 30  # seconds
    queue_max_size: int = 1000

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    pcr_failure_threshold: int = 3
    pcr_recovery_timeout: int = 300  # 5 minutes
    api_failure_threshold: int = 5
    api_recovery_timeout: int = 60   # 1 minute

@dataclass
class TradingConfig:
    """Trading algorithm configuration"""
    min_confidence: int = 7
    max_daily_signals: int = 6
    institutional_threshold: float = 2.5
    min_time_delta: float = 0.5  # seconds
    max_monitored_symbols: int = 20
    tick_processing_batch_size: int = 50

@dataclass
class MonitoringConfig:
    """System monitoring configuration"""
    health_check_interval: int = 30  # seconds
    enable_metrics: bool = True
    metrics_port: int = 9090
    log_retention_days: int = 7

@dataclass
class DataRetentionConfig:
    """Data retention configuration"""
    signal_retention_hours: int = 24
    tick_retention_hours: int = 1
    volume_history_size: int = 20

@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_cors: bool = False
    enable_xsrf_protection: bool = False
    max_upload_size: int = 50  # MB

@dataclass
class FeatureFlags:
    """Feature toggles"""
    enable_real_pcr: bool = True
    enable_signal_persistence: bool = False
    enable_performance_tracking: bool = True
    enable_alerts: bool = False

@dataclass
class DatabaseConfig:
    """Database configuration (optional)"""
    database_url: Optional[str] = None
    pool_size: int = 5
    max_overflow: int = 10

@dataclass
class AppConfig:
    """Main application configuration"""
    # Environment
    environment: str = "development"
    log_level: str = "INFO"
    timezone: str = "Asia/Kolkata"

    # Sub-configurations
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    data_retention: DataRetentionConfig = field(default_factory=DataRetentionConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Load configuration from environment variables"""
        config = cls()

        # Environment settings
        config.environment = os.getenv('ENV', 'development')
        config.log_level = os.getenv('LOG_LEVEL', 'INFO')
        config.timezone = os.getenv('TZ', 'Asia/Kolkata')

        # WebSocket configuration
        config.websocket.max_retries = int(os.getenv('WS_MAX_RETRIES', '5'))
        config.websocket.initial_backoff = int(os.getenv('WS_INITIAL_BACKOFF', '1'))
        config.websocket.connection_timeout = int(os.getenv('WS_CONNECTION_TIMEOUT', '30'))
        config.websocket.queue_max_size = int(os.getenv('QUEUE_MAX_SIZE', '1000'))

        # Circuit breaker configuration
        config.circuit_breaker.pcr_failure_threshold = int(os.getenv('PCR_FAILURE_THRESHOLD', '3'))
        config.circuit_breaker.pcr_recovery_timeout = int(os.getenv('PCR_RECOVERY_TIMEOUT', '300'))
        config.circuit_breaker.api_failure_threshold = int(os.getenv('API_FAILURE_THRESHOLD', '5'))
        config.circuit_breaker.api_recovery_timeout = int(os.getenv('API_RECOVERY_TIMEOUT', '60'))

        # Trading configuration
        config.trading.min_confidence = int(os.getenv('MIN_CONFIDENCE', '7'))
        config.trading.max_daily_signals = int(os.getenv('MAX_DAILY_SIGNALS', '6'))
        config.trading.institutional_threshold = float(os.getenv('INSTITUTIONAL_THRESHOLD', '2.5'))
        config.trading.min_time_delta = float(os.getenv('MIN_TIME_DELTA', '0.5'))
        config.trading.max_monitored_symbols = int(os.getenv('MAX_MONITORED_SYMBOLS', '20'))
        config.trading.tick_processing_batch_size = int(os.getenv('TICK_PROCESSING_BATCH_SIZE', '50'))

        # Monitoring configuration
        config.monitoring.health_check_interval = int(os.getenv('HEALTH_CHECK_INTERVAL', '30'))
        config.monitoring.enable_metrics = os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
        config.monitoring.metrics_port = int(os.getenv('METRICS_PORT', '9090'))
        config.monitoring.log_retention_days = int(os.getenv('LOG_RETENTION_DAYS', '7'))

        # Data retention configuration
        config.data_retention.signal_retention_hours = int(os.getenv('SIGNAL_RETENTION_HOURS', '24'))
        config.data_retention.tick_retention_hours = int(os.getenv('TICK_RETENTION_HOURS', '1'))
        config.data_retention.volume_history_size = int(os.getenv('VOLUME_HISTORY_SIZE', '20'))

        # Security configuration
        config.security.enable_cors = os.getenv('ENABLE_CORS', 'false').lower() == 'true'
        config.security.enable_xsrf_protection = os.getenv('ENABLE_XSRF_PROTECTION', 'false').lower() == 'true'
        config.security.max_upload_size = int(os.getenv('MAX_UPLOAD_SIZE', '50'))

        # Feature flags
        config.features.enable_real_pcr = os.getenv('ENABLE_REAL_PCR', 'true').lower() == 'true'
        config.features.enable_signal_persistence = os.getenv('ENABLE_SIGNAL_PERSISTENCE', 'false').lower() == 'true'
        config.features.enable_performance_tracking = os.getenv('ENABLE_PERFORMANCE_TRACKING', 'true').lower() == 'true'
        config.features.enable_alerts = os.getenv('ENABLE_ALERTS', 'false').lower() == 'true'

        # Database configuration
        config.database.database_url = os.getenv('DATABASE_URL')
        config.database.pool_size = int(os.getenv('DATABASE_POOL_SIZE', '5'))
        config.database.max_overflow = int(os.getenv('DATABASE_MAX_OVERFLOW', '10'))

        return config

    def validate(self) -> bool:
        """Validate configuration values"""
        errors = []

        # Validate trading parameters
        if not (1 <= self.trading.min_confidence <= 10):
            errors.append("MIN_CONFIDENCE must be between 1 and 10")

        if self.trading.max_daily_signals <= 0:
            errors.append("MAX_DAILY_SIGNALS must be positive")

        if self.trading.institutional_threshold <= 0:
            errors.append("INSTITUTIONAL_THRESHOLD must be positive")

        # Validate WebSocket parameters
        if self.websocket.max_retries <= 0:
            errors.append("WS_MAX_RETRIES must be positive")

        if self.websocket.initial_backoff <= 0:
            errors.append("WS_INITIAL_BACKOFF must be positive")

        # Validate circuit breaker parameters
        if self.circuit_breaker.pcr_failure_threshold <= 0:
            errors.append("PCR_FAILURE_THRESHOLD must be positive")

        if self.circuit_breaker.pcr_recovery_timeout <= 0:
            errors.append("PCR_RECOVERY_TIMEOUT must be positive")

        # Validate monitoring parameters
        if self.monitoring.health_check_interval <= 0:
            errors.append("HEALTH_CHECK_INTERVAL must be positive")

        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            return False

        logger.info("Configuration validation passed")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging"""
        return {
            'environment': self.environment,
            'log_level': self.log_level,
            'timezone': self.timezone,
            'websocket': {
                'max_retries': self.websocket.max_retries,
                'initial_backoff': self.websocket.initial_backoff,
                'connection_timeout': self.websocket.connection_timeout,
            },
            'trading': {
                'min_confidence': self.trading.min_confidence,
                'max_daily_signals': self.trading.max_daily_signals,
                'institutional_threshold': self.trading.institutional_threshold,
            },
            'features': {
                'enable_real_pcr': self.features.enable_real_pcr,
                'enable_signal_persistence': self.features.enable_signal_persistence,
                'enable_performance_tracking': self.features.enable_performance_tracking,
            }
        }

# Global configuration instance
_config: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = AppConfig.from_env()
        if not _config.validate():
            raise ValueError("Invalid configuration")
        logger.info("Configuration loaded and validated")
        logger.debug(f"Configuration: {_config.to_dict()}")
    return _config

def reload_config() -> AppConfig:
    """Reload configuration from environment"""
    global _config
    _config = AppConfig.from_env()
    if not _config.validate():
        raise ValueError("Invalid configuration after reload")
    logger.info("Configuration reloaded and validated")
    return _config