"""
Optional Signal Persistence for Historical Analysis

Provides SQLite-based storage for trading signals with optional
PostgreSQL support for production deployments.
"""

import sqlite3
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import os

logger = logging.getLogger(__name__)

class SignalPersistence:
    """Handles persistence of trading signals for historical analysis"""

    def __init__(self, db_path: str = "data/trading_signals.db"):
        self.db_path = db_path
        self._ensure_data_directory()
        self._initialize_database()

    def _ensure_data_directory(self):
        """Ensure data directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def _initialize_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create signals table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        confidence_score INTEGER NOT NULL,
                        current_price REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        target_price REAL NOT NULL,
                        stop_loss REAL NOT NULL,
                        risk_reward TEXT NOT NULL,
                        order_imbalance TEXT NOT NULL,
                        institutional_ratio REAL NOT NULL,
                        volume_status TEXT NOT NULL,
                        pcr REAL NOT NULL,
                        pcr_bias TEXT NOT NULL,
                        oi_change INTEGER,
                        oi_trend TEXT,
                        potential_profit_pct REAL,
                        relative_score REAL,
                        time_detected TEXT NOT NULL,
                        timestamp_created REAL NOT NULL,
                        additional_data TEXT
                    )
                """)

                # Create signal performance table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS signal_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_id INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        exit_price REAL,
                        actual_profit_pct REAL,
                        exit_time TEXT,
                        notes TEXT,
                        timestamp_updated REAL NOT NULL,
                        FOREIGN KEY (signal_id) REFERENCES signals (id)
                    )
                """)

                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp_created)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(signal_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_confidence ON signals(confidence_score)")

                conn.commit()
                logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def save_signal(self, signal: Dict[str, Any]) -> int:
        """
        Save a trading signal to database

        Args:
            signal: Signal dictionary from generate_signal function

        Returns:
            Signal ID if successful, None otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Extract signal data
                additional_data = {
                    k: v for k, v in signal.items()
                    if k not in [
                        'Stock Symbol', 'Signal Type', 'Confidence Score',
                        'Current Price', 'Entry Price', 'Target Price', 'Stop Loss',
                        'Risk:Reward', 'Order Imbalance', 'Institutional Ratio',
                        'Volume Status', 'PCR', 'PCR Bias', 'OI Change',
                        'OI Trend', 'Potential Profit %', 'Relative Score',
                        'Time Detected'
                    ]
                }

                cursor.execute("""
                    INSERT INTO signals (
                        symbol, signal_type, confidence_score, current_price,
                        entry_price, target_price, stop_loss, risk_reward,
                        order_imbalance, institutional_ratio, volume_status,
                        pcr, pcr_bias, oi_change, oi_trend,
                        potential_profit_pct, relative_score, time_detected,
                        timestamp_created, additional_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal.get('Stock Symbol'),
                    signal.get('Signal Type'),
                    signal.get('Confidence Score'),
                    signal.get('Current Price'),
                    signal.get('Entry Price'),
                    signal.get('Target Price'),
                    signal.get('Stop Loss'),
                    signal.get('Risk:Reward'),
                    signal.get('Order Imbalance'),
                    signal.get('Institutional Ratio'),
                    signal.get('Volume Status'),
                    signal.get('PCR'),
                    signal.get('PCR Bias'),
                    signal.get('OI Change'),
                    signal.get('OI Trend'),
                    signal.get('Potential Profit %'),
                    signal.get('Relative Score'),
                    signal.get('Time Detected'),
                    time.time(),
                    json.dumps(additional_data) if additional_data else None
                ))

                signal_id = cursor.lastrowid
                conn.commit()

                logger.debug(f"Signal saved: {signal.get('Stock Symbol')} - {signal.get('Signal Type')} (ID: {signal_id})")
                return signal_id

        except Exception as e:
            logger.error(f"Failed to save signal: {e}")
            return None

    def get_signals(self,
                   symbol: Optional[str] = None,
                   signal_type: Optional[str] = None,
                   min_confidence: Optional[int] = None,
                   hours_back: int = 24,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve signals from database with optional filtering

        Args:
            symbol: Filter by stock symbol
            signal_type: Filter by signal type
            min_confidence: Minimum confidence score
            hours_back: Hours to look back
            limit: Maximum number of signals to return

        Returns:
            List of signal dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Return rows as dictionaries
                cursor = conn.cursor()

                # Build query
                query = "SELECT * FROM signals WHERE timestamp_created >= ?"
                params = [time.time() - (hours_back * 3600)]

                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)

                if signal_type:
                    query += " AND signal_type = ?"
                    params.append(signal_type)

                if min_confidence:
                    query += " AND confidence_score >= ?"
                    params.append(min_confidence)

                query += " ORDER BY timestamp_created DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                signals = []
                for row in rows:
                    signal = dict(row)

                    # Parse additional data
                    if signal.get('additional_data'):
                        try:
                            additional = json.loads(signal['additional_data'])
                            signal.update(additional)
                        except:
                            pass
                    del signal['additional_data']

                    # Convert timestamp to readable time
                    signal['created_time'] = datetime.fromtimestamp(signal['timestamp_created']).strftime('%Y-%m-%d %H:%M:%S')

                    signals.append(signal)

                return signals

        except Exception as e:
            logger.error(f"Failed to retrieve signals: {e}")
            return []

    def update_signal_performance(self,
                                 signal_id: int,
                                 status: str,
                                 exit_price: Optional[float] = None,
                                 actual_profit_pct: Optional[float] = None,
                                 notes: Optional[str] = None):
        """
        Update signal performance after trade completion

        Args:
            signal_id: ID of the signal
            status: 'WIN', 'LOSS', 'PARTIAL', 'CANCELLED'
            exit_price: Exit price
            actual_profit_pct: Actual profit percentage
            notes: Additional notes
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO signal_performance (
                        signal_id, status, exit_price, actual_profit_pct,
                        notes, timestamp_updated
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    signal_id, status, exit_price, actual_profit_pct,
                    notes, time.time()
                ))

                conn.commit()
                logger.debug(f"Signal performance updated: ID {signal_id} - {status}")

        except Exception as e:
            logger.error(f"Failed to update signal performance: {e}")

    def get_statistics(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Get trading statistics for analysis

        Args:
            days_back: Number of days to analyze

        Returns:
            Statistics dictionary
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cutoff_time = time.time() - (days_back * 24 * 3600)

                # Total signals
                cursor.execute("SELECT COUNT(*) FROM signals WHERE timestamp_created >= ?", (cutoff_time,))
                total_signals = cursor.fetchone()[0]

                # Signals by type
                cursor.execute("""
                    SELECT signal_type, COUNT(*)
                    FROM signals
                    WHERE timestamp_created >= ?
                    GROUP BY signal_type
                """, (cutoff_time,))
                signals_by_type = dict(cursor.fetchall())

                # Average confidence
                cursor.execute("""
                    SELECT AVG(confidence_score)
                    FROM signals
                    WHERE timestamp_created >= ?
                """, (cutoff_time,))
                avg_confidence = cursor.fetchone()[0] or 0

                # Performance stats (if performance tracking is used)
                cursor.execute("""
                    SELECT sp.status, COUNT(*)
                    FROM signal_performance sp
                    JOIN signals s ON sp.signal_id = s.id
                    WHERE s.timestamp_created >= ?
                    GROUP BY sp.status
                """, (cutoff_time,))
                performance_by_status = dict(cursor.fetchall())

                # Win rate calculation
                wins = performance_by_status.get('WIN', 0)
                total_performance = sum(performance_by_status.values())
                win_rate = (wins / total_performance * 100) if total_performance > 0 else 0

                # Average profit
                cursor.execute("""
                    SELECT AVG(actual_profit_pct)
                    FROM signal_performance sp
                    JOIN signals s ON sp.signal_id = s.id
                    WHERE s.timestamp_created >= ? AND sp.actual_profit_pct IS NOT NULL
                """, (cutoff_time,))
                avg_profit = cursor.fetchone()[0] or 0

                return {
                    'period_days': days_back,
                    'total_signals': total_signals,
                    'signals_by_type': signals_by_type,
                    'average_confidence': round(avg_confidence, 2),
                    'performance_stats': performance_by_status,
                    'win_rate_percent': round(win_rate, 2),
                    'average_profit_pct': round(avg_profit, 2),
                    'total_tracked_performance': total_performance
                }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    def cleanup_old_data(self, days_to_keep: int = 90):
        """
        Clean up old signal data to manage database size

        Args:
            days_to_keep: Number of days to keep data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cutoff_time = time.time() - (days_to_keep * 24 * 3600)

                # Delete old signal performance records
                cursor.execute("""
                    DELETE FROM signal_performance
                    WHERE signal_id IN (
                        SELECT id FROM signals WHERE timestamp_created < ?
                    )
                """, (cutoff_time,))

                # Delete old signals
                cursor.execute("DELETE FROM signals WHERE timestamp_created < ?", (cutoff_time,))

                deleted_rows = cursor.rowcount
                conn.commit()

                logger.info(f"Cleaned up {deleted_rows} old signal records")

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")

# Global persistence instance
_persistence: Optional[SignalPersistence] = None

def get_signal_persistence() -> Optional[SignalPersistence]:
    """Get global signal persistence instance"""
    global _persistence
    return _persistence

def initialize_signal_persistence(db_path: str = "data/trading_signals.db", enabled: bool = True):
    """Initialize signal persistence"""
    global _persistence

    if not enabled:
        _persistence = None
        logger.info("Signal persistence disabled")
        return None

    try:
        _persistence = SignalPersistence(db_path)
        logger.info(f"Signal persistence initialized: {db_path}")
        return _persistence
    except Exception as e:
        logger.error(f"Signal persistence initialization failed: {e}")
        _persistence = None
        return None