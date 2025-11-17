"""
Professional Database Manager
Handles database operations for the trading dashboard with support for PostgreSQL and SQLite
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import asyncpg
import aiosqlite
import json
from contextlib import asynccontextmanager

from config import get_config

logger = logging.getLogger(__name__)
config = get_config()

class DatabaseManager:
    """Professional database manager with PostgreSQL and SQLite support"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.is_postgresql = database_url.startswith("postgresql")
        self.pool = None
        self.sqlite_connection = None

    async def initialize(self):
        """Initialize database connection and create tables"""
        try:
            if self.is_postgresql:
                await self._initialize_postgresql()
            else:
                await self._initialize_sqlite()

            await self._create_tables()
            logger.info("✅ Database initialized successfully")

        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise

    async def _initialize_postgresql(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=config.database.pool_size // 2,
                max_size=config.database.pool_size,
                command_timeout=60
            )
            logger.info("✅ PostgreSQL connection pool created")

        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            raise

    async def _initialize_sqlite(self):
        """Initialize SQLite connection"""
        try:
            # Ensure data directory exists
            import os
            os.makedirs(os.path.dirname(self.database_url.replace("sqlite:///", "")), exist_ok=True)

            self.sqlite_connection = await aiosqlite.connect(
                self.database_url.replace("sqlite:///", "")
            )
            # Enable WAL mode for better performance
            await self.sqlite_connection.execute("PRAGMA journal_mode=WAL")
            await self.sqlite_connection.execute("PRAGMA synchronous=NORMAL")
            await self.sqlite_connection.execute("PRAGMA cache_size=10000")
            await self.sqlite_connection.execute("PRAGMA temp_store=memory")

            logger.info("✅ SQLite connection established")

        except Exception as e:
            logger.error(f"SQLite connection failed: {e}")
            raise

    async def _create_tables(self):
        """Create database tables"""
        try:
            if self.is_postgresql:
                await self._create_postgresql_tables()
            else:
                await self._create_sqlite_tables()

        except Exception as e:
            logger.error(f"Table creation failed: {e}")
            raise

    async def _create_postgresql_tables(self):
        """Create PostgreSQL tables"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(255) UNIQUE NOT NULL,
                    api_key VARCHAR(255) NOT NULL,
                    api_secret VARCHAR(255) NOT NULL,
                    access_token VARCHAR(255) NOT NULL,
                    session_data JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    last_active TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    signal_id VARCHAR(255) UNIQUE NOT NULL,
                    user_id VARCHAR(255) NOT NULL,
                    symbol VARCHAR(50) NOT NULL,
                    signal_type VARCHAR(50) NOT NULL,
                    confidence_score INTEGER NOT NULL,
                    current_price DECIMAL(10,2) NOT NULL,
                    entry_price DECIMAL(10,2) NOT NULL,
                    target_price DECIMAL(10,2) NOT NULL,
                    stop_loss DECIMAL(10,2) NOT NULL,
                    risk_reward VARCHAR(20) NOT NULL,
                    order_imbalance VARCHAR(100) NOT NULL,
                    institutional_ratio DECIMAL(8,2) NOT NULL,
                    volume_status VARCHAR(50) NOT NULL,
                    pcr DECIMAL(6,2) NOT NULL,
                    pcr_bias VARCHAR(20) NOT NULL,
                    oi_change INTEGER,
                    oi_trend VARCHAR(20),
                    potential_profit_pct DECIMAL(6,2) NOT NULL,
                    relative_score DECIMAL(4,1) NOT NULL,
                    time_detected VARCHAR(10) NOT NULL,
                    timestamp_created BIGINT NOT NULL,
                    broadcasted BOOLEAN DEFAULT FALSE,
                    additional_data JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_performance (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    signal_id VARCHAR(255) NOT NULL,
                    user_id VARCHAR(255) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    exit_price DECIMAL(10,2),
                    actual_profit_pct DECIMAL(6,2),
                    notes TEXT,
                    exit_time TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    FOREIGN KEY (signal_id) REFERENCES trading_signals(signal_id)
                )
            """)

            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_trading_signals_user_id ON trading_signals(user_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol ON trading_signals(symbol)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_trading_signals_timestamp ON trading_signals(timestamp_created)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_trading_signals_broadcasted ON trading_signals(broadcasted)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_trading_signals_confidence ON trading_signals(confidence_score)")

    async def _create_sqlite_tables(self):
        """Create SQLite tables"""
        await self.sqlite_connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                user_id TEXT UNIQUE NOT NULL,
                api_key TEXT NOT NULL,
                api_secret TEXT NOT NULL,
                access_token TEXT NOT NULL,
                session_data TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_active TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await self.sqlite_connection.execute("""
            CREATE TABLE IF NOT EXISTS trading_signals (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                signal_id TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL,
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
                potential_profit_pct REAL NOT NULL,
                relative_score REAL NOT NULL,
                time_detected TEXT NOT NULL,
                timestamp_created INTEGER NOT NULL,
                broadcasted INTEGER DEFAULT 0,
                additional_data TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await self.sqlite_connection.execute("""
            CREATE TABLE IF NOT EXISTS signal_performance (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                signal_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                status TEXT NOT NULL,
                exit_price REAL,
                actual_profit_pct REAL,
                notes TEXT,
                exit_time TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (signal_id) REFERENCES trading_signals(signal_id)
            )
        """)

        # Create indexes
        await self.sqlite_connection.execute("CREATE INDEX IF NOT EXISTS idx_trading_signals_user_id ON trading_signals(user_id)")
        await self.sqlite_connection.execute("CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol ON trading_signals(symbol)")
        await self.sqlite_connection.execute("CREATE INDEX IF NOT EXISTS idx_trading_signals_timestamp ON trading_signals(timestamp_created)")
        await self.sqlite_connection.execute("CREATE INDEX IF NOT EXISTS idx_trading_signals_broadcasted ON trading_signals(broadcasted)")

    async def save_user_credentials(self, user_id: str, api_key: str, api_secret: str,
                                  access_token: str, session_data: Dict = None):
        """Save or update user credentials"""
        try:
            now = datetime.now().isoformat()

            if self.is_postgresql:
                async with self.pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO users (user_id, api_key, api_secret, access_token, session_data, updated_at, last_active)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (user_id) DO UPDATE SET
                        api_key = EXCLUDED.api_key,
                        api_secret = EXCLUDED.api_secret,
                        access_token = EXCLUDED.access_token,
                        session_data = EXCLUDED.session_data,
                        updated_at = EXCLUDED.updated_at,
                        last_active = EXCLUDED.last_active
                    """, user_id, api_key, api_secret, access_token,
                      json.dumps(session_data) if session_data else None, now, now)
            else:
                session_json = json.dumps(session_data) if session_data else None
                await self.sqlite_connection.execute("""
                    INSERT OR REPLACE INTO users
                    (user_id, api_key, api_secret, access_token, session_data, updated_at, last_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, user_id, api_key, api_secret, access_token, session_json, now, now)

            logger.debug(f"User credentials saved for {user_id}")

        except Exception as e:
            logger.error(f"Failed to save user credentials: {e}")
            raise

    async def get_user_credentials(self, user_id: str) -> Optional[Dict]:
        """Get user credentials"""
        try:
            if self.is_postgresql:
                async with self.pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        SELECT api_key, api_secret, access_token, session_data
                        FROM users WHERE user_id = $1
                    """, user_id)
            else:
                cursor = await self.sqlite_connection.execute("""
                    SELECT api_key, api_secret, access_token, session_data
                    FROM users WHERE user_id = ?
                """, (user_id,))
                row = await cursor.fetchone()

            if not row:
                return None

            session_data = json.loads(row['session_data']) if row['session_data'] else {}

            return {
                "api_key": row['api_key'],
                "api_secret": row['api_secret'],
                "access_token": row['access_token'],
                "session_data": session_data
            }

        except Exception as e:
            logger.error(f"Failed to get user credentials: {e}")
            return None

    async def save_signal(self, signal_data: Dict) -> bool:
        """Save trading signal to database"""
        try:
            if self.is_postgresql:
                async with self.pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO trading_signals (
                            signal_id, user_id, symbol, signal_type, confidence_score,
                            current_price, entry_price, target_price, stop_loss,
                            risk_reward, order_imbalance, institutional_ratio,
                            volume_status, pcr, pcr_bias, oi_change, oi_trend,
                            potential_profit_pct, relative_score, time_detected,
                            timestamp_created, additional_data
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                            $14, $15, $16, $17, $18, $19, $20, $21, $22
                        )
                        ON CONFLICT (signal_id) DO NOTHING
                    """,
                    signal_data['id'], signal_data['user_id'], signal_data['symbol'],
                    signal_data['signal_type'], signal_data['confidence_score'],
                    signal_data['current_price'], signal_data['entry_price'],
                    signal_data['target_price'], signal_data['stop_loss'],
                    signal_data['risk_reward'], signal_data['order_imbalance'],
                    signal_data['institutional_ratio'], signal_data['volume_status'],
                    signal_data['pcr'], signal_data['pcr_bias'], signal_data.get('oi_change'),
                    signal_data.get('oi_trend'), signal_data['potential_profit_pct'],
                    signal_data['relative_score'], signal_data['time_detected'],
                    signal_data['timestamp_created'],
                    json.dumps({k: v for k, v in signal_data.items()
                              if k not in ['id', 'user_id', 'symbol', 'signal_type',
                                       'confidence_score', 'current_price', 'entry_price',
                                       'target_price', 'stop_loss', 'risk_reward',
                                       'order_imbalance', 'institutional_ratio',
                                       'volume_status', 'pcr', 'pcr_bias', 'oi_change',
                                       'oi_trend', 'potential_profit_pct', 'relative_score',
                                       'time_detected', 'timestamp_created']})
                    )
            else:
                additional_data = {k: v for k, v in signal_data.items()
                                 if k not in ['id', 'user_id', 'symbol', 'signal_type',
                                          'confidence_score', 'current_price', 'entry_price',
                                          'target_price', 'stop_loss', 'risk_reward',
                                          'order_imbalance', 'institutional_ratio',
                                          'volume_status', 'pcr', 'pcr_bias', 'oi_change',
                                          'oi_trend', 'potential_profit_pct', 'relative_score',
                                          'time_detected', 'timestamp_created']}

                await self.sqlite_connection.execute("""
                    INSERT OR IGNORE INTO trading_signals (
                        signal_id, user_id, symbol, signal_type, confidence_score,
                        current_price, entry_price, target_price, stop_loss,
                        risk_reward, order_imbalance, institutional_ratio,
                        volume_status, pcr, pcr_bias, oi_change, oi_trend,
                        potential_profit_pct, relative_score, time_detected,
                        timestamp_created, additional_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    signal_data['id'], signal_data['user_id'], signal_data['symbol'],
                    signal_data['signal_type'], signal_data['confidence_score'],
                    signal_data['current_price'], signal_data['entry_price'],
                    signal_data['target_price'], signal_data['stop_loss'],
                    signal_data['risk_reward'], signal_data['order_imbalance'],
                    signal_data['institutional_ratio'], signal_data['volume_status'],
                    signal_data['pcr'], signal_data['pcr_bias'], signal_data.get('oi_change'),
                    signal_data.get('oi_trend'), signal_data['potential_profit_pct'],
                    signal_data['relative_score'], signal_data['time_detected'],
                    signal_data['timestamp_created'], json.dumps(additional_data))

            return True

        except Exception as e:
            logger.error(f"Failed to save signal: {e}")
            return False

    async def get_signals(self, user_id: str = None, limit: int = 50,
                         min_confidence: int = 7, symbol: str = None,
                         hours_back: int = 24) -> List[Dict]:
        """Get trading signals with filtering"""
        try:
            cutoff_time = int((datetime.now() - timedelta(hours=hours_back)).timestamp())

            if self.is_postgresql:
                async with self.pool.acquire() as conn:
                    query = """
                        SELECT * FROM trading_signals
                        WHERE timestamp_created >= $1 AND confidence_score >= $2
                    """
                    params = [cutoff_time, min_confidence]

                    if user_id:
                        query += " AND user_id = $3"
                        params.append(user_id)
                        if symbol:
                            query += " AND symbol = $4"
                            params.append(symbol)
                    elif symbol:
                        query += " AND symbol = $3"
                        params.append(symbol)

                    query += " ORDER BY timestamp_created DESC LIMIT $" + str(len(params) + 1)
                    params.append(limit)

                    rows = await conn.fetch(query, *params)

                    return [dict(row) for row in rows]

            else:
                query = """
                    SELECT * FROM trading_signals
                    WHERE timestamp_created >= ? AND confidence_score >= ?
                """
                params = [cutoff_time, min_confidence]

                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)
                    if symbol:
                        query += " AND symbol = ?"
                        params.append(symbol)
                elif symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)

                query += " ORDER BY timestamp_created DESC LIMIT ?"
                params.append(limit)

                cursor = await self.sqlite_connection.execute(query, params)
                rows = await cursor.fetchall()

                # Convert to list of dicts
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get signals: {e}")
            return []

    async def get_unbroadcasted_signals(self, limit: int = 100) -> List[Dict]:
        """Get signals that haven't been broadcasted"""
        try:
            if self.is_postgresql:
                async with self.pool.acquire() as conn:
                    rows = await conn.fetch("""
                        SELECT * FROM trading_signals
                        WHERE broadcasted = FALSE
                        ORDER BY timestamp_created DESC
                        LIMIT $1
                    """, limit)

                    return [dict(row) for row in rows]

            else:
                cursor = await self.sqlite_connection.execute("""
                    SELECT * FROM trading_signals
                    WHERE broadcasted = 0
                    ORDER BY timestamp_created DESC
                    LIMIT ?
                """, (limit,))

                rows = await cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get unbroadcasted signals: {e}")
            return []

    async def mark_signal_broadcasted(self, signal_id: str):
        """Mark signal as broadcasted"""
        try:
            if self.is_postgresql:
                async with self.pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE trading_signals
                        SET broadcasted = TRUE
                        WHERE signal_id = $1
                    """, signal_id)
            else:
                await self.sqlite_connection.execute("""
                    UPDATE trading_signals
                    SET broadcasted = 1
                    WHERE signal_id = ?
                """, (signal_id,))

        except Exception as e:
            logger.error(f"Failed to mark signal as broadcasted: {e}")

    async def get_statistics(self, user_id: str, days_back: int = 30) -> Dict:
        """Get trading statistics for a user"""
        try:
            cutoff_time = int((datetime.now() - timedelta(days=days_back)).timestamp())

            if self.is_postgresql:
                async with self.pool.acquire() as conn:
                    # Total signals
                    total_signals = await conn.fetchval("""
                        SELECT COUNT(*) FROM trading_signals
                        WHERE user_id = $1 AND timestamp_created >= $2
                    """, user_id, cutoff_time)

                    # Average confidence
                    avg_confidence = await conn.fetchval("""
                        SELECT AVG(confidence_score) FROM trading_signals
                        WHERE user_id = $1 AND timestamp_created >= $2
                    """, user_id, cutoff_time) or 0

                    # Signal type distribution
                    signal_types = await conn.fetch("""
                        SELECT signal_type, COUNT(*)
                        FROM trading_signals
                        WHERE user_id = $1 AND timestamp_created >= $2
                        GROUP BY signal_type
                    """, user_id, cutoff_time)

                    # Performance stats
                    performance = await conn.fetch("""
                        SELECT sp.status, COUNT(*)
                        FROM signal_performance sp
                        JOIN trading_signals ts ON sp.signal_id = ts.signal_id
                        WHERE ts.user_id = $1 AND ts.timestamp_created >= $2
                        GROUP BY sp.status
                    """, user_id, cutoff_time)

                    # Average profit
                    avg_profit = await conn.fetchval("""
                        SELECT AVG(sp.actual_profit_pct)
                        FROM signal_performance sp
                        JOIN trading_signals ts ON sp.signal_id = ts.signal_id
                        WHERE ts.user_id = $1 AND ts.timestamp_created >= $2
                        AND sp.actual_profit_pct IS NOT NULL
                    """, user_id, cutoff_time) or 0

            else:
                # Total signals
                cursor = await self.sqlite_connection.execute("""
                    SELECT COUNT(*) FROM trading_signals
                    WHERE user_id = ? AND timestamp_created >= ?
                """, (user_id, cutoff_time))
                total_signals = (await cursor.fetchone())[0]

                # Average confidence
                cursor = await self.sqlite_connection.execute("""
                    SELECT AVG(confidence_score) FROM trading_signals
                    WHERE user_id = ? AND timestamp_created >= ?
                """, (user_id, cutoff_time))
                avg_confidence = (await cursor.fetchone())[0] or 0

                # Signal types
                cursor = await self.sqlite_connection.execute("""
                    SELECT signal_type, COUNT(*) FROM trading_signals
                    WHERE user_id = ? AND timestamp_created >= ?
                    GROUP BY signal_type
                """, (user_id, cutoff_time))
                signal_types = await cursor.fetchall()

                # Performance
                cursor = await self.sqlite_connection.execute("""
                    SELECT sp.status, COUNT(*)
                    FROM signal_performance sp
                    JOIN trading_signals ts ON sp.signal_id = ts.signal_id
                    WHERE ts.user_id = ? AND ts.timestamp_created >= ?
                    GROUP BY sp.status
                """, (user_id, cutoff_time))
                performance = await cursor.fetchall()

                # Average profit
                cursor = await self.sqlite_connection.execute("""
                    SELECT AVG(sp.actual_profit_pct)
                    FROM signal_performance sp
                    JOIN trading_signals ts ON sp.signal_id = ts.signal_id
                    WHERE ts.user_id = ? AND ts.timestamp_created >= ?
                    AND sp.actual_profit_pct IS NOT NULL
                """, (user_id, cutoff_time))
                avg_profit = (await cursor.fetchone())[0] or 0

            # Calculate win rate
            performance_dict = dict(performance) if performance else {}
            wins = performance_dict.get('WIN', 0)
            total_performance = sum(performance_dict.values()) if performance else 0
            win_rate = (wins / total_performance * 100) if total_performance > 0 else 0

            return {
                'period_days': days_back,
                'total_signals': total_signals,
                'average_confidence': round(float(avg_confidence), 2),
                'signals_by_type': dict(signal_types) if signal_types else {},
                'performance_stats': performance_dict,
                'win_rate_percent': round(win_rate, 2),
                'average_profit_pct': round(float(avg_profit), 2),
                'total_tracked_performance': total_performance
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    async def get_user_signal_count(self, user_id: str) -> int:
        """Get total signal count for a user"""
        try:
            if self.is_postgresql:
                async with self.pool.acquire() as conn:
                    return await conn.fetchval(
                        "SELECT COUNT(*) FROM trading_signals WHERE user_id = $1", user_id
                    )
            else:
                cursor = await self.sqlite_connection.execute(
                    "SELECT COUNT(*) FROM trading_signals WHERE user_id = ?", (user_id,)
                )
                return (await cursor.fetchone())[0]

        except Exception as e:
            logger.error(f"Failed to get user signal count: {e}")
            return 0

    async def get_user_last_signal_time(self, user_id: str) -> Optional[float]:
        """Get last signal time for a user"""
        try:
            if self.is_postgresql:
                async with self.pool.acquire() as conn:
                    timestamp = await conn.fetchval("""
                        SELECT MAX(timestamp_created) FROM trading_signals WHERE user_id = $1
                    """, user_id)
                    return float(timestamp) if timestamp else None
            else:
                cursor = await self.sqlite_connection.execute("""
                    SELECT MAX(timestamp_created) FROM trading_signals WHERE user_id = ?
                """, (user_id,))
                result = await cursor.fetchone()
                return float(result[0]) if result and result[0] else None

        except Exception as e:
            logger.error(f"Failed to get user last signal time: {e}")
            return None

    async def health_check(self) -> Dict:
        """Perform database health check"""
        try:
            if self.is_postgresql:
                async with self.pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                    return {"status": "healthy", "type": "postgresql"}
            else:
                await self.sqlite_connection.execute("SELECT 1")
                return {"status": "healthy", "type": "sqlite"}

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    async def close(self):
        """Close database connections"""
        try:
            if self.is_postgresql and self.pool:
                await self.pool.close()
                logger.info("PostgreSQL connection pool closed")
            elif self.sqlite_connection:
                await self.sqlite_connection.close()
                logger.info("SQLite connection closed")

        except Exception as e:
            logger.error(f"Error closing database connection: {e}")