"""
Professional Authentication Manager
Handles JWT authentication and Zerodha Kite Connect integration
"""

import jwt
import hashlib
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import secrets

from kiteconnect import KiteConnect

logger = logging.getLogger(__name__)

class AuthManager:
    """Professional authentication manager with JWT and Kite Connect integration"""

    def __init__(self):
        self.secret_key = secrets.token_urlsafe(32)  # In production, use environment variable
        self.algorithm = "HS256"
        self.token_expiry_hours = 24
        self.blacklisted_tokens = set()

    def create_token(self, session_data: Dict) -> str:
        """Create JWT token for authenticated user"""
        try:
            payload = {
                "user_id": self._generate_user_id(session_data["api_key"]),
                "api_key": session_data["api_key"][:8] + "****",  # Masked API key
                "session_active": True,
                "exp": datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
                "iat": datetime.utcnow(),
                "jti": secrets.token_urlsafe(16)  # JWT ID for blacklisting
            }

            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.info(f"Token created for user {payload['user_id']}")
            return token

        except Exception as e:
            logger.error(f"Token creation failed: {e}")
            raise

    def verify_token(self, token: str) -> bool:
        """Verify JWT token"""
        try:
            if token in self.blacklisted_tokens:
                return False

            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Check if token is expired
            if datetime.utcnow() > datetime.fromtimestamp(payload["exp"]):
                return False

            return True

        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return False
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return False
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return False

    def get_user_id_from_token(self, token: str) -> Optional[str]:
        """Extract user ID from token"""
        try:
            if not self.verify_token(token):
                return None

            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload.get("user_id")

        except Exception as e:
            logger.error(f"Failed to extract user ID from token: {e}")
            return None

    def invalidate_token(self, token: str):
        """Add token to blacklist"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            self.blacklisted_tokens.add(payload["jti"])
            logger.info(f"Token invalidated for user {payload.get('user_id')}")
        except Exception as e:
            logger.error(f"Token invalidation failed: {e}")

    def _generate_user_id(self, api_key: str) -> str:
        """Generate consistent user ID from API key"""
        return hashlib.sha256(api_key.encode()).hexdigest()[:32]

    async def authenticate_with_kite(self, api_key: str, api_secret: str, request_token: str) -> Optional[Dict]:
        """Authenticate with Zerodha Kite Connect"""
        try:
            kite = KiteConnect(api_key=api_key)

            # Generate session
            data = kite.generate_session(request_token, api_secret=api_secret)

            if not data or "access_token" not in data:
                logger.error("Kite Connect authentication failed - no access token received")
                return None

            # Validate access token by making a test API call
            kite.set_access_token(data["access_token"])

            try:
                # Test API call to get profile
                profile = kite.profile()
                if not profile:
                    logger.error("Kite Connect authentication failed - invalid profile")
                    return None

                logger.info(f"✅ Kite Connect authentication successful for user: {profile.get('user_name', 'Unknown')}")

                # Create session data
                session_data = {
                    "api_key": api_key,
                    "api_secret": api_secret,
                    "access_token": data["access_token"],
                    "public_token": data.get("public_token"),
                    "user_id": profile.get("user_id"),
                    "user_name": profile.get("user_name"),
                    "user_type": profile.get("user_type"),
                    "email": profile.get("email"),
                    "broker": profile.get("broker"),
                    "products": profile.get("products", []),
                    "exchanges": profile.get("exchanges", []),
                    "order_types": profile.get("order_types", []),
                }

                return session_data

            except Exception as api_error:
                logger.error(f"Kite Connect API validation failed: {api_error}")
                return None

        except Exception as e:
            logger.error(f"Kite Connect authentication failed: {e}")
            return None

    async def refresh_token_if_needed(self, token: str) -> Optional[str]:
        """Refresh token if close to expiry (Kite Connect tokens don't expire normally)"""
        try:
            if not self.verify_token(token):
                return None

            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            exp_time = datetime.fromtimestamp(payload["exp"])
            current_time = datetime.utcnow()

            # If token expires within next hour, refresh it
            if (exp_time - current_time) < timedelta(hours=1):
                logger.info("Token close to expiry, refreshing...")
                # For Kite Connect, tokens typically don't expire unless manually revoked
                # So we just return the same token
                return token

            return token

        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return None

    def get_token_info(self, token: str) -> Optional[Dict]:
        """Get token information without verification (for debugging)"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm], options={"verify_exp": False})
            return {
                "user_id": payload.get("user_id"),
                "api_key": payload.get("api_key"),
                "session_active": payload.get("session_active"),
                "issued_at": payload.get("iat"),
                "expires_at": payload.get("exp"),
                "expired": datetime.utcnow() > datetime.fromtimestamp(payload["exp"]),
                "jti": payload.get("jti"),
                "blacklisted": payload.get("jti") in self.blacklisted_tokens
            }

        except Exception as e:
            logger.error(f"Failed to get token info: {e}")
            return None

    def cleanup_expired_tokens(self):
        """Remove expired tokens from blacklist (maintenance task)"""
        try:
            current_tokens = set()
            for token_jti in self.blacklisted_tokens.copy():
                # Check if token is expired by attempting to decode
                try:
                    payload = jwt.decode(
                        {"jti": token_jti, "exp": 0},  # Will always be expired
                        self.secret_key,
                        algorithms=[self.algorithm],
                        options={"verify_exp": True}
                    )
                except jwt.ExpiredSignatureError:
                    # Token is expired, remove from blacklist
                    self.blacklisted_tokens.discard(token_jti)
                except:
                    # Other error, keep in blacklist
                    continue

            logger.debug(f"Cleaned up expired tokens. Active blacklisted tokens: {len(self.blacklisted_tokens)}")

        except Exception as e:
            logger.error(f"Token cleanup failed: {e}")

class RateLimiter:
    """Simple rate limiter for authentication endpoints"""

    def __init__(self, max_requests: int = 10, window_seconds: int = 300):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for identifier"""
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        if identifier in self.requests:
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > window_start
            ]
        else:
            self.requests[identifier] = []

        # Check if under limit
        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(now)
            return True

        return False

    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for identifier"""
        now = time.time()
        window_start = now - self.window_seconds

        if identifier not in self.requests:
            return self.max_requests

        # Count requests in current window
        recent_requests = [
            req_time for req_time in self.requests[identifier]
            if req_time > window_start
        ]

        return max(0, self.max_requests - len(recent_requests))

# Global instances
auth_manager = AuthManager()
rate_limiter = RateLimiter()