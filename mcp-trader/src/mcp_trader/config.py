"""Enhanced configuration for MCP Trader with all new features."""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent.parent / ".env"
# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Enhanced configuration for MCP Trader with all APIs and features."""

    # Core API Keys
    tiingo_api_key: Optional[str] = None
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None
    
    # New API Keys for enhanced features
    alpha_vantage_api_key: Optional[str] = None
    news_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None
    polygon_api_key: Optional[str] = None
    
    # Social Media APIs (optional)
    twitter_bearer_token: Optional[str] = None
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    
    # Server settings
    server_name: str = "mcp-trader"
    
    # Alert system settings
    alert_check_interval: int = 60  # seconds
    max_alerts_per_symbol: int = 10
    alert_history_limit: int = 1000
    
    # Screening settings
    max_concurrent_requests: int = 10
    screening_cache_ttl: int = 300  # 5 minutes
    
    # News sentiment settings
    news_cache_ttl: int = 1800  # 30 minutes
    max_articles_per_request: int = 100
    sentiment_analysis_enabled: bool = True
    
    # Fundamental analysis settings
    fundamental_cache_ttl: int = 3600  # 1 hour
    
    # Machine Learning settings (for future price prediction)
    ml_model_path: Optional[str] = None
    enable_price_prediction: bool = False
    
    # Backtesting settings
    backtest_initial_capital: float = 100000.0
    backtest_commission: float = 0.001  # 0.1%
    
    # Risk management defaults
    default_max_risk_percent: float = 2.0
    default_position_size_method: str = "fixed_risk"  # fixed_risk, percent_portfolio, kelly
    
    # Data provider preferences
    preferred_stock_provider: str = "tiingo"
    preferred_crypto_provider: str = "tiingo"  # tiingo or binance
    
    # Feature flags
    enable_alerts: bool = True
    enable_screening: bool = True
    enable_fundamental_analysis: bool = True
    enable_sentiment_analysis: bool = True
    enable_pattern_recognition: bool = True
    enable_backtesting: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file_path: str = "mcp_trader.log"

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        
        def get_bool(key: str, default: bool = False) -> bool:
            """Get boolean from environment variable."""
            value = os.getenv(key, "").lower()
            return value in ("true", "1", "yes", "on") if value else default
        
        def get_int(key: str, default: int) -> int:
            """Get integer from environment variable."""
            try:
                return int(os.getenv(key, str(default)))
            except ValueError:
                return default
        
        def get_float(key: str, default: float) -> float:
            """Get float from environment variable."""
            try:
                return float(os.getenv(key, str(default)))
            except ValueError:
                return default
        
        return cls(
            # Core API Keys
            tiingo_api_key=os.getenv("TIINGO_API_KEY"),
            binance_api_key=os.getenv("BINANCE_API_KEY"),
            binance_api_secret=os.getenv("BINANCE_API_SECRET"),
            
            # Enhanced API Keys
            alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
            news_api_key=os.getenv("NEWS_API_KEY"),
            finnhub_api_key=os.getenv("FINNHUB_API_KEY"),
            polygon_api_key=os.getenv("POLYGON_API_KEY"),
            
            # Social Media APIs
            twitter_bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
            reddit_client_id=os.getenv("REDDIT_CLIENT_ID"),
            reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            
            # Server settings
            server_name=os.getenv("MCP_SERVER_NAME", "mcp-trader-enhanced"),
            
            # Alert system
            alert_check_interval=get_int("ALERT_CHECK_INTERVAL", 60),
            max_alerts_per_symbol=get_int("MAX_ALERTS_PER_SYMBOL", 10),
            alert_history_limit=get_int("ALERT_HISTORY_LIMIT", 1000),
            
            # Screening
            max_concurrent_requests=get_int("MAX_CONCURRENT_REQUESTS", 10),
            screening_cache_ttl=get_int("SCREENING_CACHE_TTL", 300),
            
            # News sentiment
            news_cache_ttl=get_int("NEWS_CACHE_TTL", 1800),
            max_articles_per_request=get_int("MAX_ARTICLES_PER_REQUEST", 100),
            sentiment_analysis_enabled=get_bool("SENTIMENT_ANALYSIS_ENABLED", True),
            
            # Fundamental analysis
            fundamental_cache_ttl=get_int("FUNDAMENTAL_CACHE_TTL", 3600),
            
            # ML settings
            ml_model_path=os.getenv("ML_MODEL_PATH"),
            enable_price_prediction=get_bool("ENABLE_PRICE_PREDICTION", False),
            
            # Backtesting
            backtest_initial_capital=get_float("BACKTEST_INITIAL_CAPITAL", 100000.0),
            backtest_commission=get_float("BACKTEST_COMMISSION", 0.001),
            
            # Risk management
            default_max_risk_percent=get_float("DEFAULT_MAX_RISK_PERCENT", 2.0),
            default_position_size_method=os.getenv("DEFAULT_POSITION_SIZE_METHOD", "fixed_risk"),
            
            # Data providers
            preferred_stock_provider=os.getenv("PREFERRED_STOCK_PROVIDER", "tiingo"),
            preferred_crypto_provider=os.getenv("PREFERRED_CRYPTO_PROVIDER", "tiingo"),
            
            # Feature flags
            enable_alerts=get_bool("ENABLE_ALERTS", True),
            enable_screening=get_bool("ENABLE_SCREENING", True),
            enable_fundamental_analysis=get_bool("ENABLE_FUNDAMENTAL_ANALYSIS", True),
            enable_sentiment_analysis=get_bool("ENABLE_SENTIMENT_ANALYSIS", True),
            enable_pattern_recognition=get_bool("ENABLE_PATTERN_RECOGNITION", True),
            enable_backtesting=get_bool("ENABLE_BACKTESTING", True),
            
            # Logging
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_to_file=get_bool("LOG_TO_FILE", False),
            log_file_path=os.getenv("LOG_FILE_PATH", "mcp_trader.log")
        )
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings/errors."""
        warnings = []
        
        # Check required API keys for core functionality
        if not self.tiingo_api_key:
            warnings.append("⚠️ TIINGO_API_KEY not configured - stock data will be limited")
        
        # Check optional API keys
        if self.enable_fundamental_analysis and not self.alpha_vantage_api_key:
            warnings.append("⚠️ ALPHA_VANTAGE_API_KEY not configured - fundamental analysis disabled")
        
        if self.enable_sentiment_analysis and not self.news_api_key:
            warnings.append("⚠️ NEWS_API_KEY not configured - sentiment analysis will use mock data")
        
        # Validate numeric ranges
        if not (1 <= self.alert_check_interval <= 3600):
            warnings.append("⚠️ ALERT_CHECK_INTERVAL should be between 1-3600 seconds")
        
        if not (0.01 <= self.default_max_risk_percent <= 10.0):
            warnings.append("⚠️ DEFAULT_MAX_RISK_PERCENT should be between 0.01-10.0")
        
        if not (0.0 <= self.backtest_commission <= 0.01):
            warnings.append("⚠️ BACKTEST_COMMISSION should be between 0.0-0.01 (0-1%)")
        
        # Check provider settings
        valid_providers = ["tiingo", "binance"]
        if self.preferred_crypto_provider not in valid_providers:
            warnings.append(f"⚠️ PREFERRED_CRYPTO_PROVIDER should be one of: {valid_providers}")
        
        return warnings
    
    def get_enabled_features(self) -> list[str]:
        """Get list of enabled features."""
        features = ["technical_analysis", "risk_management"]
        
        if self.enable_alerts:
            features.append("alerts")
        if self.enable_screening:
            features.append("screening")
        if self.enable_fundamental_analysis and self.alpha_vantage_api_key:
            features.append("fundamental_analysis")
        if self.enable_sentiment_analysis:
            features.append("sentiment_analysis")
        if self.enable_pattern_recognition:
            features.append("pattern_recognition")
        if self.enable_backtesting:
            features.append("backtesting")
        if self.enable_price_prediction and self.ml_model_path:
            features.append("price_prediction")
        
        return features
    
    def get_api_status(self) -> dict[str, str]:
        """Get status of all API keys."""
        return {
            "tiingo": "✅ Configured" if self.tiingo_api_key else "❌ Missing",
            "binance": "✅ Configured" if self.binance_api_key else "❌ Missing", 
            "alpha_vantage": "✅ Configured" if self.alpha_vantage_api_key else "❌ Missing",
            "news_api": "✅ Configured" if self.news_api_key else "❌ Missing",
            "finnhub": "✅ Configured" if self.finnhub_api_key else "❌ Missing",
            "polygon": "✅ Configured" if self.polygon_api_key else "❌ Missing",
            "twitter": "✅ Configured" if self.twitter_bearer_token else "❌ Missing",
            "reddit": "✅ Configured" if self.reddit_client_id else "❌ Missing"
        }


# Global config instance
config = Config.from_env()

# Validate and log warnings on startup
_config_warnings = config.validate()
if _config_warnings:
    import logging
    logger = logging.getLogger(__name__)
    for warning in _config_warnings:
        logger.warning(warning)


# Helper function to check if a feature is available
def is_feature_available(feature_name: str) -> bool:
    """Check if a specific feature is available and properly configured."""
    feature_map = {
        "fundamental_analysis": config.enable_fundamental_analysis and config.alpha_vantage_api_key,
        "sentiment_analysis": config.enable_sentiment_analysis and config.news_api_key,
        "alerts": config.enable_alerts,
        "screening": config.enable_screening,
        "pattern_recognition": config.enable_pattern_recognition,
        "backtesting": config.enable_backtesting,
        "price_prediction": config.enable_price_prediction and config.ml_model_path,
        "crypto_binance": config.binance_api_key is not None,
        "social_twitter": config.twitter_bearer_token is not None,
        "social_reddit": config.reddit_client_id is not None
    }
    
    return feature_map.get(feature_name, False)


# Configuration summary for logging
def get_config_summary() -> str:
    """Get a summary of the current configuration."""
    enabled_features = config.get_enabled_features()
    api_status = config.get_api_status()
    
    summary = f"""
🔧 MCP Trader Enhanced Configuration Summary:

📊 Enabled Features ({len(enabled_features)}):
{chr(10).join(f"  • {feature}" for feature in enabled_features)}

🔑 API Status:
{chr(10).join(f"  • {api}: {status}" for api, status in api_status.items())}

⚙️ Key Settings:
  • Alert Check Interval: {config.alert_check_interval}s
  • Max Concurrent Requests: {config.max_concurrent_requests}
  • Default Risk %: {config.default_max_risk_percent}%
  • Preferred Crypto Provider: {config.preferred_crypto_provider}
  • Backtesting Capital: ${config.backtest_initial_capital:,.0f}

🎯 Server: {config.server_name}
📝 Log Level: {config.log_level}
"""
    
    return summary.strip()