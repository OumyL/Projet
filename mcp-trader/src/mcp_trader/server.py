#!/usr/bin/env python3
"""
MCP Trader Server Extended - Version finale corrigée
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, List, Dict, Optional
import pandas as pd

from dotenv import load_dotenv
from fastmcp import Context, FastMCP

# Import configuration
from .config import config

# Import our modules
from .data import EnhancedMarketData
from .indicators import (
    PatternRecognition,
    RelativeStrength,
    RiskAnalysis,
    TechnicalAnalysis,
    VolumeProfile,
)
from aiohttp import web

def fix_volume_types(df):
    if df is None or len(df) == 0:
        return df
    try:
        if 'volume' in df.columns:
            df = df.copy()
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype('int64')
        return df
    except Exception as e:
        logger.warning(f"Error fixing volume types: {e}")
        return df
def run_http_test_server():
    """
    Pour les tests, retourne une app aiohttp très simple
    avec un endpoint /health qui répond {"status": "ok"}.
    """
    app = web.Application()
    async def health(request):
        return web.json_response({"status": "ok"})
    app.router.add_get("/health", health)
    return app

# Import optional modules avec gestion d'erreur
try:
    from .fundamental import AlphaVantageClient, FundamentalAnalyzer
    FUNDAMENTAL_AVAILABLE = True
    
except ImportError:
    FUNDAMENTAL_AVAILABLE = False

try:
    from .news_sentiment import MarketNewsAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP(config.server_name)

# Initialize service instances
market_data = EnhancedMarketData()
tech_analysis = TechnicalAnalysis()
rs_analysis = RelativeStrength()
volume_analysis = VolumeProfile()
pattern_recognition = PatternRecognition()
risk_analysis = RiskAnalysis()

# Initialize optional services 
alpha_vantage_client = None
fundamental_analyzer = None
news_analyzer = None

if FUNDAMENTAL_AVAILABLE:
    try:
        if config.alpha_vantage_api_key:
            alpha_vantage_client = AlphaVantageClient(config.alpha_vantage_api_key)
            fundamental_analyzer = FundamentalAnalyzer(alpha_vantage_client)
            logger.info("Alpha Vantage fundamental analysis enabled")
    except Exception as e:
        logger.warning(f"Alpha Vantage not available: {e}")

if SENTIMENT_AVAILABLE:
    try:
        if config.news_api_key:
            news_analyzer = MarketNewsAnalyzer(config.news_api_key)
            logger.info("News sentiment analysis enabled")
    except Exception as e:
        logger.warning(f"News API not available: {e}")

# Context-aware helper functions
def generate_request_id() -> str:
    """Generate a unique request ID for tracking."""
    return f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

async def log_request_start(ctx: Context, tool_name: str, params: dict[str, Any]) -> str:
    """Log the start of a request with context."""
    request_id = generate_request_id()
    await ctx.log(f"[{request_id}] Starting {tool_name} with params: {params}")
    return request_id

async def log_request_end(
    ctx: Context, request_id: str, tool_name: str, success: bool, message: str = ""
):
    """Log the end of a request with context."""
    status = "SUCCESS" if success else "FAILED"
    await ctx.log(f"[{request_id}] {tool_name} {status}{f': {message}' if message else ''}")

async def handle_error(ctx: Context, request_id: str, tool_name: str, error: Exception) -> str:
    """Handle and log errors with context."""
    error_msg = f"Error in {tool_name}: {str(error)}"
    await ctx.log(f"[{request_id}] ERROR: {error_msg}", level="error")

    if hasattr(error, "__traceback__"):
        import traceback
        tb_lines = traceback.format_tb(error.__traceback__)
        await ctx.log(f"[{request_id}] Traceback: {''.join(tb_lines)}", level="debug")

    return error_msg

# Tools Implementation
#tool d'analyse crypto
@mcp.tool()
async def analyze_crypto(
    ctx: Context,
    symbol: str,
    provider: str | None = None,
    lookback_days: int = 365,
    quote_currency: str = "usd",
) -> str:
    """
    Analyze a crypto asset's technical setup with enhanced indicators and automatic fallback.

    Args:
        symbol: Crypto symbol (e.g., BTC, ETH)
        provider: Data provider - 'auto' (with fallback), 'tiingo' or 'binance'
        lookback_days: Number of days to look back (default: 365)
        quote_currency: Quote currency (default: usd)
    """
    if provider is None:
        provider = "auto"  # Fallback automatique par défaut

    request_id = await log_request_start(
        ctx, "analyze_crypto", {
            "symbol": symbol, "provider": provider,
            "lookback_days": lookback_days, "quote_currency": quote_currency,
        }
    )

    try:
        ctx.report_progress(0.2, f"Fetching {symbol} data with smart fallback...")
        
        # Utiliser le fallback automatique
        if provider == "auto":
            df = await market_data.get_crypto_data_with_fallback(
                symbol=symbol,
                lookback_days=lookback_days,
                quote_currency=quote_currency
            )
            df = fix_volume_types(df)
            actual_provider = df.get('data_source', ['unknown'])[0] if 'data_source' in df.columns else 'unknown'
        else:
            # Provider spécifique demandé
            df = await market_data.get_crypto_historical_data(
                symbol=symbol, lookback_days=lookback_days,
                provider=provider, quote_currency=quote_currency,
            )
            df = fix_volume_types(df)
            actual_provider = provider

        await ctx.report_progress(0.5, "Calculating technical indicators...")
        
        # Add indicators
        df = tech_analysis.add_core_indicators(df)

        ctx.report_progress(0.8, "Analyzing trend status...")
        
        # Get trend status
        trend = tech_analysis.check_trend_status(df)
        signals = tech_analysis.get_trading_signals(df)
        
        ctx.report_progress(1.0, "Analysis complete!")

        latest = df.iloc[-1]
        data_source_emoji = "🟢" if actual_provider == "tiingo" else "🟡" if actual_provider == "binance" else "⚪"
        
        analysis = f"""
🚀 **Crypto Analysis: {symbol}** {data_source_emoji}

{signals['signal_emoji']} **Overall Signal: {signals['overall_signal']}**
📊 **Signal Strength: {signals['signal_strength']}/100**
📡 **Data Source: {actual_provider.title()}** {data_source_emoji}

📈 **Trend Analysis:**
- Above 20 SMA: {"✅" if trend["above_20sma"] else "❌"}
- Above 50 SMA: {"✅" if trend["above_50sma"] else "❌"}
- Above 200 SMA: {"✅" if trend["above_200sma"] else "❌"}
- 20/50 SMA Cross: {"✅ Bullish" if trend["20_50_bullish"] else "❌ Bearish"}
- 50/200 SMA Cross: {"✅ Bullish" if trend["50_200_bullish"] else "❌ Bearish"}

⚡ **Momentum:**
- RSI (14): {trend["rsi"]:.2f}
- MACD: {"✅ Bullish" if trend["macd_bullish"] else "❌ Bearish"}
"""

        # Add enhanced indicators if available
        if "bb_position" in trend:
            analysis += f"""
🎯 **Bollinger Bands:**
- Position: {trend["bb_position"]:.2f} (0=Lower, 1=Upper)
- Squeeze: {"⚡ YES" if trend.get("bb_squeeze") else "❌ No"}
"""

        if "stoch_k" in trend:
            analysis += f"""
📊 **Stochastic:**
- %K: {trend["stoch_k"]:.2f}, %D: {trend["stoch_d"]:.2f}
- Status: {"⚠️ Overbought" if trend.get("stoch_overbought") else "🔄 Oversold" if trend.get("stoch_oversold") else "✅ Neutral"}
"""

        analysis += f"""
💹 **Key Metrics:**
- Latest Price: {df["close"].iloc[-1]:.6f}
- ATR (14): {df["atr"].iloc[-1]:.6f}
- Volume Avg (20D): {df["avg_20d_vol"].iloc[-1]:.2f}

🎯 **Trading Signals:**
{chr(10).join(signals['signal_details'])}

💡 **Data Reliability:** {data_source_emoji} 
   🟢 Tiingo (Primary) | 🟡 Binance (Fallback) | ⚪ Unknown
"""

        await log_request_end(ctx, request_id, "analyze_crypto", True, f"Analyzed {symbol} via {actual_provider}")
        return analysis

    except Exception as e:
        error_msg = await handle_error(ctx, request_id, "analyze_crypto", e)
        return f"❌ Error analyzing {symbol}: {error_msg}\n\n💡 Both Tiingo and Binance may be experiencing issues. Try again later."


#tool d'analyse de stock
@mcp.tool()
async def analyze_stock(ctx: Context, symbol: str) -> str:
    """
    Analyze a stock's technical setup with enhanced indicators.

    Args:
        symbol: Stock symbol (e.g., NVDA)
    """
    request_id = await log_request_start(ctx, "analyze_stock", {"symbol": symbol})

    try:
        ctx.report_progress(0.2, f"Fetching {symbol} stock data...")
        
        # Fetch data
        df = await market_data.get_historical_data(symbol)
        df = fix_volume_types(df)

        await ctx.report_progress(0.5, "Calculating technical indicators...")
        
        # Add indicators
        df = tech_analysis.add_core_indicators(df)

        ctx.report_progress(0.8, "Analyzing trend status...")
        
        # Get trend status
        trend = tech_analysis.check_trend_status(df)
        signals = tech_analysis.get_trading_signals(df)

        ctx.report_progress(1.0, "Analysis complete!")

        latest = df.iloc[-1]
        analysis = f"""
📈 **Stock Analysis: {symbol}**

{signals['signal_emoji']} **Overall Signal: {signals['overall_signal']}**
📊 **Signal Strength: {signals['signal_strength']}/100**

📈 **Trend Analysis:**
- Above 20 SMA: {"✅" if trend["above_20sma"] else "❌"}
- Above 50 SMA: {"✅" if trend["above_50sma"] else "❌"}
- Above 200 SMA: {"✅" if trend["above_200sma"] else "❌"}

⚡ **Momentum:**
- RSI (14): {trend["rsi"]:.2f}
- MACD: {"✅ Bullish" if trend["macd_bullish"] else "❌ Bearish"}

💹 **Key Metrics:**
- Latest Price: ${df["close"].iloc[-1]:.2f}
- ATR (14): ${df["atr"].iloc[-1]:.2f}
- Volume Avg (20D): {int(df["avg_20d_vol"].iloc[-1]):,}

🎯 **Trading Signals:**
{chr(10).join(signals['signal_details'])}
"""

        await log_request_end(ctx, request_id, "analyze_stock", True, f"Analyzed {symbol}")
        return analysis

    except Exception as e:
        error_msg = await handle_error(ctx, request_id, "analyze_stock", e)
        return f"❌ Error analyzing {symbol}: {error_msg}"
#tool de la mesure du performance d'actif
@mcp.tool()
async def relative_strength(ctx: Context, symbol: str, benchmark: str = "SPY") -> str:
    """Calculate relative strength vs benchmark."""
    request_id = await log_request_start(ctx, "relative_strength", {"symbol": symbol, "benchmark": benchmark})

    try:
        ctx.report_progress(0.3, f"Calculating relative strength...")
        
        rs_results = await rs_analysis.calculate_rs(market_data, symbol, benchmark)

        ctx.report_progress(1.0, "Analysis complete!")

        if not rs_results:
            return f"❌ Insufficient data for {symbol} vs {benchmark}"

        rs_text = f"""
📊 **Relative Strength: {symbol} vs {benchmark}**

"""
        for period, score in rs_results.items():
            if period.startswith("RS_"):
                days = period.split("_")[1]
                rs_text += f"📈 **{days} RS Score: {score}**"

                if score >= 80:
                    rs_text += " 🚀 Strong Outperformance\n"
                elif score >= 65:
                    rs_text += " ⭐ Moderate Outperformance\n"
                elif score >= 50:
                    rs_text += " ✅ Slight Outperformance\n"
                elif score >= 35:
                    rs_text += " ⚠️ Slight Underperformance\n"
                else:
                    rs_text += " 📉 Strong Underperformance\n"

        await log_request_end(ctx, request_id, "relative_strength", True)
        return rs_text

    except Exception as e:
        error_msg = await handle_error(ctx, request_id, "relative_strength", e)
        return f"❌ Error calculating RS: {error_msg}"
#tool pour montrer la repatition des volumes échangé à chaque niveau
@mcp.tool()
async def volume_profile(ctx: Context, symbol: str, lookback_days: int = 60) -> str:
    """Analyze volume distribution by price."""
    request_id = await log_request_start(ctx, "volume_profile", {"symbol": symbol, "lookback_days": lookback_days})

    try:
        ctx.report_progress(0.2, f"Fetching {lookback_days} days of data...")
        
        df = await market_data.get_historical_data(symbol, lookback_days + 10)
        df = fix_volume_types(df)

        ctx.report_progress(0.6, "Analyzing volume profile...")
        
        profile = volume_analysis.analyze_volume_profile(df.tail(lookback_days))

        ctx.report_progress(1.0, "Analysis complete!")

        profile_text = f"""
📊 **Volume Profile: {symbol} ({lookback_days} days)**

🎯 **Key Levels:**
- Point of Control: ${profile["point_of_control"]} (Highest volume)
- Value Area: ${profile["value_area_low"]} - ${profile["value_area_high"]} (70% volume)

📈 **Top Volume Levels:**
"""

        sorted_bins = sorted(profile["bins"], key=lambda x: x["volume"], reverse=True)
        for i, bin_data in enumerate(sorted_bins[:5]):
            profile_text += f"{i + 1}. ${bin_data['price_low']:.2f}-${bin_data['price_high']:.2f}: {bin_data['volume_percent']:.1f}%\n"

        await log_request_end(ctx, request_id, "volume_profile", True)
        return profile_text

    except Exception as e:
        error_msg = await handle_error(ctx, request_id, "volume_profile", e)
        return f"❌ Error analyzing volume profile: {error_msg}"
#tool d'identification  automatique des figures chartistes
@mcp.tool()
async def detect_patterns(ctx: Context, symbol: str) -> str:
    """Detect chart patterns in price data."""
    request_id = await log_request_start(ctx, "detect_patterns", {"symbol": symbol})

    try:
        ctx.report_progress(0.2, "Fetching data for pattern detection...")
        
        df = await market_data.get_historical_data(symbol, lookback_days=90)
        df = fix_volume_types(df)

        ctx.report_progress(0.6, "Scanning for patterns...")
        
        pattern_results = pattern_recognition.detect_patterns(df)

        ctx.report_progress(1.0, "Pattern detection complete!")

        if not pattern_results["patterns"]:
            return f"📊 **Pattern Analysis: {symbol}**\n\n❌ No significant patterns detected in recent data."

        pattern_text = f"📊 **Chart Patterns: {symbol}**\n\n"

        for pattern in pattern_results["patterns"]:
            pattern_text += f"🔍 **{pattern['type']}**"
            
            if "start_date" in pattern and "end_date" in pattern:
                pattern_text += f" ({pattern['start_date']} to {pattern['end_date']})"
            
            pattern_text += f"\n- Price Level: ${pattern['price_level']}"
            
            if "confidence" in pattern:
                pattern_text += f"\n- Confidence: {pattern['confidence']}"
            
            pattern_text += "\n\n"

        pattern_text += "⚠️ **Note:** Pattern recognition should be confirmed with other analysis."

        await log_request_end(ctx, request_id, "detect_patterns", True, f"Found {len(pattern_results['patterns'])} patterns")
        return pattern_text

    except Exception as e:
        error_msg = await handle_error(ctx, request_id, "detect_patterns", e)
        return f"❌ Error detecting patterns: {error_msg}"

@mcp.tool()
async def position_size(
    ctx: Context,
    symbol: str,
    stop_price: float,
    risk_amount: float,
    account_size: float,
    price: float = 0,
) -> str:
    """Calculate optimal position size."""
    request_id = await log_request_start(ctx, "position_size", {
        "symbol": symbol, "stop_price": stop_price,
        "risk_amount": risk_amount, "account_size": account_size, "price": price
    })

    try:
        if price == 0:
            ctx.report_progress(0.2, "Fetching current price...")
            df = await market_data.get_historical_data(symbol, lookback_days=5)
            df = fix_volume_types(df)
            price = df["close"].iloc[-1]

        ctx.report_progress(0.6, "Calculating position size...")

        position_results = risk_analysis.calculate_position_size(
            price=price, stop_price=stop_price,
            risk_amount=risk_amount, account_size=account_size,
        )

        ctx.report_progress(1.0, "Position sizing complete!")

        position_text = f"""
📊 **Position Sizing: {symbol}**

💰 **Trade Setup:**
- Entry Price: ${price:.2f}
- Stop Loss: ${stop_price:.2f}
- Risk per Share: ${position_results["risk_per_share"]:.2f}

📈 **Recommended Position:**
- Shares: {position_results["recommended_shares"]}
- Position Cost: ${position_results["position_cost"]:,.2f}
- Total Risk: ${position_results["dollar_risk"]:.2f} ({position_results["account_percent_risked"]:.2f}% of account)

🎯 **Profit Targets (R-Multiples):**
- R1 (1:1): ${position_results["r_multiples"]["r1"]:.2f}
- R2 (2:1): ${position_results["r_multiples"]["r2"]:.2f}
- R3 (3:1): ${position_results["r_multiples"]["r3"]:.2f}
"""

        await log_request_end(ctx, request_id, "position_size", True)
        return position_text

    except Exception as e:
        error_msg = await handle_error(ctx, request_id, "position_size", e)
        return f"❌ Error calculating position size: {error_msg}"

@mcp.tool()
async def suggest_stops(ctx: Context, symbol: str) -> str:
    """Suggest stop loss levels."""
    request_id = await log_request_start(ctx, "suggest_stops", {"symbol": symbol})

    try:
        ctx.report_progress(0.2, "Fetching data for stop analysis...")
        
        df = await market_data.get_historical_data(symbol, lookback_days=60)
        df = fix_volume_types(df)

        ctx.report_progress(0.4, "Calculating indicators...")
        
        df = tech_analysis.add_core_indicators(df)

        ctx.report_progress(0.7, "Analyzing stop levels...")
        
        stops = risk_analysis.suggest_stop_levels(df)
        latest_close = df["close"].iloc[-1]

        ctx.report_progress(1.0, "Stop analysis complete!")

        stops_text = f"""
🛑 **Stop Loss Suggestions: {symbol}** (Current: ${latest_close:.2f})

⚡ **ATR-Based Stops:**
- Conservative (1x): ${stops["atr_1x"]:.2f} ({((latest_close - stops["atr_1x"]) / latest_close * 100):.1f}% risk)
- Moderate (2x): ${stops["atr_2x"]:.2f} ({((latest_close - stops["atr_2x"]) / latest_close * 100):.1f}% risk)
- Aggressive (3x): ${stops["atr_3x"]:.2f} ({((latest_close - stops["atr_3x"]) / latest_close * 100):.1f}% risk)

📊 **Percentage-Based:**
- Tight (2%): ${stops["percent_2"]:.2f}
- Medium (5%): ${stops["percent_5"]:.2f}
- Wide (8%): ${stops["percent_8"]:.2f}

📈 **Technical Levels:**
"""

        if "sma_20" in stops:
            stops_text += f"- 20-day SMA: ${stops['sma_20']:.2f}\n"
        if "sma_50" in stops:
            stops_text += f"- 50-day SMA: ${stops['sma_50']:.2f}\n"
        if "recent_swing" in stops:
            stops_text += f"- Recent Swing Low: ${stops['recent_swing']:.2f}\n"

        await log_request_end(ctx, request_id, "suggest_stops", True)
        return stops_text

    except Exception as e:
        error_msg = await handle_error(ctx, request_id, "suggest_stops", e)
        return f"❌ Error suggesting stops: {error_msg}"

# Sentiment analysis tools 
if SENTIMENT_AVAILABLE and news_analyzer:
    @mcp.tool()
    async def analyze_sentiment(
        ctx: Context,
        symbol: str,
        days_back: int = 7,
        max_articles: int = 50
    ) -> str:
        """Analyze news sentiment for a stock or crypto."""
        request_id = await log_request_start(ctx, "analyze_sentiment", {
            "symbol": symbol, "days_back": days_back, "max_articles": max_articles
        })

        try:
            ctx.report_progress(0.2, f"Fetching news for {symbol}...")
            
            sentiment_analysis = await news_analyzer.analyze_symbol_sentiment(
                symbol, days_back, max_articles
            )
            
            ctx.report_progress(1.0, "Sentiment analysis complete!")

            sentiment_emoji = {
                "very_positive": "🚀", "positive": "📈", "neutral": "⚖️",
                "negative": "📉", "very_negative": "🔻"
            }

            sentiment_text = f"""
📰 **News Sentiment: {symbol}**

{sentiment_emoji.get(sentiment_analysis.sentiment_label.value, "⚖️")} **Overall: {sentiment_analysis.overall_sentiment:+.3f}**
📊 **Classification: {sentiment_analysis.sentiment_label.value.replace('_', ' ').title()}**
🎯 **Confidence: {sentiment_analysis.confidence:.1%}**

📈 **Article Breakdown:**
- Total: {sentiment_analysis.total_articles}
- Positive: {sentiment_analysis.positive_count}
- Neutral: {sentiment_analysis.neutral_count}
- Negative: {sentiment_analysis.negative_count}

📊 **Trend: {sentiment_analysis.sentiment_trend.title()}** (Strength: {sentiment_analysis.trend_strength:.1%})

🔑 **Key Themes:**
{chr(10).join(f"• {theme}" for theme in sentiment_analysis.key_themes) if sentiment_analysis.key_themes else "• No specific themes"}
"""

            await log_request_end(ctx, request_id, "analyze_sentiment", True)
            return sentiment_text

        except Exception as e:
            error_msg = await handle_error(ctx, request_id, "analyze_sentiment", e)
            return f"❌ Error analyzing sentiment: {error_msg}"
        
#system de diagnostic
@mcp.tool()
async def system_diagnostic(ctx: Context) -> str:
    """Diagnostic complet du système et des APIs."""
    request_id = await log_request_start(ctx, "system_diagnostic", {})
    
    try:
        ctx.report_progress(0.1, "Starting comprehensive system diagnostic...")
        
        diagnostic_results = {
            "timestamp": datetime.now().isoformat(),
            "server_status": "operational",
            "api_tests": {},
            "feature_status": {}
        }
        
        # Test 1: Tiingo API avec protection complète
        try:
            ctx.report_progress(0.2, "Testing Tiingo API...")
            df = await asyncio.wait_for(
                market_data.get_historical_data("AAPL", lookback_days=5), 
                timeout=15
            )
            
            if df is not None and len(df) > 0:
                latest_price = float(df['close'].iloc[-1])
                diagnostic_results["api_tests"]["tiingo"] = {
                    "status": "✅ Working",
                    "data_points": len(df),
                    "latest_price": f"${latest_price:.2f}",
                    "columns": list(df.columns)
                }
            else:
                diagnostic_results["api_tests"]["tiingo"] = {
                    "status": "⚠️ No Data",
                    "data_points": 0
                }
                
        except asyncio.TimeoutError:
            diagnostic_results["api_tests"]["tiingo"] = {
                "status": "⏰ Timeout",
                "error": "Request timeout after 15 seconds"
            }
        except Exception as e:
            error_str = str(e)[:100]  
            if any(term in error_str.lower() for term in ["429", "rate", "limit", "quota"]):
                diagnostic_results["api_tests"]["tiingo"] = {
                    "status": "🔴 Rate Limited",
                    "error": "API quota exhausted - wait 10+ minutes"
                }
            else:
                diagnostic_results["api_tests"]["tiingo"] = {
                    "status": "❌ Error",
                    "error": error_str
                }
        
        # Test 2: Alpha Vantage API 
        ctx.report_progress(0.4, "Testing Alpha Vantage API...")
        if fundamental_analyzer:
            try:
                async with alpha_vantage_client:
                    overview = await asyncio.wait_for(
                        alpha_vantage_client.get_company_overview("AAPL"),
                        timeout=20
                    )
                    diagnostic_results["api_tests"]["alpha_vantage"] = {
                        "status": "✅ Working",
                        "company_name": overview.name[:30] if overview.name else "N/A",
                        "pe_ratio": overview.pe_ratio if overview.pe_ratio else "N/A"
                    }
            except asyncio.TimeoutError:
                diagnostic_results["api_tests"]["alpha_vantage"] = {
                    "status": "⏰ Timeout",
                    "error": "Request timeout after 20 seconds"
                }
            except Exception as e:
                error_str = str(e)[:100]
                diagnostic_results["api_tests"]["alpha_vantage"] = {
                    "status": "❌ Error",
                    "error": error_str
                }
        else:
            diagnostic_results["api_tests"]["alpha_vantage"] = {
                "status": "⚪ Not Configured",
                "error": "ALPHA_VANTAGE_API_KEY not provided"
            }
        
        # Test 3: News API 
        ctx.report_progress(0.6, "Testing News API...")
        if news_analyzer:
            try:
                sentiment = await asyncio.wait_for(
                    news_analyzer.analyze_symbol_sentiment("AAPL", days_back=3, max_articles=5),
                    timeout=15
                )
                diagnostic_results["api_tests"]["news_api"] = {
                    "status": "✅ Working" if sentiment.total_articles > 0 else "⚠️ No articles",
                    "articles_found": sentiment.total_articles,
                    "sentiment_score": round(sentiment.overall_sentiment, 3)
                }
            except asyncio.TimeoutError:
                diagnostic_results["api_tests"]["news_api"] = {
                    "status": "⏰ Timeout",
                    "error": "Request timeout after 15 seconds"
                }
            except Exception as e:
                diagnostic_results["api_tests"]["news_api"] = {
                    "status": "❌ Error",
                    "error": str(e)[:100]
                }
        else:
            diagnostic_results["api_tests"]["news_api"] = {
                "status": "⚪ Not Configured",
                "error": "NEWS_API_KEY not provided"
            }
        
        # Test 4: Fonctionnalités Core - VERSION ULTRA-PROTÉGÉE
        ctx.report_progress(0.8, "Testing core features...")
        
        # Test analyse technique avec protection maximale
        try:
            # Récupération des données de test
            test_df = await asyncio.wait_for(
                market_data.get_historical_data("AAPL", lookback_days=30),
                timeout=10
            )
            
            if test_df is not None and len(test_df) >= 20:
                # Test calcul des indicateurs avec gestion d'erreur totale
                try:
                    test_df_with_indicators = tech_analysis.add_core_indicators(test_df)
                    
                    # Compter les indicateurs ajoutés
                    base_columns = {"open", "high", "low", "close", "volume", "symbol", "date"}
                    all_columns = set(test_df_with_indicators.columns)
                    indicator_columns = all_columns - base_columns
                    
                    # Test analyse de tendance
                    try:
                        trend_result = tech_analysis.check_trend_status(test_df_with_indicators)
                        current_rsi = trend_result.get('rsi', 'N/A')
                        if current_rsi != 'N/A':
                            current_rsi = f"{current_rsi:.1f}"
                            
                        diagnostic_results["feature_status"]["technical_analysis"] = {
                            "status": "✅ Working",
                            "indicators_calculated": len(indicator_columns),
                            "current_rsi": current_rsi,
                            "sample_indicators": sorted(list(indicator_columns))[:5],
                            "trend_analysis": "functional"
                        }
                    except Exception as trend_error:
                        diagnostic_results["feature_status"]["technical_analysis"] = {
                            "status": "⚠️ Partial",
                            "indicators_calculated": len(indicator_columns),
                            "current_rsi": "Error",
                            "error": f"Trend analysis failed: {str(trend_error)[:50]}"
                        }
                        
                except Exception as indicator_error:
                    diagnostic_results["feature_status"]["technical_analysis"] = {
                        "status": "❌ Error",
                        "indicators_calculated": 0,
                        "error": f"Indicator calculation failed: {str(indicator_error)[:50]}"
                    }
            else:
                diagnostic_results["feature_status"]["technical_analysis"] = {
                    "status": "⚠️ Insufficient Data",
                    "indicators_calculated": 0,
                    "data_points": len(test_df) if test_df is not None else 0
                }
                
        except Exception as data_error:
            diagnostic_results["feature_status"]["technical_analysis"] = {
                "status": "❌ Data Error",
                "error": f"Could not fetch test data: {str(data_error)[:50]}"
            }
        
        # Test gestion des risques avec protection
        try:
            if 'test_df_with_indicators' in locals() and len(test_df_with_indicators) >= 20:
                test_stops = risk_analysis.suggest_stop_levels(test_df_with_indicators)
                diagnostic_results["feature_status"]["risk_management"] = {
                    "status": "✅ Working",
                    "stop_levels_generated": len(test_stops),
                    "sample_stops": list(test_stops.keys())[:3]
                }
            else:
                diagnostic_results["feature_status"]["risk_management"] = {
                    "status": "⚠️ Limited",
                    "error": "Insufficient data for risk analysis"
                }
        except Exception as risk_error:
            diagnostic_results["feature_status"]["risk_management"] = {
                "status": "❌ Error",
                "error": f"Risk analysis failed: {str(risk_error)[:50]}"
            }
        
        ctx.report_progress(1.0, "Diagnostic complete!")
        
        # Construire le rapport final avec protection
        try:
            result_text = f"""
🔧 **System Diagnostic Report**

⏰ **Timestamp:** {diagnostic_results['timestamp'][:19].replace('T', ' ')}
🟢 **Server Status:** {diagnostic_results['server_status']}

📡 **API Status Tests:**
"""
            
            # APIs avec gestion sécurisée
            for api_name, api_result in diagnostic_results["api_tests"].items():
                status = api_result.get('status', 'Unknown')
                result_text += f"**{api_name.upper().replace('_', ' ')}:** {status}\n"
                
                # Détails selon le statut
                if status.startswith('✅'):
                    if 'data_points' in api_result:
                        result_text += f"  • Data points: {api_result['data_points']}\n"
                    if 'latest_price' in api_result:
                        result_text += f"  • Latest price: {api_result['latest_price']}\n"
                    if 'articles_found' in api_result:
                        result_text += f"  • Articles: {api_result['articles_found']}\n"
                elif 'error' in api_result:
                    result_text += f"  • Issue: {api_result['error']}\n"
                result_text += "\n"
            
            result_text += "🛠️ **Feature Status Tests:**\n"
            
            # Features avec gestion sécurisée
            for feature_name, feature_result in diagnostic_results["feature_status"].items():
                clean_name = feature_name.upper().replace('_', ' ')
                status = feature_result.get('status', 'Unknown')
                result_text += f"**{clean_name}:** {status}\n"
                
                if status.startswith('✅'):
                    if 'indicators_calculated' in feature_result:
                        result_text += f"  • Indicators: {feature_result['indicators_calculated']}\n"
                    if 'current_rsi' in feature_result and feature_result['current_rsi'] != 'N/A':
                        result_text += f"  • Current RSI: {feature_result['current_rsi']}\n"
                    if 'stop_levels_generated' in feature_result:
                        result_text += f"  • Stop levels: {feature_result['stop_levels_generated']}\n"
                elif 'error' in feature_result:
                    result_text += f"  • Issue: {feature_result['error']}\n"
                result_text += "\n"
            
            # Recommandations intelligentes
            result_text += "💡 **System Recommendations:**\n"
            
            # Compter les APIs fonctionnelles
            working_apis = sum(1 for api in diagnostic_results["api_tests"].values() 
                              if api.get("status", "").startswith("✅"))
            total_apis = len(diagnostic_results["api_tests"])
            
            if working_apis >= 2:
                result_text += "• ✅ System operational with multiple data sources\n"
            elif working_apis == 1:
                result_text += "• ⚠️ System functional but limited - some APIs unavailable\n"
            else:
                result_text += "• ❌ Multiple API issues - check connectivity and API keys\n"
            
            # Compter les features fonctionnelles
            working_features = sum(1 for feature in diagnostic_results["feature_status"].values()
                                 if feature.get("status", "").startswith("✅"))
            
            if working_features >= 2:
                result_text += "• ✅ Core analysis features operational\n"
            else:
                result_text += "• ⚠️ Limited analysis capabilities - check data sources\n"
            
            result_text += f"\n📊 **Overall Health:** {working_apis}/{total_apis} APIs + {working_features}/{len(diagnostic_results['feature_status'])} Features operational"
            result_text += "\n✅ **Diagnostic Status:** Completed successfully without critical errors"
            
            await log_request_end(ctx, request_id, "system_diagnostic", True)
            return result_text
            
        except Exception as format_error:
            return f"""
🔧 **System Diagnostic - Basic Report**

⏰ **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
⚠️ **Status:** Diagnostic completed with formatting limitations

**Raw Results Available:**
• API Tests: {len(diagnostic_results.get('api_tests', {}))} completed
• Feature Tests: {len(diagnostic_results.get('feature_status', {}))} completed

**Formatting Error:** {str(format_error)[:100]}

✅ **Note:** Core diagnostic logic executed successfully
"""
        
    except Exception as e:
        error_msg = await handle_error(ctx, request_id, "system_diagnostic", e)
        return f"""❌ **System Diagnostic Failed**

**Critical Error:** {error_msg}

🔧 **Emergency Status Check:**
• **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
• **Issue:** Diagnostic system encountered critical error
• **Impact:** Unable to complete full system analysis

💡 **Alternatives:**
• Try: `api_status_check()` for simpler API testing
• Try: `analyze_stock("AAPL")` to test core functionality
• Check logs for detailed error information

⚠️ **Note:** This error does not necessarily indicate system failure
"""
#screening tool
@mcp.tool()
async def screen_momentum_stocks(
    ctx: Context,
    symbols: str = "AAPL,MSFT,GOOGL,AMZN,TSLA",
    min_price_change: float = 0.2,
    min_volume_ratio: float = 1.0,
    limit: int = 10
) -> str:
    """
    Screen stocks for momentum setups - VERSION CORRIGÉE.
    
    Args:
        symbols: Comma-separated list of symbols to screen (max 5 recommended)
        min_price_change: Minimum price change % today (default: 0.2%)
        min_volume_ratio: Minimum volume vs 20-day average (default: 1.0x)
        limit: Maximum number of results
    """
    request_id = await log_request_start(ctx, "screen_momentum_stocks", {
        "symbols": symbols, "min_price_change": min_price_change,
        "min_volume_ratio": min_volume_ratio, "limit": limit
    })
    
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        
        # Limiter à 3 symboles pour éviter les timeouts
        if len(symbol_list) > 3:
            symbol_list = symbol_list[:3]
            ctx.report_progress(0.1, f"Limited to 3 symbols to avoid timeouts...")
        
        ctx.report_progress(0.1, f"Screening {len(symbol_list)} symbols...")
        
        results = []
        processed_count = 0
        error_count = 0
        
        for i, symbol in enumerate(symbol_list):
            try:
                ctx.report_progress(0.1 + (0.7 * i / len(symbol_list)), f"Analyzing {symbol}... ({i+1}/{len(symbol_list)})")
                
                # Délai entre requêtes
                if i > 0:
                    await asyncio.sleep(2)
                
                # Timeout plus court avec retry
                df = None
                for attempt in range(2):
                    try:
                        # Utiliser l'enhanced market data avec fallback
                        if hasattr(market_data, 'get_stock_data_with_retry'):
                            df = await asyncio.wait_for(
                                market_data.get_stock_data_with_retry(symbol, lookback_days=30),
                                timeout=15
                            )
                        else:
                            df = await asyncio.wait_for(
                                market_data.get_historical_data(symbol, lookback_days=30),
                                timeout=15
                            )
                        
                        if df is not None and len(df) >= 20:
                            break
                            
                    except asyncio.TimeoutError:
                        if attempt == 0:
                            await asyncio.sleep(3)
                        continue
                    except Exception as e:
                        if attempt == 0:
                            await asyncio.sleep(3)
                        continue
                
                if df is None or len(df) < 20:
                    error_count += 1
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                # Calculs avec protection d'erreur maximale
                try:
                    # Ajouter indicateurs avec gestion d'erreur
                    df_with_indicators = tech_analysis.add_core_indicators(df.copy())
                    
                    # Vérification que les indicateurs sont présents
                    if 'avg_20d_vol' not in df_with_indicators.columns:
                        # Calcul manuel si l'indicateur n'existe pas
                        df_with_indicators['avg_20d_vol'] = df_with_indicators['volume'].rolling(20).mean()
                    
                    latest = df_with_indicators.iloc[-1]
                    prev = df_with_indicators.iloc[-2] if len(df_with_indicators) > 1 else latest
                    
                    # Métriques avec validation
                    current_price = float(latest["close"])
                    prev_price = float(prev["close"])
                    
                    # Protection division par zéro
                    if prev_price > 0:
                        price_change = ((current_price - prev_price) / prev_price) * 100
                    else:
                        price_change = 0.0
                    
                    # Volume ratio avec protection
                    avg_volume = float(latest.get("avg_20d_vol", 1))
                    current_volume = float(latest["volume"])
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                    
                    # Obtenir les signaux avec gestion d'erreur
                    try:
                        trend = tech_analysis.check_trend_status(df_with_indicators)
                        signals = tech_analysis.get_trading_signals(df_with_indicators)
                        rsi = float(trend.get("rsi", 50) or 50)
                        signal_strength = float(signals.get("signal_strength", 50) or 50)
                    except Exception as signal_error:
                        logger.warning(f"Signal calculation error for {symbol}: {signal_error}")
                        rsi = 50.0
                        signal_strength = 50.0
                    
                    # Validation des valeurs
                    if not all([
                        current_price > 0,
                        not pd.isna(current_price),
                        not pd.isna(price_change),
                        not pd.isna(volume_ratio)
                    ]):
                        error_count += 1
                        continue
                    
                    # Critères de momentum
                    meets_price_criteria = abs(price_change) >= min_price_change
                    meets_volume_criteria = volume_ratio >= min_volume_ratio
                    meets_rsi_criteria = 20 <= rsi <= 90
                    
                    # Score de momentum
                    score = 50  # Base
                    score += min(abs(price_change) * 3, 20)  # Prix
                    score += min((volume_ratio - 1) * 8, 15)  # Volume
                    score += (signal_strength - 50) * 0.3  # Signal technique
                    
                    # Normaliser le score
                    score = max(0, min(100, score))
                    
                    # Déterminer la catégorie
                    if score >= 75:
                        category = "🚀 STRONG MOMENTUM"
                    elif score >= 60:
                        category = "📈 GOOD MOMENTUM"
                    elif score >= 45:
                        category = "⚖️ MODERATE"
                    else:
                        category = "🔄 WEAK"
                    
                    # Critères globaux
                    meets_criteria = meets_price_criteria and meets_volume_criteria and meets_rsi_criteria
                    
                    result = {
                        'symbol': symbol,
                        'score': round(score, 1),
                        'current_price': round(current_price, 2),
                        'price_change_percent': round(price_change, 2),
                        'volume_ratio': round(volume_ratio, 2),
                        'rsi': round(rsi, 1),
                        'signal_strength': round(signal_strength, 0),
                        'meets_criteria': meets_criteria,
                        'category': category,
                        'price_criteria': meets_price_criteria,
                        'volume_criteria': meets_volume_criteria,
                        'rsi_criteria': meets_rsi_criteria
                    }
                    
                    results.append(result)
                    processed_count += 1
                    
                    logger.info(f"✅ {symbol}: Score {score:.1f}, Price {price_change:+.1f}%, Vol {volume_ratio:.1f}x")
                    
                except Exception as calc_error:
                    logger.error(f"Calculation error for {symbol}: {calc_error}")
                    error_count += 1
                    continue
                    
            except Exception as symbol_error:
                logger.error(f"Complete error for {symbol}: {symbol_error}")
                error_count += 1
                continue
        
        ctx.report_progress(0.8, "Compiling screening results...")
        
        # Gestion des résultats
        if not results:
            return f"""📊 **Momentum Screening - No Results**

**Symbols Attempted:** {', '.join(symbol_list)}
**Processing Issues:** {error_count} symbols failed to process

❌ **Possible Causes:**
• Network connectivity issues
• Data provider temporary issues  
• Invalid symbols or insufficient historical data

🔧 **Solutions:**
1. **Retry with different symbols:** Try `screen_momentum_stocks("AAPL,MSFT")`
2. **Check individual analysis:** Use `analyze_stock("AAPL")` 
3. **Verify system status:** Use `api_status_check()`

⏰ **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        
        # Trier les résultats par score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Identifier les candidats qui répondent aux critères
        qualifying_candidates = [r for r in results if r['meets_criteria']]
        
        ctx.report_progress(1.0, "Screening complete!")
        
        # Générer le rapport
        screen_text = f"""📊 **Momentum Screening Results**

🔢 **Analysis Summary:**
• **Symbols Processed:** {processed_count}/{len(symbol_list)}
• **Errors:** {error_count}
• **Meeting All Criteria:** {len(qualifying_candidates)}
• **Total Analyzed:** {len(results)}

**🎯 Screening Criteria:**
• **Price Movement:** ≥{min_price_change}% (absolute change)
• **Volume Surge:** ≥{min_volume_ratio}x average
• **RSI Range:** 20-90 (avoid extreme levels)

**🏆 MOMENTUM RANKINGS:**
"""
        
        for i, result in enumerate(results[:limit], 1):
            # Emojis selon la performance
            momentum_emoji = "🚀" if "STRONG" in result['category'] else "📈" if "GOOD" in result['category'] else "⚖️" if "MODERATE" in result['category'] else "🔄"
            criteria_emoji = "✅" if result['meets_criteria'] else "⚠️"
            
            screen_text += f"""
**{i}. {result['symbol']}** {momentum_emoji} {criteria_emoji}
• **Score:** {result['score']}/100 | **Category:** {result['category']}
• **Price:** ${result['current_price']:.2f} ({result['price_change_percent']:+.1f}%)
• **Volume:** {result['volume_ratio']:.1f}x avg | **RSI:** {result['rsi']}
• **Criteria:** Price {"✅" if result['price_criteria'] else "❌"} | Volume {"✅" if result['volume_criteria'] else "❌"} | RSI {"✅" if result['rsi_criteria'] else "❌"}
"""
        
        # Évaluation globale
        if len(qualifying_candidates) >= 2:
            screen_text += f"\n✅ **Strong Momentum Environment:** {len(qualifying_candidates)} stocks meet all criteria"
        elif len(qualifying_candidates) >= 1:
            screen_text += f"\n⚖️ **Moderate Momentum:** {len(qualifying_candidates)} stock meets criteria"
        else:
            screen_text += f"\n📉 **Limited Momentum:** No stocks meet all criteria - consider relaxing parameters"
        
        screen_text += f"""

📊 **Market Assessment:** {'Active momentum detected' if len(qualifying_candidates) >= 2 else 'Selective momentum' if len(qualifying_candidates) >= 1 else 'Low momentum environment'}
📅 **Analysis Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
✅ **Success Rate:** {processed_count}/{len(symbol_list)} symbols analyzed

💡 **Next Steps:** Use `analyze_stock("SYMBOL")` for detailed analysis of top candidates
"""
        
        await log_request_end(ctx, request_id, "screen_momentum_stocks", True, 
                            f"Processed {processed_count}/{len(symbol_list)} symbols")
        return screen_text
        
    except Exception as e:
        error_msg = await handle_error(ctx, request_id, "screen_momentum_stocks", e)
        return f"""❌ **Momentum Screening - Critical Error**

**Error Details:** {error_msg}

🔧 **Recovery Options:**
1. **Reduce symbols:** Try 1-2 symbols max: `screen_momentum_stocks("AAPL")`
2. **Check system:** `system_diagnostic()`
3. **Individual analysis:** `analyze_stock("AAPL")` instead

📅 **Error Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
#outil d'analyse fondamrntale

if FUNDAMENTAL_AVAILABLE and fundamental_analyzer:
    @mcp.tool()
    async def analyze_fundamentals(
        ctx: Context,
        symbol: str
    ) -> str:
        """
        Analyze company fundamentals including financials, ratios, and valuation.
        
        Args:
            symbol: Stock symbol (e.g., AAPL)
        """
        request_id = await log_request_start(ctx, "analyze_fundamentals", {"symbol": symbol})
        
        try:
            ctx.report_progress(0.2, f"Fetching fundamental data for {symbol}...")
            
            # Get comprehensive fundamental analysis
            analysis = await fundamental_analyzer.get_comprehensive_analysis(symbol)
            
            if "error" in analysis:
                return f"❌ Error analyzing {symbol}: {analysis['error']}"
            
            ctx.report_progress(0.8, "Generating analysis report...")
            
            overview = analysis["company_overview"]
            valuation = analysis["valuation_assessment"]
            quality = analysis["quality_score"]
            
            # Safe formatting helper functions
            def format_billions(value):
                return f"${value/1e9:.1f}B" if value else "N/A"
            
            def format_ratio(value):
                return f"{value:.1f}" if value else "N/A"
            
            def format_percentage(value):
                return f"{value*100:.1f}%" if value else "N/A"
            
            def format_currency(value):
                return f"${value:.2f}" if value else "N/A"
            
            # Format the analysis
            fundamental_text = f"""
💼 **Fundamental Analysis: {symbol}**

🏢 **Company Overview:**
- **Name:** {overview.name}
- **Sector:** {overview.sector}
- **Industry:** {overview.industry}
- **Market Cap:** {format_billions(overview.market_cap)}

💰 **Valuation Metrics:**
- **P/E Ratio:** {format_ratio(overview.pe_ratio)}
- **P/B Ratio:** {format_ratio(overview.pb_ratio)}
- **EPS:** {format_currency(overview.eps)}
- **Dividend Yield:** {format_percentage(overview.dividend_yield)}

📊 **Profitability:**
- **Profit Margin:** {format_percentage(overview.profit_margin)}
- **Operating Margin:** {format_percentage(overview.operating_margin)}
- **ROE:** {format_percentage(overview.roe)}
- **ROA:** {format_percentage(overview.roa)}

🏦 **Financial Health:**
- **Current Ratio:** {format_ratio(overview.current_ratio)}
- **Debt-to-Equity:** {format_ratio(overview.debt_to_equity)}
- **Beta:** {format_ratio(overview.beta)}

📈 **Valuation Assessment:**
- **Score:** {valuation["valuation_score"]}/100
- **Classification:** {valuation["classification"]}
- **Recommendation:** {valuation["recommendation"]}

✨ **Quality Score:**
- **Grade:** {quality["quality_grade"]} ({quality["quality_score"]}/100)
- **Description:** {quality["description"]}

🎯 **Key Signals:**
{chr(10).join(f"• {signal}" for signal in valuation["signals"])}

📊 **Quality Factors:**
{chr(10).join(f"• {factor}" for factor in quality["factors"])}
"""

            ctx.report_progress(1.0, "Fundamental analysis complete!")
            await log_request_end(ctx, request_id, "analyze_fundamentals", True, f"Analyzed {symbol}")
            return fundamental_text
            
        except Exception as e:
            error_msg = await handle_error(ctx, request_id, "analyze_fundamentals", e)
            return f"❌ Error in fundamental analysis: {error_msg}"
    @mcp.tool()
    async def investment_thesis(
        ctx: Context,
        symbol: str
    ) -> str:
        """
        Generate an investment thesis based on fundamental analysis.
        
        Args:
            symbol: Stock symbol (e.g., AAPL)
        """
        request_id = await log_request_start(ctx, "investment_thesis", {"symbol": symbol})
        
        try:
            ctx.report_progress(0.2, f"Generating investment thesis for {symbol}...")
            
            # Get comprehensive analysis
            analysis = await fundamental_analyzer.get_comprehensive_analysis(symbol)
            
            if "error" in analysis:
                return f"❌ Error analyzing {symbol}: {analysis['error']}"
            
            ctx.report_progress(0.6, "Creating investment thesis...")
            
            # Generate investment thesis
            thesis = fundamental_analyzer.create_investment_thesis(analysis)
            
            if "error" in thesis:
                return f"❌ Error creating thesis: {thesis['error']}"
            
            ctx.report_progress(0.9, "Formatting thesis...")
            
            # Format the thesis
            confidence_emoji = {
                "Élevée": "🎯",
                "Moyenne-Haute": "📈", 
                "Moyenne": "⚖️",
                "Faible": "⚠️"
            }.get(thesis["confidence_level"], "⚖️")
            
            recommendation_emoji = {
                "ACHAT FORT": "🚀",
                "ACHAT": "📈",
                "CONSERVER": "⚖️", 
                "VENTE": "📉",
                "VENTE FORTE": "🔻"
            }.get(thesis["final_recommendation"], "⚖️")
            
            # Safe formatting for market cap
            market_cap_text = f"${thesis['market_cap_billions']:.1f}B" if thesis.get("market_cap_billions") else "N/A"
            
            # Safe formatting for metrics
            metrics = thesis["key_metrics_summary"]
            pe_ratio_text = f"{metrics['pe_ratio']:.1f}" if metrics.get("pe_ratio") else "N/A"
            roe_text = f"{metrics['roe_percent']:.1f}%" if metrics.get("roe_percent") else "N/A"
            margin_text = f"{metrics['profit_margin_percent']:.1f}%" if metrics.get("profit_margin_percent") else "N/A"
            debt_text = f"{metrics['debt_to_equity']:.2f}" if metrics.get("debt_to_equity") else "N/A"
            
            thesis_text = f"""
📋 **Investment Thesis: {symbol}**

🏢 **Company:** {thesis["company_name"]}
📊 **Sector:** {thesis["sector"]}
💰 **Market Cap:** {market_cap_text}

{recommendation_emoji} **Final Recommendation: {thesis["final_recommendation"]}**
{confidence_emoji} **Confidence Level: {thesis["confidence_level"]}**
🎯 **Combined Score: {thesis["combined_score"]}/100**

💪 **Investment Strengths:**
{chr(10).join(f"• {strength}" for strength in thesis["investment_strengths"]) if thesis["investment_strengths"] else "• None identified"}

⚠️ **Investment Concerns:**
{chr(10).join(f"• {concern}" for concern in thesis["investment_concerns"]) if thesis["investment_concerns"] else "• None identified"}

📊 **Key Metrics Summary:**
• **Valuation Score:** {metrics["valuation_score"]}/100
• **Quality Score:** {metrics["quality_score"]}/100
• **P/E Ratio:** {pe_ratio_text}
• **ROE:** {roe_text}
• **Profit Margin:** {margin_text}
• **Debt/Equity:** {debt_text}

⚠️ **Disclaimer:** This analysis is based on available financial data and should not be considered as financial advice. Always conduct your own research and consult with financial professionals before making investment decisions.
"""
            
            ctx.report_progress(1.0, "Investment thesis complete!")
            await log_request_end(ctx, request_id, "investment_thesis", True, f"Generated thesis for {symbol}")
            return thesis_text
            
        except Exception as e:
            error_msg = await handle_error(ctx, request_id, "investment_thesis", e)
            return f"❌ Error generating investment thesis: {error_msg}"

    @mcp.tool()
    async def earnings_analysis(
        ctx: Context,
        symbol: str
    ) -> str:
        """
        Analyze earnings data and surprises for a company.
        
        Args:
            symbol: Stock symbol (e.g., AAPL)
        """
        request_id = await log_request_start(ctx, "earnings_analysis", {"symbol": symbol})
        
        try:
            ctx.report_progress(0.2, f"Fetching earnings data for {symbol}...")
            
            async with alpha_vantage_client:
                earnings_data = await alpha_vantage_client.get_earnings_data(symbol)
            
            if not earnings_data:
                return f"❌ No earnings data found for {symbol}"
            
            ctx.report_progress(0.8, "Analyzing earnings trends...")
            
            earnings_text = f"""
📈 **Earnings Analysis: {symbol}**

📊 **Recent Quarterly Results:**
"""
            
            for i, earning in enumerate(earnings_data[:4]):  # Last 4 quarters
                quarter_emoji = "✅" if earning.surprise and earning.surprise > 0 else "❌" if earning.surprise and earning.surprise < 0 else "➖"
                
                # Safe formatting for earnings data
                reported_eps = f"${earning.reported_eps:.2f}" if earning.reported_eps else "N/A"
                estimated_eps = f"${earning.estimated_eps:.2f}" if earning.estimated_eps else "N/A"
                surprise_text = f"{earning.surprise:+.2f}" if earning.surprise else "N/A"
                surprise_pct = f"({earning.surprise_percentage:+.1f}%)" if earning.surprise_percentage else ""
                
                earnings_text += f"""
**Q{4-i} - {earning.fiscal_date_ending}**
• **Reported EPS:** {reported_eps}
• **Estimated EPS:** {estimated_eps}
• **Surprise:** {quarter_emoji} {surprise_text} {surprise_pct}
• **Report Date:** {earning.reported_date}

"""
            
            # Calculate trends
            recent_surprises = [e for e in earnings_data[:4] if e.surprise_percentage is not None]
            if recent_surprises:
                avg_surprise = sum(e.surprise_percentage for e in recent_surprises) / len(recent_surprises)
                beats = len([e for e in recent_surprises if e.surprise_percentage > 0])
                
                earnings_text += f"""
📊 **Earnings Trends:**
• **Average Surprise:** {avg_surprise:+.1f}%
• **Beats vs Misses:** {beats}/{len(recent_surprises)} quarters beat estimates
• **Consistency:** {"High" if beats >= 3 else "Moderate" if beats >= 2 else "Low"}
"""
                
                if avg_surprise > 5:
                    earnings_text += "• **Assessment:** 🚀 Consistently beating expectations\n"
                elif avg_surprise > 0:
                    earnings_text += "• **Assessment:** 📈 Generally meeting/beating expectations\n"
                else:
                    earnings_text += "• **Assessment:** ⚠️ Struggling to meet expectations\n"
            
            ctx.report_progress(1.0, "Earnings analysis complete!")
            await log_request_end(ctx, request_id, "earnings_analysis", True, f"Analyzed earnings for {symbol}")
            return earnings_text
            
        except Exception as e:
            error_msg = await handle_error(ctx, request_id, "earnings_analysis", e)
            return f"❌ Error in earnings analysis: {error_msg}"



@mcp.tool()
async def market_overview(
    ctx: Context,
    symbols: str = "AAPL,MSFT,GOOGL,AMZN,TSLA"  # Réduit à 5 
) -> str:
    """
    Get a quick market overview of major symbols.
    
    Args:
        symbols: Comma-separated list of symbols for overview (max 5 recommandé)
    """
    request_id = await log_request_start(ctx, "market_overview", {"symbols": symbols})
    
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        
        # Auto-limiter pour éviter les quotas API
        original_count = len(symbol_list)
        if len(symbol_list) > 5:
            symbol_list = symbol_list[:5]
            ctx.report_progress(0.1, f"Auto-limited to {len(symbol_list)}/5 symbols to avoid API limits...")
        else:
            ctx.report_progress(0.1, f"Fetching data for {len(symbol_list)} symbols...")
        
        overview_data = []
        crypto_symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'DOT', 'LINK', 'AVAX', 'SOL', 'MATIC']
        api_errors = []
        success_count = 0
        
        for i, symbol in enumerate(symbol_list):
            try:
                progress = 0.1 + (0.7 * i / len(symbol_list))
                ctx.report_progress(progress, f"Processing {symbol}... ({i+1}/{len(symbol_list)})")
                
                # Délai intelligent entre requêtes
                if i > 0:
                    delay = 1.5 + (i * 0.5)  # Délai progressif: 1.5s, 2s, 2.5s...
                    await asyncio.sleep(delay)
                
                # Timeout plus court pour éviter les blocages
                try:
                    if symbol in crypto_symbols:
                        df = await asyncio.wait_for(
                            market_data.get_crypto_historical_data(
                                symbol, lookback_days=5, provider="tiingo", quote_currency="usd"
                            ), timeout=12
                        )
                    else:
                        df = await asyncio.wait_for(
                            market_data.get_historical_data(symbol, lookback_days=5),
                            timeout=12
                        )
                except asyncio.TimeoutError:
                    api_errors.append(f"{symbol}: Timeout (>12s)")
                    continue
                
                # Validation des données
                if df is None or len(df) < 2:
                    api_errors.append(f"{symbol}: Données insuffisantes ({len(df) if df is not None else 0} points)")
                    continue
                
                # Calculs avec gestion d'erreur
                try:
                    current_price = float(df['close'].iloc[-1])
                    prev_price = float(df['close'].iloc[-2])
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                    
                    # Volume avec fallback
                    try:
                        avg_volume = float(df['volume'].mean())
                        current_volume = float(df['volume'].iloc[-1])
                        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                    except:
                        volume_ratio = 1.0
                    
                    overview_data.append({
                        'symbol': symbol,
                        'price': current_price,
                        'change_pct': change_pct,
                        'volume_ratio': volume_ratio,
                        'is_crypto': symbol in crypto_symbols
                    })
                    
                    success_count += 1
                    logger.info(f"✅ {symbol}: ${current_price:.2f} ({change_pct:+.1f}%)")
                    
                except Exception as calc_error:
                    api_errors.append(f"{symbol}: Calculation error")
                    continue
                    
            except Exception as e:
                error_str = str(e)
                if any(term in error_str.lower() for term in ["429", "too many", "rate", "limit"]):
                    api_errors.append(f"{symbol}: API limit reached")
                    # Arrêter si limite API atteinte
                    logger.warning(f"API limit hit at {symbol}, stopping further requests")
                    break
                else:
                    api_errors.append(f"{symbol}: {error_str[:30]}...")
                continue
        
        ctx.report_progress(0.8, "Compiling overview...")
        
        # Rapport intelligent selon les résultats
        if not overview_data:
            return f"""❌ **Market Overview - No Data Retrieved**

**Attempted:** {', '.join(symbol_list)}
**API Errors:** {len(api_errors)} symbols failed

**Error Details:**
{chr(10).join(f"• {error}" for error in api_errors[:5])}

🔧 **Smart Solutions:**
1. **Reduce symbols:** `market_overview("AAPL,MSFT")` (2-3 max)
2. **Wait 5 minutes** for API limit reset
3. **Try single analysis:** `analyze_stock("AAPL")`

💡 **Status Check:** `api_status_check()` for full diagnosis
"""
        
        # Trier par performance
        overview_data.sort(key=lambda x: x['change_pct'], reverse=True)
        
        # Séparer stocks et crypto
        stocks = [item for item in overview_data if not item['is_crypto']]
        cryptos = [item for item in overview_data if item['is_crypto']]
        
        ctx.report_progress(0.9, "Formatting results...")
        
        overview_text = f"""📊 **Market Overview** - {success_count}/{original_count} symbols

🔢 **Summary:**
• **Successfully analyzed:** {len(overview_data)} symbols
• **API errors:** {len(api_errors)} symbols
• **Success rate:** {success_count/original_count*100:.0f}%
• **Data source:** Tiingo API
"""
        
        if len(api_errors) > 0:
            overview_text += f"• **⚠️ API limitations detected:** {len(api_errors)} failures\n"
        
        overview_text += "\n"
        
        # Performance des stocks
        if stocks:
            overview_text += "📈 **Stock Performance:**\n"
            for item in stocks:
                emoji = "🚀" if item['change_pct'] > 3 else "📈" if item['change_pct'] > 0 else "📉"
                price_str = f"${item['price']:.2f}" if item['price'] < 1000 else f"${item['price']:,.0f}"
                overview_text += f"**{item['symbol']}** {emoji} {item['change_pct']:+.1f}% ({price_str})\n"
        
        # Performance crypto
        if cryptos:
            overview_text += "\n🪙 **Crypto Performance:**\n"
            for item in cryptos:
                emoji = "🚀" if item['change_pct'] > 5 else "📈" if item['change_pct'] > 0 else "📉"
                price_str = f"${item['price']:,.2f}" if item['price'] > 1 else f"${item['price']:.4f}"
                overview_text += f"**{item['symbol']}** {emoji} {item['change_pct']:+.1f}% ({price_str})\n"
        
        # Volume anormal
        high_volume = [item for item in overview_data if item['volume_ratio'] > 1.8]
        if high_volume:
            overview_text += f"\n🔊 **Unusual Volume:**\n"
            for item in high_volume[:2]:
                overview_text += f"**{item['symbol']}** {item['volume_ratio']:.1f}x normal volume\n"
        
        # Sentiment global
        if len(overview_data) >= 3:
            positive = len([item for item in overview_data if item['change_pct'] > 0])
            sentiment_pct = positive / len(overview_data) * 100
            
            overview_text += f"\n🎭 **Market Sentiment:**\n"
            if sentiment_pct >= 70:
                overview_text += f"• **🟢 Bullish** ({positive}/{len(overview_data)} positive)\n"
            elif sentiment_pct >= 30:
                overview_text += f"• **🟡 Mixed** ({positive}/{len(overview_data)} positive)\n"
            else:
                overview_text += f"• **🔴 Bearish** ({positive}/{len(overview_data)} positive)\n"
        
        
        if api_errors:
            overview_text += f"\n⚠️ **API Issues ({len(api_errors)}):**\n"
            if len(api_errors) <= 3:
                for error in api_errors:
                    overview_text += f"• {error}\n"
            else:
                overview_text += f"• Multiple API timeouts/limits detected\n"
                overview_text += f"• Try fewer symbols: 2-3 maximum recommended\n"
        
        # Conseils intelligents
        overview_text += f"\n💡 **Smart Tips:**\n"
        if len(api_errors) > 0:
            overview_text += "• Use fewer symbols to avoid API limits\n"
            overview_text += "• Space out requests by 5+ minutes\n"
        else:
            overview_text += "• Analysis successful - all systems operational\n"
        
        overview_text += f"\n📅 **Updated:** {datetime.now().strftime('%H:%M:%S')}"
        
        ctx.report_progress(1.0, "Overview complete!")
        await log_request_end(ctx, request_id, "market_overview", True, 
                            f"Analyzed {success_count}/{original_count} symbols")
        return overview_text
        
    except Exception as e:
        error_msg = await handle_error(ctx, request_id, "market_overview", e)
        return f"❌ Market overview error: {error_msg}\n\n💡 **Try:** Fewer symbols or wait for API reset"


   
@mcp.tool()
async def compare_companies(
    ctx: Context,
    symbols: str,
    metrics: str = "pe_ratio,pb_ratio,roe,profit_margin,debt_to_equity"
) -> str:
    """
    Compare fundamental metrics of multiple companies.
    
    Args:
        symbols: Comma-separated list of symbols (e.g., AAPL,MSFT,GOOGL)
        metrics: Comma-separated list of metrics to compare
    """
    request_id = await log_request_start(ctx, "compare_companies", {
        "symbols": symbols, "metrics": metrics
    })
    
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        metrics_list = [m.strip() for m in metrics.split(",")]
        
        ctx.report_progress(0.1, f"Comparing {len(symbol_list)} companies...")
        
        # Analyser chaque entreprise séquentiellement pour éviter les limits API
        companies_data = {}
        
        for i, symbol in enumerate(symbol_list):
            try:
                progress = 0.1 + (0.7 * i / len(symbol_list))
                ctx.report_progress(progress, f"Analyzing {symbol}...")
                
                # Attendre un peu entre les requêtes pour respecter les limites API
                if i > 0:
                    await asyncio.sleep(1)  # 1 seconde entre les requêtes
                
                # Récupérer l'analyse individuelle
                analysis = await fundamental_analyzer.get_comprehensive_analysis(symbol)
                
                if "error" not in analysis:
                    companies_data[symbol] = analysis
                    ctx.report_progress(progress + 0.05, f"✅ {symbol} analyzed")
                else:
                    ctx.report_progress(progress + 0.05, f"⚠️ {symbol} failed: {analysis['error']}")
                    logger.warning(f"Failed to analyze {symbol}: {analysis['error']}")
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                ctx.report_progress(progress + 0.05, f"❌ {symbol} error")
                continue
        
        ctx.report_progress(0.8, "Creating comparison report...")
        
        if not companies_data:
            return f"❌ Could not retrieve fundamental data for any of the requested companies: {', '.join(symbol_list)}"
        
        # Créer le tableau de comparaison manuellement
        comparison_table = []
        
        for symbol, data in companies_data.items():
            overview = data["company_overview"]
            valuation = data["valuation_assessment"]
            quality = data["quality_score"]
            
            company_row = {
                "symbol": symbol,
                "name": overview.name,
                "sector": overview.sector,
                "market_cap": overview.market_cap,
                "valuation_score": valuation["valuation_score"],
                "quality_score": quality["quality_score"],
                "recommendation": valuation["recommendation"],
                # Ajouter les métriques demandées
                "pe_ratio": overview.pe_ratio,
                "pb_ratio": overview.pb_ratio,
                "roe": overview.roe,
                "profit_margin": overview.profit_margin,
                "debt_to_equity": overview.debt_to_equity
            }
            
            comparison_table.append(company_row)
        
        # Trier par score de valorisation
        comparison_table.sort(key=lambda x: x["valuation_score"], reverse=True)
        
        # Ajouter les rangs
        for i, company in enumerate(comparison_table):
            company["rank"] = i + 1
        
        ctx.report_progress(0.9, "Formatting results...")
        
        comparison_text = f"""
📊 **Company Comparison** ({len(comparison_table)} companies successfully analyzed)

**📈 Ranking by Valuation Score:**
"""
        
        for company in comparison_table:
            recommendation_emoji = {
                "ACHAT FORT": "🚀",
                "ACHAT": "📈", 
                "CONSERVER": "⚖️",
                "VENDRE": "📉",
                "VENTE FORTE": "🔻"
            }.get(company["recommendation"], "⚖️")
            
            comparison_text += f"""
**{company["rank"]}. {company["symbol"]}** {recommendation_emoji}
• **Company:** {company["name"]}
• **Sector:** {company["sector"]}
• **Valuation Score:** {company["valuation_score"]}/100
• **Quality Score:** {company["quality_score"]}/100
• **Recommendation:** {company["recommendation"]}
"""

            # Add requested metrics with safe formatting
            for metric in metrics_list:
                if metric in company and company[metric] is not None:
                    value = company[metric]
                    if metric in ["pe_ratio", "pb_ratio"]:
                        comparison_text += f"• **{metric.upper()}:** {value:.1f}\n"
                    elif metric in ["roe", "profit_margin"]:
                        comparison_text += f"• **{metric.upper()}:** {value*100:.1f}%\n"
                    elif metric == "debt_to_equity":
                        comparison_text += f"• **Debt/Equity:** {value:.2f}\n"
                    else:
                        comparison_text += f"• **{metric}:** {value}\n"
            
            comparison_text += "\n"
        
        # Ajouter les informations de diagnostic
        analyzed_symbols = list(companies_data.keys())
        failed_symbols = [s for s in symbol_list if s not in analyzed_symbols]
        
        comparison_text += f"""
**📅 Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}
**🎯 Metrics Compared:** {", ".join(metrics_list)}
**✅ Successfully Analyzed:** {', '.join(analyzed_symbols)}
"""
        
        if failed_symbols:
            comparison_text += f"**⚠️ Failed to Analyze:** {', '.join(failed_symbols)}\n"
            comparison_text += "**💡 Note:** Some companies may fail due to API limits or missing data.\n"
        
        ctx.report_progress(1.0, "Comparison complete!")
        await log_request_end(ctx, request_id, "compare_companies", True, f"Compared {len(comparison_table)} companies")
        return comparison_text
        
    except Exception as e:
        error_msg = await handle_error(ctx, request_id, "compare_companies", e)
        return f"❌ Error in company comparison: {error_msg}"

   

    
@mcp.tool()
async def api_status_check(ctx: Context) -> str:
    """
    Vérifier le statut de toutes les APIs et sources de données.
    """
    request_id = await log_request_start(ctx, "api_status_check", {})
    
    try:
        ctx.report_progress(0.1, "Testing Tiingo API...")
        
        api_status = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "tiingo": {"status": "unknown", "details": ""},
            "binance": {"status": "unknown", "details": ""},
            "alpha_vantage": {"status": "unknown", "details": ""},
            "news_api": {"status": "unknown", "details": ""}
        }
        
        # Test Tiingo
        try:
            df = await asyncio.wait_for(
                market_data.get_historical_data("AAPL", lookback_days=2),
                timeout=10
            )
            if df is not None and len(df) > 0:
                api_status["tiingo"]["status"] = "✅ Operational"
                api_status["tiingo"]["details"] = f"Last price: ${df['close'].iloc[-1]:.2f}"
            else:
                api_status["tiingo"]["status"] = "⚠️ No Data"
        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                api_status["tiingo"]["status"] = "🔴 Rate Limited"
                api_status["tiingo"]["details"] = "Too many requests - wait 5-10 minutes"
            else:
                api_status["tiingo"]["status"] = "❌ Error"
                api_status["tiingo"]["details"] = error_str[:50]
        
        ctx.report_progress(0.3, "Testing Binance API...")
        
        # Test Binance
        try:
            df = await asyncio.wait_for(
                market_data.get_crypto_historical_data("BTCUSDT", 2, "binance"),
                timeout=10
            )
            if df is not None and len(df) > 0:
                api_status["binance"]["status"] = "✅ Operational"
                api_status["binance"]["details"] = f"BTC price: ${df['close'].iloc[-1]:.0f}"
            else:
                api_status["binance"]["status"] = "⚠️ No Data"
        except Exception as e:
            api_status["binance"]["status"] = "❌ Error"
            api_status["binance"]["details"] = str(e)[:50]
        
        ctx.report_progress(0.6, "Testing Alpha Vantage...")
        
        # Test Alpha Vantage
        if fundamental_analyzer:
            try:
                async with alpha_vantage_client:
                    overview = await asyncio.wait_for(
                        alpha_vantage_client.get_company_overview("AAPL"),
                        timeout=15
                    )
                    api_status["alpha_vantage"]["status"] = "✅ Operational"
                    api_status["alpha_vantage"]["details"] = f"Company: {overview.name}"
            except Exception as e:
                error_str = str(e)
                if "limit" in error_str.lower():
                    api_status["alpha_vantage"]["status"] = "🔴 Rate Limited"
                else:
                    api_status["alpha_vantage"]["status"] = "❌ Error"
                api_status["alpha_vantage"]["details"] = error_str[:50]
        else:
            api_status["alpha_vantage"]["status"] = "⚪ Not Configured"
        
        ctx.report_progress(0.8, "Testing News API...")
        
        # Test News API
        if news_analyzer:
            try:
                sentiment = await asyncio.wait_for(
                    news_analyzer.analyze_symbol_sentiment("AAPL", 3, 5),
                    timeout=15
                )
                api_status["news_api"]["status"] = "✅ Operational"
                api_status["news_api"]["details"] = f"Articles found: {sentiment.total_articles}"
            except Exception as e:
                api_status["news_api"]["status"] = "❌ Error"
                api_status["news_api"]["details"] = str(e)[:50]
        else:
            api_status["news_api"]["status"] = "⚪ Not Configured"
        
        ctx.report_progress(1.0, "Status check complete!")
        
        # Générer le rapport
        status_text = f"""
📊 **API Status Report** - {api_status['timestamp']}

🔌 **Data Sources:**
• **Tiingo (Stocks/Crypto):** {api_status['tiingo']['status']}
  {api_status['tiingo']['details']}

• **Binance (Crypto Fallback):** {api_status['binance']['status']}
  {api_status['binance']['details']}

💼 **Fundamental Data:**
• **Alpha Vantage:** {api_status['alpha_vantage']['status']}
  {api_status['alpha_vantage']['details']}

📰 **News & Sentiment:**
• **NewsAPI:** {api_status['news_api']['status']}
  {api_status['news_api']['details']}

🎯 **Recommendations:**
"""
        
        if api_status['tiingo']['status'].startswith('🔴'):
            status_text += "• Wait 5-10 minutes before using stock/crypto tools\n"
            status_text += "• Crypto analysis will auto-fallback to Binance\n"
        
        if api_status['binance']['status'].startswith('✅'):
            status_text += "• Binance fallback is operational for crypto data\n"
        
        operational_count = sum(1 for api in api_status.values() 
                              if isinstance(api, dict) and api['status'].startswith('✅'))
        
        status_text += f"\n📈 **Overall Status:** {operational_count}/4 APIs operational"
        
        await log_request_end(ctx, request_id, "api_status_check", True)
        return status_text
        
    except Exception as e:
        error_msg = await handle_error(ctx, request_id, "api_status_check", e)
        return f"❌ Error checking API status: {error_msg}"

#outils de recommandation

@mcp.tool()
async def get_final_recommendation(
    ctx: Context,
    symbol: str,
    analysis_type: str = "complete"
) -> str:
    """
    Génère une recommandation finale unifiée basée sur toutes les analyses.
    
    Args:
        symbol: Symbole de l'action (e.g., AAPL)
        analysis_type: Type d'analyse ("complete", "technical", "fundamental")
    """
    request_id = await log_request_start(ctx, "get_final_recommendation", {
        "symbol": symbol, "analysis_type": analysis_type
    })
    
    try:
        ctx.report_progress(0.1, f"Analyzing {symbol} for final recommendation...")
        
        # Initialiser TOUTES les variables dès le début
        technical_score = 50  # Default
        fundamental_score = 0  # Default
        risk_assessment = "MODERATE"
        tech_factors = []
        fundamental_factors = []
        risk_factors = []
        df = None
        current_price = 0.0
        
        # Analyse technique
        ctx.report_progress(0.2, "Running technical analysis...")
        try:
            df = await market_data.get_historical_data(symbol, lookback_days=100)
            df = fix_volume_types(df)
            
            if df is not None and len(df) >= 20:
                df = tech_analysis.add_core_indicators(df)
                trend = tech_analysis.check_trend_status(df)
                signals = tech_analysis.get_trading_signals(df)
                
                # Score technique sur 100
                technical_score = signals.get('signal_strength', 50) or 50
                current_price = float(df["close"].iloc[-1])
                
                # Facteurs techniques
                if trend.get("above_200sma"):
                    tech_factors.append("✅ Tendance long terme haussière")
                else:
                    tech_factors.append("❌ Tendance long terme baissière")
                    
                rsi_value = trend.get("rsi", 50) or 50
                if 30 < rsi_value < 70:
                    tech_factors.append("✅ RSI dans zone neutre")
                elif rsi_value > 70:
                    tech_factors.append("⚠️ RSI en surachat")
                else:
                    tech_factors.append("⚠️ RSI en survente")
                    
                if trend.get("macd_bullish"):
                    tech_factors.append("✅ MACD haussier")
                else:
                    tech_factors.append("❌ MACD baissier")
            else:
                tech_factors = ["⚠️ Données insuffisantes pour analyse technique"]
                
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            tech_factors = ["⚠️ Analyse technique échouée (API limite?)"]
        
        # Analyse fondamentale (si disponible)
        if fundamental_analyzer and analysis_type in ["complete", "fundamental"]:
            await ctx.report_progress(0.5, "Running fundamental analysis...")
            try:
                analysis = await fundamental_analyzer.get_comprehensive_analysis(symbol)
                
                if "error" not in analysis:
                    valuation = analysis["valuation_assessment"]
                    quality = analysis["quality_score"]
                    
                    # Score fondamental sur 100
                    fundamental_score = (valuation["valuation_score"] + quality["quality_score"]) / 2
                    
                    # Facteurs fondamentaux
                    if valuation["valuation_score"] >= 70:
                        fundamental_factors.append("✅ Valorisation attractive")
                    elif valuation["valuation_score"] >= 40:
                        fundamental_factors.append("⚖️ Valorisation neutre")
                    else:
                        fundamental_factors.append("❌ Valorisation élevée")
                        
                    if quality["quality_score"] >= 70:
                        fundamental_factors.append("✅ Entreprise de haute qualité")
                    elif quality["quality_score"] >= 50:
                        fundamental_factors.append("⚖️ Qualité d'entreprise acceptable")
                    else:
                        fundamental_factors.append("❌ Qualité d'entreprise préoccupante")
                        
                    fundamental_factors.append(f"📊 Recommandation Alpha Vantage: {valuation['recommendation']}")
                else:
                    fundamental_factors = ["⚠️ Analyse fondamentale non disponible"]
                    
            except Exception as e:
                logger.error(f"Fundamental analysis failed: {e}")
                fundamental_factors = ["⚠️ Analyse fondamentale échouée (API limite?)"]
        
        # Analyse des risques
        ctx.report_progress(0.7, "Assessing risk factors...")
        try:
            if df is not None and len(df) >= 20:
                # Volatilité récente
                recent_vol = df["close"].pct_change().tail(20).std() * 100
                if recent_vol > 3:
                    risk_factors.append("⚠️ Forte volatilité récente")
                    risk_assessment = "HIGH"
                elif recent_vol < 1:
                    risk_factors.append("✅ Faible volatilité")
                    risk_assessment = "LOW"
                else:
                    risk_factors.append("⚖️ Volatilité modérée")
                    risk_assessment = "MODERATE"
                    
                # Volume
                latest = df.iloc[-1]
                avg_vol = latest.get("avg_20d_vol", 0)
                if avg_vol > 0:
                    vol_ratio = latest["volume"] / avg_vol
                    if vol_ratio > 2:
                        risk_factors.append("⚡ Volume élevé - attention aux mouvements")
                    elif vol_ratio < 0.5:
                        risk_factors.append("⚠️ Volume faible - liquidité réduite")
                    else:
                        risk_factors.append("✅ Volume normal")
                else:
                    risk_factors.append("⚖️ Volume non analysable")
            else:
                risk_factors = ["⚠️ Évaluation des risques non disponible"]
                
        except Exception as e:
            risk_factors = ["⚠️ Évaluation des risques échouée"]
        
        # Calcul du score final et recommandation
        ctx.report_progress(0.9, "Generating final recommendation...")
        
        if analysis_type == "technical":
            final_score = technical_score
            weight_info = "100% technique"
        elif analysis_type == "fundamental":
            final_score = fundamental_score if fundamental_score > 0 else 50
            weight_info = "100% fondamental"
        else:  # complete
            if fundamental_score > 0:
                final_score = (technical_score * 0.6) + (fundamental_score * 0.4)
                weight_info = "60% technique + 40% fondamental"
            else:
                final_score = technical_score
                weight_info = "100% technique (fondamental non disponible)"
        
        # Déterminer la recommandation finale
        if final_score >= 75:
            final_recommendation = "ACHAT FORT"
            recommendation_emoji = "🚀"
            action_text = "Je recommande fortement d'ACHETER cette action"
            confidence = "Très élevée"
        elif final_score >= 60:
            final_recommendation = "ACHAT"
            recommendation_emoji = "📈"
            action_text = "Je recommande d'ACHETER cette action"
            confidence = "Élevée"
        elif final_score >= 45:
            final_recommendation = "CONSERVER/ATTENDRE"
            recommendation_emoji = "⚖️"
            action_text = "Je recommande d'ATTENDRE ou de CONSERVER les positions existantes"
            confidence = "Modérée"
        elif final_score >= 30:
            final_recommendation = "VENDRE"
            recommendation_emoji = "📉"
            action_text = "Je recommande de VENDRE cette action"
            confidence = "Élevée"
        else:
            final_recommendation = "VENTE FORTE"
            recommendation_emoji = "🔻"
            action_text = "Je recommande fortement de VENDRE cette action"
            confidence = "Très élevée"
        
        # Ajustement selon le risque
        risk_adjustment = ""
        if risk_assessment == "HIGH" and final_recommendation in ["ACHAT FORT", "ACHAT"]:
            risk_adjustment = "\n⚠️ **Attention:** Risque élevé détecté - considérez une position plus petite"
        elif risk_assessment == "LOW" and final_recommendation in ["VENDRE", "VENTE FORTE"]:
            risk_adjustment = "\n💡 **Note:** Faible risque - la vente n'est peut-être pas urgente"
        
        # Fallback pour le prix si pas récupéré
        if current_price == 0.0:
            current_price_text = "Prix non disponible (API limite?)"
        else:
            current_price_text = f"${current_price:.2f}"
        
        # Générer le rapport final
        recommendation_text = f"""
🎯 **RECOMMANDATION FINALE POUR {symbol}**

{recommendation_emoji} **{action_text}**

📊 **Résumé de l'analyse:**
• **Score final:** {final_score:.1f}/100
• **Recommandation:** {final_recommendation}
• **Confiance:** {confidence}
• **Méthodologie:** {weight_info}
• **Prix actuel:** {current_price_text}
• **Évaluation du risque:** {risk_assessment}

📈 **Facteurs techniques:**
{chr(10).join(f"  {factor}" for factor in tech_factors) if tech_factors else "  ⚠️ Non disponibles"}

"""
        
        if fundamental_factors:
            recommendation_text += f"""💼 **Facteurs fondamentaux:**
{chr(10).join(f"  {factor}" for factor in fundamental_factors)}

"""
        
        recommendation_text += f"""🛡️ **Facteurs de risque:**
{chr(10).join(f"  {factor}" for factor in risk_factors) if risk_factors else "  ⚠️ Non disponibles"}

{risk_adjustment}

📅 **Date d'analyse:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

⚠️ **Note:** Si certaines données sont manquantes, c'est probablement dû aux limites API temporaires. Réessayez dans quelques minutes.

⚠️ **Disclaimer:** Cette recommandation est basée sur l'analyse des données disponibles et ne constitue pas un conseil financier.
"""
        
        ctx.report_progress(1.0, "Final recommendation complete!")
        await log_request_end(ctx, request_id, "get_final_recommendation", True, f"Generated recommendation for {symbol}")
        return recommendation_text
        
    except Exception as e:
        error_msg = await handle_error(ctx, request_id, "get_final_recommendation", e)
        return f"❌ Error generating final recommendation: {error_msg}"



@mcp.tool()
async def quick_recommendation(
    ctx: Context,
    symbol: str
) -> str:
    """
    Recommandation rapide en une phrase basée sur l'analyse technique - VERSION CORRIGÉE.
    
    Args:
        symbol: Symbole de l'action (e.g., AAPL)
    """
    request_id = await log_request_start(ctx, "quick_recommendation", {"symbol": symbol})
    
    try:
        ctx.report_progress(0.2, f"Quick analysis of {symbol}...")
        
        # Récupération des données avec retry et timeout
        df = None
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    await asyncio.sleep(3)
                    ctx.report_progress(0.3, f"Retry {attempt + 1} for {symbol}...")
                
                df = await asyncio.wait_for(
                    market_data.get_historical_data(symbol, lookback_days=30),
                    timeout=12
                )
                
                if df is not None and len(df) >= 20:
                    break
                    
            except asyncio.TimeoutError:
                if attempt == max_retries - 1:
                    return f"""⏰ **QUICK RECOMMENDATION FOR {symbol} - TIMEOUT**

**Current Recommendation: WAIT**

📊 **Reason:** Data retrieval timeout ({12}s)
• **Status:** API response too slow
• **Impact:** Cannot perform technical analysis

🔧 **Immediate Solutions:**
1. **Retry in 2-3 minutes:** API may be under load
2. **Try different symbol:** `quick_recommendation("AAPL")`
3. **Full analysis:** `analyze_stock("{symbol}")` (more robust)
4. **Check status:** `api_status_check()`

💡 **Note:** Temporary issue, system will recover automatically
"""
                continue
                
            except Exception as e:
                error_str = str(e)
                if any(term in error_str.lower() for term in ["429", "rate", "limit", "quota"]):
                    return f"""📊 **QUICK RECOMMENDATION FOR {symbol} - API LIMIT**

**Current Recommendation: WAIT**

📊 **Reason:** API quota temporarily exhausted
• **Provider:** Tiingo API daily limit reached
• **Reset:** Usually within 1-6 hours
• **Alternative:** Use fundamental analysis tools

🔧 **Immediate Alternatives:**
1. **Fundamental analysis:** `analyze_fundamentals("{symbol}")`
2. **Sentiment analysis:** `analyze_sentiment("{symbol}")`
3. **API status:** `api_status_check()`
4. **Wait and retry:** 30-60 minutes

⏰ **Expected Recovery:** {datetime.now().hour + 2}:00 (approximate)
"""
                break
        
        # Vérification des données
        if df is None or len(df) < 20:
            return f"""❌ **QUICK RECOMMENDATION FOR {symbol} - INSUFFICIENT DATA**

**Current Recommendation: RESEARCH NEEDED**

📊 **Reason:** Insufficient historical data
• **Data Points:** {len(df) if df is not None else 0}/20 required
• **Possible Causes:** New listing, invalid symbol, or API issues

🔧 **Solutions:**
1. **Verify symbol:** Is "{symbol}" correct?
2. **Try major stocks:** AAPL, MSFT, GOOGL
3. **System check:** `system_diagnostic()`
4. **Manual research:** Check financial websites

💡 **Note:** May be a data availability issue for this specific symbol
"""
        
        await ctx.report_progress(0.5, "Processing technical indicators...")
        
        # Calcul des indicateurs avec gestion d'erreur robuste
        try:
            
            df_analysis = df.copy()
            
            # Ajouter les indicateurs avec gestion d'erreur
            df_with_indicators = tech_analysis.add_core_indicators(df_analysis)
            
            # Vérifier que les indicateurs de base sont présents
            required_columns = ['close', 'volume']
            missing_columns = [col for col in required_columns if col not in df_with_indicators.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Calculer les signaux avec gestion d'erreur
            try:
                signals = tech_analysis.get_trading_signals(df_with_indicators)
                trend = tech_analysis.check_trend_status(df_with_indicators)
            except Exception as signal_error:
                logger.warning(f"Signal calculation error: {signal_error}")
                # Fallback vers analyse basique
                signals = {
                    'signal_strength': 50,
                    'overall_signal': 'HOLD',
                    'signal_details': ['Analysis partially limited due to calculation issues']
                }
                trend = {'rsi': 50}
            
        except Exception as indicator_error:
            logger.error(f"Indicator calculation failed: {indicator_error}")
            return f"""⚠️ **QUICK RECOMMENDATION FOR {symbol} - CALCULATION ERROR**

**Current Recommendation: WAIT**

📊 **Reason:** Technical indicator calculation failed
• **Error Type:** {type(indicator_error).__name__}
• **Impact:** Cannot calculate RSI, MACD, signals

🔧 **Immediate Solutions:**
1. **Retry in 5 minutes:** Temporary calculation issue
2. **Alternative analysis:** `analyze_fundamentals("{symbol}")`
3. **System diagnostic:** `system_diagnostic()`
4. **Different symbol:** Try a major stock like AAPL

💡 **Note:** This is typically a temporary issue with data processing
"""
        
        ctx.report_progress(0.8, "Generating recommendation...")
        
        # Extraction sécurisée des métriques avec validations
        try:
            current_price = float(df_with_indicators["close"].iloc[-1])
            
            # Validation du prix
            if current_price <= 0 or pd.isna(current_price):
                raise ValueError("Invalid current price")
            
            # Signal strength avec validation
            signal_strength = signals.get('signal_strength', 50)
            if signal_strength is None or pd.isna(signal_strength):
                signal_strength = 50
            else:
                signal_strength = float(signal_strength)
                signal_strength = max(0, min(100, signal_strength))  # Clamp entre 0-100
            
            # Overall signal avec validation  
            overall_signal = signals.get('overall_signal', 'HOLD')
            if not overall_signal or overall_signal == '':
                overall_signal = 'HOLD'
            
            # RSI avec validation
            rsi = trend.get("rsi", 50)
            if rsi is None or pd.isna(rsi):
                rsi = 50
            else:
                rsi = float(rsi)
                rsi = max(0, min(100, rsi))  # Clamp entre 0-100
            
        except Exception as extraction_error:
            logger.error(f"Metric extraction failed: {extraction_error}")
            return f"""⚠️ **QUICK RECOMMENDATION FOR {symbol} - DATA PROCESSING ERROR**

**Current Recommendation: WAIT**

📊 **Reason:** Unable to extract valid metrics from data
• **Error:** Data validation failed
• **Impact:** Cannot generate reliable signals

🔧 **Solutions:**
1. **Data quality issue:** Try different time period
2. **Alternative:** `analyze_stock("{symbol}")` for robust analysis
3. **Different symbol:** Try `quick_recommendation("AAPL")`
4. **Check system:** `system_diagnostic()`

💡 **Note:** May indicate data quality issues for this symbol
"""
        
        ctx.report_progress(1.0, "Recommendation ready!")
        
        # Logique de recommandation avec validation finale
        try:
            # Déterminer l'action recommandée
            if signal_strength >= 70:
                action = "ACHETER"
                emoji = "🚀"
                reasoning = "signaux techniques très positifs"
                confidence = "Élevée"
            elif signal_strength >= 55:
                action = "ACHETER prudemment"
                emoji = "📈"
                reasoning = "signaux techniques positifs"
                confidence = "Modérée"
            elif signal_strength >= 45:
                action = "ATTENDRE"
                emoji = "⚖️"
                reasoning = "signaux techniques neutres"
                confidence = "Faible"
            elif signal_strength >= 30:
                action = "VENDRE"
                emoji = "📉"
                reasoning = "signaux techniques négatifs"
                confidence = "Modérée"
            else:
                action = "VENDRE rapidement"
                emoji = "🔻"
                reasoning = "signaux techniques très négatifs"
                confidence = "Élevée"
            
            # Ajustement selon RSI
            rsi_adjustment = ""
            if rsi > 80:
                rsi_adjustment = "\n⚠️ **RSI Alert:** Niveau de surachat élevé - prudence recommandée"
                if action in ["ACHETER", "ACHETER prudemment"]:
                    rsi_adjustment += " (considérez d'attendre une correction)"
            elif rsi < 20:
                rsi_adjustment = "\n💡 **RSI Alert:** Niveau de survente - potentiel de rebond"
                if action in ["VENDRE", "VENDRE rapidement"]:
                    rsi_adjustment += " (potentiel rebond technique possible)"
            
            # Générer la recommandation finale
            quick_text = f"""
{emoji} **QUICK RECOMMENDATION FOR {symbol}**

**Based on this analysis, I recommend to {action}.**

📊 **Express Analysis:**
• **Current Price:** ${current_price:.2f}
• **Overall Signal:** {overall_signal}
• **Signal Strength:** {signal_strength:.0f}/100
• **RSI:** {rsi:.1f}
• **Confidence:** {confidence}

📈 **Reasoning:** {reasoning}{rsi_adjustment}

⏰ **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

💡 **For detailed analysis:** `get_final_recommendation("{symbol}")`

⚠️ **Disclaimer:** Quick recommendation based on technical analysis. Not financial advice.
"""
            
            await log_request_end(ctx, request_id, "quick_recommendation", True)
            return quick_text
            
        except Exception as recommendation_error:
            logger.error(f"Recommendation generation failed: {recommendation_error}")
            return f"""⚠️ **QUICK RECOMMENDATION FOR {symbol} - GENERATION ERROR**

**Current Recommendation: WAIT**

📊 **Reason:** Recommendation logic failed
• **Error:** {type(recommendation_error).__name__}
• **Data Available:** Price=${current_price:.2f if 'current_price' in locals() else 'N/A'}

🔧 **Alternatives:**
1. **Full analysis:** `analyze_stock("{symbol}")`
2. **System check:** `system_diagnostic()`
3. **Different approach:** `get_final_recommendation("{symbol}")`

⚠️ **Note:** System issue - use alternative analysis methods

📅 **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        
    except Exception as e:
        error_msg = await handle_error(ctx, request_id, "quick_recommendation", e)
        return f"""❌ **QUICK RECOMMENDATION FOR {symbol} - SYSTEM ERROR**

**Current Recommendation: WAIT**

🔧 **System Error:** {error_msg[:100]}...

**Immediate Actions:**
1. **Check system status:** `api_status_check()`
2. **Try alternative:** `analyze_stock("{symbol}")`
3. **System diagnostic:** `system_diagnostic()`
4. **Retry later:** Wait 10-15 minutes

⚠️ **Note:** Temporary system issue - use alternative analysis tools

📅 **Error Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
# System status resource
@mcp.resource("mcp://system/status")
async def get_system_status(ctx: Context) -> str:
    """Get current system status."""
    status = {
        "server": "MCP Trader Server",
        "version": "2.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "market_data": "ready",
            "technical_analysis": "ready",
            "relative_strength": "ready",
            "volume_profile": "ready",
            "pattern_recognition": "ready",
            "risk_analysis": "ready",
            "fundamental_analysis": "ready" if fundamental_analyzer else "disabled",
            "sentiment_analysis": "ready" if news_analyzer else "disabled",
        }
    }
    return json.dumps(status, indent=2)

# Startup logging
def log_startup():
    """Log startup information."""
    logger.info("🎯 MCP Trader Server Enhanced starting...")
    logger.info(f"📡 Server name: {config.server_name}")
    
    features = ["crypto_analysis", "stock_analysis", "technical_indicators", "relative_strength", "volume_profile", "pattern_recognition", "risk_management"]
    
    if fundamental_analyzer:
        features.extend([
            "fundamental_analysis",
            "company_comparison", 
            "investment_thesis",
            "earnings_analysis"
        ])
        logger.info("✅ Alpha Vantage fundamental analysis enabled")
    else:
        logger.info("⚠️ Alpha Vantage fundamental analysis disabled (no API key)")
        
    if news_analyzer:
        features.append("sentiment_analysis")
        logger.info("✅ News sentiment analysis enabled")
    else:
        logger.info("⚠️ News sentiment analysis disabled (no API key)")
    
    logger.info(f"✅ Features enabled ({len(features)}): {', '.join(features)}")
    # Log des clés API configurées
    api_status = []
    if config.tiingo_api_key:
        api_status.append("Tiingo ✅")
    if config.alpha_vantage_api_key:
        api_status.append("Alpha Vantage ✅")  
    if config.news_api_key:
        api_status.append("NewsAPI ✅")
        
    logger.info(f"🔑 API Keys: {', '.join(api_status) if api_status else 'None configured'}")
    logger.info("🚀 Server ready!")
# Main function
def main():
    """Main entry point for the MCP server."""
    log_startup()
    mcp.run()

if __name__ == "__main__":
    main()