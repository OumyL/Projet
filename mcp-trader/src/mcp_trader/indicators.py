from typing import Any

import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta

class TechnicalAnalysis:
    """Technical analysis toolkit with improved performance and readability."""

    @staticmethod
    def add_core_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add a core set of technical indicators."""
        try:
            # Adding trend indicators
            df["sma_20"] = ta.sma(df["close"], length=20)
            df["sma_50"] = ta.sma(df["close"], length=50)
            df["sma_200"] = ta.sma(df["close"], length=200)

            # Adding volatility indicators and volume
            daily_range = df["high"].sub(df["low"])
            adr = daily_range.rolling(window=20).mean()
            df["adrp"] = adr.div(df["close"]).mul(100)
            df["avg_20d_vol"] = df["volume"].rolling(window=20).mean()

            # Adding momentum indicators
            df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
            df["rsi"] = ta.rsi(df["close"], length=14)
            macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
            if macd is not None:
                df = pd.concat([df, macd], axis=1)
            
            #Bollinger Bands
            bb = ta.bbands(df["close"], length=20, std=2)
            if bb is not None:
                df = pd.concat([df, bb], axis=1)
                #Rename columns for clarity
                df.rename(columns={
                    'BBL_20_2.0' : 'bb_lower',
                    'BBM_20_2.0' : 'bb_middle',
                    'BBU_20_2.0' : 'bb_upper',
                    'BBB_20_2.0' : 'bb_bandwidth',
                    'BBP_20_2.0' : 'bb_percent'
                }, inplace=True, errors='ignore')
            #Stochastic Oscillator
            stoch = ta.stoch(df["high"], df["low"], df["close"], K= 14, d=3, smooth_k=3)
            if stoch is not None:
                df = pd.concat([df, stoch], axis=1)
                #Renaming for clarity
                df.rename(columns={
                    'STOCHk_14_3_3' : 'stoch_k',
                    'STOCHd_14_3_3' : 'stoch_d'
                }, inplace=True,  errors='ignore')
            #Williams %R
            df["williams_r"] = ta.willr(df["high"], df["low"], df["close"], length=14)
            #Commodity Channel Index (CCI)
            df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)
            #Money Flow Index (MFI)
            df["mfi"] = ta.mfi(df["high"],df["low"], df["close"], df["volume"], length=14)
            #Volume Weighted Average Price (VWAP)
            
            if len(df) > 0:
                typical_price = (df["high"] + df["low"] + df["close"]) / 3
                df["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
                
            return df

        except KeyError as e:
            raise KeyError(f"Missing column in input DataFrame: {str(e)}") from e
        except Exception as e:
            raise Exception(f"Error calculating indicators: {str(e)}") from e

    @staticmethod
    def check_trend_status(df: pd.DataFrame) -> dict[str, Any]:
        """Analyze the current trend status with enhanced indicators - VERSION CORRIGÉE."""
        if df.empty:
            raise ValueError("DataFrame is empty. Ensure it contains valid data.")

        latest = df.iloc[-1]
    
        #  NOUVELLE FONCTION: Protection contre les erreurs
        def safe_comparison(val1, val2, default=False):
            """Comparaison sécurisée évitant les erreurs NaN/None."""
            try:
                 if pd.isna(val1) or pd.isna(val2):
                    return default
                 if val1 is None or val2 is None:
                    return default
                 return float(val1) > float(val2)
            except (TypeError, ValueError):
                return default
    
        def safe_get(series_data, key, default=None):
            """Récupération sécurisée d'une valeur."""
            try:
                value = series_data.get(key, default) if hasattr(series_data, 'get') else getattr(series_data, key, default)
                return value if not pd.isna(value) else default
            except (AttributeError, KeyError):
                return default

        #  TREND ANALYSIS 
        trend_status = {
            "above_20sma": safe_comparison(latest["close"], safe_get(latest, "sma_20")),
            "above_50sma": safe_comparison(latest["close"], safe_get(latest, "sma_50")),
            "above_200sma": safe_comparison(latest["close"], safe_get(latest, "sma_200")),
            "20_50_bullish": safe_comparison(safe_get(latest, "sma_20"), safe_get(latest, "sma_50")),
            "50_200_bullish": safe_comparison(safe_get(latest, "sma_50"), safe_get(latest, "sma_200")),
            "rsi": safe_get(latest, "rsi", 50.0),
            "macd_bullish": safe_comparison(
                safe_get(latest, "MACD_12_26_9", 0), 
                safe_get(latest, "MACDs_12_26_9", 0)
            ),
        }
    
        #  BOLLINGER BANDS
        bb_upper = safe_get(latest, "bb_upper")
        bb_lower = safe_get(latest, "bb_lower")
        if bb_upper and bb_lower:
            try:
                bb_position = (latest["close"] - bb_lower) / (bb_upper - bb_lower)
                trend_status.update({
                    "bb_squeeze": safe_get(latest, "bb_bandwidth", 1) < 0.1,
                    "bb_upper_touch": latest["close"] > bb_upper * 0.98,
                    "bb_lower_touch": latest["close"] < bb_lower * 1.02,
                    "bb_position": max(0, min(1, bb_position))
                })
            except (ZeroDivisionError, TypeError):
                trend_status.update({
                    "bb_squeeze": False, "bb_upper_touch": False,
                    "bb_lower_touch": False, "bb_position": 0.5
                })
        
        # STOCHASTIC 
        stoch_k = safe_get(latest, "stoch_k")
        stoch_d = safe_get(latest, "stoch_d")
        if stoch_k is not None:
            trend_status.update({
                "stoch_k": stoch_k,
                "stoch_d": stoch_d or stoch_k,
                "stoch_overbought": stoch_k > 80,
                "stoch_oversold": stoch_k < 20,
                "stoch_bullish_cross": safe_comparison(stoch_k, stoch_d)
            })
        
       
        williams_r = safe_get(latest, "williams_r")
        if williams_r is not None:
            trend_status.update({
                "williams_r": williams_r,
                "williams_overbought": williams_r > -20,
                "williams_oversold": williams_r < -80
            })
        
        cci = safe_get(latest, "cci")
        if cci is not None:
            trend_status.update({
                "cci": cci,
                "cci_overbought": cci > 100,
                "cci_oversold": cci < -100
            })
        
        mfi = safe_get(latest, "mfi")
        if mfi is not None:
            trend_status.update({
                "mfi": mfi,
                "mfi_overbought": mfi > 80,
                "mfi_oversold": mfi < 20
            })
        
        vwap = safe_get(latest, "vwap")
        if vwap is not None:
            try:
                trend_status.update({
                    "above_vwap": latest["close"] > vwap,
                    "vwap_distance": ((latest["close"] - vwap) / vwap) * 100
                })
            except (ZeroDivisionError, TypeError):
                trend_status.update({"above_vwap": False, "vwap_distance": 0.0})
    
        return trend_status
    @staticmethod
    def get_trading_signals(df: pd.DataFrame) -> dict[str, Any]:
        """Generate comprehensive trading signals based on all indicators."""
        trend_status = TechnicalAnalysis.check_trend_status(df)
        
        #Bull/Bear signal scoring
        bull_signals = 0
        bear_signals = 0
        total_signals = 0
        
        signal_details = []
        
        #Trend signals (high weight)
        if trend_status.get("above_200sma"):
            bull_signals += 3
            signal_details.append("Above 200 SMA (Strong Bullish)")
        else:
            bear_signals += 3
            signal_details.append("Below 200 SMA (Strong Bearish)")
        total_signals += 3
        
        if trend_status.get("20_50_bullish"):
            bull_signals += 2
            signal_details.append("20/50 SMA Bearish Cross")
        total_signals += 2
        
        #Momentum signals
        rsi = trend_status.get("rsi" ,  50)
        if 30 < rsi < 70:
            bull_signals += 1
            signal_details.append(f"RSI Neutral Zone ({rsi:.1f})")
        elif rsi > 70:
            bear_signals += 1
            signal_details.append(f" RSI Ovebought ({rsi:.1f})")
        elif rsi < 30:
            bull_signals += 1 #Oversold can be bullish for reversal
            signal_details.append(f" RSI Oversold - Potential Reversal ({rsi:.1f})")
        total_signals += 1
        
        #Bollinger Bands signals
        if trend_status.get("bb_squeeze"):
            signal_details.append(" Bollinger Band Squeeze - Breakout Expected")
            
        if trend_status.get("bb_upper_touch"):
            bear_signals += 1
            signal_details.append(" Touching Upper Bollinger Band")
            total_signals += 1
        elif trend_status.get("bb_lower_touch"):
            bull_signals += 1
            signal_details.append(" Touching Lower Bollinger Band - Potential Bounce")
            total_signals += 1
        
        #Stochastic signals
        if trend_status.get("stoch_overbought"):
            bear_signals += 1
            signal_details.append(f" Stochastic Overbought ({trend_status.get('stoch_k', 0):.1f})")
            total_signals += 1
        elif trend_status.get("stoch_oversold"):
            bull_signals += 1
            signal_details.append(f" Stochastic Oversold ({trend_status.get('stoch_k', 0):.1f})")
            total_signals += 1
        
        #Calculate overall signal strength
        if total_signals > 0:
            bull_percentage = (bull_signals / total_signals) * 100
        else:
            bull_percentage = 50
        
        #Determine overall signal
        if bull_percentage >= 70:
            overall_signal = "Strong BUY"
            signal_emoji = "🚀"
        elif bull_percentage >= 60:
            overall_signal = "BUY"
            signal_emoji = "📈"
        elif bull_percentage >= 40:
            overall_signal = "HOLD"
            signal_emoji = "⚖️"
        elif bull_percentage >= 30:
            overall_signal = "SELL"
            signal_emoji = "📉"
        else:
            overall_signal = "STRONG SELL"
            signal_emoji = "🔻"
        return {
            "overall_signal" : overall_signal,
            "signal_emoji": signal_emoji,
            "bull_percentage" : bull_percentage,
            "signal_strength" : int(bull_percentage),
            "signal_details" : signal_details,
            "bull_signals" : bull_signals,
            "bear_signals" : bear_signals,
            "total_signals" : total_signals
        }
            

class RelativeStrength:
    """Tools for calculating relative strength metrics."""

    @staticmethod
    async def calculate_rs(
        market_data,
        symbol: str,
        benchmark: str = "SPY",
        lookback_periods: list[int] = None,
    ) -> dict[str, float]:
        """
        Calculate relative strength compared to a benchmark across multiple timeframes.

        Args:
            market_data: Our market data fetcher instance
            symbol (str): The stock symbol to analyze
            benchmark (str): The benchmark symbol (default: SPY for S&P 500 ETF)
            lookback_periods (List[int]): Periods in trading days to calculate RS (default: [21, 63, 126, 252])

        Returns:
            Dict[str, float]: Relative strength scores for each timeframe
        """
        try:
            if lookback_periods is None:
                lookback_periods = [21, 63, 126, 252]

            # Get data for both the stock and benchmark
            stock_df = await market_data.get_historical_data(symbol, max(lookback_periods) + 10)
            benchmark_df = await market_data.get_historical_data(
                benchmark, max(lookback_periods) + 10
            )

            # Calculate returns for different periods
            rs_scores = {}

            for period in lookback_periods:
                # Check if we have enough data for this period
                if len(stock_df) <= period or len(benchmark_df) <= period:
                    # Skip this period if we don't have enough data
                    continue

                # Calculate the percent change for both
                stock_return = (
                    stock_df["close"].iloc[-1] / stock_df["close"].iloc[-period] - 1
                ) * 100
                benchmark_return = (
                    benchmark_df["close"].iloc[-1] / benchmark_df["close"].iloc[-period] - 1
                ) * 100

                # Calculate relative strength (stock return minus benchmark return)
                relative_performance = stock_return - benchmark_return

                
                # sophisticated distribution model based on historical data)
                rs_score = min(max(50 + relative_performance, 1), 99)

                rs_scores[f"RS_{period}d"] = round(rs_score, 2)
                rs_scores[f"Return_{period}d"] = round(stock_return, 2)
                rs_scores[f"Benchmark_{period}d"] = round(benchmark_return, 2)
                rs_scores[f"Excess_{period}d"] = round(relative_performance, 2)

            return rs_scores

        except Exception as e:
            raise Exception(f"Error calculating relative strength: {str(e)}") from e


class VolumeProfile:
    """Tools for analyzing volume distribution by price."""

    @staticmethod
    def analyze_volume_profile(df: pd.DataFrame, num_bins: int = 10) -> dict[str, Any]:
        """
        Create a volume profile analysis by price level.

        Args:
            df (pd.DataFrame): Historical price and volume data
            num_bins (int): Number of price bins to create (default: 10)

        Returns:
            Dict[str, Any]: Volume profile analysis
        """
        try:
            if len(df) < 20:
                raise ValueError("Not enough data for volume profile analysis")

            # Find the price range for the period
            price_min = df["low"].min()
            price_max = df["high"].max()

            # Create price bins
            bin_width = (price_max - price_min) / num_bins

            # Initialize the profile
            profile = {
                "price_min": price_min,
                "price_max": price_max,
                "bin_width": bin_width,
                "bins": [],
            }

            # Calculate volume by price bin
            for i in range(num_bins):
                bin_low = price_min + i * bin_width
                bin_high = bin_low + bin_width
                bin_mid = (bin_low + bin_high) / 2

                # Filter data in this price range
                mask = (df["low"] <= bin_high) & (df["high"] >= bin_low)
                volume_in_bin = df.loc[mask, "volume"].sum()

                # Calculate percentage of total volume
                volume_percent = (
                    (volume_in_bin / df["volume"].sum()) * 100 if df["volume"].sum() > 0 else 0
                )

                profile["bins"].append(
                    {
                        "price_low": round(bin_low, 2),
                        "price_high": round(bin_high, 2),
                        "price_mid": round(bin_mid, 2),
                        "volume": int(volume_in_bin),
                        "volume_percent": round(volume_percent, 2),
                    }
                )

            # Find the Point of Control (POC) - the price level with the highest volume
            poc_bin = max(profile["bins"], key=lambda x: x["volume"])
            profile["point_of_control"] = round(poc_bin["price_mid"], 2)

            # Find Value Area (70% of volume)
            sorted_bins = sorted(profile["bins"], key=lambda x: x["volume"], reverse=True)
            cumulative_volume = 0
            value_area_bins = []

            for bin_data in sorted_bins:
                value_area_bins.append(bin_data)
                cumulative_volume += bin_data["volume_percent"]
                if cumulative_volume >= 70:
                    break

            if value_area_bins:
                profile["value_area_low"] = round(min([b["price_low"] for b in value_area_bins]), 2)
                profile["value_area_high"] = round(
                    max([b["price_high"] for b in value_area_bins]), 2
                )

            return profile

        except Exception as e:
            raise Exception(f"Error analyzing volume profile: {str(e)}") from e


class PatternRecognition:
    """Tools for detecting common chart patterns with enhanced detection."""

    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> dict[str, Any]:
        """
        Detect common chart patterns in price data with improved algorithms.

        Args:
            df (pd.DataFrame): Historical price data

        Returns:
            Dict[str, Any]: Detected patterns and their properties
        """
        try:
            if len(df) < 60:  
                return {
                    "patterns": [],
                    "message": "Not enough data for pattern detection",
                }

            patterns = []

           
            recent_df = df.tail(60).copy()

            recent_df["is_min"] = (
                recent_df["low"].rolling(window=5, center=True).min() == recent_df["low"]
            )
            recent_df["is_max"] = (
                recent_df["high"].rolling(window=5, center=True).max() == recent_df["high"]
            )

            # Get the indices, prices, and dates of local minima and maxima
            minima = recent_df[recent_df["is_min"]].copy()
            maxima = recent_df[recent_df["is_max"]].copy()
            
            #Support and Resistance Levels
            patterns.extend(PatternRecognition._detect_support_resistance(recent_df))
            
          
            # Double Bottom Detection
            if len(minima) >= 2:
                for i in range(len(minima) - 1):
                    for j in range(i + 1, len(minima)):
                        price1 = minima.iloc[i]["low"]
                        price2 = minima.iloc[j]["low"]
                        date1 = minima.iloc[i].name
                        date2 = minima.iloc[j].name
                        

                        # Check if the two bottoms are at similar price levels
                        if abs(price1 - price2) / price1 < 0.03:
                            # Check if they're at least 10 days apart
                            days_apart = (date2 - date1).days
                            if days_apart >= 10 and days_apart <= 60:
                                # Check if there's a peak in between that's at least 5% higher
                                mask = (recent_df.index > date1) & (recent_df.index < date2)
                                if mask.any():
                                    max_between = recent_df.loc[mask, "high"].max()
                                    if max_between > price1 * 1.05:
                                        patterns.append(
                                            {
                                                "type": "Double Bottom",
                                                "start_date": date1.strftime("%Y-%m-%d"),
                                                "end_date": date2.strftime("%Y-%m-%d"),
                                                "price_level": round((price1 + price2) / 2, 2),
                                                "confidence": "Medium",
                                            }
                                        )

            # Double Top Detection (similar logic, but for maxima)
            if len(maxima) >= 2:
                for i in range(len(maxima) - 1):
                    for j in range(i + 1, len(maxima)):
                        price1 = maxima.iloc[i]["high"]
                        price2 = maxima.iloc[j]["high"]
                        date1 = maxima.iloc[i].name
                        date2 = maxima.iloc[j].name

                        if abs(price1 - price2) / price1 < 0.03:
                            days_apart = (date2 - date1).days
                            if days_apart >= 10 and days_apart <= 60:
                                mask = (recent_df.index > date1) & (recent_df.index < date2)
                                if mask.any():
                                    min_between = recent_df.loc[mask, "low"].min()
                                    if min_between < price1 * 0.95:
                                        patterns.append(
                                            {
                                                "type": "Double Top",
                                                "start_date": date1.strftime("%Y-%m-%d"),
                                                "end_date": date2.strftime("%Y-%m-%d"),
                                                "price_level": round((price1 + price2) / 2, 2),
                                                "confidence": "Medium",
                                            }
                                        )

            # Check for potential breakouts
            close = df["close"].iloc[-1]
            recent_high = df["high"].iloc[-20:].max()
            recent_low = df["low"].iloc[-20:].min()

            # Resistance breakout
            if close > recent_high * 0.99 and close < recent_high * 1.02:
                patterns.append(
                    {
                        "type": "Resistance Breakout",
                        "price_level": round(recent_high, 2),
                        "confidence": "Medium",
                    }
                )

            # Support breakout
            if close < recent_low * 1.01 and close > recent_low * 0.98:
                patterns.append(
                    {
                        "type": "Support Breakdown",
                        "price_level": round(recent_low, 2),
                        "confidence": "Medium",
                    }
                )

            
            patterns.extend(PatternRecognition._detect_double_patterns(minima, maxima))
            #Triangle patterns
            patterns.extend(PatternRecognition._detect_triangles(recent_df))
            #Breakout detection
            patterns.extend(PatternRecognition._detect_breakouts(df, recent_df))
             
            return{"patterns": patterns}
        except Exception as e : 
            raise Exception(f"Error detecting patterns: {str(e)}") from e
    @staticmethod
    def _detect_support_resistance(df: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect support and resistance levels."""
        patterns = []
        
        #Simple support/resistance using pivot points
        highs = df["high"].rolling(window=5, center= True).max()
        lows = df["low"].rolling(window=5, center=True).min()
        
        resistance_levels = df[df["high"] == highs] ["high"].values
        support_levels = df[df["low"] == lows] ["low"].values
        
        if len(resistance_levels) > 0:
            avg_resistance = np.mean(resistance_levels[-3:])
            patterns.append({
                "type": "Resistance Level",
                "price_level": round(avg_resistance, 2),
                "confidence": "Medium",
                "additional_info": {"level_count": len(resistance_levels)}
            })
        if len(support_levels) > 0:
            avg_support = np.mean(support_levels[-3:])
            patterns.append({
                "type": "Support Level",
                "price_level": round(avg_support, 2),
                "confidence": "Medium",
                "additional_info": {"level_count": len(support_levels)}
            })
        return patterns
    @staticmethod
    def _detect_double_patterns(minima: pd.DataFrame, maxima: pd.DataFrame) ->list[dict[str, Any]]:
        """Detect double top/bottom patterns."""
        patterns = []
        
        #Double Bottom Detection 
        if len(minima) >= 2:
            for i in range(len(minima) - 1):
                for j in range(i  + 1, len(minima)):
                    price1 = minima.iloc[i]["low"]
                    price2 = minima .iloc[j]["low"]
                    date1 = minima.iloc[i].name
                    date2 = minima.iloc[j].name
                    if abs(price1 - price2) / price1 < 0.03:
                        days_apart = (date2 - date1).days
                        if 10 <= days_apart <= 60:
                            patterns.append({
                                "type": "Double Bottom",
                                "start_date": date1.strftime("%Y-%m-%d"),
                                "end_date": date2.strftime("%Y-%m-%d"),
                                "price_level": round((price1 + price2) / 2, 2),
                                "confidence": "High" if abs(price1 - price2) / price1 < 0.01 else "Medium",
                            })
        
        if len(maxima) >= 2:
            for i in range(len(maxima) - 1):
                for j in range(i +1, len(maxima)):
                    price1 = maxima.iloc[i]["high"]
                    price2 = maxima.iloc[j]["high"]
                    date1 = maxima.iloc[i].name
                    date2 = maxima.iloc[j].name
                    
                    if abs(price1 -price2) /price1 < 0.03:
                        days_apart = (date2 - date1).days
                        if 10 <= days_apart <= 60:
                            patterns.append({
                                "type": "Double Top",
                                "start_date": date1.strftime("%Y-%m-%d"),
                                "end_date": date2.strftime("%Y-%m-%d"),
                                "price_level": round((price1 + price2) /2, 2),
                                "confidence": "High" if abs(price1 -price2) /price1 < 0.01 else "Medium",
                            })
                            
        return patterns
    @staticmethod
    def _detect_triangles(df: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect triangle patterns (ascending,, descending, symmetric)."""
        patterns = []
        
        if len(df) < 20:
            return patterns
        #Simple triangle detection using trend lines
        highs = df["high"].values
        lows = df["low"].values
        dates = df.index
        #Look for converging trend lines
        
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]
        
        #Check if highs are declining and lows are rising 
        if len(recent_highs) > 10 and len(recent_lows) > 10:
            high_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
            low_trend = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
            
            if high_trend < -0.01 and low_trend > 0.01: 
                patterns.append({
                    "type": "Symmetric Triangle" ,
                    "price_level": round((recent_highs[-1] + recent_lows[-1]) / 2,2),
                    "confidence": "Medium",
                    "additional_info": {
                        "high_trend" : round(high_trend, 4),
                        "low_trend" :  round(low_trend, 4)
                    }
                })
        return patterns
    @staticmethod
    def _detect_breakouts(full_df: pd.DataFrame, recent_df: pd.DataFrame) -> list[dict[str, Any]]:
        """Enhanced breakout detection."""
        patterns = []
        
        current_price = full_df["close"].iloc[-1]
        
        #20-day breakout
        high_20 = full_df["high"].iloc[-20:].max()
        low_20 = full_df["low"].iloc[-20:].min()
        
        #Volume confirmation
        avg_volume = full_df["volume"].iloc[-20:].mean()
        current_volume = full_df["volume"].iloc[-1]
        volume_surge = current_volume > avg_volume *1.5
        
        if current_price > high_20 * 0.999:
            confidence = "High" if volume_surge else "Medium"
            patterns.append({
                "type" : "Resistance Breakout",
                "price_level" : round(high_20, 2),
                "confidence" : confidence,
                "additional_info" :{
                    "volume_surge" : volume_surge,
                    "volume_ration" : round(current_volume / avg_volume,  2)
                }
            })
        if current_price < low_20 *1.001:
            confidence = "High" if volume_surge else "Medium"
            patterns.append({
                "type" : "Support Breakdown" , 
                "price_level" : round(low_20, 2),
                "confidence" : confidence,
                "additional_info" : {
                    "volume_surge" : volume_surge,
                    "volume_ratio" : round(current_volume / avg_volume, 2)
                    
                }
            })
        return patterns
class RiskAnalysis:
    """Tools for risk management and position sizing."""

    @staticmethod
    def calculate_position_size(
        price: float,
        stop_price: float,
        risk_amount: float,
        account_size: float,
        max_risk_percent: float = 2.0,
    ) -> dict[str, Any]:
        """
        Calculate appropriate position size based on risk parameters.

        Args:
            price (float): Current stock price
            stop_price (float): Stop loss price
            risk_amount (float): Amount willing to risk in dollars
            account_size (float): Total trading account size
            max_risk_percent (float): Maximum percentage of account to risk

        Returns:
            Dict[str, Any]: Position sizing recommendations
        """
        try:
            # Validate inputs
            if price <= 0 or account_size <= 0:
                raise ValueError("Price and account size must be positive")

            if price <= stop_price and stop_price != 0:
                raise ValueError("For long positions, stop price must be below entry price")

            # Calculate risk per share
            risk_per_share = abs(price - stop_price)

            if risk_per_share == 0:
                raise ValueError(
                    "Risk per share cannot be zero. Entry and stop prices must differ."
                )

            # Calculate position size based on dollar risk
            shares_based_on_risk = int(risk_amount / risk_per_share)

            # Calculate maximum position size based on account risk percentage
            max_risk_dollars = account_size * (max_risk_percent / 100)
            max_shares = int(max_risk_dollars / risk_per_share)

            # Take the smaller of the two
            recommended_shares = min(shares_based_on_risk, max_shares)
            actual_dollar_risk = recommended_shares * risk_per_share

            # Calculate position cost
            position_cost = recommended_shares * price

            # Calculate R-Multiples (potential reward to risk ratios)
            r1_target = price + risk_per_share
            r2_target = price + 2 * risk_per_share
            r3_target = price + 3 * risk_per_share

            return {
                "recommended_shares": recommended_shares,
                "dollar_risk": round(actual_dollar_risk, 2),
                "risk_per_share": round(risk_per_share, 2),
                "position_cost": round(position_cost, 2),
                "account_percent_risked": round((actual_dollar_risk / account_size) * 100, 2),
                "r_multiples": {
                    "r1": round(r1_target, 2),
                    "r2": round(r2_target, 2),
                    "r3": round(r3_target, 2),
                },
            }

        except Exception as e:
            raise Exception(f"Error calculating position size: {str(e)}") from e

    @staticmethod
    def suggest_stop_levels(df: pd.DataFrame) -> dict[str, float]:
        """
        Suggest appropriate stop-loss levels based on technical indicators.

        Args:
            df (pd.DataFrame): Historical price data with technical indicators

        Returns:
            Dict[str, float]: Suggested stop levels
        """
        try:
            if len(df) < 20:
                raise ValueError("Not enough data for stop level analysis")

            latest = df.iloc[-1]
            close = latest["close"]

            # Calculate ATR-based stops
            atr = latest.get("atr", df["high"].iloc[-20:] - df["low"].iloc[-20:]).mean()

            # Different stop strategies
            stops = {
                "atr_1x": round(close - 1 * atr, 2),
                "atr_2x": round(close - 2 * atr, 2),
                "atr_3x": round(close - 3 * atr, 2),
                "percent_2": round(close * 0.98, 2),
                "percent_5": round(close * 0.95, 2),
                "percent_8": round(close * 0.92, 2),
            }

            # Add SMA-based stops if available
            for sma in ["sma_20", "sma_50", "sma_200"]:
                if sma in latest and not pd.isna(latest[sma]):
                    stops[sma] = round(latest[sma], 2)
                    
            #Bollinger Band Based stops
            if "bb_lower" in latest and not pd.isna(latest["bb_lower"]):
                stops["bb_lower"] = round(latest["bb_lower"], 2)
            
            #VWAP based stop
            if "vwap" in latest and not pd.isna(latest["vwap"]):
                stops["vwap"] = round (latest["vwap"], 2)
                

            # Check for recent swing low
            recent_lows = df["low"].iloc[-20:].sort_values()
            if not recent_lows.empty:
                stops["recent_swing"] = round(recent_lows.iloc[0], 2)

            return stops

        except Exception as e:
            raise Exception(f"Error suggesting stop levels: {str(e)}") from e
