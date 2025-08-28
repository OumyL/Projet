"""
Système de screening avancé pour analyser plusieurs actifs simultanément.
Permet de filtrer et classer les actions/crypto selon des critères techniques.
"""

import asyncio
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)


class ScreeningCriteria(Enum):
    """Critères de screening disponibles."""
    
    # Critères de prix et volume
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    VOLUME_ABOVE_AVERAGE = "volume_above_average"
    PRICE_CHANGE_PERCENT = "price_change_percent"
    
    # Critères techniques
    RSI_RANGE = "rsi_range"
    ABOVE_SMA = "above_sma"
    BOLLINGER_POSITION = "bollinger_position"
    MACD_BULLISH = "macd_bullish"
    STOCH_OVERSOLD = "stoch_oversold"
    STOCH_OVERBOUGHT = "stoch_overbought"
    
    # Critères de momentum
    RELATIVE_STRENGTH = "relative_strength"
    PRICE_MOMENTUM = "price_momentum"
    VOLUME_MOMENTUM = "volume_momentum"
    
    # Patterns
    BREAKOUT_PATTERN = "breakout_pattern"
    SUPPORT_BOUNCE = "support_bounce"
    DOUBLE_BOTTOM = "double_bottom"
    
    # Critères composites
    BULLISH_SETUP = "bullish_setup"
    BEARISH_SETUP = "bearish_setup"
    CONSOLIDATION = "consolidation"


@dataclass
class ScreeningFilter:
    """Structure d'un filtre de screening."""
    criteria: ScreeningCriteria
    value: Union[float, int, bool, tuple]
    operator: str = ">="  
    weight: float = 1.0  # Poids du critère dans le score final


@dataclass
class ScreeningResult:
    """Résultat de screening pour un symbole."""
    symbol: str
    score: float
    current_price: float
    price_change_percent: float
    volume_ratio: float
    criteria_met: Dict[str, bool]
    technical_data: Dict[str, Any]
    signals: Dict[str, Any]
    rank: Optional[int] = None


class AssetScreener:
    """Moteur de screening multi-assets."""
    
    def __init__(self, market_data, tech_analysis):
        self.market_data = market_data
        self.tech_analysis = tech_analysis
        self.max_concurrent = 10  # Limite de requêtes simultanées
        
    async def screen_symbols(
        self,
        symbols: List[str],
        filters: List[ScreeningFilter],
        sort_by: str = "score",
        limit: int = 50
    ) -> List[ScreeningResult]:
        """
        Screener une liste de symboles selon les critères donnés.
        
        Args:
            symbols: Liste des symboles à analyser
            filters: Liste des filtres à appliquer
            sort_by: Critère de tri (score, price_change, volume_ratio)
            limit: Nombre maximum de résultats
        """
        logger.info(f"Starting screening of {len(symbols)} symbols with {len(filters)} filters")
        
        # Analyser les symboles par batches pour éviter la surcharge
        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def analyze_symbol(symbol: str) -> Optional[ScreeningResult]:
            async with semaphore:
                try:
                    return await self._analyze_single_symbol(symbol, filters)
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    return None
        
        # Exécuter l'analyse en parallèle
        tasks = [analyze_symbol(symbol) for symbol in symbols]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrer les résultats valides
        for result in raw_results:
            if isinstance(result, ScreeningResult):
                results.append(result)
        
        # Trier et limiter les résultats
        results = self._sort_results(results, sort_by)
        
        # Ajouter le rang
        for i, result in enumerate(results[:limit]):
            result.rank = i + 1
        
        logger.info(f"Screening completed: {len(results)} valid results")
        return results[:limit]
    
    async def _analyze_single_symbol(
        self, 
        symbol: str, 
        filters: List[ScreeningFilter]
    ) -> Optional[ScreeningResult]:
        """Analyser un symbole individuel."""
        try:
            # Récupérer les données historiques
            df = await self.market_data.get_historical_data(symbol, lookback_days=100)
            
            if df.empty or len(df) < 50:
                return None
            
            # Calculer les indicateurs techniques
            df = self.tech_analysis.add_core_indicators(df)
            trend_status = self.tech_analysis.check_trend_status(df)
            signals = self.tech_analysis.get_trading_signals(df)
            
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            # Données de base
            current_price = latest["close"]
            price_change = ((current_price - prev["close"]) / prev["close"]) * 100
            volume_ratio = latest["volume"] / latest["avg_20d_vol"] if latest["avg_20d_vol"] > 0 else 1
            
            # Évaluer chaque critère
            criteria_met = {}
            total_score = 0
            total_weight = 0
            
            for filter_item in filters:
                met, score_contribution = self._evaluate_criteria(
                    filter_item, df, trend_status, latest
                )
                criteria_met[filter_item.criteria.value] = met
                
                if met:
                    total_score += score_contribution * filter_item.weight
                total_weight += filter_item.weight
            
            # Score final normalisé
            final_score = (total_score / total_weight * 100) if total_weight > 0 else 0
            
            # Données techniques supplémentaires
            technical_data = {
                "rsi": trend_status.get("rsi", 50),
                "macd_bullish": trend_status.get("macd_bullish", False),
                "above_sma_20": trend_status.get("above_20sma", False),
                "above_sma_50": trend_status.get("above_50sma", False),
                "above_sma_200": trend_status.get("above_200sma", False),
                "bb_position": trend_status.get("bb_position", 0.5),
                "atr": latest.get("atr", 0),
                "volume_20d_avg": latest.get("avg_20d_vol", 0)
            }
            
            return ScreeningResult(
                symbol=symbol,
                score=round(final_score, 2),
                current_price=current_price,
                price_change_percent=round(price_change, 2),
                volume_ratio=round(volume_ratio, 2),
                criteria_met=criteria_met,
                technical_data=technical_data,
                signals=signals
            )
            
        except Exception as e:
            logger.error(f"Error in _analyze_single_symbol for {symbol}: {e}")
            return None
    
    def _evaluate_criteria(
        self, 
        filter_item: ScreeningFilter, 
        df: pd.DataFrame, 
        trend_status: Dict[str, Any], 
        latest: pd.Series
    ) -> tuple[bool, float]:
        """Évaluer un critère spécifique."""
        criteria = filter_item.criteria
        value = filter_item.value
        operator = filter_item.operator
        
        try:
            if criteria == ScreeningCriteria.PRICE_ABOVE:
                current_val = latest["close"]
                met = current_val >= value
                score = min(current_val / value, 2.0) if met else 0
                
            elif criteria == ScreeningCriteria.PRICE_BELOW:
                current_val = latest["close"]
                met = current_val <= value
                score = min(value / current_val, 2.0) if met else 0
                
            elif criteria == ScreeningCriteria.VOLUME_ABOVE_AVERAGE:
                volume_ratio = latest["volume"] / latest["avg_20d_vol"]
                met = volume_ratio >= value
                score = min(volume_ratio / value, 3.0) if met else 0
                
            elif criteria == ScreeningCriteria.PRICE_CHANGE_PERCENT:
                if len(df) > 1:
                    prev_close = df.iloc[-2]["close"]
                    change_pct = ((latest["close"] - prev_close) / prev_close) * 100
                    
                    if operator == ">=":
                        met = change_pct >= value
                        score = max(change_pct / value, 1.0) if met else 0
                    elif operator == "<=":
                        met = change_pct <= value
                        score = abs(value / change_pct) if met and change_pct != 0 else 0
                    else:
                        met = False
                        score = 0
                else:
                    met = False
                    score = 0
                    
            elif criteria == ScreeningCriteria.RSI_RANGE:
                rsi = trend_status.get("rsi", 50)
                if isinstance(value, tuple) and len(value) == 2:
                    met = value[0] <= rsi <= value[1]
                    # Score basé sur la proximité du centre de la range
                    center = (value[0] + value[1]) / 2
                    score = 1.0 - abs(rsi - center) / (value[1] - value[0]) if met else 0
                else:
                    met = False
                    score = 0
                    
            elif criteria == ScreeningCriteria.ABOVE_SMA:
                sma_key = f"above_{int(value)}sma"
                met = trend_status.get(sma_key, False)
                score = 1.0 if met else 0
                
            elif criteria == ScreeningCriteria.BOLLINGER_POSITION:
                bb_pos = trend_status.get("bb_position", 0.5)
                if operator == ">=":
                    met = bb_pos >= value
                    score = bb_pos if met else 0
                elif operator == "<=":
                    met = bb_pos <= value
                    score = (1 - bb_pos) if met else 0
                else:
                    met = False
                    score = 0
                    
            elif criteria == ScreeningCriteria.MACD_BULLISH:
                met = trend_status.get("macd_bullish", False)
                score = 1.0 if met else 0
                
            elif criteria == ScreeningCriteria.STOCH_OVERSOLD:
                stoch_k = trend_status.get("stoch_k", 50)
                met = stoch_k < value
                score = (value - stoch_k) / value if met else 0
                
            elif criteria == ScreeningCriteria.STOCH_OVERBOUGHT:
                stoch_k = trend_status.get("stoch_k", 50)
                met = stoch_k > value
                score = (stoch_k - value) / (100 - value) if met else 0
                
            elif criteria == ScreeningCriteria.RELATIVE_STRENGTH:
                # Score basé sur la performance relative 
                rs_score = self._calculate_simple_rs(df)
                met = rs_score >= value
                score = rs_score / 100 if met else 0
                
            elif criteria == ScreeningCriteria.PRICE_MOMENTUM:
                # Momentum sur N jours
                periods = int(value) if isinstance(value, (int, float)) else 20
                if len(df) > periods:
                    momentum = ((latest["close"] - df.iloc[-periods]["close"]) / df.iloc[-periods]["close"]) * 100
                    met = momentum > 0
                    score = max(momentum / 10, 1.0) if met else 0
                else:
                    met = False
                    score = 0
                    
            elif criteria == ScreeningCriteria.VOLUME_MOMENTUM:
                # Volume momentum (volume récent vs historique)
                recent_vol = df["volume"].tail(5).mean()
                historical_vol = df["volume"].tail(20).mean()
                vol_momentum = recent_vol / historical_vol if historical_vol > 0 else 1
                met = vol_momentum >= value
                score = min(vol_momentum, 3.0) if met else 0
                
            elif criteria == ScreeningCriteria.BREAKOUT_PATTERN:
                # Détection de cassure simple
                high_20 = df["high"].tail(20).max()
                current_price = latest["close"]
                met = current_price > high_20 * 0.999
                score = (current_price / high_20) if met else 0
                
            elif criteria == ScreeningCriteria.SUPPORT_BOUNCE:
                # Rebond sur support (prix près du bas récent)
                low_20 = df["low"].tail(20).min()
                current_price = latest["close"]
                distance_from_low = (current_price - low_20) / low_20
                met = distance_from_low <= value  # value = % max du low
                score = (value - distance_from_low) / value if met else 0
                
            elif criteria == ScreeningCriteria.BULLISH_SETUP:
                # Setup haussier composite
                conditions = [
                    trend_status.get("above_20sma", False),
                    trend_status.get("rsi", 50) > 30,
                    trend_status.get("rsi", 50) < 70,
                    latest["volume"] > latest["avg_20d_vol"] * 1.2
                ]
                met_count = sum(conditions)
                met = met_count >= 3
                score = met_count / 4
                
            elif criteria == ScreeningCriteria.BEARISH_SETUP:
                # Setup baissier composite
                conditions = [
                    not trend_status.get("above_20sma", True),
                    trend_status.get("rsi", 50) < 70,
                    trend_status.get("rsi", 50) > 30,
                    not trend_status.get("macd_bullish", True)
                ]
                met_count = sum(conditions)
                met = met_count >= 3
                score = met_count / 4
                
            elif criteria == ScreeningCriteria.CONSOLIDATION:
                # Détection de consolidation (faible volatilité)
                atr = latest.get("atr", 0)
                price = latest["close"]
                atr_percent = (atr / price) * 100 if price > 0 else 0
                met = atr_percent <= value  # value = ATR % max
                score = (value - atr_percent) / value if met else 0
                
            else:
                met = False
                score = 0
                
            return met, max(score, 0)
            
        except Exception as e:
            logger.error(f"Error evaluating criteria {criteria}: {e}")
            return False, 0
    
    def _calculate_simple_rs(self, df: pd.DataFrame) -> float:
        """Calcul simple de force relative (0-100)."""
        if len(df) < 63:
            return 50
        
        # Performance sur 3 mois
        performance = ((df["close"].iloc[-1] - df["close"].iloc[-63]) / df["close"].iloc[-63]) * 100
        
        # Convertir en score 0-100 (50 = neutre)
        return min(max(50 + performance, 0), 100)
    
    def _sort_results(
        self, 
        results: List[ScreeningResult], 
        sort_by: str
    ) -> List[ScreeningResult]:
        """Trier les résultats selon le critère donné."""
        if sort_by == "score":
            return sorted(results, key=lambda x: x.score, reverse=True)
        elif sort_by == "price_change":
            return sorted(results, key=lambda x: x.price_change_percent, reverse=True)
        elif sort_by == "volume_ratio":
            return sorted(results, key=lambda x: x.volume_ratio, reverse=True)
        elif sort_by == "symbol":
            return sorted(results, key=lambda x: x.symbol)
        else:
            return results


class PrebuiltScreeners:
    """Screeners prédéfinis pour différentes stratégies."""
    
    @staticmethod
    def momentum_breakout() -> List[ScreeningFilter]:
        """Screener pour les cassures avec momentum."""
        return [
            ScreeningFilter(ScreeningCriteria.BREAKOUT_PATTERN, 1.0, ">=", 3.0),
            ScreeningFilter(ScreeningCriteria.VOLUME_ABOVE_AVERAGE, 1.5, ">=", 2.0),
            ScreeningFilter(ScreeningCriteria.RSI_RANGE, (50, 80), "between", 2.0),
            ScreeningFilter(ScreeningCriteria.ABOVE_SMA, 20, "==", 1.5),
            ScreeningFilter(ScreeningCriteria.PRICE_MOMENTUM, 20, ">=", 1.0)
        ]
    
    @staticmethod
    def oversold_bounce() -> List[ScreeningFilter]:
        """Screener pour les rebonds sur survendu."""
        return [
            ScreeningFilter(ScreeningCriteria.RSI_RANGE, (20, 35), "between", 3.0),
            ScreeningFilter(ScreeningCriteria.STOCH_OVERSOLD, 30, "<=", 2.0),
            ScreeningFilter(ScreeningCriteria.SUPPORT_BOUNCE, 0.05, "<=", 2.0),  # 5% du low
            ScreeningFilter(ScreeningCriteria.VOLUME_ABOVE_AVERAGE, 1.2, ">=", 1.0),
            ScreeningFilter(ScreeningCriteria.ABOVE_SMA, 200, "==", 1.5)  # Tendance LT haussière
        ]
    
    @staticmethod
    def high_momentum() -> List[ScreeningFilter]:
        """Screener pour les actions à fort momentum."""
        return [
            ScreeningFilter(ScreeningCriteria.PRICE_CHANGE_PERCENT, 2.0, ">=", 2.0),
            ScreeningFilter(ScreeningCriteria.VOLUME_ABOVE_AVERAGE, 2.0, ">=", 2.0),
            ScreeningFilter(ScreeningCriteria.RELATIVE_STRENGTH, 70, ">=", 2.0),
            ScreeningFilter(ScreeningCriteria.PRICE_MOMENTUM, 10, ">=", 1.5),
            ScreeningFilter(ScreeningCriteria.RSI_RANGE, (40, 85), "between", 1.0)
        ]
    
    @staticmethod
    def consolidation_setup() -> List[ScreeningFilter]:
        """Screener pour les setups de consolidation."""
        return [
            ScreeningFilter(ScreeningCriteria.CONSOLIDATION, 2.0, "<=", 3.0),  # ATR < 2%
            ScreeningFilter(ScreeningCriteria.ABOVE_SMA, 50, "==", 2.0),
            ScreeningFilter(ScreeningCriteria.RSI_RANGE, (40, 60), "between", 2.0),
            ScreeningFilter(ScreeningCriteria.BOLLINGER_POSITION, (0.3, 0.7), "between", 1.5),
            ScreeningFilter(ScreeningCriteria.VOLUME_MOMENTUM, 0.8, "<=", 1.0)  # Volume faible
        ]
    
    @staticmethod
    def gap_up_continuation() -> List[ScreeningFilter]:
        """Screener pour les continuations après gap up."""
        return [
            ScreeningFilter(ScreeningCriteria.PRICE_CHANGE_PERCENT, 3.0, ">=", 3.0),
            ScreeningFilter(ScreeningCriteria.VOLUME_ABOVE_AVERAGE, 2.5, ">=", 2.5),
            ScreeningFilter(ScreeningCriteria.RSI_RANGE, (55, 75), "between", 2.0),
            ScreeningFilter(ScreeningCriteria.BULLISH_SETUP, True, "==", 2.0),
            ScreeningFilter(ScreeningCriteria.ABOVE_SMA, 20, "==", 1.0)
        ]


class MultiAssetWatchlist:
    """Gestionnaire de watchlists avec screening automatique."""
    
    def __init__(self, screener: AssetScreener):
        self.screener = screener
        self.watchlists: Dict[str, Dict[str, Any]] = {}
    
    def create_watchlist(
        self, 
        name: str, 
        symbols: List[str], 
        filters: List[ScreeningFilter],
        auto_update_minutes: int = 60
    ):
        """Créer une nouvelle watchlist avec screening automatique."""
        self.watchlists[name] = {
            "symbols": symbols,
            "filters": filters,
            "auto_update_minutes": auto_update_minutes,
            "last_update": None,
            "results": [],
            "task": None
        }
        
        logger.info(f"Created watchlist '{name}' with {len(symbols)} symbols")
    
    async def update_watchlist(self, name: str) -> List[ScreeningResult]:
        """Mettre à jour une watchlist spécifique."""
        if name not in self.watchlists:
            raise ValueError(f"Watchlist '{name}' not found")
        
        watchlist = self.watchlists[name]
        
        logger.info(f"Updating watchlist '{name}'...")
        results = await self.screener.screen_symbols(
            watchlist["symbols"],
            watchlist["filters"],
            limit=len(watchlist["symbols"])
        )
        
        watchlist["results"] = results
        watchlist["last_update"] = pd.Timestamp.now()
        
        return results
    
    def get_watchlist_summary(self, name: str) -> Dict[str, Any]:
        """Obtenir un résumé d'une watchlist."""
        if name not in self.watchlists:
            return {}
        
        watchlist = self.watchlists[name]
        results = watchlist["results"]
        
        if not results:
            return {
                "name": name,
                "symbol_count": len(watchlist["symbols"]),
                "last_update": None,
                "top_performers": []
            }
        
        # Top 5 performers
        top_5 = sorted(results, key=lambda x: x.score, reverse=True)[:5]
        
        return {
            "name": name,
            "symbol_count": len(watchlist["symbols"]),
            "results_count": len(results),
            "last_update": watchlist["last_update"],
            "avg_score": round(np.mean([r.score for r in results]), 2),
            "top_performers": [
                {
                    "symbol": r.symbol,
                    "score": r.score,
                    "price_change": r.price_change_percent
                }
                for r in top_5
            ]
        }
    
    def start_auto_updates(self):
        """Démarrer les mises à jour automatiques pour toutes les watchlists."""
        for name, watchlist in self.watchlists.items():
            if watchlist["auto_update_minutes"] > 0:
                task = asyncio.create_task(
                    self._auto_update_loop(name, watchlist["auto_update_minutes"])
                )
                watchlist["task"] = task
        
        logger.info("Started auto-updates for all watchlists")
    
    async def _auto_update_loop(self, name: str, interval_minutes: int):
        """Boucle de mise à jour automatique."""
        while name in self.watchlists:
            try:
                await self.update_watchlist(name)
                await asyncio.sleep(interval_minutes * 60)
            except Exception as e:
                logger.error(f"Error in auto-update for watchlist '{name}': {e}")
                await asyncio.sleep(300)  # Attendre 5 minutes en cas d'erreur


# Listes de symboles prédéfinies
class SymbolLists:
    """Listes de symboles prédéfinies pour le screening."""
    
    SP500_TOP_50 = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 'UNH', 'JNJ',
        'JPM', 'V', 'PG', 'HD', 'MA', 'CVX', 'ABBV', 'PFE', 'LLY', 'BAC',
        'KO', 'AVGO', 'PEP', 'TMO', 'COST', 'WMT', 'DIS', 'ABT', 'DHR', 'VZ',
        'ADBE', 'CMCSA', 'NKE', 'NFLX', 'CRM', 'ACN', 'TXN', 'LIN', 'PM', 'NEE',
        'UPS', 'RTX', 'T', 'BMY', 'SPGI', 'LOW', 'QCOM', 'HON', 'UNP', 'INTU'
    ]
    
    CRYPTO_MAJOR = [
        'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'MATIC', 'SOL', 'DOT', 'AVAX',
        'SHIB', 'TRX', 'WBTC', 'LEO', 'DAI', 'LTC', 'LINK', 'ATOM', 'UNI', 'ETC'
    ]
    
    TECH_STOCKS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX', 'ADBE', 'CRM',
        'ORCL', 'INTC', 'AMD', 'PYPL', 'SHOP', 'SNOW', 'PLTR', 'ZM', 'DOCU', 'UBER'
    ]
    
    GROWTH_STOCKS = [
        'TSLA', 'NVDA', 'AMD', 'PLTR', 'SNOW', 'CRWD', 'ZS', 'OKTA', 'DDOG', 'NET',
        'ROKU', 'SQ', 'SHOP', 'TWLO', 'ZOOM', 'PELOTON', 'RBLX', 'U', 'COIN', 'RIVN'
    ]


# Fonction utilitaire pour créer des screeners rapidement
def create_quick_screener(
    screener_type: str,
    symbols: List[str] = None,
    custom_filters: List[ScreeningFilter] = None
) -> tuple[List[str], List[ScreeningFilter]]:
    """
    Créer rapidement un screener avec des paramètres prédéfinis.
    
    Args:
        screener_type: Type de screener ('momentum', 'oversold', 'high_momentum', etc.)
        symbols: Liste de symboles (par défaut: SP500_TOP_50)
        custom_filters: Filtres personnalisés (remplacent les prédéfinis)
    
    Returns:
        Tuple (symboles, filtres)
    """
    if symbols is None:
        symbols = SymbolLists.SP500_TOP_50
    
    if custom_filters is not None:
        return symbols, custom_filters
    
    screener_map = {
        'momentum': PrebuiltScreeners.momentum_breakout(),
        'oversold': PrebuiltScreeners.oversold_bounce(),
        'high_momentum': PrebuiltScreeners.high_momentum(),
        'consolidation': PrebuiltScreeners.consolidation_setup(),
        'gap_up': PrebuiltScreeners.gap_up_continuation()
    }
    
    filters = screener_map.get(screener_type, PrebuiltScreeners.momentum_breakout())
    return symbols, filters