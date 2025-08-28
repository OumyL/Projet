"""
Système de backtesting simple pour tester des stratégies de trading.
Permet de simuler des stratégies sur données historiques.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Types d'ordres supportés."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    """Côté de l'ordre."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Statut de l'ordre."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Représente un ordre de trading."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    commission: float = 0.0


@dataclass
class Position:
    """Représente une position ouverte."""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def market_value(self) -> float:
        """Valeur de marché de la position."""
        return self.quantity * self.current_price
    
    @property
    def is_long(self) -> bool:
        """Position longue."""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Position courte."""
        return self.quantity < 0


@dataclass
class Trade:
    """Représente un trade complet (entrée + sortie)."""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # "long" or "short"
    pnl: float
    pnl_percent: float
    commission: float
    duration_days: float
    
    @property
    def is_winner(self) -> bool:
        """Trade gagnant."""
        return self.pnl > 0


@dataclass
class BacktestResult:
    """Résultats du backtest."""
    # Métriques de performance
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    
    # Statistiques des trades
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float
    
    # Courbes de performance
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    
    # Détails des trades
    trades: List[Trade]
    
    # Métriques supplémentaires
    initial_capital: float
    final_capital: float
    total_commission: float
    total_trades_value: float


class TradingStrategy:
    """Classe de base pour les stratégies de trading."""
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Générer les signaux de trading.
        
        Args:
            data: DataFrame avec OHLCV et indicateurs
            
        Returns:
            DataFrame avec colonnes 'signal' (1=buy, -1=sell, 0=hold)
        """
        raise NotImplementedError("Must implement generate_signals method")
    
    def set_parameters(self, **kwargs):
        """Définir les paramètres de la stratégie."""
        self.parameters.update(kwargs)


class MovingAverageCrossStrategy(TradingStrategy):
    """Stratégie de croisement de moyennes mobiles."""
    
    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__("MA Cross")
        self.parameters = {
            "fast_period": fast_period,
            "slow_period": slow_period
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Générer signaux basés sur croisement de moyennes."""
        fast_period = self.parameters["fast_period"]
        slow_period = self.parameters["slow_period"]
        
        # Calculer les moyennes mobiles
        data[f"sma_{fast_period}"] = data["close"].rolling(fast_period).mean()
        data[f"sma_{slow_period}"] = data["close"].rolling(slow_period).mean()
        
        # Générer les signaux
        data["signal"] = 0.0
        
        # Signal d'achat: moyenne rapide croise au-dessus de la lente
        data.loc[
            (data[f"sma_{fast_period}"] > data[f"sma_{slow_period}"]) &
            (data[f"sma_{fast_period}"].shift(1) <= data[f"sma_{slow_period}"].shift(1)),
            "signal"
        ] = 1.0
        
        # Signal de vente: moyenne rapide croise en-dessous de la lente
        data.loc[
            (data[f"sma_{fast_period}"] < data[f"sma_{slow_period}"]) &
            (data[f"sma_{fast_period}"].shift(1) >= data[f"sma_{slow_period}"].shift(1)),
            "signal"
        ] = -1.0
        
        return data


class RSIStrategy(TradingStrategy):
    """Stratégie basée sur le RSI."""
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__("RSI Strategy")
        self.parameters = {
            "rsi_period": rsi_period,
            "oversold": oversold,
            "overbought": overbought
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Générer signaux basés sur RSI."""
        oversold = self.parameters["oversold"]
        overbought = self.parameters["overbought"]
        
        # Le RSI doit déjà être calculé dans les données
        if "rsi" not in data.columns:
            # Calcul simple du RSI si pas présent
            delta = data["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data["rsi"] = 100 - (100 / (1 + rs))
        
        data["signal"] = 0.0
        
        # Signal d'achat: RSI sort de la zone de survente
        data.loc[
            (data["rsi"] > oversold) & (data["rsi"].shift(1) <= oversold),
            "signal"
        ] = 1.0
        
        # Signal de vente: RSI entre en zone de surachat
        data.loc[
            (data["rsi"] > overbought),
            "signal"
        ] = -1.0
        
        return data


class BollingerBandsStrategy(TradingStrategy):
    """Stratégie basée sur les Bollinger Bands."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__("Bollinger Bands")
        self.parameters = {
            "period": period,
            "std_dev": std_dev
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Générer signaux basés sur Bollinger Bands."""
        period = self.parameters["period"]
        std_dev = self.parameters["std_dev"]
        
        # Calculer Bollinger Bands si pas présentes
        if "bb_upper" not in data.columns:
            rolling_mean = data["close"].rolling(period).mean()
            rolling_std = data["close"].rolling(period).std()
            data["bb_upper"] = rolling_mean + (rolling_std * std_dev)
            data["bb_lower"] = rolling_mean - (rolling_std * std_dev)
            data["bb_middle"] = rolling_mean
        
        data["signal"] = 0.0
        
        # Signal d'achat: prix touche la bande inférieure
        data.loc[
            (data["close"] <= data["bb_lower"]) & (data["close"].shift(1) > data["bb_lower"].shift(1)),
            "signal"
        ] = 1.0
        
        # Signal de vente: prix touche la bande supérieure
        data.loc[
            (data["close"] >= data["bb_upper"]) & (data["close"].shift(1) < data["bb_upper"].shift(1)),
            "signal"
        ] = -1.0
        
        return data


class BacktestEngine:
    """Moteur de backtesting principal."""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,  # 0.1%
        slippage_rate: float = 0.0005    # 0.05%
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        # État du portefeuille
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        
        # Historique de performance
        self.equity_history: List[Tuple[datetime, float]] = []
        self.order_id_counter = 0
    
    def reset(self):
        """Réinitialiser l'état du backtesting."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.orders.clear()
        self.trades.clear()
        self.equity_history.clear()
        self.order_id_counter = 0
    
    def get_portfolio_value(self) -> float:
        """Calculer la valeur totale du portefeuille."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> str:
        """Placer un ordre."""
        order_id = f"order_{self.order_id_counter}"
        self.order_id_counter += 1
        
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )
        
        self.orders.append(order)
        return order_id
    
    def process_orders(self, current_data: pd.Series):
        """Traiter les ordres en attente."""
        current_price = current_data["close"]
        current_time = current_data.name
        
        for order in self.orders:
            if order.status != OrderStatus.PENDING:
                continue
            
            if order.symbol != current_data.get("symbol", ""):
                continue
            
            should_fill = False
            fill_price = current_price
            
            if order.order_type == OrderType.MARKET:
                should_fill = True
                # Appliquer le slippage
                if order.side == OrderSide.BUY:
                    fill_price *= (1 + self.slippage_rate)
                else:
                    fill_price *= (1 - self.slippage_rate)
            
            elif order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and current_price <= order.price:
                    should_fill = True
                    fill_price = order.price
                elif order.side == OrderSide.SELL and current_price >= order.price:
                    should_fill = True
                    fill_price = order.price
            
            elif order.order_type == OrderType.STOP_LOSS:
                if order.side == OrderSide.SELL and current_price <= order.stop_price:
                    should_fill = True
                    fill_price = current_price
            
            if should_fill:
                self._execute_order(order, fill_price, current_time)
    
    def _execute_order(self, order: Order, fill_price: float, fill_time: datetime):
        """Exécuter un ordre."""
        total_value = order.quantity * fill_price
        commission = total_value * self.commission_rate
        
        if order.side == OrderSide.BUY:
            # Vérifier si on a assez de cash
            total_cost = total_value + commission
            if self.cash < total_cost:
                order.status = OrderStatus.REJECTED
                return
            
            # Exécuter l'achat
            self.cash -= total_cost
            self._add_to_position(order.symbol, order.quantity, fill_price)
            
        else:  # SELL
            # Vérifier si on a la position
            if order.symbol not in self.positions:
                order.status = OrderStatus.REJECTED
                return
            
            position = self.positions[order.symbol]
            if position.quantity < order.quantity:
                order.status = OrderStatus.REJECTED
                return
            
            # Exécuter la vente
            self.cash += total_value - commission
            old_position = self._remove_from_position(order.symbol, order.quantity, fill_price)
            
            # Enregistrer le trade si position fermée
            if old_position:
                self._record_trade(old_position, order.quantity, fill_price, fill_time, commission)
        
        # Marquer l'ordre comme exécuté
        order.status = OrderStatus.FILLED
        order.filled_at = fill_time
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.commission = commission
    
    def _add_to_position(self, symbol: str, quantity: float, price: float):
        """Ajouter à une position."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_quantity = pos.quantity + quantity
            total_cost = (pos.quantity * pos.avg_price) + (quantity * price)
            pos.avg_price = total_cost / total_quantity
            pos.quantity = total_quantity
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=price
            )
    
    def _remove_from_position(self, symbol: str, quantity: float, price: float) -> Optional[Position]:
        """Retirer d'une position."""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        old_position = Position(
            symbol=position.symbol,
            quantity=quantity,
            avg_price=position.avg_price
        )
        
        position.quantity -= quantity
        
        if position.quantity <= 0:
            del self.positions[symbol]
        
        return old_position
    
    def _record_trade(
        self,
        position: Position,
        quantity: float,
        exit_price: float,
        exit_time: datetime,
        commission: float
    ):
        """Enregistrer un trade complet."""
        entry_price = position.avg_price
        
        # Calculer P&L
        if quantity > 0:  # Long trade
            pnl = (exit_price - entry_price) * quantity - commission
        else:  # Short trade
            pnl = (entry_price - exit_price) * abs(quantity) - commission
        
        pnl_percent = (pnl / (entry_price * quantity)) * 100
        
        # Estimer la date d'entrée (simplification)
        entry_time = exit_time - timedelta(days=30)  # Approximation
        
        trade = Trade(
            symbol=position.symbol,
            entry_date=entry_time,
            exit_date=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            side="long" if quantity > 0 else "short",
            pnl=pnl,
            pnl_percent=pnl_percent,
            commission=commission,
            duration_days=(exit_time - entry_time).days
        )
        
        self.trades.append(trade)
    
    def update_positions(self, market_data: Dict[str, float]):
        """Mettre à jour les positions avec les prix actuels."""
        for symbol, position in self.positions.items():
            if symbol in market_data:
                position.current_price = market_data[symbol]
                position.unrealized_pnl = (
                    position.current_price - position.avg_price
                ) * position.quantity
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy: TradingStrategy,
        symbol: str = "BACKTEST"
    ) -> BacktestResult:
        """Exécuter le backtest complet."""
        self.reset()
        
        # Ajouter le symbole aux données
        data = data.copy()
        data["symbol"] = symbol
        
        # Générer les signaux
        logger.info(f"Generating signals with {strategy.name} strategy...")
        data = strategy.generate_signals(data)
        
        # Simuler le trading
        logger.info("Running backtest simulation...")
        position_size_pct = 0.1  # 10% du capital par trade
        
        for i, (date, row) in enumerate(data.iterrows()):
            # Mettre à jour les positions
            self.update_positions({symbol: row["close"]})
            
            # Traiter les ordres en attente
            self.process_orders(row)
            
            # Gérer les signaux
            if "signal" in row and not np.isnan(row["signal"]):
                signal = row["signal"]
                current_price = row["close"]
                
                if signal == 1.0:  # Signal d'achat
                    if symbol not in self.positions:
                        # Calculer la taille de position
                        position_value = self.cash * position_size_pct
                        quantity = position_value / current_price
                        
                        self.place_order(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            quantity=quantity,
                            order_type=OrderType.MARKET
                        )
                
                elif signal == -1.0:  # Signal de vente
                    if symbol in self.positions:
                        position = self.positions[symbol]
                        self.place_order(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            quantity=position.quantity,
                            order_type=OrderType.MARKET
                        )
            
            # Enregistrer l'équité
            portfolio_value = self.get_portfolio_value()
            self.equity_history.append((date, portfolio_value))
        
        # Calculer les résultats
        return self._calculate_results(data)
    
    def _calculate_results(self, data: pd.DataFrame) -> BacktestResult:
        """Calculer les métriques de performance."""
        if not self.equity_history:
            raise ValueError("No equity history to analyze")
        
        # Convertir l'historique en Series
        dates, values = zip(*self.equity_history)
        equity_curve = pd.Series(values, index=dates)
        
        # Calculer les returns
        returns = equity_curve.pct_change().dropna()
        
        # Métriques de base
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        
        # Rendement annualisé
        days_total = (equity_curve.index[-1] - equity_curve.index[0]).days
        annual_return = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (365 / days_total) - 1) * 100
        
        # Maximum Drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # Sharpe Ratio (approximation)
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        # Sortino Ratio
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            sortino_ratio = (returns.mean() * 252) / (negative_returns.std() * np.sqrt(252))
        else:
            sortino_ratio = 0
        
        # Statistiques des trades
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.is_winner])
        losing_trades = total_trades - winning_trades
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # P&L moyens
        if winning_trades > 0:
            avg_win = np.mean([t.pnl for t in self.trades if t.is_winner])
            largest_win = max([t.pnl for t in self.trades if t.is_winner])
        else:
            avg_win = 0
            largest_win = 0
        
        if losing_trades > 0:
            avg_loss = np.mean([t.pnl for t in self.trades if not t.is_winner])
            largest_loss = min([t.pnl for t in self.trades if not t.is_winner])
        else:
            avg_loss = 0
            largest_loss = 0
        
        # Profit Factor
        gross_profit = sum([t.pnl for t in self.trades if t.is_winner])
        gross_loss = abs(sum([t.pnl for t in self.trades if not t.is_winner]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Durée moyenne des trades
        if self.trades:
            avg_trade_duration = np.mean([t.duration_days for t in self.trades])
        else:
            avg_trade_duration = 0
        
        # Commission totale
        total_commission = sum([order.commission for order in self.orders if order.status == OrderStatus.FILLED])
        
        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration=avg_trade_duration,
            equity_curve=equity_curve,
            drawdown_curve=drawdown * 100,
            trades=self.trades,
            initial_capital=self.initial_capital,
            final_capital=equity_curve.iloc[-1],
            total_commission=total_commission,
            total_trades_value=sum([order.filled_quantity * order.filled_price for order in self.orders if order.status == OrderStatus.FILLED])
        )


class StrategyOptimizer:
    """Optimiseur de paramètres de stratégie."""
    
    def __init__(self, backtest_engine: BacktestEngine):
        self.engine = backtest_engine
    
    def optimize_strategy(
        self,
        data: pd.DataFrame,
        strategy_class: type,
        parameter_ranges: Dict[str, List],
        optimization_metric: str = "sharpe_ratio",
        symbol: str = "BACKTEST"
    ) -> Tuple[Dict, BacktestResult]:
        """
        Optimiser les paramètres d'une stratégie.
        
        Args:
            data: Données historiques
            strategy_class: Classe de stratégie à optimiser
            parameter_ranges: Ranges de paramètres à tester
            optimization_metric: Métrique à optimiser
            symbol: Symbole pour le backtest
            
        Returns:
            Tuple (meilleurs paramètres, meilleur résultat)
        """
        best_params = None
        best_result = None
        best_metric_value = float('-inf')
        
        # Générer toutes les combinaisons de paramètres
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        from itertools import product
        
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
        
        logger.info(f"Testing {total_combinations} parameter combinations...")
        
        for i, param_combo in enumerate(product(*param_values)):
            params = dict(zip(param_names, param_combo))
            
            try:
                # Créer la stratégie avec ces paramètres
                if strategy_class == MovingAverageCrossStrategy:
                    strategy = strategy_class(
                        fast_period=params.get("fast_period", 20),
                        slow_period=params.get("slow_period", 50)
                    )
                elif strategy_class == RSIStrategy:
                    strategy = strategy_class(
                        rsi_period=params.get("rsi_period", 14),
                        oversold=params.get("oversold", 30),
                        overbought=params.get("overbought", 70)
                    )
                else:
                    strategy = strategy_class()
                    strategy.set_parameters(**params)
                
                # Exécuter le backtest
                result = self.engine.run_backtest(data, strategy, symbol)
                
                # Évaluer la métrique
                metric_value = getattr(result, optimization_metric)
                
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_params = params.copy()
                    best_result = result
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Tested {i + 1}/{total_combinations} combinations...")
                    
            except Exception as e:
                logger.error(f"Error testing parameters {params}: {e}")
                continue
        
        logger.info(f"Optimization complete. Best {optimization_metric}: {best_metric_value:.4f}")
        return best_params, best_result


# Fonctions helper pour usage facile
def run_simple_backtest(
    data: pd.DataFrame,
    strategy_name: str = "ma_cross",
    **strategy_params
) -> BacktestResult:
    """Exécuter un backtest simple."""
    
    # Créer la stratégie
    if strategy_name == "ma_cross":
        strategy = MovingAverageCrossStrategy(
            fast_period=strategy_params.get("fast_period", 20),
            slow_period=strategy_params.get("slow_period", 50)
        )
    elif strategy_name == "rsi":
        strategy = RSIStrategy(
            rsi_period=strategy_params.get("rsi_period", 14),
            oversold=strategy_params.get("oversold", 30),
            overbought=strategy_params.get("overbought", 70)
        )
    elif strategy_name == "bollinger":
        strategy = BollingerBandsStrategy(
            period=strategy_params.get("period", 20),
            std_dev=strategy_params.get("std_dev", 2.0)
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Exécuter le backtest
    engine = BacktestEngine(
        initial_capital=strategy_params.get("initial_capital", 100000),
        commission_rate=strategy_params.get("commission_rate", 0.001)
    )
    
    return engine.run_backtest(data, strategy)


def format_backtest_results(result: BacktestResult) -> str:
    """Formater les résultats de backtest."""
    
    return f"""
📊 **Backtest Results Summary**

**💰 Performance:**
- Total Return: {result.total_return:+.2f}%
- Annual Return: {result.annual_return:+.2f}%
- Final Capital: ${result.final_capital:,.2f}
- Max Drawdown: {result.max_drawdown:.2f}%

**📈 Risk Metrics:**
- Sharpe Ratio: {result.sharpe_ratio:.3f}
- Sortino Ratio: {result.sortino_ratio:.3f}

**🎯 Trading Statistics:**
- Total Trades: {result.total_trades}
- Win Rate: {result.win_rate:.1f}%
- Profit Factor: {result.profit_factor:.2f}

**💹 Trade Analysis:**
- Winning Trades: {result.winning_trades}
- Losing Trades: {result.losing_trades}
- Average Win: ${result.avg_win:.2f}
- Average Loss: ${result.avg_loss:.2f}
- Largest Win: ${result.largest_win:.2f}
- Largest Loss: ${result.largest_loss:.2f}

**⏱️ Other Metrics:**
- Avg Trade Duration: {result.avg_trade_duration:.1f} days
- Total Commission: ${result.total_commission:.2f}
- Total Trade Value: ${result.total_trades_value:,.2f}

**🎯 Strategy Assessment:**
"""
    
    # Ajouter une évaluation qualitative
    if result.sharpe_ratio > 1.5:
        return result + "✅ Excellent strategy with strong risk-adjusted returns"
    elif result.sharpe_ratio > 1.0:
        return result + "✅ Good strategy with decent risk-adjusted returns"
    elif result.sharpe_ratio > 0.5:
        return result + "⚖️ Moderate strategy, consider improvements"
    else:
        return result + "❌ Poor strategy, significant improvements needed"