"""
Module de prédiction de prix avec Machine Learning.
Utilise plusieurs modèles ML pour prédire les mouvements de prix à court terme.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import warnings
warnings.filterwarnings('ignore')


try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML libraries not available. Install scikit-learn for price prediction functionality.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)


class PredictionHorizon(Enum):
    """Horizons de prédiction supportés."""
    NEXT_DAY = "1d"
    NEXT_WEEK = "7d"
    NEXT_MONTH = "30d"
    NEXT_QUARTER = "90d"


class ModelType(Enum):
    """Types de modèles ML supportés."""
    LINEAR_REGRESSION = "linear"
    RANDOM_FOREST = "rf"
    GRADIENT_BOOSTING = "gb"
    XGBOOST = "xgb"
    ENSEMBLE = "ensemble"


@dataclass
class PredictionResult:
    """Résultat de prédiction."""
    symbol: str
    current_price: float
    predicted_price: float
    prediction_change: float  # %
    confidence: float  # 0-1
    horizon: PredictionHorizon
    model_used: str
    prediction_date: datetime
    features_used: List[str]
    
    # Métadonnées du modèle
    model_accuracy: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None


@dataclass
class ModelPerformance:
    """Performance d'un modèle ML."""
    model_name: str
    mse: float
    mae: float
    r2_score: float
    accuracy_direction: float  
    training_samples: int
    test_samples: int
    cross_val_score: Optional[float] = None


class FeatureEngineer:
    """Ingénieur de features pour la prédiction de prix."""
    
    @staticmethod
    def create_technical_features(data: pd.DataFrame) -> pd.DataFrame:
        """Créer des features techniques."""
        df = data.copy()
        
        # Prix features
        df["price_change"] = df["close"].pct_change()
        df["price_change_2d"] = df["close"].pct_change(2)
        df["price_change_5d"] = df["close"].pct_change(5)
        
        # Volatilité features
        df["volatility_5d"] = df["price_change"].rolling(5).std()
        df["volatility_20d"] = df["price_change"].rolling(20).std()
        
        # Volume features
        df["volume_change"] = df["volume"].pct_change()
        df["volume_ma_5"] = df["volume"].rolling(5).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma_5"]
        
        # Price position features
        df["high_low_pct"] = (df["close"] - df["low"]) / (df["high"] - df["low"])
        df["close_open_pct"] = (df["close"] - df["open"]) / df["open"]
        
        return df
    
    @staticmethod
    def create_momentum_features(data: pd.DataFrame) -> pd.DataFrame:
        """Créer des features de momentum."""
        df = data.copy()
        
        # RSI-like momentum
        df["rsi_14"] = FeatureEngineer._calculate_rsi(df["close"], 14)
        df["rsi_7"] = FeatureEngineer._calculate_rsi(df["close"], 7)
        
        # MACD-like features
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        
        # Stochastic-like features
        df["stoch_k"] = FeatureEngineer._calculate_stochastic(df, 14)
        
        # Price momentum
        df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
        df["momentum_10"] = df["close"] / df["close"].shift(10) - 1
        df["momentum_20"] = df["close"] / df["close"].shift(20) - 1
        
        return df
    
    @staticmethod
    def create_trend_features(data: pd.DataFrame) -> pd.DataFrame:
        """Créer des features de tendance."""
        df = data.copy()
        
        # Moving averages
        df["sma_5"] = df["close"].rolling(5).mean()
        df["sma_10"] = df["close"].rolling(10).mean()
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        
        # Price vs MA
        df["price_vs_sma5"] = df["close"] / df["sma_5"] - 1
        df["price_vs_sma10"] = df["close"] / df["sma_10"] - 1
        df["price_vs_sma20"] = df["close"] / df["sma_20"] - 1
        df["price_vs_sma50"] = df["close"] / df["sma_50"] - 1
        
        # MA slopes 
        df["sma5_slope"] = df["sma_5"].pct_change(5)
        df["sma20_slope"] = df["sma_20"].pct_change(5)
        
        # Bollinger-like features
        df["bb_upper"] = df["sma_20"] + (df["close"].rolling(20).std() * 2)
        df["bb_lower"] = df["sma_20"] - (df["close"].rolling(20).std() * 2)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["sma_20"]
        
        return df
    
    @staticmethod
    def create_lag_features(data: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Créer des features de lag."""
        df = data.copy()
        
        for lag in lags:
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
            df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
            df[f"change_lag_{lag}"] = df["price_change"].shift(lag)
        
        return df
    
    @staticmethod
    def create_all_features(data: pd.DataFrame) -> pd.DataFrame:
        """Créer toutes les features."""
        df = data.copy()
        
        # Ajouter toutes les catégories de features
        df = FeatureEngineer.create_technical_features(df)
        df = FeatureEngineer.create_momentum_features(df)
        df = FeatureEngineer.create_trend_features(df)
        df = FeatureEngineer.create_lag_features(df)
        
        # Features temporelles
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculer RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calculate_stochastic(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculer Stochastic %K."""
        low_min = data["low"].rolling(period).min()
        high_max = data["high"].rolling(period).max()
        return 100 * (data["close"] - low_min) / (high_max - low_min)


class PricePredictionModel:
    """Modèle de prédiction de prix."""
    
    def __init__(
        self,
        model_type: ModelType = ModelType.RANDOM_FOREST,
        prediction_horizon: PredictionHorizon = PredictionHorizon.NEXT_DAY
    ):
        if not ML_AVAILABLE:
            raise ImportError("ML libraries not available. Install scikit-learn.")
        
        self.model_type = model_type
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False
        self.performance_metrics = None
        
        # Initialiser le modèle
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialiser le modèle ML."""
        if self.model_type == ModelType.LINEAR_REGRESSION:
            self.model = LinearRegression()
            self.scaler = StandardScaler()
        elif self.model_type == ModelType.RANDOM_FOREST:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.scaler = None  # RF doesn't need scaling
        elif self.model_type == ModelType.GRADIENT_BOOSTING:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            self.scaler = StandardScaler()
        elif self.model_type == ModelType.XGBOOST:
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available. Install xgboost.")
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Préparer les données pour l'entraînement."""
        # Créer toutes les features
        df = FeatureEngineer.create_all_features(data)
        
        # Créer la target variable selon l'horizon
        horizon_days = int(self.prediction_horizon.value.replace('d', ''))
        df["target"] = df["close"].shift(-horizon_days)
        
        
        df = df.dropna()
        
        # Sélectionner les features 
        feature_cols = [col for col in df.columns if col not in [
            "open", "high", "low", "close", "volume", "target",
            "sma_5", "sma_10", "sma_20", "sma_50",  
            "bb_upper", "bb_lower"  
        ]]
        
        X = df[feature_cols]
        y = df["target"]
        
        self.feature_names = feature_cols
        return X, y
    
    def train(self, data: pd.DataFrame, test_size: float = 0.2) -> ModelPerformance:
        """Entraîner le modèle."""
        logger.info(f"Training {self.model_type.value} model for {self.prediction_horizon.value} prediction...")
        
        # Préparer les données
        X, y = self.prepare_data(data)
        
        if len(X) < 100:
            raise ValueError("Not enough data for training (minimum 100 samples required)")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False, random_state=42
        )
        
        # Scaling si nécessaire
        if self.scaler:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Entraîner le modèle
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Évaluer les performances
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculer les métriques
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Direction accuracy
        actual_direction = (y_test > X_test.iloc[:, -1]).astype(int)  # Prix actuel vs prédit
        pred_direction = (y_pred > X_test.iloc[:, -1]).astype(int)
        direction_accuracy = (actual_direction == pred_direction).mean() * 100
        
        # Cross-validation score
        if len(X_train) > 50:  
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='r2')
            cv_score = cv_scores.mean()
        else:
            cv_score = None
        
        self.performance_metrics = ModelPerformance(
            model_name=self.model_type.value,
            mse=mse,
            mae=mae,
            r2_score=r2,
            accuracy_direction=direction_accuracy,
            training_samples=len(X_train),
            test_samples=len(X_test),
            cross_val_score=cv_score
        )
        
        logger.info(f"Model trained. R² Score: {r2:.4f}, Direction Accuracy: {direction_accuracy:.1f}%")
        return self.performance_metrics
    
    def predict(self, data: pd.DataFrame, current_price: float) -> PredictionResult:
        """Faire une prédiction."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Préparer les features pour la dernière observation
        df = FeatureEngineer.create_all_features(data)
        
        # Prendre la dernière ligne 
        latest_features = df[self.feature_names].iloc[-1:].fillna(0)
        
        # Scaling si nécessaire
        if self.scaler:
            latest_features_scaled = self.scaler.transform(latest_features)
        else:
            latest_features_scaled = latest_features
        
        # Prédiction
        predicted_price = self.model.predict(latest_features_scaled)[0]
        
        # Calculer le changement en pourcentage
        prediction_change = ((predicted_price - current_price) / current_price) * 100
        
        # Calculer la confiance (basée sur la performance du modèle)
        confidence = min(max(self.performance_metrics.r2_score, 0), 1) if self.performance_metrics else 0.5
        
        # Feature importance
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            # Top 10 features les plus importantes
            feature_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return PredictionResult(
            symbol=data.get("symbol", "UNKNOWN"),
            current_price=current_price,
            predicted_price=predicted_price,
            prediction_change=prediction_change,
            confidence=confidence,
            horizon=self.prediction_horizon,
            model_used=self.model_type.value,
            prediction_date=datetime.now(),
            features_used=self.feature_names,
            model_accuracy=self.performance_metrics.r2_score if self.performance_metrics else None,
            feature_importance=feature_importance
        )
    
    def save_model(self, filepath: str):
        """Sauvegarder le modèle."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "model_type": self.model_type,
            "prediction_horizon": self.prediction_horizon,
            "feature_names": self.feature_names,
            "performance_metrics": self.performance_metrics
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> "PricePredictionModel":
        """Charger un modèle sauvegardé."""
        model_data = joblib.load(filepath)
        
        instance = cls(
            model_type=model_data["model_type"],
            prediction_horizon=model_data["prediction_horizon"]
        )
        
        instance.model = model_data["model"]
        instance.scaler = model_data["scaler"]
        instance.feature_names = model_data["feature_names"]
        instance.performance_metrics = model_data["performance_metrics"]
        instance.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
        return instance


class EnsemblePredictor:
    """Prédicteur ensemble combinant plusieurs modèles."""
    
    def __init__(self, prediction_horizon: PredictionHorizon = PredictionHorizon.NEXT_DAY):
        self.prediction_horizon = prediction_horizon
        self.models: Dict[ModelType, PricePredictionModel] = {}
        self.weights: Dict[ModelType, float] = {}
        self.is_trained = False
    
    def add_model(self, model_type: ModelType, weight: float = 1.0):
        """Ajouter un modèle à l'ensemble."""
        model = PricePredictionModel(model_type, self.prediction_horizon)
        self.models[model_type] = model
        self.weights[model_type] = weight
    
    def train_all(self, data: pd.DataFrame) -> Dict[ModelType, ModelPerformance]:
        """Entraîner tous les modèles."""
        performances = {}
        
        for model_type, model in self.models.items():
            try:
                logger.info(f"Training {model_type.value} model...")
                performance = model.train(data)
                performances[model_type] = performance
                
                # Ajuster le poids basé sur la performance
                self.weights[model_type] = max(performance.r2_score, 0.1)
                
            except Exception as e:
                logger.error(f"Error training {model_type.value}: {e}")
                # Retirer le modèle défaillant
                del self.models[model_type]
                del self.weights[model_type]
        
        self.is_trained = len(self.models) > 0
        return performances
    
    def predict(self, data: pd.DataFrame, current_price: float) -> PredictionResult:
        """Prédiction ensemble."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = []
        total_weight = 0
        
        for model_type, model in self.models.items():
            try:
                pred = model.predict(data, current_price)
                weight = self.weights[model_type]
                predictions.append((pred, weight))
                total_weight += weight
            except Exception as e:
                logger.error(f"Error in {model_type.value} prediction: {e}")
                continue
        
        if not predictions:
            raise ValueError("No models could make predictions")
        
        # Calculer la prédiction pondérée
        weighted_price = sum(pred.predicted_price * weight for pred, weight in predictions) / total_weight
        weighted_change = ((weighted_price - current_price) / current_price) * 100
        
        # Calculer la confiance moyenne pondérée
        weighted_confidence = sum(pred.confidence * weight for pred, weight in predictions) / total_weight
        
        # Collecter les features les plus importantes
        all_features = set()
        for pred, _ in predictions:
            if pred.feature_importance:
                all_features.update(pred.feature_importance.keys())
        
        return PredictionResult(
            symbol=data.get("symbol", "UNKNOWN"),
            current_price=current_price,
            predicted_price=weighted_price,
            prediction_change=weighted_change,
            confidence=weighted_confidence,
            horizon=self.prediction_horizon,
            model_used="ensemble",
            prediction_date=datetime.now(),
            features_used=list(all_features),
            feature_importance=None  
        )


class MarketRegimeDetector:
    """Détecteur de régime de marché pour ajuster les prédictions."""
    
    @staticmethod
    def detect_regime(data: pd.DataFrame) -> str:
        """Détecter le régime de marché actuel."""
        # Calcul de volatilité récente vs historique
        recent_vol = data["close"].pct_change().tail(20).std()
        historical_vol = data["close"].pct_change().std()
        
        # Calcul de tendance
        sma_20 = data["close"].rolling(20).mean().iloc[-1]
        sma_50 = data["close"].rolling(50).mean().iloc[-1]
        current_price = data["close"].iloc[-1]
        
        # Classification du régime
        if recent_vol > historical_vol * 1.5:
            if current_price > sma_20 > sma_50:
                return "volatile_bullish"
            elif current_price < sma_20 < sma_50:
                return "volatile_bearish"
            else:
                return "high_volatility"
        else:
            if current_price > sma_20 > sma_50:
                return "trending_bullish"
            elif current_price < sma_20 < sma_50:
                return "trending_bearish"
            else:
                return "sideways"


# Fonctions helper pour usage facile
async def predict_stock_price(
    data: pd.DataFrame,
    symbol: str,
    model_type: str = "ensemble",
    horizon: str = "1d"
) -> PredictionResult:
    """Prédire le prix d'une action."""
    if not ML_AVAILABLE:
        raise ImportError("ML libraries not available")
    
    current_price = data["close"].iloc[-1]
    horizon_enum = PredictionHorizon(horizon)
    
    if model_type == "ensemble":
        # Créer un ensemble de modèles
        predictor = EnsemblePredictor(horizon_enum)
        predictor.add_model(ModelType.RANDOM_FOREST)
        predictor.add_model(ModelType.GRADIENT_BOOSTING)
        if XGBOOST_AVAILABLE:
            predictor.add_model(ModelType.XGBOOST)
        
        # Entraîner et prédire
        predictor.train_all(data)
        result = predictor.predict(data, current_price)
    else:
        # Modèle unique
        model_type_enum = ModelType(model_type)
        model = PricePredictionModel(model_type_enum, horizon_enum)
        model.train(data)
        result = model.predict(data, current_price)
    
    result.symbol = symbol
    return result


def format_prediction_result(result: PredictionResult) -> str:
    """Formater le résultat de prédiction."""
    direction_emoji = "📈" if result.prediction_change > 0 else "📉" if result.prediction_change < 0 else "➡️"
    confidence_stars = "⭐" * int(result.confidence * 5)
    
    prediction_text = f"""
🤖 **AI Price Prediction for {result.symbol}**

**📊 Current Analysis:**
- Current Price: ${result.current_price:.2f}
- Predicted Price ({result.horizon.value}): ${result.predicted_price:.2f}
- Expected Change: {result.prediction_change:+.2f}% {direction_emoji}

**🎯 Model Information:**
- Model Used: {result.model_used.title()}
- Confidence: {result.confidence:.1%} {confidence_stars}
- Prediction Date: {result.prediction_date.strftime('%Y-%m-%d %H:%M')}

**📈 Model Performance:**
- Accuracy: {result.model_accuracy:.1%} (R² Score)
"""
    
    if result.feature_importance:
        prediction_text += f"""
**🔍 Key Factors Influencing Prediction:**
"""
        for feature, importance in list(result.feature_importance.items())[:5]:
            prediction_text += f"• {feature.replace('_', ' ').title()}: {importance:.3f}\n"
    
    # Ajouter une interprétation
    if result.confidence > 0.7:
        prediction_text += "\n✅ **High confidence prediction** - Strong signal from multiple indicators"
    elif result.confidence > 0.5:
        prediction_text += "\n⚖️ **Moderate confidence** - Mixed signals, use with other analysis"
    else:
        prediction_text += "\n⚠️ **Low confidence** - Uncertain market conditions, exercise caution"
    
    prediction_text += f"""

**⚠️ Disclaimer:** 
AI predictions are based on historical patterns and should not be used as sole investment advice.
Market conditions can change rapidly and past performance doesn't guarantee future results.
Always combine with fundamental analysis and risk management.
"""
    
    return prediction_text


# Classes d'alertes basées sur les prédictions
class PredictionAlert:
    """Alertes basées sur les prédictions ML."""
    
    @staticmethod
    def create_bullish_prediction_alert(confidence_threshold: float = 0.6):
        """Alerte pour prédictions haussières."""
        return {
            "name": "Prédiction Haussière IA",
            "criteria": {
                "prediction_change": {"min": 2.0},  # >2% de hausse prédite
                "confidence": {"min": confidence_threshold},
                "horizon": "1d"
            }
        }
    
    @staticmethod
    def create_bearish_prediction_alert(confidence_threshold: float = 0.6):
        """Alerte pour prédictions baissières."""
        return {
            "name": "Prédiction Baissière IA", 
            "criteria": {
                "prediction_change": {"max": -2.0},  # >2% de baisse prédite
                "confidence": {"min": confidence_threshold},
                "horizon": "1d"
            }
        }