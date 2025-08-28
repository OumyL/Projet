"""
Module d'analyse des actualités et sentiment pour les marchés financiers.
Intègre plusieurs sources d'actualités et analyse le sentiment avec NLP.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import os
import json
import re
from collections import defaultdict, Counter

# Pour l'analyse de sentiment (optionnel - installer avec pip install textblob vaderSentiment)
try:
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    logging.warning("Sentiment analysis libraries not available. Install textblob and vaderSentiment for full functionality.")

logger = logging.getLogger(__name__)


class SentimentScore(Enum):
    """Classification du sentiment."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


class NewsSource(Enum):
    """Sources d'actualités supportées."""
    NEWS_API = "newsapi"
    ALPHA_VANTAGE = "alphavantage"
    FINNHUB = "finnhub"
    POLYGON = "polygon"
    REDDIT = "reddit"  # Via Reddit API
    TWITTER = "twitter"  # Via Twitter API


@dataclass
class NewsArticle:
    """Structure d'un article d'actualité."""
    title: str
    description: str
    content: str
    url: str
    published_at: datetime
    source: str
    author: Optional[str] = None
    symbol: Optional[str] = None
    
    # Analyse de sentiment
    sentiment_score: Optional[float] = None  # -1 à +1
    sentiment_label: Optional[SentimentScore] = None
    sentiment_confidence: Optional[float] = None
    
    # Métadonnées
    relevance_score: Optional[float] = None
    keywords: List[str] = None
    category: Optional[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


@dataclass
class SentimentAnalysis:
    """Résultat d'analyse de sentiment."""
    symbol: str
    overall_sentiment: float  # Score moyen -1 à +1
    sentiment_label: SentimentScore
    confidence: float
    
    # Métriques détaillées
    positive_count: int
    negative_count: int
    neutral_count: int
    total_articles: int
    
    # Tendances temporelles
    sentiment_trend: str  # "improving", "declining", "stable"
    trend_strength: float  # 0 à 1
    
    # Sources de sentiment
    news_sentiment: float
    analysis_period: str  # ← DÉPLACÉ AVANT LES DEFAULTS
    last_updated: datetime  # ← DÉPLACÉ AVANT LES DEFAULTS
    
     # Arguments avec valeurs par défaut à la fin
    social_sentiment: Optional[float] = None
    analyst_sentiment: Optional[float] = None
    key_themes: List[str] = None
    
    def __post_init__(self):
        if self.key_themes is None:
            self.key_themes = []


class NewsAPIClient:
    """Client pour l'API NewsAPI."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NEWS_API_KEY")
        if not self.api_key:
            raise ValueError("NEWS_API_KEY is required")
        
        self.base_url = "https://newsapi.org/v2"
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={"X-API-Key": self.api_key},
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_everything(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 100
    ) -> List[NewsArticle]:
        """Récupérer des articles avec une requête spécifique."""
        
        params = {
            "q": query,
            "language": language,
            "sortBy": sort_by,
            "pageSize": min(page_size, 100)  # Limite API
        }
        
        if from_date:
            params["from"] = from_date.strftime("%Y-%m-%dT%H:%M:%S")
        if to_date:
            params["to"] = to_date.strftime("%Y-%m-%dT%H:%M:%S")
        
        try:
            async with self.session.get(f"{self.base_url}/everything", params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                if data.get("status") != "ok":
                    raise ValueError(f"NewsAPI Error: {data.get('message', 'Unknown error')}")
                
                articles = []
                for article_data in data.get("articles", []):
                    if article_data.get("title") and article_data.get("publishedAt"):
                        articles.append(self._parse_article(article_data))
                
                return articles
                
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Network error: {e}")
    
    def _parse_article(self, article_data: Dict[str, Any]) -> NewsArticle:
        """Parser un article depuis les données API."""
        published_at = datetime.fromisoformat(
            article_data["publishedAt"].replace("Z", "+00:00")
        )
        
        return NewsArticle(
            title=article_data.get("title", ""),
            description=article_data.get("description", ""),
            content=article_data.get("content", ""),
            url=article_data.get("url", ""),
            published_at=published_at,
            source=article_data.get("source", {}).get("name", "Unknown"),
            author=article_data.get("author")
        )


class FinancialSentimentAnalyzer:
    """Analyseur de sentiment spécialisé pour la finance."""
    
    def __init__(self):
        self.financial_keywords = {
            "positive": [
                "growth", "profit", "gain", "increase", "rise", "surge", "boost", "strong",
                "bullish", "outperform", "exceed", "beat", "record", "milestone", "expansion",
                "acquisition", "merger", "partnership", "innovation", "breakthrough", "success"
            ],
            "negative": [
                "loss", "decline", "drop", "fall", "crash", "plunge", "weak", "bearish",
                "underperform", "miss", "disappoint", "concern", "risk", "uncertainty",
                "bankruptcy", "layoffs", "scandal", "investigation", "fraud", "lawsuit"
            ],
            "neutral": [
                "report", "announce", "release", "update", "forecast", "estimate", "guidance",
                "meeting", "conference", "earnings", "dividend", "split", "trading", "volume"
            ]
        }
        
        # Multiplicateurs pour les mots-clés financiers
        self.keyword_multipliers = {
            "earnings beat": 1.5,
            "earnings miss": -1.5,
            "revenue growth": 1.3,
            "profit margin": 1.2,
            "debt reduction": 1.1,
            "market share": 1.1,
            "cost cutting": -0.8,
            "guidance cut": -1.4,
            "investigation": -1.3,
            "lawsuit": -1.2,
            "regulatory": -0.9,
            "competition": -0.7
        }
        
        if SENTIMENT_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
    
    def analyze_article_sentiment(self, article: NewsArticle) -> NewsArticle:
        """Analyser le sentiment d'un article."""
        text = f"{article.title} {article.description}"
        
        if not SENTIMENT_AVAILABLE:
            # Analyse basique par mots-clés
            sentiment_score = self._keyword_based_sentiment(text)
        else:
            # Analyse avancée avec NLP
            sentiment_score = self._advanced_sentiment_analysis(text)
        
        # Appliquer les multiplicateurs financiers
        financial_modifier = self._calculate_financial_modifier(text)
        adjusted_score = sentiment_score * financial_modifier
        
        # Normaliser entre -1 et 1
        article.sentiment_score = max(-1, min(1, adjusted_score))
        article.sentiment_label = self._score_to_label(article.sentiment_score)
        article.sentiment_confidence = abs(article.sentiment_score)
        
        # Extraire les mots-clés
        article.keywords = self._extract_keywords(text)
        
        return article
    
    def _keyword_based_sentiment(self, text: str) -> float:
        """Analyse de sentiment basée sur les mots-clés."""
        text_lower = text.lower()
        
        positive_count = sum(
            text_lower.count(word) for word in self.financial_keywords["positive"]
        )
        negative_count = sum(
            text_lower.count(word) for word in self.financial_keywords["negative"]
        )
        
        total_keywords = positive_count + negative_count
        if total_keywords == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_keywords
    
    def _advanced_sentiment_analysis(self, text: str) -> float:
        """Analyse de sentiment avancée avec NLP."""
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_score = blob.sentiment.polarity
        
        # VADER sentiment
        vader_scores = self.vader_analyzer.polarity_scores(text)
        vader_score = vader_scores['compound']
        
        # Moyenne pondérée (VADER est meilleur pour les réseaux sociaux)
        combined_score = (textblob_score * 0.6) + (vader_score * 0.4)
        
        return combined_score
    
    def _calculate_financial_modifier(self, text: str) -> float:
        """Calculer le modificateur basé sur les termes financiers."""
        text_lower = text.lower()
        modifier = 1.0
        
        for phrase, mult in self.keyword_multipliers.items():
            if phrase in text_lower:
                modifier *= (1 + mult * 0.1)  # Appliquer un facteur d'ajustement
        
        return max(0.1, min(2.0, modifier))  # Limiter entre 0.1 et 2.0
    
    def _score_to_label(self, score: float) -> SentimentScore:
        """Convertir un score en label."""
        if score > 0.6:
            return SentimentScore.VERY_POSITIVE
        elif score > 0.2:
            return SentimentScore.POSITIVE
        elif score > -0.2:
            return SentimentScore.NEUTRAL
        elif score > -0.6:
            return SentimentScore.NEGATIVE
        else:
            return SentimentScore.VERY_NEGATIVE
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extraire les mots-clés pertinents."""
        # Mots-clés financiers trouvés
        keywords = []
        text_lower = text.lower()
        
        all_financial_words = (
            self.financial_keywords["positive"] + 
            self.financial_keywords["negative"] + 
            list(self.keyword_multipliers.keys())
        )
        
        for word in all_financial_words:
            if word in text_lower:
                keywords.append(word)
        
        return keywords[:10]  # Limiter à 10 mots-clés


class MarketNewsAnalyzer:
    """Analyseur principal des actualités de marché."""
    
    def __init__(self, news_api_key: str = None):
        self.news_client = None
        if news_api_key or os.getenv("NEWS_API_KEY"):
            self.news_api_key = news_api_key or os.getenv("NEWS_API_KEY")
        
        self.sentiment_analyzer = FinancialSentimentAnalyzer()
        self._cache = {}
        self._cache_ttl = 1800  # 30 minutes
    
    async def analyze_symbol_sentiment(
        self,
        symbol: str,
        days_back: int = 7,
        max_articles: int = 100
    ) -> SentimentAnalysis:
        """Analyser le sentiment pour un symbole spécifique."""
        
        # Vérifier le cache
        cache_key = f"{symbol}_{days_back}_{max_articles}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]["data"]
        
        # Récupérer les articles
        articles = await self._fetch_symbol_news(symbol, days_back, max_articles)
        
        if not articles:
            return self._create_empty_sentiment_analysis(symbol, days_back)
        
        # Analyser le sentiment de chaque article
        analyzed_articles = []
        for article in articles:
            analyzed_article = self.sentiment_analyzer.analyze_article_sentiment(article)
            analyzed_articles.append(analyzed_article)
        
        # Calculer les métriques globales
        sentiment_analysis = self._calculate_overall_sentiment(
            symbol, analyzed_articles, days_back
        )
        
        # Mettre en cache
        self._cache[cache_key] = {
            "data": sentiment_analysis,
            "timestamp": datetime.now().timestamp()
        }
        
        return sentiment_analysis
    
    async def _fetch_symbol_news(
        self, 
        symbol: str, 
        days_back: int, 
        max_articles: int
    ) -> List[NewsArticle]:
        """Récupérer les actualités pour un symbole avec recherche élargie."""
        
        if not self.news_api_key:
            logger.warning("No NEWS_API_KEY provided, using mock data")
            return self._generate_mock_articles(symbol, days_back)
        
        from_date = datetime.now() - timedelta(days=days_back * 2)
        to_date = datetime.now()
        
        # Requêtes de recherche plus larges et variées
        company_map = {
        'AAPL': 'Apple',
        'TSLA': 'Tesla',
        'MSFT': 'Microsoft',
        'GOOGL': 'Google',
        'AMZN': 'Amazon',
        'META': 'Meta',
        'NVDA': 'Nvidia',
        'BTC': 'Bitcoin',
        'ETH': 'Ethereum'
        }
    
        company_name = company_map.get(symbol.upper(), symbol)
        
        # Construire la requête de recherche
        search_queries = [
            f'{company_name}',
            f'"{symbol}"',
            f"{symbol} stock",
            f"{symbol} earnings",
            f"{symbol} financial"
        ]
        
        all_articles = []
        
        async with NewsAPIClient(self.news_api_key) as client:
            for query in search_queries:
                try:
                    for source_domain in [None, 'reuters.com', 'bloomberg.com', 'cnbc.com']:
                        articles = await client.get_everything(
                        query=query,
                        from_date=from_date,
                        to_date=to_date,
                        page_size=min(max_articles // len(search_queries), 20),
                        domains = source_domain
                        )
                    
                    # Ajouter le symbole aux articles
                    for article in articles:
                        article.symbol = symbol
                        article.relevance_score = self._calculate_relevance(article, symbol)
                    
                    all_articles.extend(articles)
                    
                    if len(all_articles) >= max_articles:
                        break
                    
                except Exception as e:
                    logger.error(f"Error fetching news for query '{query}': {e}")
                    continue
                if len(all_articles) >= max_articles:
                 break
    
    # Si toujours aucun article, utiliser les données mock
        if not all_articles:
            logger.info(f"No articles found for {symbol}, using mock data for demonstration")
            return self._generate_mock_articles(symbol, days_back)
        
        # Dédupliquer et trier par pertinence
        unique_articles = self._deduplicate_articles(all_articles)
        unique_articles.sort(key=lambda x: x.relevance_score or 0, reverse=True)
        
        return unique_articles[:max_articles]
    
    def _calculate_relevance(self, article: NewsArticle, symbol: str) -> float:
        """Calculer la pertinence d'un article pour un symbole."""
        text = f"{article.title} {article.description}".lower()
        symbol_lower = symbol.lower()
        
        # Score de base selon les mentions
        mentions = text.count(symbol_lower)
        base_score = min(mentions * 0.3, 1.0)
        
        # Bonus pour mention dans le titre
        if symbol_lower in article.title.lower():
            base_score += 0.5
        
        # Bonus pour sources financières fiables
        financial_sources = [
            "reuters", "bloomberg", "marketwatch", "cnbc", "yahoo finance",
            "seeking alpha", "financial times", "wall street journal"
        ]
        
        source_lower = article.source.lower()
        if any(fs in source_lower for fs in financial_sources):
            base_score += 0.3
        
        # Malus pour articles trop anciens
        days_old = (datetime.now() - article.published_at).days
        if days_old > 3:
            base_score *= 0.8
        
        return min(base_score, 1.0)
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Supprimer les articles en double."""
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            title_key = article.title.lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)
        
        return unique_articles
    
    def _calculate_overall_sentiment(
        self,
        symbol: str,
        articles: List[NewsArticle],
        days_back: int
    ) -> SentimentAnalysis:
        """Calculer le sentiment global."""
        
        if not articles:
            return self._create_empty_sentiment_analysis(symbol, days_back)
        
        # Calculer les métriques de base
        sentiments = [a.sentiment_score for a in articles if a.sentiment_score is not None]
        
        if not sentiments:
            return self._create_empty_sentiment_analysis(symbol, days_back)
        
        # Score global pondéré par la pertinence et l'âge
        weighted_sentiments = []
        weights = []
        
        for article in articles:
            if article.sentiment_score is not None:
                # Poids basé sur la pertinence et l'âge
                relevance_weight = article.relevance_score or 0.5
                
                days_old = (datetime.now() - article.published_at).days
                recency_weight = max(0.1, 1.0 - (days_old / 7))  # Décroît sur 7 jours
                
                total_weight = relevance_weight * recency_weight
                
                weighted_sentiments.append(article.sentiment_score * total_weight)
                weights.append(total_weight)
        
        if not weighted_sentiments:
            overall_sentiment = 0.0
        else:
            overall_sentiment = sum(weighted_sentiments) / sum(weights)
        
        # Compter les catégories
        positive_count = len([s for s in sentiments if s > 0.2])
        negative_count = len([s for s in sentiments if s < -0.2])
        neutral_count = len(sentiments) - positive_count - negative_count
        
        # Calculer la tendance
        sentiment_trend, trend_strength = self._calculate_sentiment_trend(articles)
        
        # Extraire les thèmes clés
        key_themes = self._extract_key_themes(articles)
        
        return SentimentAnalysis(
            symbol=symbol,
            overall_sentiment=round(overall_sentiment, 3),
            sentiment_label=self.sentiment_analyzer._score_to_label(overall_sentiment),
            confidence=round(abs(overall_sentiment), 3),
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            total_articles=len(articles),
            sentiment_trend=sentiment_trend,
            trend_strength=trend_strength,
            news_sentiment=round(overall_sentiment, 3),
            analysis_period=f"{days_back} days",
            last_updated=datetime.now(),
            key_themes=key_themes
        )
    
    def _calculate_sentiment_trend(
        self, 
        articles: List[NewsArticle]
    ) -> Tuple[str, float]:
        """Calculer la tendance du sentiment."""
        
        if len(articles) < 4:
            return "stable", 0.0
        
        # Trier par date
        sorted_articles = sorted(articles, key=lambda x: x.published_at)
        
        # Diviser en deux moitiés temporelles
        mid_point = len(sorted_articles) // 2
        early_articles = sorted_articles[:mid_point]
        recent_articles = sorted_articles[mid_point:]
        
        # Calculer sentiment moyen pour chaque période
        early_sentiment = np.mean([
            a.sentiment_score for a in early_articles 
            if a.sentiment_score is not None
        ])
        
        recent_sentiment = np.mean([
            a.sentiment_score for a in recent_articles 
            if a.sentiment_score is not None
        ])
        
        # Calculer la différence
        sentiment_change = recent_sentiment - early_sentiment
        
        # Déterminer la tendance
        if abs(sentiment_change) < 0.1:
            trend = "stable"
        elif sentiment_change > 0:
            trend = "improving"
        else:
            trend = "declining"
        
        trend_strength = min(abs(sentiment_change), 1.0)
        
        return trend, trend_strength
    
    def _extract_key_themes(self, articles: List[NewsArticle]) -> List[str]:
        """Extraire les thèmes clés des articles."""
        all_keywords = []
        for article in articles:
            if article.keywords:
                all_keywords.extend(article.keywords)
        
        # Compter les occurrences
        keyword_counts = Counter(all_keywords)
        
        # Retourner les 5 thèmes les plus fréquents
        return [keyword for keyword, count in keyword_counts.most_common(5)]
    
    def _create_empty_sentiment_analysis(
        self, 
        symbol: str, 
        days_back: int
    ) -> SentimentAnalysis:
        """Créer une analyse vide en cas d'absence de données."""
        return SentimentAnalysis(
            symbol=symbol,
            overall_sentiment=0.0,
            sentiment_label=SentimentScore.NEUTRAL,
            confidence=0.0,
            positive_count=0,
            negative_count=0,
            neutral_count=0,
            total_articles=0,
            sentiment_trend="stable",
            trend_strength=0.0,
            news_sentiment=0.0,
            analysis_period=f"{days_back} days",
            last_updated=datetime.now(),
            key_themes=[]
        )
    
    def _generate_mock_articles(
        self, 
        symbol: str, 
        days_back: int
    ) -> List[NewsArticle]:
        """Générer des articles factices pour les tests."""
        mock_articles = []
        
        base_date = datetime.now() - timedelta(days=days_back)
        
        # Templates d'articles avec différents sentiments
        templates = [
            {
                "title": f"{symbol} Reports Strong Q4 Earnings, Beats Expectations",
                "description": f"{symbol} announced record quarterly profits with revenue growth of 15%.",
                "sentiment": 0.7
            },
            {
                "title": f"{symbol} Faces Regulatory Scrutiny Over Market Practices",
                "description": f"Federal regulators launch investigation into {symbol}'s business practices.",
                "sentiment": -0.6
            },
            {
                "title": f"{symbol} Announces Strategic Partnership with Tech Giant",
                "description": f"{symbol} enters major partnership to expand digital capabilities.",
                "sentiment": 0.5
            },
            {
                "title": f"{symbol} Stock Price Remains Volatile Amid Market Uncertainty",
                "description": f"Analysts remain divided on {symbol}'s near-term prospects.",
                "sentiment": 0.0
            },
            {
                "title": f"{symbol} Cuts Guidance Due to Supply Chain Disruptions",
                "description": f"{symbol} reduces full-year guidance citing ongoing supply chain challenges.",
                "sentiment": -0.4
            }
        ]
        
        for i, template in enumerate(templates):
            article_date = base_date + timedelta(days=i * (days_back // len(templates)))
            
            mock_articles.append(NewsArticle(
                title=template["title"],
                description=template["description"],
                content=template["description"] + " Full article content would be here.",
                url=f"https://example.com/news/{symbol.lower()}-{i}",
                published_at=article_date,
                source="Mock Financial News",
                symbol=symbol,
                relevance_score=0.8,
                sentiment_score=template["sentiment"]
            ))
        
        return mock_articles
    
    def _is_cache_valid(self, key: str) -> bool:
        """Vérifier la validité du cache."""
        if key not in self._cache:
            return False
        
        cache_age = datetime.now().timestamp() - self._cache[key]["timestamp"]
        return cache_age < self._cache_ttl
    
    async def get_market_sentiment_overview(
        self, 
        symbols: List[str], 
        days_back: int = 7
    ) -> Dict[str, Any]:
        """Obtenir un aperçu du sentiment pour plusieurs symboles."""
        
        sentiment_results = {}
        
        # Analyser chaque symbole
        tasks = [
            self.analyze_symbol_sentiment(symbol, days_back) 
            for symbol in symbols
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error analyzing sentiment for {symbol}: {result}")
                continue
            
            sentiment_results[symbol] = result
        
        # Créer le résumé global
        if sentiment_results:
            overall_sentiment = np.mean([
                r.overall_sentiment for r in sentiment_results.values()
            ])
            
            total_articles = sum([
                r.total_articles for r in sentiment_results.values()
            ])
            
            # Compter les tendances
            trend_counts = Counter([
                r.sentiment_trend for r in sentiment_results.values()
            ])
            
            # Symboles les plus positifs/négatifs
            sorted_by_sentiment = sorted(
                sentiment_results.items(),
                key=lambda x: x[1].overall_sentiment,
                reverse=True
            )
            
            most_positive = sorted_by_sentiment[:3]
            most_negative = sorted_by_sentiment[-3:]
            
        else:
            overall_sentiment = 0.0
            total_articles = 0
            trend_counts = Counter()
            most_positive = []
            most_negative = []
        
        return {
            "market_sentiment_score": round(overall_sentiment, 3),
            "total_articles_analyzed": total_articles,
            "symbols_analyzed": len(sentiment_results),
            "sentiment_trends": dict(trend_counts),
            "most_positive_symbols": [
                {"symbol": symbol, "sentiment": result.overall_sentiment}
                for symbol, result in most_positive
            ],
            "most_negative_symbols": [
                {"symbol": symbol, "sentiment": result.overall_sentiment}
                for symbol, result in most_negative
            ],
            "individual_results": sentiment_results,
            "analysis_date": datetime.now().isoformat()
        }


class SentimentBasedScreener:
    """Screener basé sur l'analyse de sentiment."""
    
    def __init__(self, news_analyzer: MarketNewsAnalyzer):
        self.news_analyzer = news_analyzer
    
    async def screen_by_sentiment(
        self,
        symbols: List[str],
        min_sentiment: float = 0.3,
        min_articles: int = 5,
        sentiment_trend: str = "improving"
    ) -> List[Dict[str, Any]]:
        """Screener les symboles selon le sentiment."""
        
        results = []
        
        for symbol in symbols:
            try:
                sentiment_analysis = await self.news_analyzer.analyze_symbol_sentiment(symbol)
                
                # Appliquer les filtres
                if (sentiment_analysis.overall_sentiment >= min_sentiment and
                    sentiment_analysis.total_articles >= min_articles and
                    (sentiment_trend == "any" or sentiment_analysis.sentiment_trend == sentiment_trend)):
                    
                    results.append({
                        "symbol": symbol,
                        "sentiment_score": sentiment_analysis.overall_sentiment,
                        "sentiment_label": sentiment_analysis.sentiment_label.value,
                        "total_articles": sentiment_analysis.total_articles,
                        "trend": sentiment_analysis.sentiment_trend,
                        "trend_strength": sentiment_analysis.trend_strength,
                        "key_themes": sentiment_analysis.key_themes,
                        "confidence": sentiment_analysis.confidence
                    })
                    
            except Exception as e:
                logger.error(f"Error screening sentiment for {symbol}: {e}")
                continue
        
        # Trier par score de sentiment
        results.sort(key=lambda x: x["sentiment_score"], reverse=True)
        
        return results


# Fonctions helper pour usage facile
async def analyze_stock_sentiment(
    symbol: str,
    days_back: int = 7,
    news_api_key: str = None
) -> SentimentAnalysis:
    """Analyser rapidement le sentiment d'une action."""
    analyzer = MarketNewsAnalyzer(news_api_key)
    return await analyzer.analyze_symbol_sentiment(symbol, days_back)


async def get_market_mood(
    symbols: List[str] = None,
    days_back: int = 7,
    news_api_key: str = None
) -> Dict[str, Any]:
    """Obtenir l'humeur générale du marché."""
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
    
    analyzer = MarketNewsAnalyzer(news_api_key)
    return await analyzer.get_market_sentiment_overview(symbols, days_back)


# Classe d'alerte basée sur le sentiment
class SentimentAlert:
    """Alertes basées sur l'analyse de sentiment."""
    
    @staticmethod
    def create_sentiment_spike_alert(threshold: float = 0.6):
        """Alerte pour les pics de sentiment positif."""
        return {
            "name": "Pic de Sentiment Positif",
            "criteria": {
                "overall_sentiment": {"min": threshold},
                "total_articles": {"min": 3},
                "trend": "improving"
            }
        }
    
    @staticmethod
    def create_sentiment_crash_alert(threshold: float = -0.5):
        """Alerte pour les chutes de sentiment."""
        return {
            "name": "Chute de Sentiment",
            "criteria": {
                "overall_sentiment": {"max": threshold},
                "total_articles": {"min": 5},
                "trend": "declining"
            }
        }
    
    @staticmethod
    def create_controversy_alert(min_articles: int = 10):
        """Alerte pour les controverses (beaucoup d'articles négatifs)."""
        return {
            "name": "Controverse Détectée",
            "criteria": {
                "total_articles": {"min": min_articles},
                "negative_count": {"min": 7},
                "overall_sentiment": {"max": -0.2}
            }
        }