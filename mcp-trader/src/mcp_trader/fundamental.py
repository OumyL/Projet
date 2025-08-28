"""
Module d'analyse fondamentale avec Alpha Vantage API.
Fournit des données financières, ratios et métriques fondamentales.
"""

import asyncio
import aiohttp
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class CompanyOverview:
    """Vue d'ensemble d'une entreprise."""
    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: float
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    dividend_yield: Optional[float]
    eps: Optional[float]
    revenue_ttm: Optional[float]
    profit_margin: Optional[float]
    operating_margin: Optional[float]
    roe: Optional[float]  # Return on Equity
    roa: Optional[float]  # Return on Assets
    debt_to_equity: Optional[float]
    current_ratio: Optional[float]
    beta: Optional[float]
    week_52_high: Optional[float]
    week_52_low: Optional[float]


@dataclass
class EarningsData:
    """Données de résultats trimestriels."""
    symbol: str
    fiscal_date_ending: str
    reported_date: str
    reported_eps: Optional[float]
    estimated_eps: Optional[float]
    surprise: Optional[float]
    surprise_percentage: Optional[float]


@dataclass
class FinancialRatios:
    """Ratios financiers calculés."""
    symbol: str
    
    # Ratios de valorisation
    pe_ratio: Optional[float]
    peg_ratio: Optional[float]
    pb_ratio: Optional[float]
    ps_ratio: Optional[float]  # Price to Sales
    pcf_ratio: Optional[float]  # Price to Cash Flow
    
    # Ratios de rentabilité
    gross_margin: Optional[float]
    operating_margin: Optional[float]
    net_margin: Optional[float]    
    roe: Optional[float]
    roa: Optional[float]
    roic: Optional[float]  # Return on Invested Capital
    
    # Ratios de liquidité
    current_ratio: Optional[float]
    quick_ratio: Optional[float]
    cash_ratio: Optional[float]
    
    # Ratios d'endettement
    debt_to_equity: Optional[float]
    debt_to_assets: Optional[float]
    interest_coverage: Optional[float]
    
    # Ratios d'efficacité
    asset_turnover: Optional[float]
    inventory_turnover: Optional[float]
    receivables_turnover: Optional[float]


class AlphaVantageClient:
    """Client pour l'API Alpha Vantage."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY is required")
        
        self.base_url = "https://www.alphavantage.co/query"
        self.session: Optional[aiohttp.ClientSession] = None
        
        
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 3600 
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _make_request(
        self, 
        function: str, 
        symbol: str = None, 
        **kwargs
    ) -> Dict[str, Any]:
        """Faire une requête vers l'API Alpha Vantage."""
        params = {
            "function": function,
            "apikey": self.api_key,
            **kwargs
        }
        
        if symbol:
            params["symbol"] = symbol.upper()
        
        # Vérifier le cache
        cache_key = f"{function}_{symbol}_{hash(str(sorted(params.items())))}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        
        try:
            async with self.session.get(self.base_url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Vérifier les erreurs API
                if "Error Message" in data:
                    raise ValueError(f"Alpha Vantage API Error: {data['Error Message']}")
                
                if "Note" in data:
                    raise ValueError(f"Alpha Vantage API Limit: {data['Note']}")
                
                # Mettre en cache
                self._save_to_cache(cache_key, data)
                return data
                
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Network error: {e}")
    
    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Récupérer données du cache."""
        if key in self._cache:
            entry = self._cache[key]
            if datetime.now().timestamp() - entry["timestamp"] < self._cache_ttl:
                return entry["data"]
            else:
                del self._cache[key]
        return None
    
    def _save_to_cache(self, key: str, data: Dict[str, Any]):
        """Sauvegarder dans le cache."""
        self._cache[key] = {
            "data": data,
            "timestamp": datetime.now().timestamp()
        }
    
    async def get_company_overview(self, symbol: str) -> CompanyOverview:
        """Récupérer la vue d'ensemble d'une entreprise."""
        data = await self._make_request("OVERVIEW", symbol)
        
        def safe_float(value: str) -> Optional[float]:
            try:
                return float(value) if value and value != "None" else None
            except (ValueError, TypeError):
                return None
        
        return CompanyOverview(
            symbol=symbol.upper(),
            name=data.get("Name", ""),
            sector=data.get("Sector", ""),
            industry=data.get("Industry", ""),
            market_cap=safe_float(data.get("MarketCapitalization")),
            pe_ratio=safe_float(data.get("PERatio")),
            pb_ratio=safe_float(data.get("PriceToBookRatio")),
            dividend_yield=safe_float(data.get("DividendYield")),
            eps=safe_float(data.get("EPS")),
            revenue_ttm=safe_float(data.get("RevenueTTM")),
            profit_margin=safe_float(data.get("ProfitMargin")),
            operating_margin=safe_float(data.get("OperatingMarginTTM")),
            roe=safe_float(data.get("ReturnOnEquityTTM")),
            roa=safe_float(data.get("ReturnOnAssetsTTM")),
            debt_to_equity=safe_float(data.get("DebtToEquityRatio")),
            current_ratio=safe_float(data.get("CurrentRatio")),
            beta=safe_float(data.get("Beta")),
            week_52_high=safe_float(data.get("52WeekHigh")),
            week_52_low=safe_float(data.get("52WeekLow"))
        )
    
    async def get_earnings_data(self, symbol: str) -> List[EarningsData]:
        """Récupérer les données de résultats."""
        data = await self._make_request("EARNINGS", symbol)
        
        earnings_list = []
        quarterly_earnings = data.get("quarterlyEarnings", [])
        
        for earning in quarterly_earnings[:8]:  # 8 derniers trimestres
            def safe_float(value: str) -> Optional[float]:
                try:
                    return float(value) if value and value not in ["None", ""] else None
                except (ValueError, TypeError):
                    return None
            
            reported_eps = safe_float(earning.get("reportedEPS"))
            estimated_eps = safe_float(earning.get("estimatedEPS"))
            
            surprise = None
            surprise_percentage = None
            if reported_eps is not None and estimated_eps is not None:
                surprise = reported_eps - estimated_eps
                if estimated_eps != 0:
                    surprise_percentage = (surprise / estimated_eps) * 100
            
            earnings_list.append(EarningsData(
                symbol=symbol.upper(),
                fiscal_date_ending=earning.get("fiscalDateEnding", ""),
                reported_date=earning.get("reportedDate", ""),
                reported_eps=reported_eps,
                estimated_eps=estimated_eps,
                surprise=surprise,
                surprise_percentage=surprise_percentage
            ))
        
        return earnings_list
    
    async def get_income_statement(self, symbol: str) -> Dict[str, Any]:
        """Récupérer le compte de résultat."""
        return await self._make_request("INCOME_STATEMENT", symbol)
    
    async def get_balance_sheet(self, symbol: str) -> Dict[str, Any]:
        """Récupérer le bilan."""
        return await self._make_request("BALANCE_SHEET", symbol)
    
    async def get_cash_flow(self, symbol: str) -> Dict[str, Any]:
        """Récupérer le tableau de flux de trésorerie."""
        return await self._make_request("CASH_FLOW", symbol)


class FundamentalAnalyzer:
    """Analyseur de données fondamentales."""
    
    def __init__(self, alpha_vantage_client: AlphaVantageClient):
        self.client = alpha_vantage_client
    
    async def get_comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """Analyse fondamentale complète."""
        try:
            # Récupérer toutes les données en parallèle
            overview_task = self.client.get_company_overview(symbol)
            earnings_task = self.client.get_earnings_data(symbol)
            
            overview, earnings = await asyncio.gather(
                overview_task, earnings_task, return_exceptions=True
            )
            
            if isinstance(overview, Exception):
                logger.error(f"Error getting overview for {symbol}: {overview}")
                return {"error": str(overview)}
            
            if isinstance(earnings, Exception):
                logger.error(f"Error getting earnings for {symbol}: {earnings}")
                earnings = []
            
            # Calculer les ratios et métriques
            ratios = self._calculate_financial_ratios(overview)
            valuation = self._assess_valuation(overview, earnings)
            quality_score = self._calculate_quality_score(overview)
            
            return {
                "symbol": symbol.upper(),
                "company_overview": overview,
                "earnings_data": earnings,
                "financial_ratios": ratios,
                "valuation_assessment": valuation,
                "quality_score": quality_score,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {symbol}: {e}")
            return {"error": str(e)}
    
    def _calculate_financial_ratios(self, overview: CompanyOverview) -> FinancialRatios:
        """Calculer les ratios financiers détaillés."""
        
        # Calcul du PEG ratio
        peg_ratio = None
        if overview.pe_ratio and overview.eps:
            # Estimation simple du taux de croissance (nécessiterait des données historiques)
            estimated_growth = 10 
            if estimated_growth > 0:
                peg_ratio = overview.pe_ratio / estimated_growth
        
        # Price to Sales ratio
        ps_ratio = None
        if overview.market_cap and overview.revenue_ttm:
            ps_ratio = overview.market_cap / overview.revenue_ttm
        
        return FinancialRatios(
            symbol=overview.symbol,
            pe_ratio=overview.pe_ratio,
            peg_ratio=peg_ratio,
            pb_ratio=overview.pb_ratio,
            ps_ratio=ps_ratio,
            pcf_ratio=None,  # Nécessiterait données de cash flow
            gross_margin=None,  # Nécessiterait données détaillées
            operating_margin=overview.operating_margin,
            net_margin=overview.profit_margin,
            roe=overview.roe,
            roa=overview.roa,
            roic=None,  # Nécessiterait calcul complexe
            current_ratio=overview.current_ratio,
            quick_ratio=None,  # Nécessiterait données détaillées
            cash_ratio=None,  # Nécessiterait données de bilan
            debt_to_equity=overview.debt_to_equity,
            debt_to_assets=None,  # Nécessiterait calcul
            interest_coverage=None,  # Nécessiterait données détaillées
            asset_turnover=None,  # Nécessiterait calcul
            inventory_turnover=None,  # Nécessiterait données détaillées
            receivables_turnover=None  # Nécessiterait données détaillées
        )
    
    def _assess_valuation(
        self, 
        overview: CompanyOverview, 
        earnings: List[EarningsData]
    ) -> Dict[str, Any]:
        """Évaluer la valorisation de l'entreprise."""
        
        valuation_signals = []
        valuation_score = 0
        max_score = 0
        
        # Analyse du P/E ratio
        if overview.pe_ratio:
            max_score += 20
            if overview.pe_ratio < 15:
                valuation_signals.append("✅ P/E Attractif (< 15)")
                valuation_score += 20
            elif overview.pe_ratio < 25:
                valuation_signals.append("⚖️ P/E Modéré (15-25)")
                valuation_score += 10
            else:
                valuation_signals.append("⚠️ P/E Élevé (> 25)")
        
        # Analyse du P/B ratio
        if overview.pb_ratio:
            max_score += 15
            if overview.pb_ratio < 1.5:
                valuation_signals.append("✅ P/B Attractif (< 1.5)")
                valuation_score += 15
            elif overview.pb_ratio < 3:
                valuation_signals.append("⚖️ P/B Modéré (1.5-3)")
                valuation_score += 8
            else:
                valuation_signals.append("⚠️ P/B Élevé (> 3)")
        
        # Analyse de la marge bénéficiaire
        if overview.profit_margin:
            max_score += 15
            margin_pct = overview.profit_margin * 100
            if margin_pct > 20:
                valuation_signals.append(f"✅ Marge Excellente ({margin_pct:.1f}%)")
                valuation_score += 15
            elif margin_pct > 10:
                valuation_signals.append(f"⚖️ Marge Correcte ({margin_pct:.1f}%)")
                valuation_score += 8
            else:
                valuation_signals.append(f"⚠️ Marge Faible ({margin_pct:.1f}%)")
        
        # Analyse du ROE
        if overview.roe:
            max_score += 15
            roe_pct = overview.roe * 100
            if roe_pct > 15:
                valuation_signals.append(f"✅ ROE Excellent ({roe_pct:.1f}%)")
                valuation_score += 15
            elif roe_pct > 10:
                valuation_signals.append(f"⚖️ ROE Correct ({roe_pct:.1f}%)")
                valuation_score += 8
            else:
                valuation_signals.append(f"⚠️ ROE Faible ({roe_pct:.1f}%)")
        
        # Analyse des surprises de résultats
        if earnings:
            max_score += 15
            recent_surprises = [e for e in earnings[:4] if e.surprise_percentage is not None]
            if len(recent_surprises) >= 2:
                avg_surprise = sum(e.surprise_percentage for e in recent_surprises) / len(recent_surprises)
                if avg_surprise > 5:
                    valuation_signals.append(f"✅ Surpasse Souvent Attentes ({avg_surprise:+.1f}%)")
                    valuation_score += 15
                elif avg_surprise > -5:
                    valuation_signals.append(f"⚖️ Résultats Conformes ({avg_surprise:+.1f}%)")
                    valuation_score += 8
                else:
                    valuation_signals.append(f"⚠️ Déçoit Souvent ({avg_surprise:+.1f}%)")
        
        # Analyse de l'endettement
        if overview.debt_to_equity:
            max_score += 10
            if overview.debt_to_equity < 0.3:
                valuation_signals.append(f"✅ Endettement Faible ({overview.debt_to_equity:.2f})")
                valuation_score += 10
            elif overview.debt_to_equity < 0.6:
                valuation_signals.append(f"⚖️ Endettement Modéré ({overview.debt_to_equity:.2f})")
                valuation_score += 5
            else:
                valuation_signals.append(f"⚠️ Endettement Élevé ({overview.debt_to_equity:.2f})")
        
        # Analyse du dividende
        if overview.dividend_yield:
            max_score += 10
            div_pct = overview.dividend_yield * 100
            if 2 <= div_pct <= 6:
                valuation_signals.append(f"✅ Dividende Attractif ({div_pct:.2f}%)")
                valuation_score += 10
            elif div_pct > 0:
                valuation_signals.append(f"⚖️ Dividende Présent ({div_pct:.2f}%)")
                valuation_score += 5
        
        # Score final
        final_score = (valuation_score / max_score * 100) if max_score > 0 else 0
        
        # Classification
        if final_score >= 80:
            classification = "SOUS-ÉVALUÉE"
            recommendation = "ACHAT FORT"
        elif final_score >= 60:
            classification = "VALORISATION CORRECTE"
            recommendation = "ACHAT"
        elif final_score >= 40:
            classification = "VALORISATION NEUTRE"
            recommendation = "CONSERVER"
        elif final_score >= 20:
            classification = "SURÉVALUÉE"
            recommendation = "VENDRE"
        else:
            classification = "TRÈS SURÉVALUÉE"
            recommendation = "VENTE FORTE"
        
        return {
            "valuation_score": round(final_score, 1),
            "classification": classification,
            "recommendation": recommendation,
            "signals": valuation_signals,
            "key_metrics": {
                "pe_ratio": overview.pe_ratio,
                "pb_ratio": overview.pb_ratio,
                "profit_margin_pct": overview.profit_margin * 100 if overview.profit_margin else None,
                "roe_pct": overview.roe * 100 if overview.roe else None,
                "debt_to_equity": overview.debt_to_equity,
                "dividend_yield_pct": overview.dividend_yield * 100 if overview.dividend_yield else None
            }
        }
    
    def _calculate_quality_score(self, overview: CompanyOverview) -> Dict[str, Any]:
        """Calculer un score de qualité de l'entreprise."""
        
        quality_factors = []
        quality_score = 0
        max_score = 0
        
        # Rentabilité
        if overview.roe and overview.roa:
            max_score += 25
            roe_pct = overview.roe * 100
            roa_pct = overview.roa * 100
            
            if roe_pct > 15 and roa_pct > 8:
                quality_factors.append("✅ Rentabilité Excellente")
                quality_score += 25
            elif roe_pct > 10 and roa_pct > 5:
                quality_factors.append("⚖️ Rentabilité Correcte")
                quality_score += 15
            else:
                quality_factors.append("⚠️ Rentabilité Faible")
                quality_score += 5
        
        # Stabilité financière
        if overview.debt_to_equity and overview.current_ratio:
            max_score += 20
            if overview.debt_to_equity < 0.4 and overview.current_ratio > 1.5:
                quality_factors.append("✅ Structure Financière Solide")
                quality_score += 20
            elif overview.debt_to_equity < 0.7 and overview.current_ratio > 1.2:
                quality_factors.append("⚖️ Structure Financière Acceptable")
                quality_score += 12
            else:
                quality_factors.append("⚠️ Structure Financière Préoccupante")
                quality_score += 4
        
        # Taille et liquidité
        if overview.market_cap:
            max_score += 15
            if overview.market_cap > 10e9:  # > 10B
                quality_factors.append("✅ Grande Capitalisation")
                quality_score += 15
            elif overview.market_cap > 2e9:  # > 2B
                quality_factors.append("⚖️ Capitalisation Moyenne")
                quality_score += 10
            else:
                quality_factors.append("⚠️ Petite Capitalisation")
                quality_score += 5
        
        # Profitabilité des marges
        if overview.operating_margin and overview.profit_margin:
            max_score += 20
            op_margin = overview.operating_margin * 100
            net_margin = overview.profit_margin * 100
            
            if op_margin > 15 and net_margin > 10:
                quality_factors.append("✅ Marges Excellentes")
                quality_score += 20
            elif op_margin > 8 and net_margin > 5:
                quality_factors.append("⚖️ Marges Correctes")
                quality_score += 12
            else:
                quality_factors.append("⚠️ Marges Préoccupantes")
                quality_score += 4
        
        # Secteur (classification simplifiée)
        max_score += 10
        defensive_sectors = ["Consumer Staples", "Utilities", "Healthcare"]
        growth_sectors = ["Technology", "Consumer Discretionary"]
        
        if overview.sector in defensive_sectors:
            quality_factors.append("✅ Secteur Défensif")
            quality_score += 8
        elif overview.sector in growth_sectors:
            quality_factors.append("⚖️ Secteur de Croissance")
            quality_score += 6
        else:
            quality_factors.append("⚖️ Secteur Cyclique")
            quality_score += 5
        
        # Dividende (stabilité)
        max_score += 10
        if overview.dividend_yield and overview.dividend_yield > 0:
            div_pct = overview.dividend_yield * 100
            if 1 <= div_pct <= 8:
                quality_factors.append("✅ Politique de Dividende Stable")
                quality_score += 10
            else:
                quality_factors.append("⚖️ Dividende Présent")
                quality_score += 5
        else:
            quality_factors.append("⚖️ Pas de Dividende")
            quality_score += 3
        
        # Score final
        final_score = (quality_score / max_score * 100) if max_score > 0 else 0
        
        # Classification de qualité
        if final_score >= 85:
            quality_grade = "A+"
            description = "Entreprise de Très Haute Qualité"
        elif final_score >= 75:
            quality_grade = "A"
            description = "Entreprise de Haute Qualité"
        elif final_score >= 65:
            quality_grade = "B+"
            description = "Entreprise de Qualité Supérieure"
        elif final_score >= 55:
            quality_grade = "B"
            description = "Entreprise de Qualité Moyenne"
        elif final_score >= 45:
            quality_grade = "C+"
            description = "Entreprise de Qualité Acceptable"
        elif final_score >= 35:
            quality_grade = "C"
            description = "Entreprise de Qualité Faible"
        else:
            quality_grade = "D"
            description = "Entreprise de Qualité Préoccupante"
        
        return {
            "quality_score": round(final_score, 1),
            "quality_grade": quality_grade,
            "description": description,
            "factors": quality_factors
        }
    
    async def compare_companies(
        self, 
        symbols: List[str], 
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Comparer plusieurs entreprises."""
        if metrics is None:
            metrics = ["pe_ratio", "pb_ratio", "roe", "profit_margin", "debt_to_equity"]
        
        companies_data = {}
        
        # Récupérer les données pour chaque entreprise
        for symbol in symbols:
            try:
                analysis = await self.get_comprehensive_analysis(symbol)
                if "error" not in analysis:
                    companies_data[symbol] = analysis
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        if not companies_data:
            return {"error": "No valid company data retrieved"}
        
        # Créer le tableau de comparaison
        comparison_table = []
        
        for symbol, data in companies_data.items():
            overview = data["company_overview"]
            valuation = data["valuation_assessment"]
            quality = data["quality_score"]
            
            row = {
                "symbol": symbol,
                "name": overview.name,
                "sector": overview.sector,
                "market_cap": overview.market_cap,
                "valuation_score": valuation["valuation_score"],
                "quality_score": quality["quality_score"],
                "recommendation": valuation["recommendation"]
            }
            
            # Ajouter les métriques demandées
            for metric in metrics:
                if hasattr(overview, metric):
                    row[metric] = getattr(overview, metric)
            
            comparison_table.append(row)
        
        # Trier par score de valorisation
        comparison_table.sort(key=lambda x: x["valuation_score"], reverse=True)
        
        # Ajouter les rangs
        for i, row in enumerate(comparison_table):
            row["rank"] = i + 1
        
        return {
            "comparison_table": comparison_table,
            "analysis_date": datetime.now().isoformat(),
            "metrics_compared": metrics,
            "companies_analyzed": len(comparison_table)
        }
    
    def create_investment_thesis(
        self, 
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Créer une thèse d'investissement."""
        if "error" in analysis:
            return {"error": analysis["error"]}
        
        overview = analysis["company_overview"]
        valuation = analysis["valuation_assessment"]
        quality = analysis["quality_score"]
        
        # Points forts
        strengths = []
        if quality["quality_score"] >= 70:
            strengths.append(f"Entreprise de haute qualité ({quality['quality_grade']})")
        
        if valuation["valuation_score"] >= 60:
            strengths.append("Valorisation attractive")
        
        if overview.roe and overview.roe > 0.15:
            strengths.append(f"Excellent retour sur capitaux propres ({overview.roe*100:.1f}%)")
        
        if overview.profit_margin and overview.profit_margin > 0.15:
            strengths.append(f"Marges bénéficiaires élevées ({overview.profit_margin*100:.1f}%)")
        
        # Points faibles
        weaknesses = []
        if quality["quality_score"] < 50:
            weaknesses.append("Qualité d'entreprise préoccupante")
        
        if valuation["valuation_score"] < 40:
            weaknesses.append("Valorisation élevée")
        
        if overview.debt_to_equity and overview.debt_to_equity > 0.6:
            weaknesses.append(f"Endettement élevé ({overview.debt_to_equity:.2f})")
        
        if overview.pe_ratio and overview.pe_ratio > 30:
            weaknesses.append(f"P/E élevé ({overview.pe_ratio:.1f})")
        
        # Recommandation finale
        combined_score = (valuation["valuation_score"] + quality["quality_score"]) / 2
        
        if combined_score >= 75:
            final_recommendation = "ACHAT FORT"
            confidence = "Élevée"
        elif combined_score >= 60:
            final_recommendation = "ACHAT"
            confidence = "Moyenne-Haute"
        elif combined_score >= 45:
            final_recommendation = "CONSERVER"  
            confidence = "Moyenne"
        elif combined_score >= 30:
            final_recommendation = "VENTE"
            confidence = "Moyenne-Haute"
        else:
            final_recommendation = "VENTE FORTE"
            confidence = "Élevée"
        
        return {
            "symbol": overview.symbol,
            "company_name": overview.name,
            "final_recommendation": final_recommendation,
            "confidence_level": confidence,
            "combined_score": round(combined_score, 1),
            "investment_strengths": strengths,
            "investment_concerns": weaknesses,
            "key_metrics_summary": {
                "valuation_score": valuation["valuation_score"],
                "quality_score": quality["quality_score"],
                "pe_ratio": overview.pe_ratio,
                "roe_percent": overview.roe * 100 if overview.roe else None,
                "profit_margin_percent": overview.profit_margin * 100 if overview.profit_margin else None,
                "debt_to_equity": overview.debt_to_equity
            },
            "sector": overview.sector,
            "market_cap_billions": overview.market_cap / 1e9 if overview.market_cap else None
        }


# Helper functions pour usage facile
async def analyze_company_fundamentals(
    symbol: str, 
    api_key: str = None
) -> Dict[str, Any]:
    """Fonction helper pour analyser rapidement une entreprise."""
    async with AlphaVantageClient(api_key) as client:
        analyzer = FundamentalAnalyzer(client)
        analysis = await analyzer.get_comprehensive_analysis(symbol)
        
        if "error" not in analysis:
            thesis = analyzer.create_investment_thesis(analysis)
            analysis["investment_thesis"] = thesis
        
        return analysis


async def compare_sector_companies(
    symbols: List[str], 
    api_key: str = None
) -> Dict[str, Any]:
    """Comparer plusieurs entreprises du même secteur."""
    async with AlphaVantageClient(api_key) as client:
        analyzer = FundamentalAnalyzer(client)
        return await analyzer.compare_companies(symbols)


# Classes pour l'intégration avec le système d'alertes
class FundamentalAlert:
    """Alertes basées sur les critères fondamentaux."""
    
    @staticmethod
    def create_undervalued_alert(pe_threshold: float = 15, pb_threshold: float = 1.5):
        """Créer une alerte pour les actions sous-évaluées."""
        return {
            "name": "Sous-évaluation Fondamentale",
            "criteria": {
                "pe_ratio": {"max": pe_threshold},
                "pb_ratio": {"max": pb_threshold},
                "roe": {"min": 0.10},  # ROE > 10%
                "debt_to_equity": {"max": 0.5}
            }
        }
    
    @staticmethod  
    def create_quality_growth_alert():
        """Alerte pour les entreprises de qualité en croissance."""
        return {
            "name": "Croissance de Qualité",
            "criteria": {
                "roe": {"min": 0.15},  # ROE > 15%
                "profit_margin": {"min": 0.10},  # Marge > 10%
                "debt_to_equity": {"max": 0.4},
                "market_cap": {"min": 2e9}  # > 2B
            }
        }
    
    @staticmethod
    def create_dividend_aristocrat_alert():
        """Alerte pour les aristocrates du dividende."""
        return {
            "name": "Aristocrate Dividende",
            "criteria": {
                "dividend_yield": {"min": 0.02, "max": 0.08},  # 2-8%
                "pe_ratio": {"max": 20},
                "debt_to_equity": {"max": 0.6},
                "current_ratio": {"min": 1.2}
            }
        }