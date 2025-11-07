"""Specialized AI agents for marketing analysis"""

from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .schemas import InsightRecommendation, InsightCategory, AdStatus, AdInsight
import json


class BaseAgent:
    """Base class for specialized agents"""

    def __init__(self, llm_model: str = "gpt-4o-mini", temperature: float = 0.1):
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)

    def analyze(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Override in subclass"""
        raise NotImplementedError


class ROASAgent(BaseAgent):
    """Agent specialized in ROAS analysis"""

    def analyze(self, data: List[Dict[str, Any]]) -> List[InsightRecommendation]:
        """Analyze ROAS performance"""

        insights = []

        for row in data:
            roas = self._get_value(row, "roas", 0)
            spend = self._get_value(row, "spend", 0)
            frequency = self._get_value(row, "frequency", 0)
            ad_id = str(row.get("ad_id", "unknown"))

            # ROAS 1-2: Test new creatives
            if 1.0 <= roas <= 2.0:
                insights.append(InsightRecommendation(
                    category=InsightCategory.ROAS,
                    condition=f"ROAS {roas:.2f} (1-2 range)",
                    recommendation="Test 2-3 new hooks/thumbnails; rotate in new ad creative; cap frequency to avoid fatigue",
                    priority="high",
                    affected_ads=[ad_id],
                    metrics={"roas": roas, "spend": spend, "frequency": frequency}
                ))

            # ROAS < 1: Immediate action needed
            elif roas < 1.0 and spend > 50:
                insights.append(InsightRecommendation(
                    category=InsightCategory.ROAS,
                    condition=f"ROAS {roas:.2f} (unprofitable)",
                    recommendation="Pause immediately or reduce budget by 75%; audit targeting and creative quality",
                    priority="high",
                    affected_ads=[ad_id],
                    metrics={"roas": roas, "spend": spend}
                ))

            # ROAS > 3: Scale opportunity
            elif roas > 3.0:
                insights.append(InsightRecommendation(
                    category=InsightCategory.ROAS,
                    condition=f"ROAS {roas:.2f} (high performance)",
                    recommendation="Scale budget by 20-30%; create lookalike audiences; duplicate winning creative",
                    priority="medium",
                    affected_ads=[ad_id],
                    metrics={"roas": roas, "spend": spend}
                ))

        return insights

    def _get_value(self, row: Dict, key: str, default: float = 0) -> float:
        """Safely extract numeric value"""
        value = row.get(key, default)
        if isinstance(value, str):
            value = value.replace("%", "").replace(",", "").strip()
            try:
                return float(value)
            except:
                return default
        return float(value) if value else default


class CTRAgent(BaseAgent):
    """Agent specialized in CTR analysis"""

    def analyze(self, data: List[Dict[str, Any]]) -> List[InsightRecommendation]:
        """Analyze CTR performance and trends"""

        insights = []

        for row in data:
            ctr = self._get_value(row, "ctr", 0)
            ctr_7d = self._get_value(row, "ctr_7d", 0)
            ctr_prev_7d = self._get_value(row, "ctr_prev_7d", 0)
            ctr_drop = self._get_value(row, "ctr_drop", 0)
            impressions = self._get_value(row, "impressions", 0)
            ad_id = str(row.get("ad_id", "unknown"))

            # CTR declining significantly
            if ctr_drop > 30 or (ctr_prev_7d > 0 and ctr_7d < ctr_prev_7d * 0.7):
                insights.append(InsightRecommendation(
                    category=InsightCategory.CTR,
                    condition=f"CTR dropped {ctr_drop:.1f}% vs previous period",
                    recommendation="Creative fatigue detected; rotate in fresh ad creative; test new angles/hooks",
                    priority="high",
                    affected_ads=[ad_id],
                    metrics={"ctr": ctr, "ctr_7d": ctr_7d, "ctr_prev_7d": ctr_prev_7d, "ctr_drop": ctr_drop}
                ))

            # Low CTR overall
            elif ctr < 1.0 and impressions > 1000:
                insights.append(InsightRecommendation(
                    category=InsightCategory.CTR,
                    condition=f"Low CTR {ctr:.2f}% with {int(impressions)} impressions",
                    recommendation="Poor creative engagement; test stronger hooks, better visuals, or clearer CTA",
                    priority="high",
                    affected_ads=[ad_id],
                    metrics={"ctr": ctr, "impressions": impressions}
                ))

            # High CTR
            elif ctr > 3.0:
                insights.append(InsightRecommendation(
                    category=InsightCategory.CTR,
                    condition=f"Strong CTR {ctr:.2f}%",
                    recommendation="High engagement; scale impressions, create similar variants, analyze winning elements",
                    priority="medium",
                    affected_ads=[ad_id],
                    metrics={"ctr": ctr, "impressions": impressions}
                ))

        return insights

    def _get_value(self, row: Dict, key: str, default: float = 0) -> float:
        """Safely extract numeric value"""
        value = row.get(key, default)
        if isinstance(value, str):
            value = value.replace("%", "").replace(",", "").strip()
            try:
                return float(value)
            except:
                return default
        return float(value) if value else default


class ConversionAgent(BaseAgent):
    """Agent specialized in conversion funnel analysis"""

    def analyze(self, data: List[Dict[str, Any]]) -> List[InsightRecommendation]:
        """Analyze conversion funnel performance"""

        insights = []

        for row in data:
            atc_to_purchase = self._get_value(row, "atc_to_purchase", 0)
            ctr = self._get_value(row, "ctr", 0)
            add_to_cart = self._get_value(row, "add_to_cart", 0)
            purchases = self._get_value(row, "purchases", 0)
            ad_id = str(row.get("ad_id", "unknown"))

            # Healthy CTR but poor conversion
            if ctr > 1.5 and atc_to_purchase < 20 and add_to_cart > 10:
                insights.append(InsightRecommendation(
                    category=InsightCategory.CONVERSION,
                    condition=f"CTR {ctr:.2f}% healthy but ATC→Purchase {atc_to_purchase:.1f}% < 20%",
                    recommendation="Audit landing page and checkout flow; check for friction, page speed, trust signals; review pricing and shipping",
                    priority="high",
                    affected_ads=[ad_id],
                    metrics={"ctr": ctr, "atc_to_purchase": atc_to_purchase, "add_to_cart": add_to_cart}
                ))

            # Low ATC rate
            elif add_to_cart < 5 and ctr > 1.0:
                insights.append(InsightRecommendation(
                    category=InsightCategory.CONVERSION,
                    condition=f"Low cart adds ({int(add_to_cart)}) despite clicks",
                    recommendation="Landing page issue; ensure message match, improve product presentation, add urgency/scarcity",
                    priority="high",
                    affected_ads=[ad_id],
                    metrics={"add_to_cart": add_to_cart, "ctr": ctr}
                ))

            # Good conversion rate
            elif atc_to_purchase > 30:
                insights.append(InsightRecommendation(
                    category=InsightCategory.CONVERSION,
                    condition=f"Strong ATC→Purchase rate {atc_to_purchase:.1f}%",
                    recommendation="Excellent conversion funnel; scale traffic to this flow, document winning elements",
                    priority="low",
                    affected_ads=[ad_id],
                    metrics={"atc_to_purchase": atc_to_purchase, "purchases": purchases}
                ))

        return insights

    def _get_value(self, row: Dict, key: str, default: float = 0) -> float:
        """Safely extract numeric value"""
        value = row.get(key, default)
        if isinstance(value, str):
            value = value.replace("%", "").replace(",", "").strip()
            try:
                return float(value)
            except:
                return default
        return float(value) if value else default


class FrequencyAgent(BaseAgent):
    """Agent specialized in frequency and ad fatigue analysis"""

    def analyze(self, data: List[Dict[str, Any]]) -> List[InsightRecommendation]:
        """Analyze frequency and ad fatigue"""

        insights = []

        for row in data:
            frequency = self._get_value(row, "frequency", 0)
            ctr = self._get_value(row, "ctr", 0)
            ctr_drop = self._get_value(row, "ctr_drop", 0)
            ad_id = str(row.get("ad_id", "unknown"))

            # High frequency with declining performance
            if frequency > 3.5 and (ctr_drop > 20 or ctr < 1.0):
                insights.append(InsightRecommendation(
                    category=InsightCategory.FREQUENCY,
                    condition=f"Frequency {frequency:.2f} with performance decline",
                    recommendation="Ad fatigue detected; cap frequency at 3, expand audience, or rotate creative",
                    priority="high",
                    affected_ads=[ad_id],
                    metrics={"frequency": frequency, "ctr": ctr, "ctr_drop": ctr_drop}
                ))

            # Very high frequency
            elif frequency > 5.0:
                insights.append(InsightRecommendation(
                    category=InsightCategory.FREQUENCY,
                    condition=f"Very high frequency {frequency:.2f}",
                    recommendation="Audience saturation; expand targeting, add exclusions, or pause for creative refresh",
                    priority="high",
                    affected_ads=[ad_id],
                    metrics={"frequency": frequency}
                ))

        return insights

    def _get_value(self, row: Dict, key: str, default: float = 0) -> float:
        """Safely extract numeric value"""
        value = row.get(key, default)
        if isinstance(value, str):
            value = value.replace("%", "").replace(",", "").strip()
            try:
                return float(value)
            except:
                return default
        return float(value) if value else default


class StatusDecisionAgent(BaseAgent):
    """Agent that decides ad status based on all insights"""

    def decide_status(self, row: Dict[str, Any], insights: List[InsightRecommendation]) -> AdInsight:
        """Decide status for an ad based on insights"""

        ad_id = str(row.get("ad_id", "unknown"))
        ad_name = str(row.get("ad_name", "unknown"))

        # Find insights for this ad
        ad_insights = [i for i in insights if ad_id in i.affected_ads]

        # Extract metrics
        roas = self._get_value(row, "roas", 0)
        ctr = self._get_value(row, "ctr", 0)
        spend = self._get_value(row, "spend", 0)
        frequency = self._get_value(row, "frequency", 0)

        # Decision logic
        high_priority_issues = [i for i in ad_insights if i.priority == "high"]

        status = AdStatus.KEEP
        recommendations = []
        issues = []

        # PAUSE conditions
        if roas < 1.0 and spend > 50:
            status = AdStatus.PAUSE
            issues.append(f"Unprofitable: ROAS {roas:.2f}")
        elif frequency > 5.0 and ctr < 0.5:
            status = AdStatus.PAUSE
            issues.append(f"Severe fatigue: Frequency {frequency:.2f}, CTR {ctr:.2f}%")

        # FIX conditions
        elif len(high_priority_issues) >= 2:
            status = AdStatus.FIX
            for issue in high_priority_issues:
                issues.append(issue.condition)
                recommendations.append(issue.recommendation)

        # TEST conditions
        elif 1.0 <= roas <= 2.5 or (ctr > 2.0 and roas < 2.0):
            status = AdStatus.TEST
            recommendations.extend([i.recommendation for i in ad_insights])

        # KEEP - performing well
        elif roas > 2.5 and ctr > 1.5:
            status = AdStatus.KEEP
            recommendations.append("Continue monitoring; performing well")

        # Default recommendations from insights
        if not recommendations:
            recommendations = [i.recommendation for i in ad_insights[:3]]

        return AdInsight(
            ad_id=ad_id,
            ad_name=ad_name,
            status=status,
            recommendations=recommendations[:5],  # Limit to top 5
            metrics={
                "roas": roas,
                "ctr": ctr,
                "spend": spend,
                "frequency": frequency
            },
            issues=issues
        )

    def _get_value(self, row: Dict, key: str, default: float = 0) -> float:
        """Safely extract numeric value"""
        value = row.get(key, default)
        if isinstance(value, str):
            value = value.replace("%", "").replace(",", "").strip()
            try:
                return float(value)
            except:
                return default
        return float(value) if value else default
