"""LangGraph orchestration for InsightAgent Engine"""

from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from .agents import ROASAgent, CTRAgent, ConversionAgent, FrequencyAgent, StatusDecisionAgent
from .schemas import InsightRecommendation, AdInsight
import operator


class AnalysisState(TypedDict):
    """State for analysis workflow"""
    data: List[Dict[str, Any]]
    normalized_data: List[Dict[str, Any]]
    column_mapping: Dict[str, str]

    # Agent outputs
    roas_insights: Annotated[List[InsightRecommendation], operator.add]
    ctr_insights: Annotated[List[InsightRecommendation], operator.add]
    conversion_insights: Annotated[List[InsightRecommendation], operator.add]
    frequency_insights: Annotated[List[InsightRecommendation], operator.add]

    # Final outputs
    all_insights: List[InsightRecommendation]
    ad_insights: List[AdInsight]
    summary: str
    metrics_overview: Dict[str, Any]


class InsightGraph:
    """LangGraph workflow for marketing insights"""

    def __init__(self, llm_model: str = "gpt-4o-mini", temperature: float = 0.1):
        self.llm_model = llm_model
        self.temperature = temperature

        # Initialize agents
        self.roas_agent = ROASAgent(llm_model, temperature)
        self.ctr_agent = CTRAgent(llm_model, temperature)
        self.conversion_agent = ConversionAgent(llm_model, temperature)
        self.frequency_agent = FrequencyAgent(llm_model, temperature)
        self.status_agent = StatusDecisionAgent(llm_model, temperature)

        # Build graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""

        workflow = StateGraph(AnalysisState)

        # Add nodes
        workflow.add_node("analyze_roas", self._analyze_roas)
        workflow.add_node("analyze_ctr", self._analyze_ctr)
        workflow.add_node("analyze_conversion", self._analyze_conversion)
        workflow.add_node("analyze_frequency", self._analyze_frequency)
        workflow.add_node("aggregate_insights", self._aggregate_insights)
        workflow.add_node("decide_status", self._decide_status)
        workflow.add_node("generate_summary", self._generate_summary)

        # Define edges - parallel execution of specialized agents
        workflow.set_entry_point("analyze_roas")
        workflow.add_edge("analyze_roas", "analyze_ctr")
        workflow.add_edge("analyze_ctr", "analyze_conversion")
        workflow.add_edge("analyze_conversion", "analyze_frequency")
        workflow.add_edge("analyze_frequency", "aggregate_insights")
        workflow.add_edge("aggregate_insights", "decide_status")
        workflow.add_edge("decide_status", "generate_summary")
        workflow.add_edge("generate_summary", END)

        return workflow.compile()

    def _analyze_roas(self, state: AnalysisState) -> Dict[str, Any]:
        """Node: ROAS analysis"""
        insights = self.roas_agent.analyze(state["normalized_data"])
        return {"roas_insights": insights}

    def _analyze_ctr(self, state: AnalysisState) -> Dict[str, Any]:
        """Node: CTR analysis"""
        insights = self.ctr_agent.analyze(state["normalized_data"])
        return {"ctr_insights": insights}

    def _analyze_conversion(self, state: AnalysisState) -> Dict[str, Any]:
        """Node: Conversion analysis"""
        insights = self.conversion_agent.analyze(state["normalized_data"])
        return {"conversion_insights": insights}

    def _analyze_frequency(self, state: AnalysisState) -> Dict[str, Any]:
        """Node: Frequency analysis"""
        insights = self.frequency_agent.analyze(state["normalized_data"])
        return {"frequency_insights": insights}

    def _aggregate_insights(self, state: AnalysisState) -> Dict[str, Any]:
        """Node: Aggregate all insights"""

        all_insights = []
        all_insights.extend(state.get("roas_insights", []))
        all_insights.extend(state.get("ctr_insights", []))
        all_insights.extend(state.get("conversion_insights", []))
        all_insights.extend(state.get("frequency_insights", []))

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        all_insights.sort(key=lambda x: priority_order.get(x.priority, 3))

        # Calculate metrics overview
        data = state["normalized_data"]

        total_spend = sum(self._get_value(row, "spend", 0) for row in data)
        total_revenue = sum(self._get_value(row, "purchase_value", 0) for row in data)
        total_purchases = sum(self._get_value(row, "purchases", 0) for row in data)
        total_clicks = sum(self._get_value(row, "clicks", 0) for row in data)
        total_impressions = sum(self._get_value(row, "impressions", 0) for row in data)

        metrics_overview = {
            "total_spend": total_spend,
            "total_revenue": total_revenue,
            "overall_roas": total_revenue / total_spend if total_spend > 0 else 0,
            "total_purchases": total_purchases,
            "total_clicks": total_clicks,
            "total_impressions": total_impressions,
            "overall_ctr": (total_clicks / total_impressions * 100) if total_impressions > 0 else 0,
            "total_ads": len(data)
        }

        return {
            "all_insights": all_insights,
            "metrics_overview": metrics_overview
        }

    def _decide_status(self, state: AnalysisState) -> Dict[str, Any]:
        """Node: Decide status for each ad"""

        ad_insights = []

        for row in state["normalized_data"]:
            ad_insight = self.status_agent.decide_status(row, state["all_insights"])
            ad_insights.append(ad_insight)

        return {"ad_insights": ad_insights}

    def _generate_summary(self, state: AnalysisState) -> Dict[str, Any]:
        """Node: Generate executive summary"""

        llm = ChatOpenAI(model=self.llm_model, temperature=self.temperature)

        metrics = state["metrics_overview"]
        high_priority = [i for i in state["all_insights"] if i.priority == "high"]

        pause_count = sum(1 for ad in state["ad_insights"] if ad.status == "pause")
        fix_count = sum(1 for ad in state["ad_insights"] if ad.status == "fix")
        test_count = sum(1 for ad in state["ad_insights"] if ad.status == "test")

        summary_prompt = f"""Generate a concise executive summary (2-3 sentences) for this marketing analysis:

Total Spend: ${metrics['total_spend']:.2f}
Total Revenue: ${metrics['total_revenue']:.2f}
Overall ROAS: {metrics['overall_roas']:.2f}
Overall CTR: {metrics['overall_ctr']:.2f}%

High Priority Issues: {len(high_priority)}
Ads to Pause: {pause_count}
Ads to Fix: {fix_count}
Ads to Test: {test_count}

Focus on actionable insights and overall account health."""

        response = llm.invoke(summary_prompt)
        summary = response.content.strip()

        return {"summary": summary}

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

    def run(self, initial_state: Dict[str, Any]) -> AnalysisState:
        """Execute the graph"""
        return self.graph.invoke(initial_state)
