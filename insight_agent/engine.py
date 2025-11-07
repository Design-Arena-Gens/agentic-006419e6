"""Main InsightAgent Engine"""

from typing import List, Dict, Any, Optional
import time
from .schemas import AnalysisRequest, AnalysisResponse
from .column_mapper import ColumnMapper
from .graph import InsightGraph


class InsightAgentEngine:
    """
    Headless Python library for advanced marketing analysis.

    The engine semantically identifies required data columns, uses a team of
    specialized AI agents to execute formula-based insights, and returns
    structured JSON responses.

    Example:
        >>> engine = InsightAgentEngine()
        >>> request = AnalysisRequest(data=[...])
        >>> response = engine.analyze(request)
        >>> print(response.summary)
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.1
    ):
        """
        Initialize the InsightAgent Engine.

        Args:
            llm_model: OpenAI model to use for LLM calls
            temperature: Temperature for LLM generation
        """
        self.llm_model = llm_model
        self.temperature = temperature
        self.column_mapper = ColumnMapper(llm_model)
        self.graph = InsightGraph(llm_model, temperature)

    def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """
        Analyze marketing data and generate insights.

        Args:
            request: AnalysisRequest with raw data and optional configuration

        Returns:
            AnalysisResponse with insights, recommendations, and status decisions
        """
        start_time = time.time()

        # Step 1: Normalize columns
        if request.column_mapping:
            # Use provided mapping
            column_mapping = request.column_mapping
            normalized_data = self.column_mapper.normalize_data(
                request.data,
                column_mapping
            )
        else:
            # Auto-detect mapping
            columns = list(request.data[0].keys()) if request.data else []
            column_mapping = self.column_mapper.map_columns_with_llm(columns)
            normalized_data = self.column_mapper.normalize_data(
                request.data,
                column_mapping
            )

        # Step 2: Run LangGraph workflow
        initial_state = {
            "data": request.data,
            "normalized_data": normalized_data,
            "column_mapping": column_mapping,
            "roas_insights": [],
            "ctr_insights": [],
            "conversion_insights": [],
            "frequency_insights": [],
            "all_insights": [],
            "ad_insights": [],
            "summary": "",
            "metrics_overview": {}
        }

        result = self.graph.run(initial_state)

        # Step 3: Build response
        execution_time = time.time() - start_time

        response = AnalysisResponse(
            summary=result["summary"],
            insights=result["all_insights"],
            ad_insights=result["ad_insights"],
            metrics_overview=result["metrics_overview"],
            execution_metadata={
                "execution_time_seconds": round(execution_time, 2),
                "llm_model": self.llm_model,
                "total_ads_analyzed": len(normalized_data),
                "column_mapping": column_mapping,
                "insights_generated": len(result["all_insights"])
            }
        )

        return response

    def analyze_dict(self, data: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Convenience method for dict-based analysis.

        Args:
            data: List of data rows as dictionaries
            **kwargs: Additional AnalysisRequest parameters

        Returns:
            Response as dictionary
        """
        request = AnalysisRequest(data=data, **kwargs)
        response = self.analyze(request)
        return response.model_dump()
