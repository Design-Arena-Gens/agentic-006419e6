"""Pydantic V2 schemas for InsightAgent Engine"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class AdStatus(str, Enum):
    """Ad status recommendation"""
    PAUSE = "pause"
    FIX = "fix"
    TEST = "test"
    KEEP = "keep"


class InsightCategory(str, Enum):
    """Categories of marketing insights"""
    ROAS = "roas"
    CTR = "ctr"
    CONVERSION = "conversion"
    FREQUENCY = "frequency"
    SPEND = "spend"
    CREATIVE = "creative"


class InsightRecommendation(BaseModel):
    """Individual insight recommendation"""
    model_config = ConfigDict(use_enum_values=True)

    category: InsightCategory = Field(..., description="Type of insight")
    condition: str = Field(..., description="Condition that triggered this insight")
    recommendation: str = Field(..., description="Actionable recommendation")
    priority: Literal["high", "medium", "low"] = Field(..., description="Priority level")
    affected_ads: List[str] = Field(default_factory=list, description="Ad IDs affected")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Relevant metrics")


class AdInsight(BaseModel):
    """Insight for a specific ad"""
    ad_id: str
    ad_name: str
    status: AdStatus
    recommendations: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    issues: List[str] = Field(default_factory=list)


class AnalysisRequest(BaseModel):
    """Request schema for marketing analysis"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: List[Dict[str, Any]] = Field(..., description="Raw marketing data rows")
    column_mapping: Optional[Dict[str, str]] = Field(
        None,
        description="Optional mapping of data columns to standard names"
    )
    analysis_focus: Optional[List[InsightCategory]] = Field(
        None,
        description="Specific categories to focus analysis on"
    )
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model to use")
    temperature: float = Field(default=0.1, description="LLM temperature")


class AnalysisResponse(BaseModel):
    """Response schema for marketing analysis"""
    model_config = ConfigDict(use_enum_values=True)

    summary: str = Field(..., description="Executive summary of findings")
    insights: List[InsightRecommendation] = Field(..., description="Generated insights")
    ad_insights: List[AdInsight] = Field(..., description="Per-ad analysis")
    metrics_overview: Dict[str, Any] = Field(..., description="Overall metrics summary")
    execution_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about analysis execution"
    )
