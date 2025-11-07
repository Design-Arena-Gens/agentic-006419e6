"""FastAPI microservice wrapper for InsightAgent Engine"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List
import os
from insight_agent import InsightAgentEngine, AnalysisRequest, AnalysisResponse

app = FastAPI(
    title="InsightAgent Engine API",
    description="Headless AI-powered marketing analysis microservice",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine
engine = None


@app.on_event("startup")
async def startup_event():
    """Initialize the engine on startup"""
    global engine

    # Validate OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    engine = InsightAgentEngine(
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1"))
    )


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "InsightAgent Engine",
        "version": "0.1.0"
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "engine_initialized": engine is not None,
        "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    """
    Analyze marketing data and generate insights.

    Example payload:
    ```json
    {
        "data": [
            {
                "Campaign name": "Summer Sale",
                "Ad name": "Beach Creative",
                "Ad ID": "12345",
                "Spend": 1000,
                "ROAS": 1.5,
                "CTR %": 2.1,
                "Frequency": 3.2
            }
        ]
    }
    ```
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        response = engine.analyze(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/simple")
async def analyze_simple(data: List[Dict[str, Any]]):
    """
    Simplified endpoint that accepts raw data array.

    Example payload:
    ```json
    [
        {
            "Campaign name": "Summer Sale",
            "Ad name": "Beach Creative",
            "Ad ID": "12345",
            "Spend": 1000,
            "ROAS": 1.5,
            "CTR %": 2.1
        }
    ]
    ```
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        result = engine.analyze_dict(data)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


class ColumnMappingRequest(BaseModel):
    """Request for column mapping"""
    columns: List[str]


@app.post("/columns/map")
async def map_columns(request: ColumnMappingRequest):
    """
    Map custom column names to standard schema.

    Example:
    ```json
    {
        "columns": ["Daily Budget", "Link Clicks", "Revenue"]
    }
    ```
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        mapping = engine.column_mapper.map_columns_with_llm(request.columns)
        return {"mapping": mapping}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mapping failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
