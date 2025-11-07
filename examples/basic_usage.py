"""Basic usage example of InsightAgent Engine"""

import os
from insight_agent import InsightAgentEngine, AnalysisRequest

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Sample marketing data
sample_data = [
    {
        "Campaign name": "Summer Sale 2024",
        "Ad set name": "US Audience",
        "Ad name": "Beach Creative A",
        "Ad ID": "ad_001",
        "Spend": 1500,
        "Impressions": 75000,
        "Clicks": 1200,
        "CTR %": 1.6,
        "Frequency": 2.8,
        "ROAS": 1.8,
        "Purchases": 45,
        "Purchase value": 2700,
        "Adds to cart": 120,
        "ATC→Purchase %": 37.5,
        "CTR 7d %": 1.8,
        "CTR prev7 %": 2.1,
        "CTR drop vs prev7 %": 14.3
    },
    {
        "Campaign name": "Summer Sale 2024",
        "Ad set name": "US Audience",
        "Ad name": "Beach Creative B",
        "Ad ID": "ad_002",
        "Spend": 2000,
        "Impressions": 120000,
        "Clicks": 800,
        "CTR %": 0.67,
        "Frequency": 4.2,
        "ROAS": 0.8,
        "Purchases": 20,
        "Purchase value": 1600,
        "Adds to cart": 85,
        "ATC→Purchase %": 23.5,
        "CTR 7d %": 0.65,
        "CTR prev7 %": 1.2,
        "CTR drop vs prev7 %": 45.8
    },
    {
        "Campaign name": "Fall Collection",
        "Ad set name": "CA Audience",
        "Ad name": "Product Showcase",
        "Ad ID": "ad_003",
        "Spend": 800,
        "Impressions": 50000,
        "Clicks": 1500,
        "CTR %": 3.0,
        "Frequency": 2.1,
        "ROAS": 4.2,
        "Purchases": 56,
        "Purchase value": 3360,
        "Adds to cart": 180,
        "ATC→Purchase %": 31.1,
        "CTR 7d %": 3.1,
        "CTR prev7 %": 2.9,
        "CTR drop vs prev7 %": -6.9
    }
]

def main():
    # Initialize engine
    print("Initializing InsightAgent Engine...")
    engine = InsightAgentEngine(llm_model="gpt-4o-mini")

    # Create analysis request
    request = AnalysisRequest(data=sample_data)

    # Run analysis
    print("\\nAnalyzing marketing data...\\n")
    response = engine.analyze(request)

    # Print results
    print("=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    print(response.summary)
    print()

    print("=" * 80)
    print("METRICS OVERVIEW")
    print("=" * 80)
    for key, value in response.metrics_overview.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    print()

    print("=" * 80)
    print("TOP INSIGHTS")
    print("=" * 80)
    for i, insight in enumerate(response.insights[:5], 1):
        print(f"\\n{i}. [{insight.priority.upper()}] {insight.category.upper()}")
        print(f"   Condition: {insight.condition}")
        print(f"   Action: {insight.recommendation}")
    print()

    print("=" * 80)
    print("AD STATUS RECOMMENDATIONS")
    print("=" * 80)
    for ad in response.ad_insights:
        print(f"\\n{ad.ad_name} (ID: {ad.ad_id})")
        print(f"Status: {ad.status.upper()}")
        if ad.issues:
            print(f"Issues: {', '.join(ad.issues)}")
        print("Recommendations:")
        for rec in ad.recommendations[:3]:
            print(f"  • {rec}")
    print()

    print("=" * 80)
    print("EXECUTION METADATA")
    print("=" * 80)
    for key, value in response.execution_metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
