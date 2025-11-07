"""Example API client for InsightAgent Engine"""

import requests
import json

API_URL = "http://localhost:8000"

sample_data = [
    {
        "Campaign name": "Summer Sale 2024",
        "Ad name": "Beach Creative A",
        "Ad ID": "ad_001",
        "Spend": 1500,
        "ROAS": 1.8,
        "CTR %": 1.6,
        "Frequency": 2.8,
        "Purchases": 45,
        "Purchase value": 2700,
        "Adds to cart": 120,
        "ATC→Purchase %": 37.5,
        "CTR 7d %": 1.8,
        "CTR prev7 %": 2.1,
        "CTR drop vs prev7 %": 14.3
    }
]


def check_health():
    """Check API health"""
    response = requests.get(f"{API_URL}/health")
    print("Health Check:", response.json())


def analyze_simple():
    """Simple analysis endpoint"""
    response = requests.post(
        f"{API_URL}/analyze/simple",
        json=sample_data
    )

    if response.status_code == 200:
        result = response.json()
        print("\n" + "=" * 80)
        print("ANALYSIS RESULTS")
        print("=" * 80)
        print("\nSummary:", result["summary"])
        print(f"\nTotal Insights: {len(result['insights'])}")
        print(f"Ads Analyzed: {len(result['ad_insights'])}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def map_columns():
    """Test column mapping"""
    columns = ["Daily Budget", "Link Clicks", "Revenue", "Click Rate"]

    response = requests.post(
        f"{API_URL}/columns/map",
        json={"columns": columns}
    )

    if response.status_code == 200:
        result = response.json()
        print("\n" + "=" * 80)
        print("COLUMN MAPPING")
        print("=" * 80)
        for original, mapped in result["mapping"].items():
            print(f"{original:20} -> {mapped}")


if __name__ == "__main__":
    print("Testing InsightAgent Engine API\n")

    check_health()
    print()

    analyze_simple()
    print()

    map_columns()
