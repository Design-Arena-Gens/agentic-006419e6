"""Semantic column identification using LLM"""

from typing import Dict, List, Optional
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class ColumnMapper:
    """Semantically map data columns to standard schema"""

    STANDARD_COLUMNS = {
        "campaign_name": ["campaign name", "campaign", "campaign_name"],
        "ad_set_name": ["ad set name", "ad set", "adset", "ad_set_name"],
        "ad_name": ["ad name", "ad", "ad_name", "creative name"],
        "ad_id": ["ad id", "ad_id", "adid", "creative id"],
        "spend": ["spend", "cost", "amount spent", "budget spent"],
        "impressions": ["impressions", "impr", "views"],
        "clicks": ["clicks", "link clicks", "click"],
        "ctr": ["ctr", "ctr %", "click through rate", "clickthrough rate"],
        "frequency": ["frequency", "freq", "avg frequency"],
        "roas": ["roas", "return on ad spend", "roi"],
        "purchases": ["purchases", "conversions", "sales", "purchase"],
        "purchase_value": ["purchase value", "revenue", "conversion value", "sales value"],
        "add_to_cart": ["adds to cart", "add to cart", "atc", "cart adds"],
        "atc_to_purchase": ["atc→purchase %", "atc to purchase", "cart to purchase", "conversion rate"],
        "ctr_7d": ["ctr 7d %", "ctr 7d", "ctr last 7 days"],
        "ctr_prev_7d": ["ctr prev7 %", "ctr prev 7d", "ctr previous 7 days"],
        "ctr_drop": ["ctr drop vs prev7 %", "ctr drop", "ctr decline"],
        "status": ["status", "state", "ad status"],
    }

    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0)

    def fuzzy_match(self, column: str) -> Optional[str]:
        """Fast fuzzy matching before LLM"""
        column_lower = column.lower().strip()

        for standard, variations in self.STANDARD_COLUMNS.items():
            if column_lower in variations:
                return standard
            # Check partial matches
            for variant in variations:
                if variant in column_lower or column_lower in variant:
                    return standard

        return None

    def map_columns_with_llm(self, columns: List[str]) -> Dict[str, str]:
        """Use LLM to semantically map columns"""

        unmapped = []
        mapping = {}

        # First pass: fuzzy matching
        for col in columns:
            matched = self.fuzzy_match(col)
            if matched:
                mapping[col] = matched
            else:
                unmapped.append(col)

        # Second pass: LLM for unmapped columns
        if unmapped:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a data column mapper for marketing analytics.
Map the given column names to standard schema fields.

Standard fields available:
{standard_fields}

Return ONLY a JSON object mapping input columns to standard fields.
If a column doesn't match any standard field, map it to "unknown".

Example:
Input: ["Daily Budget", "Link Clicks", "Revenue"]
Output: {{"Daily Budget": "spend", "Link Clicks": "clicks", "Revenue": "purchase_value"}}
"""),
                ("user", "Map these columns: {columns}")
            ])

            chain = prompt | self.llm

            try:
                response = chain.invoke({
                    "standard_fields": ", ".join(self.STANDARD_COLUMNS.keys()),
                    "columns": str(unmapped)
                })

                # Parse LLM response
                import json
                content = response.content.strip()

                # Extract JSON from markdown if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                llm_mapping = json.loads(content)
                mapping.update(llm_mapping)
            except Exception as e:
                print(f"LLM mapping failed: {e}")
                # Fallback: mark as unknown
                for col in unmapped:
                    mapping[col] = "unknown"

        return mapping

    def normalize_data(self, data: List[Dict], column_mapping: Optional[Dict[str, str]] = None) -> List[Dict]:
        """Normalize data using column mapping"""

        if not data:
            return []

        # Auto-detect mapping if not provided
        if column_mapping is None:
            columns = list(data[0].keys())
            column_mapping = self.map_columns_with_llm(columns)

        # Create reverse mapping (original -> standard)
        reverse_map = {orig: std for orig, std in column_mapping.items() if std != "unknown"}

        # Normalize all rows
        normalized = []
        for row in data:
            new_row = {}
            for orig_col, value in row.items():
                standard_col = reverse_map.get(orig_col, orig_col)
                new_row[standard_col] = value
            normalized.append(new_row)

        return normalized
