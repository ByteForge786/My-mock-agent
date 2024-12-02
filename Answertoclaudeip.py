import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
import yaml
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToolType(Enum):
    SCHEMA = "schema_lookup"
    SQL = "sql_generation" 
    VALIDATION = "sql_validation"
    EXECUTION = "sql_execution"
    VISUALIZATION = "visualization"

@dataclass
class PlanStep:
    tool: str
    inputs: Dict[str, Any]
    reason: str
    status: str = "pending"
    result: Optional[Dict] = None
    error: Optional[str] = None
    order: int = 0

class Tool:
    def execute(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

class SchemaLookupTool(Tool):
    def __init__(self, schema_path: str):
        with open(schema_path) as f:
            self.schema_config = yaml.safe_load(f)
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self._init_embeddings()

    def _init_embeddings(self):
        self.table_embeddings = {}
        for table_name, info in self.schema_config['tables'].items():
            text = f"{table_name} {info['description']}"
            self.table_embeddings[table_name] = self.embed_model.encode(text)

    def execute(self, query: str) -> Dict[str, Any]:
        query_embedding = self.embed_model.encode(query)
        relevant_tables = []
        
        for table_name, embedding in self.table_embeddings.items():
            similarity = np.dot(query_embedding, embedding)
            if similarity > 0.5:
                table_info = self.schema_config['tables'][table_name]
                relevant_tables.append({
                    "name": table_name,
                    "create_statement": table_info['create_statement'],
                    "description": table_info['description'],
                    "sample_questions": table_info.get('sample_questions', [])
                })
        
        return {
            "status": "success",
            "relevant_tables": relevant_tables,
            "schema_context": "\n".join(t["create_statement"] for t in relevant_tables)
        }

class SQLGenerationTool(Tool):
    def execute(self, user_query: str, schema_context: str) -> Dict[str, Any]:
        prompt = f"""
Generate a Snowflake SQL query based on:
User Question: {user_query}

Available Schema:
{schema_context}

Requirements:
1. Use only SELECT statements (no DML/DDL)
2. Include proper table aliases
3. Use appropriate JOINs
4. Add necessary filters
5. Use Snowflake date functions if needed

Return format:
{{
    "status": "success",
    "sql": "your SQL query",
    "explanation": "step by step explanation"
}}
"""
        return get_llm_response(prompt)

class SQLValidationTool(Tool):
    MAX_RETRIES = 3

    def execute(self, sql: str, schema_context: str, retry_count: int = 0) -> Dict[str, Any]:
        prompt = f"""
Validate this SQL query:
{sql}

Against schema:
{schema_context}

Check for:
1. SQL injection risks
2. DML/DDL statements (not allowed)
3. Proper JOIN conditions
4. Syntax correctness

Return format:
{{
    "status": "success" or "error",
    "is_safe": boolean,
    "issues": ["list of issues"],
    "feedback": "improvement suggestions if any"
}}
"""
        validation_result = get_llm_response(prompt)
        
        # If validation failed and we haven't exceeded retries
        if not validation_result.get("is_safe", False) and retry_count < self.MAX_RETRIES:
            validation_result["status"] = "retry"
            validation_result["retry_count"] = retry_count + 1
            
        return validation_result

class SQLExecutionTool(Tool):
    def execute(self, sql: str) -> Dict[str, Any]:
        # Mock execution with sample data
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30),
            'sales': np.random.uniform(1000, 5000, 30),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 30)
        })

        return {
            "status": "success",
            "data": df,
            "metadata": {
                "columns": list(df.columns),
                "row_count": len(df),
                "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
        }

class VisualizationTool(Tool):
    def execute(self, user_query: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
Create visualization config for:
User Question: {user_query}

Available Data:
Columns: {metadata['columns']}
Types: {metadata['column_types']}

Return format:
{{
    "status": "success",
    "viz_type": "line" or "bar",
    "config": {{
        "x": "column_name",
        "y": "column_name",
        "color": "optional_grouping_column",
        "title": "chart_title"
    }}
}}
"""
        return get_llm_response(prompt)

class Agent:
    def __init__(self, schema_path: str):
        self.tools = {
            ToolType.SCHEMA.value: SchemaLookupTool(schema_path),
            ToolType.SQL.value: SQLGenerationTool(),
            ToolType.VALIDATION.value: SQLValidationTool(),
            ToolType.EXECUTION.value: SQLExecutionTool(),
            ToolType.VISUALIZATION.value: VisualizationTool()
        }
        
        with open(schema_path) as f:
            self.schema_config = yaml.safe_load(f)

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Determine query intent and create execution plan"""
        prompt = f"""
Analyze query: "{query}"

Available tools: {[t.value for t in ToolType]}

Determine:
1. Is this a greeting/help request or data analysis question?
2. If data analysis, what tools are needed in what order?
3. Will visualization help answer this question?

Return format:
{{
    "query_type": "greeting" or "help" or "data_analysis" or "off_topic",
    "needs_visualization": boolean,
    "plan": [
        {{
            "tool": "tool_name",
            "reason": "why this step is needed"
        }}
    ]
}}
"""
        return get_llm_response(prompt)

    def handle_greeting_or_help(self, query: str) -> Dict[str, Any]:
        """Handle non-data analysis queries"""
        # Get schema context for better response
        schema_info = self.tools[ToolType.SCHEMA.value].execute(query)
        
        prompt = f"""
Respond to: "{query}"

Available data/capabilities:
{json.dumps(self.schema_config.get('analysis_capabilities', []), indent=2)}

Tables:
{[table['name'] for table in schema_info.get('relevant_tables', [])]}

Return format:
{{
    "status": "success",
    "response": "your helpful response"
}}
"""
        response = get_llm_response(prompt)
        return {
            "type": "conversation",
            "response": response.get("response", "How can I help you analyze the data?")
        }

    def process_query(self, query: str) -> Dict[str, Any]:
        """Main query processing pipeline"""
        try:
            # Step 1: Analyze query intent and get plan
            analysis = self.analyze_query(query)
            logger.info(f"Query analysis: {analysis}")

            # Handle non-data queries
            if analysis["query_type"] in ["greeting", "help"]:
                return self.handle_greeting_or_help(query)
            elif analysis["query_type"] == "off_topic":
                return {
                    "type": "conversation",
                    "response": "I can only help with data analysis questions about the available data."
                }

            # Step 2: Execute the plan
            context = {"query": query}
            executed_steps = []

            for step in analysis["plan"]:
                tool_name = step["tool"]
                tool = self.tools[tool_name]

                if tool_name == ToolType.SCHEMA.value:
                    result = tool.execute(query=query)
                    context.update(result)

                elif tool_name == ToolType.SQL.value:
                    result = tool.execute(
                        user_query=query,
                        schema_context=context["schema_context"]
                    )
                    context.update(result)

                elif tool_name == ToolType.VALIDATION.value:
                    retry_count = 0
                    while retry_count < SQLValidationTool.MAX_RETRIES:
                        result = tool.execute(
                            sql=context["sql"],
                            schema_context=context["schema_context"],
                            retry_count=retry_count
                        )
                        if result["status"] == "success":
                            break
                        if result["status"] == "retry":
                            # Regenerate SQL with feedback
                            sql_result = self.tools[ToolType.SQL.value].execute(
                                user_query=f"{query} [Feedback: {result['feedback']}]",
                                schema_context=context["schema_context"]
                            )
                            context.update(sql_result)
                            retry_count += 1
                        else:
                            break
                    context.update(result)

                elif tool_name == ToolType.EXECUTION.value:
                    result = tool.execute(sql=context["sql"])
                    context.update(result)

                elif tool_name == ToolType.VISUALIZATION.value and analysis.get("needs_visualization"):
                    result = tool.execute(
                        user_query=query,
                        metadata=context["metadata"]
                    )
                    if result["status"] == "success":
                        # Create visualization using the config
                        viz_config = result["config"]
                        df = context["data"]
                        if result["viz_type"] == "line":
                            fig = px.line(df, **viz_config)
                        else:
                            fig = px.bar(df, **viz_config)
                        context["visualization"] = fig

                executed_steps.append({
                    "tool": tool_name,
                    "status": result["status"],
                    "result": result
                })

            return {
                "type": "analysis",
                "steps": executed_steps,
                "context": context,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Error processing query: {traceback.format_exc()}")
            return {
                "type": "error",
                "message": str(e),
                "status": "error"
            }
