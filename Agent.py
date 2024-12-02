```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
import yaml
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

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
        # Get relevant tables using embeddings
        query_embedding = self.embed_model.encode(query)
        relevant_tables = []
        
        for table_name, embedding in self.table_embeddings.items():
            similarity = np.dot(query_embedding, embedding)
            if similarity > 0.5:
                table_info = self.schema_config['tables'][table_name]
                relevant_tables.append({
                    "name": table_name,
                    "create_statement": table_info['create_statement'],
                    "description": table_info['description']
                })
        
        return {
            "status": "success",
            "relevant_tables": relevant_tables,
            "schema_context": "\n".join(t["create_statement"] for t in relevant_tables)
        }

class SQLGenerationTool(Tool):
    def execute(self, query: str, schema_context: str) -> Dict[str, Any]:
        prompt = f"""
Generate a Snowflake SQL query for: "{query}"

Available Schema:
{schema_context}

Requirements:
1. Use Snowflake SQL syntax
2. Include proper table aliases
3. Use appropriate JOINs
4. Add necessary filters
5. Format dates using SNOWFLAKE functions (e.g., DATE_TRUNC, DATEADD)
6. Include aggregations if needed
7. Optimize for performance

Example Snowflake SQL patterns:
- Date filtering: DATE_TRUNC('month', date_column)
- Time windows: DATEADD(month, -1, CURRENT_DATE())
- String patterns: ILIKE for case-insensitive matching
- Aggregations with QUALIFY for row limits

Response format:
{{
    "sql": "Complete SQL query",
    "explanation": "Step by step explanation",
    "tables_used": ["table1", "table2"]
}}
"""
        return get_llm_response(prompt)

class SQLValidationTool(Tool):
    def execute(self, sql: str) -> Dict[str, Any]:
        prompt = f"""
Analyze this SQL query for safety and correctness:

{sql}

Check for:
1. DROP, DELETE, TRUNCATE, ALTER statements
2. Data modification (INSERT, UPDATE)
3. System table access
4. Proper JOIN conditions
5. Appropriate WHERE clauses
6. Resource-intensive operations
7. Snowflake syntax compliance

Response format:
{{
    "is_safe": boolean,
    "issues": ["list of issues found"],
    "suggestions": ["improvement suggestions"],
    "requires_modification": boolean
}}
"""
        return get_llm_response(prompt)

class SQLExecutionTool(Tool):
    def __init__(self, mock: bool = True):
        self.mock = mock
        if mock:
            self.mock_data = self._generate_mock_data()

    def _generate_mock_data(self):
        return pd.DataFrame({
            'category': ['Electronics', 'Clothing', 'Books'] * 10,
            'revenue': np.random.uniform(1000, 5000, 30),
            'date': pd.date_range('2024-01-01', periods=30)
        })

    def execute(self, sql: str) -> Dict[str, Any]:
        try:
            if self.mock:
                df = self.mock_data
            else:
                # df = pd.read_sql(sql, conn)
                pass

            return {
                "status": "success",
                "columns": list(df.columns),
                "row_count": len(df),
                "has_data": len(df) > 0,
                "column_types": {col: str(df[col].dtype) for col in df.columns},
                # Don't include actual data in the response
                "summary": {
                    "numeric_columns": [col for col, dtype in df.dtypes.items() if np.issubdtype(dtype, np.number)],
                    "categorical_columns": [col for col, dtype in df.dtypes.items() if dtype == 'object'],
                    "temporal_columns": [col for col, dtype in df.dtypes.items() if np.issubdtype(dtype, np.datetime64)]
                }
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

class VisualizationTool(Tool):
    def execute(self, chart_type: str, config: Dict, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            if chart_type == "line":
                fig = px.line(data, 
                            x=config["x"],
                            y=config["y"],
                            title=config.get("title", ""))
            else:
                fig = px.bar(data,
                            x=config["x"],
                            y=config["y"],
                            title=config.get("title", ""))
            
            return {
                "status": "success",
                "figure": fig
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

class Agent:
    def __init__(self, schema_path: str):
        self.tools = {
            ToolType.SCHEMA.value: SchemaLookupTool(schema_path),
            ToolType.SQL.value: SQLGenerationTool(),
            ToolType.VALIDATION.value: SQLValidationTool(),
            ToolType.EXECUTION.value: SQLExecutionTool(),
            ToolType.VISUALIZATION.value: VisualizationTool()
        }

    def analyze_query(self, query: str) -> Dict[str, Any]:
        prompt = f"""
Analyze this user query: "{query}"

Available tools:
{list(self.tools.keys())}

Consider:
1. Is this a greeting or help request?
2. Is this a data analysis question?
3. Is this off-topic?
4. What tools would be needed to answer?

For greetings/help: Only schema_lookup needed to explain capabilities
For data analysis: Plan complete analysis flow
For off-topic: Explain why it's not supported

Response format:
{{
    "query_type": "greeting|help|data_analysis|off_topic",
    "direct_response": string or null,
    "plan": [
        {{
            "tool": "tool_name",
            "inputs_needed": ["input1", "input2"],
            "reason": "Why this step is needed"
        }}
    ],
    "requires_visualization": boolean,
    "visualization_type": "line|bar|none",
    "reasoning": "Full explanation of the decision"
}}
"""
        return get_llm_response(prompt)

    def get_next_step(self, 
                     query: str, 
                     current_plan: List[PlanStep], 
                     executed_steps: List[PlanStep], 
                     available_tools: List[str]) -> Dict[str, Any]:
        prompt = f"""
Analyze current state:
Query: "{query}"
Executed steps: {[{
    "tool": step.tool,
    "status": step.status,
    "result_summary": "success" if step.result else "pending"
} for step in executed_steps]}
Remaining plan: {[{
    "tool": step.tool,
    "status": step.status
} for step in current_plan if step.status == "pending"]}

Determine next action:
1. What tool should be called next?
2. What inputs are needed?
3. Should we modify the plan?

Response format:
{{
    "next_tool": "tool_name or null",
    "required_inputs": {{"input_name": "source_of_input"}},
    "reason": "Why this is the next step",
    "plan_modification": "none|modify|stop"
}}
"""
        return get_llm_response(prompt)

    def process_query(self, query: str) -> Dict[str, Any]:
        # 1. Initial Analysis
        analysis = self.analyze_query(query)
        
        # Handle non-data queries
        if analysis["query_type"] != "data_analysis":
            if analysis["query_type"] in ["greeting", "help"]:
                # Get schema info for capabilities
                schema_result = self.tools[ToolType.SCHEMA.value].execute(query)
                return {
                    "type": "conversation",
                    "response": analysis["direct_response"],
                    "capabilities": schema_result
                }
            return {
                "type": "conversation",
                "response": analysis["direct_response"]
            }

        # 2. Execute Plan
        plan = [PlanStep(**step) for step in analysis["plan"]]
        executed_steps = []
        context = {"query": query}
        
        while True:
            next_action = self.get_next_step(
                query=query,
                current_plan=plan,
                executed_steps=executed_steps,
                available_tools=list(self.tools.keys())
            )
            
            if not next_action["next_tool"]:
                break
                
            # Execute next tool
            tool = self.tools[next_action["next_tool"]]
            inputs = {k: context.get(v) for k, v in next_action["required_inputs"].items()}
            
            try:
                result = tool.execute(**inputs)
                step = PlanStep(
                    tool=next_action["next_tool"],
                    inputs=inputs,
                    reason=next_action["reason"],
                    status="completed",
                    result=result
                )
                executed_steps.append(step)
                context.update(result)
                
            except Exception as e:
                logger.error(f"Tool execution failed: {str(e)}")
                return {
                    "type": "error",
                    "message": f"Failed to execute {next_action['next_tool']}: {str(e)}"
                }

        return {
            "type": "analysis",
            "context": context,
            "steps": executed_steps,
            "needs_visualization": analysis.get("requires_visualization", False)
        }
```
