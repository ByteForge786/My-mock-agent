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
    CONVERSATION = "conversation"

@dataclass
class PlanStep:
    tool: str
    inputs: Dict[str, Any]
    reason: str
    status: str = "pending"
    result: Optional[Dict] = None
    error: Optional[str] = None
    order: int = 0  # Track step order in plan

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
                    "description": table_info['description']
                })
        
        return {
            "status": "success",
            "relevant_tables": relevant_tables,
            "schema_context": "\n".join(t["create_statement"] for t in relevant_tables)
        }

class SQLGenerationTool(Tool):
    def execute(self, query: str, schema_context: str, current_step: int, total_steps: int) -> Dict[str, Any]:
        prompt = f"""
Current Step: {current_step} of {total_steps}
Purpose: Generate SQL for query

User Query: "{query}"

Available Schema:
{schema_context}

Requirements:
1. Use Snowflake SQL syntax
2. Include proper table aliases
3. Use appropriate JOINs
4. Add necessary filters
5. Format dates using Snowflake functions

Previous steps have provided schema context. Generate SQL that will be validated next.

Response format:
{{
    "sql": "Complete SQL query",
    "explanation": "Step by step explanation",
    "tables_used": ["table1", "table2"]
}}
"""
        return get_llm_response(prompt)

class SQLValidationTool(Tool):
    def execute(self, sql: str, current_step: int, total_steps: int) -> Dict[str, Any]:
        prompt = f"""
Current Step: {current_step} of {total_steps}
Purpose: Validate SQL for safety and correctness

SQL Query:
{sql}

Check for:
1. Injection risks
2. Proper JOIN conditions
3. Appropriate WHERE clauses
4. Resource usage
5. Syntax correctness

Next step will execute this SQL if validated.

Response format:
{{
    "is_safe": boolean,
    "issues": ["list of issues"],
    "suggestions": ["improvements"],
    "requires_modification": boolean
}}
"""
        return get_llm_response(prompt)

class SQLExecutionTool(Tool):
    def execute(self, sql: str, current_step: int, total_steps: int) -> Dict[str, Any]:
        # Mock data for demonstration
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30),
            'sales': np.random.uniform(1000, 5000, 30),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 30)
        })

        return {
            "status": "success",
            "data": df,
            "summary": {
                "row_count": len(df),
                "columns": list(df.columns),
                "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
                "temporal_columns": df.select_dtypes(include=['datetime64']).columns.tolist()
            }
        }

class VisualizationTool(Tool):
    def execute(self, data: pd.DataFrame, viz_type: str, current_step: int, total_steps: int) -> Dict[str, Any]:
        prompt = f"""
Current Step: {current_step} of {total_steps}
Purpose: Configure visualization

Data Summary:
- Columns: {list(data.columns)}
- Types: {data.dtypes.to_dict()}

Determine visualization configuration for type: {viz_type}

Response format:
{{
    "config": {{
        "x": "column_name",
        "y": "column_name",
        "color": "optional_column",
        "title": "chart_title"
    }}
}}
"""
        config = get_llm_response(prompt)
        
        try:
            if viz_type == "line":
                fig = px.line(data, **config["config"])
            else:
                fig = px.bar(data, **config["config"])
            
            return {
                "status": "success",
                "figure": fig
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

class ConversationTool(Tool):
    def execute(self, query: str, capabilities: List[str]) -> Dict[str, Any]:
        prompt = f"""
Handle user query: "{query}"

Available capabilities:
{json.dumps(capabilities, indent=2)}

If greeting/help: Provide friendly response explaining capabilities
If off-topic: Redirect to data analysis capabilities
If unclear: Ask for clarification about data analysis needs

Response format:
{{
    "response": "complete response",
    "query_type": "greeting|help|off_topic|unclear",
    "suggestions": ["list of suggested queries"]
}}
"""
        return get_llm_response(prompt)

class Agent:
    def __init__(self, schema_path: str, max_steps: int = 10):
        self.max_steps = max_steps
        self.execution_history = []
        
        # Initialize tools
        self.tools = {
            ToolType.SCHEMA.value: SchemaLookupTool(schema_path),
            ToolType.SQL.value: SQLGenerationTool(),
            ToolType.VALIDATION.value: SQLValidationTool(),
            ToolType.EXECUTION.value: SQLExecutionTool(),
            ToolType.VISUALIZATION.value: VisualizationTool(),
            ToolType.CONVERSATION.value: ConversationTool()
        }
        
        with open(schema_path) as f:
            self.schema_config = yaml.safe_load(f)

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Initial analysis to determine type and create execution plan"""
        prompt = f"""
Analyze query: "{query}"

Available tools:
{json.dumps([t.value for t in ToolType], indent=2)}

Determine:
1. Query type (data_analysis, greeting, help, off_topic)
2. Required analysis steps
3. Visualization needs

For data analysis, create complete execution plan considering dependencies.

Response format:
{{
    "query_type": "data_analysis|greeting|help|off_topic",
    "plan": [
        {{
            "step": 1,
            "tool": "tool_name",
            "reason": "why this step",
            "depends_on": ["previous step numbers"],
            "required_inputs": ["input names"],
            "expected_outputs": ["output names"]
        }}
    ],
    "visualization": {{
        "required": boolean,
        "type": "line|bar|none",
        "reason": "why this visualization"
    }}
}}
"""
        return get_llm_response(prompt)

    def create_execution_plan(self, analysis: Dict[str, Any]) -> List[PlanStep]:
        """Convert analysis into executable steps with dependencies"""
        steps = []
        for step_info in analysis.get("plan", []):
            step = PlanStep(
                tool=step_info["tool"],
                inputs={},  # Will be populated during execution
                reason=step_info["reason"],
                order=step_info["step"]
            )
            steps.append(step)
        return sorted(steps, key=lambda x: x.order)

    def determine_next_step(self, 
                          current_step: PlanStep,
                          executed_steps: List[PlanStep], 
                          context: Dict[str, Any],
                          total_steps: int) -> Dict[str, Any]:
        """Determine inputs for next step execution with full context"""
        prompt = f"""
Planning Step {current_step.order} of {total_steps}
Tool: {current_step.tool}
Purpose: {current_step.reason}

Context available:
{json.dumps(list(context.keys()), indent=2)}

Previous steps executed:
{json.dumps([{'tool': s.tool, 'status': s.status} for s in executed_steps], indent=2)}

Determine exact inputs needed for current step.

Response format:
{{
    "inputs": {{
        "param_name": "context_key"
    }},
    "reason": "Why these inputs were chosen"
}}
"""
        return get_llm_response(prompt)

    def process_query(self, query: str) -> Dict[str, Any]:
        """Main query processing pipeline with step tracking"""
        try:
            # Initial analysis and plan creation
            analysis = self.analyze_query(query)
            if analysis["query_type"] != "data_analysis":
                return self._handle_conversation(query)

            # Create initial plan
            plan = self.create_execution_plan(analysis)
            total_steps = len(plan)
            context = {"query": query}
            executed_steps = []

            # Execute plan
            for step in plan:
                if len(executed_steps) >= self.max_steps:
                    break

                # Determine inputs for current step
                next_action = self.determine_next_step(
                    current_step=step,
                    executed_steps=executed_steps,
                    context=context,
                    total_steps=total_steps
                )

                # Execute step with awareness of position in plan
                result = self.tools[step.tool].execute(
                    **next_action["inputs"],
                    current_step=step.order,
                    total_steps=total_steps
                )

                step.result = result
                step.status = "completed" if result["status"] == "success" else "error"
                executed_steps.append(step)

                # Update context with results
                if isinstance(result, dict):
                    context.update(result)

                # Check for errors
                if step.status == "error":
                    return self._handle_error(
                        f"Failed at step {step.order}: {step.tool}",
                        result.get("error", "Unknown error"),
                        executed_steps
                    )

            # Prepare final response
            return {
                "type": "analysis",
                "context": context,
                "steps": executed_steps,
                "visualization": analysis.get("visualization", {}),
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Query processing failed: {traceback.format_exc()}")
            return self._handle_error("Failed to process query", str(e))

    def _handle_conversation(self, query: str) -> Dict[str, Any]:
        """Handle non-data analysis queries"""
        capabilities = self.schema_config.get("capabilities", [])
        result = self.tools[ToolType.CONVERSATION.value].execute(
            query=query,
            capabilities=capabilities
        )
        return {
            "type": "conversation",
            "response": result.get("response", "I can help you analyze data."),
            "suggestions": result.get("suggestions", [])
        }

    def _handle_error(self, message: str, details: str, steps: List[PlanStep] = None) -> Dict[str, Any]:
        """Standardized error handling"""
        return {
            "type": "error",
            "message": message,
            "details": details,
            "steps": steps
        }
