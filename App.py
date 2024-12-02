```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
import yaml
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import sqlparse
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Tool(ABC):
    """Base class for all tools"""
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        pass

class UserIntentTool(Tool):
    """LLM-based tool for analyzing user intent and determining next steps"""
    def execute(self, query: str) -> Dict[str, Any]:
        prompt = f"""
Analyze this user query: "{query}"

Determine the appropriate response and next steps.

Consider:
1. Is this a general question about capabilities or a greeting?
2. Is this a data analysis question requiring SQL?
3. Does this question require database schema understanding?
4. Is this question relevant to business data analysis?
5. Would visualization help answer this question effectively?

Respond in JSON format:
{{
    "query_type": "conversation|data_analysis|off_topic",
    "needs_sql": boolean,
    "needs_schema_lookup": boolean,
    "needs_visualization": boolean,
    "response": string,  # For direct responses to greetings/help
    "reasoning": string,
    "next_steps": ["schema_lookup"|"sql_generation"|"visualization"],
    "visualization_type": "line|bar|pie|scatter|none"  # If visualization needed
}}

Stay focused on business data analysis domain. Only process relevant queries.
"""
        # Mock LLM response based on query patterns
        # In production, replace with actual LLM call
        mock_response = {
            "query_type": "data_analysis",
            "needs_sql": True,
            "needs_schema_lookup": True,
            "needs_visualization": "trend" in query.lower() or "compare" in query.lower(),
            "reasoning": "Query requires data analysis and potential visualization",
            "next_steps": ["schema_lookup", "sql_generation"],
            "visualization_type": "line" if "trend" in query.lower() else "bar"
        }
        
        return mock_response

class SchemaLookupTool(Tool):
    """Tool for finding relevant schema based on query"""
    def __init__(self, schema_path: str):
        with open(schema_path) as f:
            self.schema_config = yaml.safe_load(f)
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embeddings for schema matching"""
        self.table_embeddings = {}
        for table_name, info in self.schema_config['tables'].items():
            text = f"{table_name} {info['description']}"
            self.table_embeddings[table_name] = self.embed_model.encode(text)

    def execute(self, query: str) -> Dict[str, Any]:
        prompt = f"""
Given the user query: "{query}"

Analyze the query and available schema to determine:
1. Which tables are relevant to this query?
2. What relationships between tables are needed?
3. What columns are required for the analysis?

Available tables and their purposes:
{json.dumps([{k: v['description']} for k, v in self.schema_config['tables'].items()], indent=2)}

Response format:
{{
    "relevant_tables": ["table1", "table2"],
    "required_relationships": ["table1.col = table2.col"],
    "explanation": "Why these tables are needed"
}}
"""
        # Mock LLM schema selection
        query_embedding = self.embed_model.encode(query)
        
        # Find relevant tables using embedding similarity
        relevant_tables = []
        for table_name, embedding in self.table_embeddings.items():
            similarity = np.dot(query_embedding, embedding)
            if similarity > 0.5:  # Threshold
                relevant_tables.append({
                    "name": table_name,
                    "create_statement": self.schema_config['tables'][table_name]['create_statement'],
                    "description": self.schema_config['tables'][table_name]['description']
                })

        return {
            "status": "success",
            "relevant_tables": relevant_tables,
            "schema_context": "\n".join(t["create_statement"] for t in relevant_tables)
        }

class SQLGenerationTool(Tool):
    """Tool for generating SQL based on schema context"""
    def execute(self, query: str, schema_context: str) -> Dict[str, Any]:
        prompt = f"""
Generate SQL for: "{query}"

Available Schema:
{schema_context}

Requirements:
1. Use only available tables/columns
2. Include proper joins
3. Add appropriate filters/aggregations
4. Optimize for performance

Response format:
{{
    "sql": "SQL query",
    "explanation": "Query logic explanation",
    "needs_visualization": boolean,
    "visualization_suggestion": {{
        "type": "line|bar|pie|scatter",
        "x_column": "column_name",
        "y_column": "column_name"
    }}
}}
"""
        # Mock SQL generation
        mock_sql = """
        SELECT 
            category,
            SUM(revenue) as total_revenue
        FROM sales
        GROUP BY category
        ORDER BY total_revenue DESC
        """
        
        return {
            "sql": mock_sql,
            "explanation": "Generated SQL based on schema and query intent",
            "needs_visualization": True,
            "visualization_suggestion": {
                "type": "bar",
                "x_column": "category",
                "y_column": "total_revenue"
            }
        }

class SQLValidationTool(Tool):
    """Tool for validating SQL safety and providing feedback"""
    def execute(self, sql: str) -> Dict[str, Any]:
        prompt = f"""
Analyze this SQL query for safety and correctness:

{sql}

Check for:
1. Dangerous operations (DROP, DELETE, TRUNCATE, etc.)
2. Syntax correctness
3. Basic query structure

Respond with:
{{
    "is_safe": boolean,
    "needs_regeneration": boolean,
    "feedback": "Explanation of issues if any",
    "suggested_fixes": ["fix1", "fix2"]
}}
"""
        # Basic validation
        dangerous_patterns = ["DROP", "DELETE", "TRUNCATE", "ALTER"]
        is_dangerous = any(pattern.lower() in sql.lower() for pattern in dangerous_patterns)
        
        return {
            "is_safe": not is_dangerous,
            "needs_regeneration": is_dangerous,
            "feedback": "Query contains dangerous operations" if is_dangerous else "Query is safe",
            "suggested_fixes": [] if not is_dangerous else ["Remove dangerous operations"]
        }

class VisualizationTool(Tool):
    """Tool for generating appropriate visualizations"""
    def execute(self, query: str, data: pd.DataFrame, suggestion: Dict) -> Dict[str, Any]:
        prompt = f"""
Analyze query and data to create visualization:

Query: "{query}"
Data Columns: {list(data.columns)}
Data Sample: {data.head(2).to_dict()}
Suggestion: {json.dumps(suggestion)}

Determine:
1. Best visualization type
2. Required data transformations
3. Aesthetic configurations

Response format:
{{
    "type": "line|bar|pie|scatter",
    "config": {{
        "x": "column_name",
        "y": "column_name",
        "title": "chart_title"
    }},
    "explanation": "Why this visualization works best"
}}
"""
        # Create visualization based on suggestion
        if suggestion["type"] == "line":
            fig = px.line(data, 
                         x=suggestion["x_column"], 
                         y=suggestion["y_column"],
                         title=f"Trend of {suggestion['y_column']} over {suggestion['x_column']}")
        else:
            fig = px.bar(data,
                        x=suggestion["x_column"],
                        y=suggestion["y_column"],
                        title=f"{suggestion['y_column']} by {suggestion['x_column']}")
        
        return {
            "figure": fig,
            "explanation": f"Created {suggestion['type']} chart based on data characteristics"
        }

class MockDatabase:
    """Mock database for testing"""
    def __init__(self):
        self.generate_mock_data()

    def generate_mock_data(self):
        self.data = pd.DataFrame({
            'category': ['A', 'B', 'C'] * 10,
            'revenue': np.random.uniform(1000, 5000, 30),
            'date': pd.date_range('2024-01-01', periods=30)
        })

    def execute_query(self, sql: str) -> pd.DataFrame:
        return self.data

class SQLAssistant:
    """Main assistant class orchestrating all tools"""
    def __init__(self, schema_path: str):
        self.tools = {
            "intent": UserIntentTool(),
            "schema": SchemaLookupTool(schema_path),
            "sql": SQLGenerationTool(),
            "validation": SQLValidationTool(),
            "visualization": VisualizationTool()
        }
        self.db = MockDatabase()

    def process_query(self, query: str) -> Dict[str, Any]:
        flow = []
        
        try:
            # 1. Analyze user intent
            intent_result = self.tools["intent"].execute(query)
            flow.append({"step": "intent_analysis", "result": intent_result})

            if intent_result["query_type"] == "conversation":
                return {
                    "type": "conversation",
                    "response": intent_result["response"],
                    "flow": flow
                }

            # 2. Schema lookup if needed
            if intent_result["needs_schema_lookup"]:
                schema_result = self.tools["schema"].execute(query)
                flow.append({"step": "schema_lookup", "result": schema_result})

                # 3. Generate SQL
                sql_result = self.tools["sql"].execute(
                    query, 
                    schema_result["schema_context"]
                )
                flow.append({"step": "sql_generation", "result": sql_result})

                # 4. Validate SQL
                validation_result = self.tools["validation"].execute(sql_result["sql"])
                flow.append({"step": "sql_validation", "result": validation_result})

                if not validation_result["is_safe"]:
                    return {
                        "type": "error",
                        "response": validation_result["feedback"],
                        "flow": flow
                    }

                # 5. Execute SQL
                data = self.db.execute_query(sql_result["sql"])
                
                # 6. Generate visualization if needed
                viz_result = None
                if sql_result["needs_visualization"]:
                    viz_result = self.tools["visualization"].execute(
                        query, 
                        data, 
                        sql_result["visualization_suggestion"]
                    )
                    flow.append({"step": "visualization", "result": viz_result})

                return {
                    "type": "analysis",
                    "data": data,
                    "sql": sql_result["sql"],
                    "visualization": viz_result["figure"] if viz_result else None,
                    "flow": flow
                }

        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {
                "type": "error",
                "response": str(e),
                "flow": flow
            }

def create_streamlit_app():
    st.set_page_config(page_title="SQL Assistant", layout="wide")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "assistant" not in st.session_state:
        st.session_state.assistant = SQLAssistant("schema_config.yaml")

    st.title("SQL Analytics Assistant")
    
    # Chat input
    if prompt := st.chat_input("Ask about your data"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Processing..."):
            response = st.session_state.assistant.process_query(prompt)
            
            if response["type"] == "conversation":
                content = response["response"]
            elif response["type"] == "analysis":
                content = f"Here's what I found:\n\nSQL Query:\n```sql\n{response['sql']}\n```"
            else:
                content = f"Error: {response['response']}"

            message = {
                "role": "assistant",
                "content": content,
                "response": response
            }
            st.session_state.messages.append(message)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "response" in message:
                response = message["response"]
                if response["type"] == "analysis":
                    st.dataframe(response["data"])
                    if response.get("visualization"):
                        st.plotly_chart(response["visualization"])
                
                with st.expander("Show execution flow"):
                    st.json(response["flow"])

if __name__ == "__main__":
    create_streamlit_app()
```
