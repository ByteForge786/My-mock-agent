```python
# app.py
import streamlit as st
from typing import Dict, Any
import json
import yaml

def initialize_session_state():
    """Initialize Streamlit session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = Agent("schema_config.yaml")

def format_step_details(step: PlanStep) -> str:
    """Format execution step details for display"""
    details = f"**Tool**: {step.tool}\n"
    details += f"**Reason**: {step.reason}\n"
    details += f"**Status**: {step.status}\n"
    
    if step.result:
        if step.tool == ToolType.SCHEMA.value:
            details += "\n**Found Tables**:\n"
            for table in step.result.get("relevant_tables", []):
                details += f"- {table['name']}: {table['description']}\n"
        
        elif step.tool == ToolType.SQL.value:
            details += f"\n**Generated SQL**:\n```sql\n{step.result.get('sql', '')}\n```"
            details += f"\n**Explanation**:\n{step.result.get('explanation', '')}"
        
        elif step.tool == ToolType.VALIDATION.value:
            details += "\n**Validation Results**:\n"
            if step.result.get("is_safe"):
                details += "✅ SQL is safe to execute\n"
            else:
                details += "❌ SQL has issues:\n"
                for issue in step.result.get("issues", []):
                    details += f"- {issue}\n"
        
        elif step.tool == ToolType.EXECUTION.value:
            details += f"\n**Execution Results**:\n"
            details += f"- Rows: {step.result.get('row_count', 0)}\n"
            details += f"- Columns: {', '.join(step.result.get('columns', []))}\n"
    
    return details

def create_streamlit_app():
    st.set_page_config(
        page_title="SQL Analytics Assistant",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    initialize_session_state()
    
    st.title("SQL Analytics Assistant")
    
    # Show capabilities in expander
    with st.expander("💡 What can I help you with?"):
        st.markdown("""
        I can help you analyze your business data! Try asking:
        
        📊 **Data Analysis**
        - Show me total sales by category
        - What's the trend of daily revenue?
        - Who are our top 10 customers?
        
        📈 **Trends and Patterns**
        - How have sales changed over time?
        - Show me monthly revenue trends
        - Compare sales across regions
        
        🔍 **Specific Queries**
        - What products had the highest sales last month?
        - Show me customer distribution by segment
        - Which category generates most revenue?
        """)
    
    # Chat input
    if query := st.chat_input("Ask about your data..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Process query
        with st.spinner("Thinking..."):
            response = st.session_state.agent.process_query(query)
            
            # Format response
            if response["type"] == "conversation":
                content = response["response"]
                if "capabilities" in response:
                    content += "\n\nI can help you analyze data from these tables:\n"
                    for table in response["capabilities"]["relevant_tables"]:
                        content += f"\n- {table['name']}: {table['description']}"
            
            elif response["type"] == "error":
                content = f"❌ {response['message']}"
            
            else:  # analysis
                content = "I've analyzed your request. Here's what I found:"
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": content,
                "response": response
            })
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show additional details for assistant responses
            if message["role"] == "assistant" and "response" in message:
                response = message["response"]
                
                # For analysis responses
                if response["type"] == "analysis":
                    # Show execution flow
                    with st.expander("🔍 View Analysis Steps"):
                        for step in response["steps"]:
                            st.markdown(format_step_details(step))
                            st.markdown("---")
                    
                    # Show results
                    if "context" in response:
                        context = response["context"]
                        if context.get("data") is not None:
                            st.dataframe(context["data"])
                        
                        # Show visualization if available
                        if response.get("needs_visualization") and "figure" in context:
                            st.plotly_chart(context["figure"], use_container_width=True)
                
                # For errors or warnings
                elif response["type"] == "error":
                    st.error(response["message"])
                    if "details" in response:
                        with st.expander("See Error Details"):
                            st.json(response["details"])

def main():
    create_streamlit_app()

if __name__ == "__main__":
    main()
```
