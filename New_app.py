import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import yaml
import json
from agent import Agent
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'agent' not in st.session_state:
    st.session_state.agent = Agent('schema.yaml')

def load_schema_examples():
    """Load sample questions from schema"""
    with open('schema.yaml') as f:
        schema = yaml.safe_load(f)
    
    examples = []
    for table, info in schema['tables'].items():
        examples.extend(info.get('sample_questions', []))
    
    capabilities = schema.get('analysis_capabilities', [])
    examples.extend([cap['example'] for cap in capabilities])
    
    return examples

def display_visualization(result):
    """Display visualization based on agent result"""
    if result.get('visualization', {}).get('required'):
        if 'figure' in result.get('context', {}):
            st.plotly_chart(result['context']['figure'], use_container_width=True)
        else:
            st.warning("Visualization was requested but could not be generated")

def display_data_summary(result):
    """Display data summary from analysis"""
    if 'summary' in result.get('context', {}):
        summary = result['context']['summary']
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Rows Analyzed", summary.get('row_count', 0))
        with col2:
            st.metric("Columns Used", len(summary.get('columns', [])))
        
        # Show column types if available
        if 'numeric_columns' in summary:
            st.write("**Column Types:**")
            col_types = {
                "Numeric": summary.get('numeric_columns', []),
                "Categorical": summary.get('categorical_columns', []),
                "Temporal": summary.get('temporal_columns', [])
            }
            for type_name, cols in col_types.items():
                if cols:
                    st.write(f"- {type_name}: {', '.join(cols)}")

def display_query_steps(result):
    """Display execution steps in an expander"""
    if 'steps' in result:
        with st.expander("View Analysis Steps", expanded=False):
            for idx, step in enumerate(result['steps'], 1):
                st.markdown(f"**Step {idx}: {step.tool}**")
                st.write(f"Reason: {step.reason}")
                if step.status == "completed":
                    st.success("✓ Completed")
                elif step.status == "error":
                    st.error(f"✗ Failed: {step.error}")

def main():
    st.title("Data Analysis Assistant")
    
    # Sidebar with examples
    st.sidebar.title("Example Questions")
    examples = load_schema_examples()
    selected_example = st.sidebar.selectbox(
        "Try an example query:",
        [""] + examples
    )
    
    if selected_example:
        st.session_state.current_query = selected_example

    # Main chat interface
    st.write("Ask me anything about your sales data!")
    
    # Query input
    query = st.chat_input("Enter your query here...")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        with st.spinner("Analyzing..."):
            try:
                result = st.session_state.agent.process_query(query)
                
                # Display results based on type
                if result['type'] == 'analysis':
                    # Show visualization if available
                    display_visualization(result)
                    
                    # Show data summary
                    display_data_summary(result)
                    
                    # Show execution steps
                    display_query_steps(result)
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "Here's your analysis!"
                    })
                
                elif result['type'] == 'conversation':
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result['response']
                    })
                    if 'suggestions' in result:
                        st.info("Try asking: " + " | ".join(result['suggestions']))
                
                elif result['type'] == 'error':
                    st.error(f"Error: {result['message']}")
                    if 'details' in result:
                        st.write("Details:", result['details'])
            
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                st.error("Sorry, I encountered an error processing your query.")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

if __name__ == "__main__":
    main()
