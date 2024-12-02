class VisualizationTool(Tool):
    def execute(self, user_query: str, metadata: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        prompt = f"""
Generate Plotly Express code to visualize:
User Question: {user_query}

Available Data in 'df' DataFrame:
Columns: {metadata['columns']}
Types: {metadata['column_types']}
Sample Values: {', '.join(f"{col}: {data[col].iloc[0]}" for col in data.columns)}

Requirements:
1. Use plotly.express (imported as px)
2. DataFrame is available as 'df'
3. Choose appropriate chart type (line, bar, scatter, etc.)
4. Include proper title and labels
5. Return only the code to create the figure

Example format:
"fig = px.line(df, x='date', y='sales', title='Sales Trend')"

Return format:
{{
    "status": "success",
    "plotly_code": "complete plotly code here",
    "explanation": "why this visualization"
}}
"""
        try:
            result = get_llm_response(prompt)
            
            if result["status"] == "success" and "plotly_code" in result:
                # Create a local namespace with required imports and data
                namespace = {
                    'px': px,
                    'df': data
                }
                
                # Execute the plotly code in the namespace
                exec(result["plotly_code"], namespace)
                
                # Get the figure from namespace
                if 'fig' in namespace:
                    fig = namespace['fig']
                    return {
                        "status": "success",
                        "figure": fig,
                        "code": result["plotly_code"],
                        "explanation": result.get("explanation", "")
                    }
                else:
                    return {
                        "status": "error",
                        "message": "No figure was created by the code"
                    }
            else:
                return {
                    "status": "error",
                    "message": "Invalid response from LLM"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to create visualization: {str(e)}"
            }



elif tool_name == ToolType.VISUALIZATION.value and analysis.get("needs_visualization"):
    if "metadata" not in context or "data" not in context:
        logger.error("Missing data/metadata for visualization")
        continue
        
    result = tool.execute(
        user_query=query,
        metadata=context["metadata"],
        data=context["data"]
    )
    
    if result["status"] == "success":
        context["visualization"] = result["figure"]
        context["viz_code"] = result["code"]
        context["viz_explanation"] = result.get("explanation", "")


def display_analysis_result(result: dict):
    if result["status"] == "success":
        # Show visualization if available
        if "visualization" in result["context"]:
            st.plotly_chart(result["context"]["visualization"], use_container_width=True)
            
            # Show the code used to create visualization
            if "viz_code" in result["context"]:
                with st.expander("View Visualization Code"):
                    st.code(result["context"]["viz_code"], language="python")
                    if "viz_explanation" in result["context"]:
                        st.markdown(f"**Explanation**: {result['context']['viz_explanation']}")
