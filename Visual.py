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



class Agent:
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

            # Important: Check if plan exists in analysis
            for step_info in analysis.get("plan", []):
                # Check if tool exists in step_info
                if "tool" not in step_info:
                    logger.error(f"Missing tool in step info: {step_info}")
                    continue

                tool_name = step_info["tool"]
                if tool_name not in self.tools:
                    logger.error(f"Unknown tool: {tool_name}")
                    continue

                tool = self.tools[tool_name]
                result = {"status": "error", "message": "Step not executed"}

                try:
                    if tool_name == ToolType.SCHEMA.value:
                        result = tool.execute(query=query)
                        if result["status"] == "success":
                            context.update(result)

                    elif tool_name == ToolType.SQL.value:
                        if "schema_context" not in context:
                            logger.error("Missing schema context for SQL generation")
                            continue
                        result = tool.execute(
                            user_query=query,
                            schema_context=context["schema_context"]
                        )
                        if result["status"] == "success":
                            context["sql"] = result["sql"]

                    elif tool_name == ToolType.VALIDATION.value:
                        if "sql" not in context:
                            logger.error("Missing SQL for validation")
                            continue
                            
                        retry_count = 0
                        while retry_count < SQLValidationTool.MAX_RETRIES:
                            result = tool.execute(
                                sql=context["sql"],
                                schema_context=context["schema_context"],
                                retry_count=retry_count
                            )
                            if result["status"] == "success" and result.get("is_safe", False):
                                break
                            if retry_count < SQLValidationTool.MAX_RETRIES - 1:
                                sql_result = self.tools[ToolType.SQL.value].execute(
                                    user_query=f"{query} [Feedback: {result.get('feedback', '')}]",
                                    schema_context=context["schema_context"]
                                )
                                if sql_result["status"] == "success":
                                    context["sql"] = sql_result["sql"]
                            retry_count += 1

                    elif tool_name == ToolType.EXECUTION.value:
                        if "sql" not in context:
                            logger.error("Missing SQL for execution")
                            continue
                        result = tool.execute(sql=context["sql"])
                        if result["status"] == "success":
                            context.update(result)

                    elif tool_name == ToolType.VISUALIZATION.value and analysis.get("needs_visualization"):
                        if "metadata" not in context or "data" not in context:
                            logger.error("Missing data/metadata for visualization")
                            continue
                            
                        result = tool.execute(
                            user_query=query,
                            metadata=context["metadata"],
                            data=context["data"]
                        )
                        
                        if result["status"] == "success" and "figure" in result:
                            context["visualization"] = result["figure"]
                            context["viz_code"] = result.get("code", "")
                            context["viz_explanation"] = result.get("explanation", "")

                except Exception as step_error:
                    logger.error(f"Error in {tool_name}: {str(step_error)}")
                    result = {
                        "status": "error",
                        "message": str(step_error)
                    }

                # Store step execution result
                current_step = {
                    "tool": tool_name,  # Use tool_name directly from the loop
                    "status": result.get("status", "error"),
                    "result": result,
                    "reason": step_info.get("reason", "")  # Include reason from plan
                }
                
                executed_steps.append(current_step)

                # Break if step failed
                if result.get("status") != "success":
                    break

            # Return final result
            return {
                "type": "analysis",
                "steps": executed_steps,
                "context": context,
                "status": "success" if all(step["status"] == "success" for step in executed_steps) else "error"
            }

        except Exception as e:
            logger.error(f"Error processing query: {traceback.format_exc()}")
            return {
                "type": "error",
                "message": str(e),
                "status": "error"
            }
