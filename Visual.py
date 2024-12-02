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








def process_query(self, query: str) -> Dict[str, Any]:
    """Main query processing pipeline with comprehensive error handling and logging"""
    try:
        # Step 1: Analyze query intent and get plan
        logger.info(f"Starting query analysis for: {query}")
        analysis = self.analyze_query(query)
        logger.info(f"Query analysis result: {analysis}")

        # Handle non-data queries
        if analysis.get("query_type") in ["greeting", "help"]:
            logger.info("Handling greeting/help query")
            return self.handle_greeting_or_help(query)
        elif analysis.get("query_type") == "off_topic":
            logger.info("Handling off-topic query")
            return {
                "type": "conversation",
                "response": "I can only help with data analysis questions about the available data."
            }

        # Initialize context and steps tracking
        context = {"query": query}
        executed_steps = []

        # Validate plan exists
        if not analysis.get("plan"):
            logger.error("No execution plan in analysis")
            return {
                "type": "error",
                "message": "Failed to create execution plan",
                "status": "error"
            }

        # Execute each step in the plan
        for step_info in analysis["plan"]:
            logger.info(f"Processing step: {step_info}")
            
            # Validate step structure
            if not isinstance(step_info, dict) or "tool" not in step_info:
                logger.error(f"Invalid step information: {step_info}")
                continue

            tool_name = step_info["tool"]
            if tool_name not in self.tools:
                logger.error(f"Unknown tool requested: {tool_name}")
                continue

            tool = self.tools[tool_name]
            result = {"status": "error", "message": "Step not executed"}

            try:
                # Schema Lookup
                if tool_name == ToolType.SCHEMA.value:
                    result = tool.execute(query=query)
                    if result["status"] == "success":
                        logger.info(f"Schema lookup successful. Found tables: {[t['name'] for t in result.get('relevant_tables', [])]}")
                        context.update(result)

                # SQL Generation
                elif tool_name == ToolType.SQL.value:
                    if "schema_context" not in context:
                        raise ValueError("Missing schema context for SQL generation")
                    result = tool.execute(
                        user_query=query,
                        schema_context=context["schema_context"]
                    )
                    if result["status"] == "success":
                        context["sql"] = result["sql"]
                        logger.info("SQL generation successful")

                # SQL Validation
                elif tool_name == ToolType.VALIDATION.value:
                    if "sql" not in context:
                        raise ValueError("Missing SQL for validation")
                    
                    retry_count = 0
                    while retry_count < SQLValidationTool.MAX_RETRIES:
                        logger.info(f"Validation attempt {retry_count + 1}")
                        result = tool.execute(
                            sql=context["sql"],
                            schema_context=context["schema_context"],
                            retry_count=retry_count
                        )
                        
                        if result["status"] == "success" and result.get("is_safe", False):
                            logger.info("SQL validation successful")
                            break
                            
                        if retry_count < SQLValidationTool.MAX_RETRIES - 1:
                            logger.info(f"Retrying SQL generation with feedback: {result.get('feedback', '')}")
                            sql_result = self.tools[ToolType.SQL.value].execute(
                                user_query=f"{query} [Feedback: {result.get('feedback', '')}]",
                                schema_context=context["schema_context"]
                            )
                            if sql_result["status"] == "success":
                                context["sql"] = sql_result["sql"]
                        
                        retry_count += 1

                # SQL Execution
                elif tool_name == ToolType.EXECUTION.value:
                    if "sql" not in context:
                        raise ValueError("Missing SQL for execution")
                    result = tool.execute(sql=context["sql"])
                    if result["status"] == "success":
                        logger.info(f"SQL execution successful. Rows: {result.get('metadata', {}).get('row_count', 0)}")
                        context.update(result)

                # Visualization
                elif tool_name == ToolType.VISUALIZATION.value:
                    if not analysis.get("needs_visualization"):
                        logger.info("Skipping visualization - not needed")
                        continue
                        
                    if "metadata" not in context or "data" not in context:
                        raise ValueError("Missing data/metadata for visualization")
                    
                    result = tool.execute(
                        user_query=query,
                        metadata=context["metadata"]
                    )
                    
                    if result["status"] == "success" and "config" in result:
                        try:
                            df = context["data"]
                            viz_type = result.get("viz_type", "bar")
                            viz_config = result["config"]
                            
                            logger.info(f"Creating {viz_type} visualization")
                            fig = px.line(df, **viz_config) if viz_type == "line" else px.bar(df, **viz_config)
                            context["visualization"] = fig
                            result["visualization_created"] = True
                            logger.info("Visualization created successfully")
                        except Exception as viz_error:
                            logger.error(f"Visualization creation failed: {viz_error}")
                            result = {
                                "status": "error",
                                "message": str(viz_error)
                            }

            except Exception as step_error:
                logger.error(f"Error in {tool_name}: {str(step_error)}")
                result = {
                    "status": "error",
                    "message": str(step_error)
                }

            # Record step execution
            step_record = {
                "tool": tool_name,
                "status": result.get("status", "error"),
                "result": {k: v for k, v in result.items() if k != "tool"},
                "reason": step_info.get("reason", "")
            }
            executed_steps.append(step_record)

            # Break on failure
            if step_record["status"] != "success":
                logger.error(f"Step failed: {tool_name}")
                break

        # Prepare final response
        success = all(step["status"] == "success" for step in executed_steps)
        logger.info(f"Query processing completed. Success: {success}")
        
        return {
            "type": "analysis",
            "steps": executed_steps,
            "context": context,
            "status": "success" if success else "error"
        }

    except Exception as e:
        logger.error(f"Error processing query: {traceback.format_exc()}")
        return {
            "type": "error",
            "message": str(e),
            "status": "error"
        }



class SchemaLookupTool(Tool):
    def execute(self, query: str) -> Dict[str, Any]:
        query_embedding = self.embed_model.encode(query)
        relevant_tables = []
        
        logger.info(f"Searching relevant tables for query: {query}")
        
        for table_name, embedding in self.table_embeddings.items():
            similarity = np.dot(query_embedding, embedding)
            logger.info(f"Table: {table_name}, Similarity: {similarity:.4f}")
            
            if similarity > 0.5:
                table_info = self.schema_config['tables'][table_name]
                relevant_tables.append({
                    "name": table_name,
                    "create_statement": table_info['create_statement'],
                    "description": table_info['description'],
                    "sample_questions": table_info.get('sample_questions', [])
                })
        
        logger.info(f"Selected tables: {[t['name'] for t in relevant_tables]}")
        
        result = {
            "status": "success",
            "relevant_tables": relevant_tables,
            "schema_context": "\n".join(t["create_statement"] for t in relevant_tables)
        }
        
        return result




def process_query(self, query: str) -> Dict[str, Any]:
    """Main query processing pipeline"""
    try:
        # Step 1: Analyze query intent and get plan
        logger.info(f"Starting query analysis for: {query}")
        analysis = self.analyze_query(query)
        logger.info(f"Query analysis result: {analysis}")

        # Handle non-data queries
        if isinstance(analysis, dict) and analysis.get("query_type") in ["greeting", "help"]:
            return self.handle_greeting_or_help(query)
        elif isinstance(analysis, dict) and analysis.get("query_type") == "off_topic":
            return {
                "type": "conversation",
                "response": "I can only help with data analysis questions about the available data."
            }

        # Initialize context and steps tracking
        context = {"query": query}
        executed_steps = []

        # Validate plan exists and is list
        plan = analysis.get("plan", []) if isinstance(analysis, dict) else []
        if not plan:
            logger.error("No valid plan received")
            return {
                "type": "error",
                "message": "Failed to create execution plan",
                "status": "error"
            }

        # Execute each step in the plan
        for step_dict in plan:
            if not isinstance(step_dict, dict):
                continue
                
            tool_name = step_dict.get("tool")
            if not tool_name or tool_name not in self.tools:
                continue

            # Execute schema lookup
            if tool_name == ToolType.SCHEMA.value:
                result = self.tools[tool_name].execute(query=query)
                if result["status"] == "success":
                    context.update(result)

            # Execute SQL generation
            elif tool_name == ToolType.SQL.value:
                if "schema_context" not in context:
                    continue
                result = self.tools[tool_name].execute(
                    user_query=query,
                    schema_context=context["schema_context"]
                )
                if result["status"] == "success":
                    context["sql"] = result["sql"]

            # Execute SQL validation
            elif tool_name == ToolType.VALIDATION.value:
                if "sql" not in context:
                    continue
                retry_count = 0
                while retry_count < SQLValidationTool.MAX_RETRIES:
                    result = self.tools[tool_name].execute(
                        sql=context["sql"],
                        schema_context=context["schema_context"],
                        retry_count=retry_count
                    )
                    if result["status"] == "success" and result.get("is_safe", False):
                        break
                    if retry_count < SQLValidationTool.MAX_RETRIES - 1 and result.get("feedback"):
                        sql_result = self.tools[ToolType.SQL.value].execute(
                            user_query=f"{query} [Feedback: {result['feedback']}]",
                            schema_context=context["schema_context"]
                        )
                        if sql_result["status"] == "success":
                            context["sql"] = sql_result["sql"]
                    retry_count += 1

            # Execute SQL
            elif tool_name == ToolType.EXECUTION.value:
                if "sql" not in context:
                    continue
                result = self.tools[tool_name].execute(sql=context["sql"])
                if result["status"] == "success":
                    context.update(result)

            # Execute visualization if needed
            elif tool_name == ToolType.VISUALIZATION.value and analysis.get("needs_visualization"):
                if "metadata" not in context or "data" not in context:
                    continue
                result = self.tools[tool_name].execute(
                    user_query=query,
                    metadata=context["metadata"]
                )
                if result["status"] == "success" and "config" in result:
                    try:
                        df = context["data"]
                        viz_type = result.get("viz_type", "bar")
                        viz_config = result["config"]
                        fig = px.line(df, **viz_config) if viz_type == "line" else px.bar(df, **viz_config)
                        context["visualization"] = fig
                        result["visualization_created"] = True
                    except Exception as viz_error:
                        logger.error(f"Visualization creation failed: {viz_error}")
                        result = {"status": "error", "message": str(viz_error)}

            # Record step execution
            executed_steps.append({
                "tool": tool_name,
                "status": result.get("status", "error"),
                "result": {k: v for k, v in result.items() if k != "tool"}
            })

            # Break if step failed
            if result.get("status") != "success":
                break

        return {
            "type": "analysis",
            "steps": executed_steps,
            "context": context,
            "status": "success" if executed_steps and all(s["status"] == "success" for s in executed_steps) else "error"
        }

    except Exception as e:
        logger.error(f"Error processing query: {traceback.format_exc()}")
        return {
            "type": "error",
            "message": str(e),
            "status": "error"
        }
