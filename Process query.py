def process_query(self, query: str) -> Dict[str, Any]:
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

        for step in analysis.get("plan", []):
            # Ensure the step is valid
            if not isinstance(step, dict) or "tool" not in step:
                logger.error(f"Invalid step in plan: {step}")
                continue

            tool_name = step["tool"]
            if tool_name not in self.tools:
                logger.error(f"Unknown tool: {tool_name}")
                continue

            tool = self.tools[tool_name]
            result = {"status": "error", "message": "Step not executed"}

            try:
                if tool_name == ToolType.SCHEMA.value:
                    # Schema lookup tool
                    result = tool.execute(query=query)
                    if result["status"] == "success":
                        context.update(result)

                elif tool_name == ToolType.SQL.value:
                    # SQL generation tool
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
                    # SQL validation tool
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
                            # Regenerate SQL with feedback
                            sql_result = self.tools[ToolType.SQL.value].execute(
                                user_query=f"{query} [Feedback: {result.get('feedback', '')}]",
                                schema_context=context["schema_context"]
                            )
                            context["sql"] = sql_result["sql"]
                            retry_count += 1
                        else:
                            break

                elif tool_name == ToolType.EXECUTION.value:
                    # SQL execution tool
                    if "sql" not in context:
                        logger.error("Missing SQL for execution")
                        continue
                    result = tool.execute(sql=context["sql"])
                    context["data"] = result["data"]
                    context["metadata"] = result["metadata"]

                elif tool_name == ToolType.VISUALIZATION.value and analysis.get("needs_visualization"):
                    # Visualization tool
                    if "metadata" not in context:
                        logger.error("Missing metadata for visualization")
                        continue
                    result = tool.execute(
                        user_query=query,
                        metadata=context["metadata"]
                    )
                    if result["status"] == "success":
                        viz_config = result["config"]
                        df = context["data"]
                        fig = px.line(df, **viz_config) if result["viz_type"] == "line" else px.bar(df, **viz_config)
                        context["visualization"] = fig

            except Exception as e:
                logger.error(f"Error in step '{tool_name}': {traceback.format_exc()}")

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
