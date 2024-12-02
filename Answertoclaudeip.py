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

            # Prepare execution inputs based on tool type
            if step.tool == ToolType.SCHEMA.value:
                execution_inputs = {
                    "query": context["query"]
                }
            elif step.tool == ToolType.SQL.value:
                if "schema_context" not in context:
                    return self._handle_error(
                        "Missing schema context",
                        "Schema lookup step must be executed before SQL generation",
                        executed_steps
                    )
                execution_inputs = {
                    "query": context["query"],
                    "schema_context": context["schema_context"],
                    "current_step": step.order,
                    "total_steps": total_steps
                }
            elif step.tool == ToolType.VALIDATION.value:
                if "sql" not in context:
                    return self._handle_error(
                        "Missing SQL query",
                        "SQL generation step must be executed before validation",
                        executed_steps
                    )
                execution_inputs = {
                    "sql": context["sql"],
                    "current_step": step.order,
                    "total_steps": total_steps
                }
            elif step.tool == ToolType.EXECUTION.value:
                if "sql" not in context:
                    return self._handle_error(
                        "Missing SQL query",
                        "SQL validation step must be executed before execution",
                        executed_steps
                    )
                execution_inputs = {
                    "sql": context["sql"],
                    "current_step": step.order,
                    "total_steps": total_steps
                }
            elif step.tool == ToolType.VISUALIZATION.value:
                if "data" not in context:
                    return self._handle_error(
                        "Missing data",
                        "SQL execution step must be executed before visualization",
                        executed_steps
                    )
                execution_inputs = {
                    "data": context["data"],
                    "viz_type": analysis.get("visualization", {}).get("type", "bar"),
                    "current_step": step.order,
                    "total_steps": total_steps
                }
            else:  # CONVERSATION tool
                execution_inputs = {
                    "query": query,
                    "capabilities": self.schema_config.get("capabilities", [])
                }

            # Execute step
            logger.info(f"Executing {step.tool} with inputs: {execution_inputs}")
            result = self.tools[step.tool].execute(**execution_inputs)
            
            # Update step status and store result
            step.result = result
            step.status = "completed" if result.get("status") == "success" else "error"
            executed_steps.append(step)

            # Update context with results
            if isinstance(result, dict):
                # For SQL Generation tool, extract SQL from result
                if step.tool == ToolType.SQL.value and "sql" in result:
                    context["sql"] = result["sql"]
                # For execution tool, ensure data is properly stored
                elif step.tool == ToolType.EXECUTION.value and "data" in result:
                    context["data"] = result["data"]
                # For other tools, update all results
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
