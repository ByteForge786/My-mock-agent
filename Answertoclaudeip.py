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

            # Get base inputs from next_action
            execution_inputs = next_action["inputs"]

            # Add current_step and total_steps only for tools that need them
            if step.tool in [
                ToolType.SQL.value,
                ToolType.VALIDATION.value,
                ToolType.EXECUTION.value,
                ToolType.VISUALIZATION.value
            ]:
                execution_inputs.update({
                    "current_step": step.order,
                    "total_steps": total_steps
                })

            # Execute step with appropriate inputs
            result = self.tools[step.tool].execute(**execution_inputs)

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
