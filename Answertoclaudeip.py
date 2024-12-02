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

                # Get inputs for current step
                next_action = self.determine_next_step(
                    current_step=step,
                    executed_steps=executed_steps,
                    context=context,
                    total_steps=total_steps
                )

                # Prepare execution inputs based on tool type
                execution_inputs = {}
                
                if step.tool == ToolType.SCHEMA.value:
                    execution_inputs = {
                        "query": context["query"]
                    }
                elif step.tool == ToolType.SQL.value:
                    execution_inputs = {
                        "query": context["query"],
                        "schema_context": context["schema_context"],
                        "current_step": step.order,
                        "total_steps": total_steps
                    }
                elif step.tool == ToolType.VALIDATION.value:
                    execution_inputs = {
                        "sql": context.get("sql", ""),
                        "current_step": step.order,
                        "total_steps": total_steps
                    }
                elif step.tool == ToolType.EXECUTION.value:
                    execution_inputs = {
                        "sql": context.get("sql", ""),
                        "current_step": step.order,
                        "total_steps": total_steps
                    }
                elif step.tool == ToolType.VISUALIZATION.value:
                    execution_inputs = {
                        "data": context.get("data", pd.DataFrame()),
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
                result = self.tools[step.tool].execute(**execution_inputs)

                # Update step status
                step.result = result
                step.status = "completed" if result.get("status") == "success" else "error"
                executed_steps.append(step)

                # Update context with results
                if isinstance(result, dict):
                    # For SQL Generation tool, extract SQL from result
                    if step.tool == ToolType.SQL.value and "sql" in result:
                        context["sql"] = result["sql"]
                    # For other tools, update all results
                    else:
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
