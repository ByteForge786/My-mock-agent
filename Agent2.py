```python
class Agent:
    """Agent for orchestrating tools and managing query processing flow"""
    
    TOOL_INPUT_REQUIREMENTS = {
        "schema_lookup": {
            "required": ["query"],
            "optional": []
        },
        "sql_generation": {
            "required": ["query", "schema_context"],
            "optional": []
        },
        "sql_validation": {
            "required": ["sql"],
            "optional": []
        },
        "sql_execution": {
            "required": ["sql"],
            "optional": []
        },
        "visualization": {
            "required": ["chart_type", "config", "data"],
            "optional": []
        }
    }

    def __init__(self, schema_path: str):
        # Initialize tools
        self.tools = {
            ToolType.SCHEMA.value: SchemaLookupTool(schema_path),
            ToolType.SQL.value: SQLGenerationTool(),
            ToolType.VALIDATION.value: SQLValidationTool(),
            ToolType.EXECUTION.value: SQLExecutionTool(),
            ToolType.VISUALIZATION.value: VisualizationTool()
        }
        # Load schema config
        with open(schema_path) as f:
            self.schema_config = yaml.safe_load(f)

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Initial query analysis to determine intent and create plan"""
        prompt = f"""
Analyze this user query: "{query}"

Available tools and their required inputs:
{json.dumps(self.TOOL_INPUT_REQUIREMENTS, indent=2)}

Consider:
1. Is this a greeting or help request?
2. Is this a data analysis question?
3. Is this off-topic?

For data analysis questions, create a plan considering:
1. Required tools and their inputs
2. Order of operations
3. Visualization needs

Response must be valid JSON in format:
{{
    "query_type": "greeting|help|data_analysis|off_topic",
    "direct_response": string if greeting/help/off-topic else null,
    "plan": [
        {{
            "tool": "tool_name",
            "inputs": {{}},  # Empty dict for initial plan
            "reason": "Why this step is needed"
        }}
    ] if data_analysis else [],
    "requires_visualization": boolean if data_analysis else false,
    "visualization_type": "line|bar|none",
    "reasoning": "Full explanation of decision"
}}
"""
        try:
            response = get_llm_response(prompt)
            if isinstance(response, str):
                response = json.loads(response)
            return response
        except Exception as e:
            logger.error(f"Query analysis failed: {str(e)}")
            return {
                "query_type": "error",
                "direct_response": "I had trouble understanding your query. Could you rephrase it?"
            }

    def determine_next_step(self, 
                          query: str, 
                          plan: List[PlanStep], 
                          executed_steps: List[PlanStep], 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Determine next action based on current state and available context"""
        
        # Find next pending step
        pending_steps = [step for step in plan if step.status == "pending"]
        if not pending_steps:
            return {"next_tool": None}

        next_step = pending_steps[0]
        tool_requirements = self.TOOL_INPUT_REQUIREMENTS.get(next_step.tool, {})
        
        prompt = f"""
Determine inputs for next tool execution:

Query: "{query}"
Next Tool: {next_step.tool}
Required Inputs: {tool_requirements.get('required', [])}
Optional Inputs: {tool_requirements.get('optional', [])}
Available Context Keys: {list(context.keys())}

Last executed step result: 
{executed_steps[-1].result if executed_steps else 'None'}

Provide exact context keys for required tool inputs.
Response format:
{{
    "next_tool": "{next_step.tool}",
    "required_inputs": {{
        "param_name": "exact_context_key"
    }},
    "reason": "Why this step with these inputs"
}}
"""
        try:
            response = get_llm_response(prompt)
            if isinstance(response, str):
                response = json.loads(response)
            return response
        except Exception as e:
            logger.error(f"Next step determination failed: {str(e)}")
            return {
                "next_tool": None,
                "reason": f"Failed to determine inputs: {str(e)}"
            }

    def process_query(self, query: str) -> Dict[str, Any]:
        """Main query processing pipeline"""
        try:
            # 1. Initial Analysis
            analysis = self.analyze_query(query)
            if not isinstance(analysis, dict):
                return {
                    "type": "error",
                    "message": "Failed to analyze query",
                    "details": str(analysis)
                }

            # 2. Handle non-data queries
            query_type = analysis.get("query_type", "error")
            if query_type not in ["data_analysis"]:
                if query_type in ["greeting", "help"]:
                    # Use schema for help responses
                    try:
                        schema_result = self.tools[ToolType.SCHEMA.value].execute(query=query)
                        response = analysis.get("direct_response", "")
                        if schema_result.get("status") == "success":
                            capabilities = self.schema_config.get("analysis_capabilities", [])
                            response += "\n\nI can help you analyze:\n"
                            for capability in capabilities:
                                response += f"\n- {capability.get('description', '')}"
                    except Exception as e:
                        logger.error(f"Schema lookup failed: {str(e)}")
                        schema_result = {"status": "error", "error": str(e)}
                    
                    return {
                        "type": "conversation",
                        "response": response
                    }
                
                return {
                    "type": "conversation",
                    "response": analysis.get("direct_response", "I can only help with data analysis questions.")
                }

            # 3. Create execution plan
            plan = [
                PlanStep(
                    tool=step["tool"],
                    inputs={},  # Will be filled during execution
                    reason=step["reason"],
                    status="pending"
                )
                for step in analysis.get("plan", [])
            ]

            # 4. Execute plan
            executed_steps = []
            context = {"query": query}
            
            while True:
                # Get next step
                next_action = self.determine_next_step(
                    query=query,
                    plan=plan,
                    executed_steps=executed_steps,
                    context=context
                )
                
                next_tool = next_action.get("next_tool")
                if not next_tool:
                    break

                # Validate tool exists
                if next_tool not in self.tools:
                    return {
                        "type": "error",
                        "message": f"Unknown tool: {next_tool}"
                    }

                # Get tool and its requirements
                tool = self.tools[next_tool]
                required_inputs = self.TOOL_INPUT_REQUIREMENTS[next_tool]["required"]
                
                # Validate and prepare inputs
                input_mapping = next_action.get("required_inputs", {})
                available_inputs = {}
                
                for param_name in required_inputs:
                    context_key = input_mapping.get(param_name)
                    if not context_key or context_key not in context:
                        return {
                            "type": "error",
                            "message": f"Missing required input '{param_name}' for {next_tool}"
                        }
                    available_inputs[param_name] = context[context_key]

                # Execute tool
                try:
                    result = tool.execute(**available_inputs)
                    
                    # Record execution
                    step = PlanStep(
                        tool=next_tool,
                        inputs=available_inputs,
                        reason=next_action["reason"],
                        status="completed",
                        result=result
                    )
                    executed_steps.append(step)
                    
                    # Update context
                    if isinstance(result, dict):
                        context.update(result)
                    
                except Exception as e:
                    logger.error(f"Tool execution failed: {str(e)}")
                    return {
                        "type": "error",
                        "message": f"Failed to execute {next_tool}: {str(e)}",
                        "steps": executed_steps
                    }

            # 5. Return results
            return {
                "type": "analysis",
                "context": context,
                "steps": executed_steps,
                "needs_visualization": analysis.get("requires_visualization", False),
                "visualization_type": analysis.get("visualization_type", "none")
            }

        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {
                "type": "error",
                "message": "Failed to process query",
                "details": str(e)
            }
```
