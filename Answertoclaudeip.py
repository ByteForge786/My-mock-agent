def determine_next_step(self, 
                       query: str, 
                       plan: List[PlanStep], 
                       executed_steps: List[PlanStep], 
                       context: Dict[str, Any]) -> Dict[str, Any]:
    """Determine next action based on current state and available context"""
    
    # Find next pending step
    pending_steps = [step for step in plan if step.status == "pending"]
    if not pending_steps:
        # If there are no more pending steps, we're done
        return {"next_tool": None}

    next_step = pending_steps[0]
    tool_requirements = self.TOOL_INPUT_REQUIREMENTS.get(next_step.tool, {})
    
    # Determine the required inputs for the next step
    required_inputs = {}
    for required_input in tool_requirements.get('required', []):
        if required_input in context:
            required_inputs[required_input] = context[required_input]
        else:
            # If a required input is not available in the context, we can't proceed
            return {
                "next_tool": None,
                "reason": f"Missing required input: {required_input}"
            }

    # Determine the optional inputs for the next step
    optional_inputs = {}
    for optional_input in tool_requirements.get('optional', []):
        if optional_input in context:
            optional_inputs[optional_input] = context[optional_input]

    # Construct the response
    return {
        "next_tool": next_step.tool,
        "required_inputs": required_inputs,
        "reason": next_step.reason
    }
