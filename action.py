from llm import call_llm, extract_structured_json, LLMConnection, get_connection
from calculator import add, multiply

async def action(decision_text: str, connection=None):
    """
    Take action based on a decision.
    
    Args:
        decision_text: The decision text from LLM
        connection: Optional LLMConnection instance
        
    Returns:
        Result of the action
    """
    try:
        # First, extract the task and parameters from the decision
        from models import PerceptionOutput
        decision = extract_structured_json(decision_text, PerceptionOutput)
        
        if "add" in decision.task:
            # Extract parameters for addition
            from models import AddInput
            add_params = AddInput.parse_obj(decision.function_call_params)
            result = add(add_params.a, add_params.b)
            return {"result": result}
        
        elif "multiply" in decision.task:
            # Extract parameters for multiplication
            from models import MultiplyInput
            multiply_params = MultiplyInput.parse_obj(decision.function_call_params)
            result = multiply(multiply_params.a, multiply_params.b)
            return {"result": result}
        
        else:
            # Use the provided connection or get the singleton
            if connection is None:
                connection = get_connection()
                
            return await call_llm(decision_text, connection=connection)
    
    except ValueError as e:
        print(f"Error extracting structured data: {e}")
        
        # Use the provided connection or get the singleton
        if connection is None:
            connection = get_connection()
            
        return await call_llm(decision_text, connection=connection)