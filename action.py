from llm import call_llm, extract_structured_json, LLMConnection, get_connection
from calculator import add, multiply

async def action(decision_text: str, connection=None):
    """
    Take action based on a decision.
    
    Args:
        decision_text: The decision text from LLM
        connection: Optional LLMConnection instance
        
    Returns:
        Result of the action as a simple value
    """
    try:
        # First, extract the task and parameters from the decision
        from models import PerceptionOutput

        # Print more diagnostic info
        print(f"Decision text type: {type(decision_text)}")
        if isinstance(decision_text, str):
            print(f"Decision text contains: {decision_text[:100]}...")
        
        decision = extract_structured_json(decision_text, PerceptionOutput)
        print(f"Extracted decision: {decision}")
        
        
        if "add" in decision.task:
            # Extract parameters for addition
            from models import AddInput
            add_params = AddInput.parse_obj(decision.function_call_params)
            print(f"Add parameters: {add_params.a}, {add_params.b}")    
            result = add(add_params)
            # Return the simple integer result
            return result.result
        
        elif "multiply" in decision.task:
            # Extract parameters for multiplication
            from models import MultiplyInput
            multiply_params = MultiplyInput.parse_obj(decision.function_call_params)
            print(f"Multiply parameters: {multiply_params.a}, {multiply_params.b}")
            result = multiply(multiply_params)
            # Return the simple integer result
            return result.result
        
        else:
            # Use the provided connection or get the singleton
            if connection is None:
                connection = get_connection()
                
            text_result = await call_llm(decision_text, connection=connection)
            return text_result
    
    except ValueError as e:
        print(f"Error extracting structured data: {e}")
        
        # Use the provided connection or get the singleton
        if connection is None:
            connection = get_connection()
            
        text_result = await call_llm(decision_text, connection=connection)
        return text_result