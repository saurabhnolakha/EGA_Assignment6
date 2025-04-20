from llm import call_llm, extract_structured_json, LLMConnection, get_connection
import importlib
from typing import Dict, Any, Optional

async def execute_function(function_name: str, params: Dict[str, Any]):
    """
    Dynamically execute a function from calculator.py with the appropriate input model.
    
    Args:
        function_name: Name of the function to execute (e.g., 'add', 'multiply')
        params: Dictionary of parameters to pass to the function
        
    Returns:
        Result of the function execution
    """
    try:
        # Dynamically import the function from calculator module
        calculator_module = importlib.import_module('calculator')
        if not hasattr(calculator_module, function_name):
            raise ValueError(f"Function '{function_name}' not found in calculator module")
        
        function = getattr(calculator_module, function_name)
        
        # Convert function name to CamelCase for input model (e.g., 'add' -> 'AddInput')
        model_name = function_name[0].upper() + function_name[1:] + 'Input'
        
        # Dynamically import the input model from models module
        models_module = importlib.import_module('models')
        if not hasattr(models_module, model_name):
            raise ValueError(f"Model '{model_name}' not found in models module")
        
        input_model = getattr(models_module, model_name)
        
        # Validate and convert parameters using the model
        model_instance = input_model.parse_obj(params)
        print(f"{function_name.capitalize()} parameters: {model_instance}")
        
        # Execute the function with the validated model
        result = function(model_instance)
        
        # Return the result (most calculator functions return an object with a 'result' attribute)
        return result.result if hasattr(result, 'result') else result
        
    except Exception as e:
        raise ValueError(f"Error executing function '{function_name}': {e}")

async def action(decision_text: str, connection: Optional[LLMConnection] = None):
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
        from models import DecisionOutput

        # Print more diagnostic info
        print(f"Decision text type: {type(decision_text)}")
        if isinstance(decision_text, str):
            print(f"Decision text contains: {decision_text[:100]}...")
        
        decision = extract_structured_json(decision_text, DecisionOutput)
        print(f"Extracted decision: {decision}")
        
        # Check if we have a valid function call
        if decision.function_call and decision.function_call_params:
            # Extract the function name (remove any parameters in parentheses)
            function_name = decision.function_call.split('(')[0].strip().lower()
            return await execute_function(function_name, decision.function_call_params)
        else:
            # Fallback to text processing if no valid function call
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