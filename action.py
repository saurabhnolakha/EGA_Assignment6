from llm import call_llm, LLMConnection, get_connection
from utils import extract_json, extract_structured_json, get_logger
import importlib
from typing import Dict, Any, Optional

# Get a logger for this module
logger = get_logger(__name__)

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
        logger.info(f"Executing function: '{function_name}'")
        logger.debug(f"Function parameters: {params}")
        
        # Dynamically import the function from calculator module
        calculator_module = importlib.import_module('calculator')
        if not hasattr(calculator_module, function_name):
            logger.error(f"Function '{function_name}' not found in calculator module")
            raise ValueError(f"Function '{function_name}' not found in calculator module")
        
        function = getattr(calculator_module, function_name)
        logger.debug(f"Successfully imported function '{function_name}'")
        
        # Convert function name to CamelCase for input model (e.g., 'add' -> 'AddInput')
        model_name = function_name[0].upper() + function_name[1:] + 'Input'
        
        # Dynamically import the input model from models module
        models_module = importlib.import_module('models')
        if not hasattr(models_module, model_name):
            logger.error(f"Model '{model_name}' not found in models module")
            raise ValueError(f"Model '{model_name}' not found in models module")
        
        input_model = getattr(models_module, model_name)
        logger.debug(f"Successfully imported model '{model_name}'")
        
        # Validate and convert parameters using the model
        model_instance = input_model.parse_obj(params)
        logger.info(f"{function_name.capitalize()} parameters: {model_instance}")
        
        # Execute the function with the validated model
        logger.debug(f"Executing {function_name} with validated parameters")
        result = function(model_instance)
        
        # Return the result (most calculator functions return an object with a 'result' attribute)
        result_value = result.result if hasattr(result, 'result') else result
        logger.info(f"Function execution successful. Result: {result_value}")
        return result_value
        
    except Exception as e:
        logger.error(f"Error executing function '{function_name}': {e}")
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
        logger.info("Processing action from decision")
        
        # First, extract the task and parameters from the decision
        from models import DecisionOutput

        # Print more diagnostic info
        logger.debug(f"Decision text type: {type(decision_text)}")
        if isinstance(decision_text, str):
            truncated_text = decision_text[:100] + "..." if len(decision_text) > 100 else decision_text
            logger.debug(f"Decision text contains: {truncated_text}")
        
        logger.debug("Extracting structured JSON from decision text")
        decision = extract_structured_json(decision_text, DecisionOutput)
        logger.info(f"Extracted decision: {decision}")
        
        # Check if we have a valid function call
        if decision.function_call and decision.function_call_params:
            logger.info(f"Valid function call found: {decision.function_call}")
            # Extract the function name (remove any parameters in parentheses)
            function_name = decision.function_call.split('(')[0].strip().lower()
            logger.debug(f"Extracted function name: {function_name}")
            return await execute_function(function_name, decision.function_call_params)
        else:
            logger.info("No valid function call found, using text processing fallback")
            # Fallback to text processing if no valid function call
            if connection is None:
                logger.debug("No connection provided, getting singleton")
                connection = get_connection()
                
            logger.debug("Calling LLM for text processing")
            text_result = await call_llm(decision_text, connection=connection)
            logger.info("Text processing completed")
            return text_result
    
    except ValueError as e:
        logger.error(f"Error extracting structured data: {e}")
        
        # Use the provided connection or get the singleton
        if connection is None:
            logger.debug("No connection provided, getting singleton for error fallback")
            connection = get_connection()
            
        logger.info("Using LLM fallback due to error")
        text_result = await call_llm(decision_text, connection=connection)
        logger.debug("LLM fallback completed")
        return text_result