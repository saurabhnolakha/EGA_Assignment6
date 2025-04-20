import json
from typing import Any, Dict

def to_json(user_input: str, data: Any, **kwargs) -> dict:
    """
    Convert data to a simplified JSON format with just user_input and result.
    
    Args:
        user_input: The original user input
        data: The data to convert (can be a Pydantic model, dict, or other type)
        **kwargs: Additional key-value pairs (ignored in simplified format)
        
    Returns:
        A dictionary with user_input and result in the format {"user_input": "...", "result": ...}
    """
    # Extract the actual result value from different possible formats
    if isinstance(data, dict):
        # If data is a dict like {"result": 40} or {"result": {"result": 40}}
        if "result" in data:
            if isinstance(data["result"], dict) and "result" in data["result"]:
                result_value = data["result"]["result"]
            else:
                result_value = data["result"]
        else:
            # Use the whole dict as is
            result_value = data
    elif hasattr(data, "result") and not callable(data.result):
        # If data is an object with a result attribute
        result_value = data.result
    elif hasattr(data, "dict") and callable(data.dict):
        # If data is a Pydantic model, convert to dict
        data_dict = data.dict()
        if "result" in data_dict:
            result_value = data_dict["result"]
        else:
            result_value = data_dict
    else:
        # Otherwise use data as is
        result_value = data
    
    # Return the simplified format
    return {
        "user_input": user_input,
        "result": result_value
    }

def combine_json(*args, **kwargs) -> Dict[str, Any]:
    """
    Combine multiple objects into a single JSON-serializable dictionary.
    
    Args:
        *args: Objects to include in the result (will be indexed numerically if not dictionaries)
        **kwargs: Key-value pairs to include in the result
        
    Returns:
        A dictionary that can be serialized to JSON
    """
    result = {}
    
    # Process positional arguments
    for i, arg in enumerate(args):
        # If the arg is a dictionary, merge it into the result
        if isinstance(arg, dict):
            for key, value in arg.items():
                # Make sure the value is serializable
                if hasattr(value, 'dict') and callable(value.dict):
                    result[key] = value.dict()
                else:
                    try:
                        json.dumps(value)
                        result[key] = value
                    except (TypeError, OverflowError):
                        result[key] = str(value)
        else:
            # Otherwise, store it with a numeric key
            key = f"item_{i}"
            # Make sure the value is serializable
            if hasattr(arg, 'dict') and callable(arg.dict):
                result[key] = arg.dict()
            else:
                try:
                    json.dumps(arg)
                    result[key] = arg
                except (TypeError, OverflowError):
                    result[key] = str(arg)
    
    # Process keyword arguments
    for key, value in kwargs.items():
        # Make sure the value is serializable
        if hasattr(value, 'dict') and callable(value.dict):
            result[key] = value.dict()
        else:
            try:
                json.dumps(value)
                result[key] = value
            except (TypeError, OverflowError):
                result[key] = str(value)
    
    return result