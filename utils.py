import json
from typing import Any, Dict, TypeVar, Type, Union
from pydantic import BaseModel, ValidationError
import re
import logging


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



def extract_json(llm_output: Union[str, Any]) -> Dict[str, Any]:
    """
    Extract JSON from LLM output text using regex pattern matching.
    
    Args:
        llm_output: The raw text output from the LLM or a dict
        
    Returns:
        A dictionary parsed from the JSON in the output
        
    Raises:
        ValueError: If no valid JSON is found in the text
    """
    # If the input is already a dictionary, return it directly
    if isinstance(llm_output, dict):
        return llm_output
        
    # Strip the text attribute if the input is a Gemini response object
    if hasattr(llm_output, 'text'):
        llm_output = llm_output.text
    
    # Convert to string if it's not already
    if not isinstance(llm_output, str):
        llm_output = str(llm_output)
    
    # Check for markdown code blocks with JSON
    code_block_matches = re.findall(r'```(?:json)?\s*\n(.*?)\n```', llm_output, re.DOTALL)
    if code_block_matches:
        for match in code_block_matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    # Try to find JSON pattern with curly braces
    json_matches = re.findall(r'\{.*\}', llm_output, re.DOTALL)
    
    if json_matches:
        # Try the matches in order, from longest to shortest (assuming the longest is the most complete)
        sorted_matches = sorted(json_matches, key=len, reverse=True)
        
        for match in sorted_matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # If we didn't find any valid JSON with the first pattern, try a more aggressive approach
    # Look for anything that might be JSON-like
    try:
        # Try to find the most JSON-like substring
        start_idx = llm_output.find('{')
        end_idx = llm_output.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            json_substring = llm_output[start_idx:end_idx+1]
            return json.loads(json_substring)
    except json.JSONDecodeError:
        pass
    
    # If all else fails, try to create a simple key-value JSON
    # This might be useful for plain text responses
    return {"text": llm_output}

T = TypeVar('T', bound=BaseModel)

def extract_structured_json(llm_output: Union[str, Dict, Any], model_class: Type[T]) -> T:
    """
    Extract JSON from LLM output and validate it against a Pydantic model.
    
    Args:
        llm_output: The raw text output from the LLM, dict, or response object
        model_class: The Pydantic model class to validate against
        
    Returns:
        An instance of the Pydantic model
        
    Raises:
        ValueError: If validation fails
    """
    try:
        # If input is already a model instance of the correct type, return it
        if isinstance(llm_output, model_class):
            return llm_output
        
        # First extract the JSON
        json_data = extract_json(llm_output)
        
        # Pre-process data for common model structures
        if "function_call_params" in json_data and isinstance(json_data["function_call_params"], list):
            # Convert list params to dictionary with keys a, b, etc.
            params = {}
            for i, value in enumerate(json_data["function_call_params"]):
                key = chr(97 + i)  # 'a', 'b', 'c', etc.
                # Convert string numbers to integers if possible
                if isinstance(value, str) and value.isdigit():
                    params[key] = int(value)
                else:
                    params[key] = value
            json_data["function_call_params"] = params
        
        # Add standardized parameter name mapping
        if "function_call_params" in json_data and isinstance(json_data["function_call_params"], dict):
            # Map common parameter names to standardized names
            param_mapping = {
                "param1": "a", "parameter1": "a", "p1": "a", "x": "a", "first": "a", "1": "a", "integer1": "a", "num1": "a", "number1": "a",
                "param2": "b", "parameter2": "b", "p2": "b", "y": "b", "second": "b", "2": "b", "integer2": "b", "num2": "b", "number2": "b",
                "param3": "c", "parameter3": "c", "p3": "c", "z": "c", "third": "c", "3": "c", "integer3": "c", "num3": "c", "number3": "c",
                # Add more mappings as needed
            }
            
            new_params = {}
            for key, value in json_data["function_call_params"].items():
                # If the key has a mapping, use the mapped key
                mapped_key = param_mapping.get(key, key)
                # Convert string numbers to integers if possible
                if isinstance(value, str) and value.isdigit():
                    new_params[mapped_key] = int(value)
                else:
                    new_params[mapped_key] = value
            
            json_data["function_call_params"] = new_params
        
        # Generic handling for task-based function calls in any model
        if "task" in json_data:
            if "function_call" not in json_data:
                json_data["function_call"] = json_data["task"].split("(")[0] if "(" in json_data["task"] else json_data["task"]
            
            # If we have input parameters but no function_call_params
            if "function_call_params" not in json_data and "input" in json_data:
                params = {}
                if isinstance(json_data["input"], list):
                    for i, value in enumerate(json_data["input"]):
                        key = chr(97 + i)  # 'a', 'b', 'c', etc.
                        # Convert string numbers to integers if possible
                        params[key] = int(value) if isinstance(value, str) and value.isdigit() else value
                elif isinstance(json_data["input"], dict):
                    params = json_data["input"]
                else:
                    params = {"a": json_data["input"]}
                json_data["function_call_params"] = params
        
        # Extract parameters from function_call string if function_call_params is missing
        if "function_call" in json_data and "function_call_params" not in json_data:
            func_call = json_data["function_call"]
            if "(" in func_call and ")" in func_call:
                params_str = func_call.split("(", 1)[1].rsplit(")", 1)[0]
                params_list = [p.strip() for p in params_str.split(",")]
                params = {}
                for i, value in enumerate(params_list):
                    if value:  # Skip empty values
                        key = chr(97 + i)  # 'a', 'b', 'c', etc.
                        # Convert string numbers to integers if possible
                        if value.isdigit():
                            params[key] = int(value)
                        else:
                            # Remove quotes if present
                            if (value.startswith('"') and value.endswith('"')) or \
                               (value.startswith("'") and value.endswith("'")):
                                value = value[1:-1]
                            params[key] = value
                json_data["function_call_params"] = params
        
        # Handle the case where we just have text but need a model
        if len(json_data) == 1 and "text" in json_data:
            # Try to determine if the text field matches any field in the model
            if model_class.__fields__:
                first_field = next(iter(model_class.__fields__.keys()))
                # If "text" is not a field in the model, map it to the first field
                if "text" not in model_class.__fields__:
                    return model_class(**{first_field: json_data["text"]})
        
        # Try to validate with Pydantic
        try:
            return model_class.parse_obj(json_data)
        except ValidationError as e:
            # In case of validation error, print more details for debugging
            print(f"Validation error: {e}")
            print(f"JSON data: {json_data}")
            # Check which fields are missing and set them to default values if possible
            missing_fields = []
            invalid_fields = []
            
            for error in e.errors():
                if error["type"] == "missing":
                    field_name = error["loc"][0]
                    missing_fields.append(field_name)
                    if field_name not in json_data:
                        # Set a default value based on field type
                        field_info = model_class.__fields__[field_name]
                        if field_info.type_ == str:
                            json_data[field_name] = ""
                        elif field_info.type_ == int:
                            json_data[field_name] = 0
                        elif field_info.type_ == dict:
                            json_data[field_name] = {}
                        elif field_info.type_ == list:
                            json_data[field_name] = []
                # Handle type errors
                elif error["type"] in ["dict_type", "list_type", "string_type"]:
                    field_path = error["loc"]
                    field_name = field_path[0] if field_path else None
                    if field_name:
                        invalid_fields.append(field_name)
                        # Fix common type issues
                        if field_name == "function_call_params" and "function_call_params" in json_data:
                            # If function_call_params is not a dict, convert it
                            if not isinstance(json_data["function_call_params"], dict):
                                value = json_data["function_call_params"]
                                if isinstance(value, list):
                                    # Convert list to dict
                                    params = {}
                                    for i, v in enumerate(value):
                                        key = chr(97 + i)  # 'a', 'b', 'c', etc.
                                        params[key] = int(v) if isinstance(v, str) and v.isdigit() else v
                                    json_data["function_call_params"] = params
                                elif isinstance(value, str):
                                    # Convert string to dict
                                    try:
                                        json_data["function_call_params"] = json.loads(value)
                                    except:
                                        json_data["function_call_params"] = {"text": value}
                                else:
                                    # Any other type, convert to simple dict
                                    json_data["function_call_params"] = {"value": value}
            
            # Try again with the fixed data
            if missing_fields or invalid_fields:
                print(f"Attempting to fix fields: Missing: {missing_fields}, Invalid: {invalid_fields}")
                return model_class.parse_obj(json_data)
            else:
                raise
            
    except ValidationError as e:
        raise ValueError(f"JSON validation failed: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error extracting or validating JSON: {str(e)}")

