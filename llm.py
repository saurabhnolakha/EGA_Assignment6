import asyncio
import re
import json
from typing import TypeVar, Type, Any, Optional, Dict, Union
from pydantic import BaseModel, ValidationError
from google import genai
from dotenv import load_dotenv
import os
import logging

# Configure logging to only show warnings and errors
logging.basicConfig(level=logging.WARNING)
logging.getLogger("google.generativeai").setLevel(logging.WARNING)

load_dotenv()

class LLMConnection:
    """
    Class to manage a single LLM connection instance that can be reused for multiple queries.
    """
    _instance = None
    
    @classmethod
    def get_instance(cls, api_key=None, model="gemini-2.0-flash"):
        """Singleton pattern to ensure only one connection is created"""
        if cls._instance is None:
            cls._instance = cls(api_key, model)
        return cls._instance
    
    def __init__(self, api_key=None, model="gemini-2.0-flash"):
        """Initialize the LLM connection with the given API key and model"""
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        print(f"Initialized LLM connection with model: {model}")
    
    async def generate(self, prompt, timeout=30):
        """Generate content with timeout"""
        print(f"Generating with model: {self.model}, timeout: {timeout}s")
        try:
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None, 
                    lambda: self.client.models.generate_content(
                        model=self.model,
                        contents=prompt
                    )
                ),
                timeout=timeout
            )
            print("LLM generation completed")
            return response
        except asyncio.TimeoutError:
            print(f"LLM generation timed out after {timeout} seconds!")
            raise
        except Exception as e:
            print(f"Error in LLM generation: {e}")
            raise

    async def __aenter__(self):
        """Context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - no cleanup needed for now"""
        pass


# Get the global connection instance (for use across modules)
def get_connection():
    """Get the singleton LLM connection instance"""
    return LLMConnection.get_instance()


# Maintain backward compatibility with existing code
async def generate_with_timeout(client, prompt, timeout=100):
    """Legacy function for backward compatibility"""
    print("WARNING: Using deprecated generate_with_timeout function")
    connection = LLMConnection.get_instance()
    response = await connection.generate(prompt, timeout)
    print("LLM Response: ", response.text)
    return response.text


# Create a global client for backward compatibility
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


async def call_llm(prompt: str, timeout=30, connection=None):
    """
    Call the LLM with the given prompt using the singleton connection or a provided one.
    
    Args:
        prompt: The text prompt to send to the LLM
        timeout: Maximum time to wait for a response in seconds
        connection: Optional connection to use (if None, uses singleton)
        
    Returns:
        The text response from the LLM
    """
    if connection is None:
        connection = LLMConnection.get_instance()
    response = await connection.generate(prompt, timeout)
    return response.text


# Reuse the connection in a more explicit way
async def call_llm_with_connection(connection, prompt, timeout=30):
    """
    Call the LLM with an explicitly provided connection.
    
    Args:
        connection: An instance of LLMConnection
        prompt: The text prompt to send to the LLM
        timeout: Maximum time to wait for a response in seconds
        
    Returns:
        The text response from the LLM
    """
    response = await connection.generate(prompt, timeout)
    return response.text


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

