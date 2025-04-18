import asyncio
import re
import json
from typing import TypeVar, Type, Any, Optional, Dict, Union
from pydantic import BaseModel, ValidationError
from google import genai
from dotenv import load_dotenv
import os

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


def extract_json(llm_output: str) -> Dict[str, Any]:
    """
    Extract JSON from LLM output text using regex pattern matching.
    
    Args:
        llm_output: The raw text output from the LLM
        
    Returns:
        A dictionary parsed from the JSON in the output
        
    Raises:
        ValueError: If no valid JSON is found in the text
    """
    # Strip the text attribute if the input is a Gemini response object
    if hasattr(llm_output, 'text'):
        llm_output = llm_output.text
    
    # Try to find JSON pattern with curly braces
    json_pattern = r'\{(?:[^{}]|(?R))*\}'
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
        
    raise ValueError("No valid JSON found in the LLM output")

T = TypeVar('T', bound=BaseModel)

def extract_structured_json(llm_output: Union[str, Any], model_class: Type[T]) -> T:
    """
    Extract JSON from LLM output and validate it against a Pydantic model.
    
    Args:
        llm_output: The raw text output from the LLM or response object
        model_class: The Pydantic model class to validate against
        
    Returns:
        An instance of the Pydantic model
        
    Raises:
        ValueError: If no valid JSON is found or if validation fails
    """
    try:
        # First extract the JSON
        json_data = extract_json(llm_output)
        
        # Then validate with Pydantic
        return model_class.parse_obj(json_data)
    except ValidationError as e:
        raise ValueError(f"JSON validation failed: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error extracting or validating JSON: {str(e)}")

