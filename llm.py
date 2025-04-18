import asyncio
import re
import json
from typing import TypeVar, Type, Any, Optional, Dict, Union
from pydantic import BaseModel, ValidationError
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

# Access your API key and initialize Gemini client correctly
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


async def generate_with_timeout(client, prompt, timeout=100):
    """Generate content with a timeout"""
    print("Calling LLM-Gemini Now...")
    try:
        # Convert the synchronous generate_content call to run in a thread
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None, 
                lambda: client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
            ),
            timeout=timeout
        )
        print("LLM generation completed")
        print("LLM Response: ", response.text)
        return response.text
    except TimeoutError:
        print("LLM generation timed out!")
        raise
    except Exception as e:
        print(f"Error in LLM generation: {e}")
        raise

async def call_llm(prompt: str):

    return await (generate_with_timeout(client, prompt))

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

