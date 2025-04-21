import asyncio
import re
import json
from typing import TypeVar, Type, Any, Optional, Dict, Union
from pydantic import BaseModel, ValidationError
from google import genai
from dotenv import load_dotenv
import os
from utils import get_logger

# Get a logger for this module
logger = get_logger(__name__)

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
        logger.info(f"Initialized LLM connection with model: {model}")
    
    async def generate(self, prompt, timeout=30):
        """Generate content with timeout"""
        logger.info(f"Generating with model: {self.model}, timeout: {timeout}s")
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
            logger.info("LLM generation completed")
            return response
        except asyncio.TimeoutError:
            logger.error(f"LLM generation timed out after {timeout} seconds!")
            raise
        except Exception as e:
            logger.error(f"Error in LLM generation: {e}")
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
    logger.warning("Using deprecated generate_with_timeout function")
    connection = LLMConnection.get_instance()
    response = await connection.generate(prompt, timeout)
    logger.debug(f"LLM Response: {response.text}")
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
    
    prompt_length = len(prompt)
    logger.info(f"Calling LLM with prompt length: {prompt_length} chars")
    if prompt_length > 200:
        truncated = prompt[:100] + "..." + prompt[-100:]
        logger.debug(f"Truncated prompt: {truncated}")
    else:
        logger.debug(f"Prompt: {prompt}")
        
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
    logger.info("Calling LLM with explicit connection")
    response = await connection.generate(prompt, timeout)
    return response.text

