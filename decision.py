from llm import call_llm, LLMConnection, get_connection
from utils import extract_json, get_logger
import json

# Get a logger for this module
logger = get_logger(__name__)

async def decision(facts, tools_description, memory=None, connection=None):
    """
    Make a decision based on facts and optional memory.
    
    Args:
        facts: The perceived facts
        tools_description: Available tools description
        memory: Optional memory information
        connection: Optional LLMConnection instance
        
    Returns:
        Decision response from LLM
    """
    logger.info("Making decision based on facts and memory")
    logger.debug(f"Facts: {facts}")
    
    if memory is None or "No information found" in memory:
        logger.info("No relevant memory found, creating decision prompt")
        context = f"""Facts: {facts}\n Available Tools: {tools_description}.
        Given the above information, decide what action should be performed next.
        
        Respond strictly in JSON format as shown below:     
        {{
            "task": "add", 
            "function_call": "add(1, 2)",
            "function_call_params": [1, 2], 
            "output": null
        }}
        
        "task" : function to be called from the available tools.
        "function_call" : function call to be performed.
        "function_call_params" : parameters for the function call.
        "output" : output of the function call. It should always be null when no memory match is found.

        Note: Do NOT modify or add explanations or descriptions. Return strictly ONLY the JSON object.
        """
    else:
        logger.info("Using memory for decision")
        # First attempt to extract structured memory data
        try:
            logger.debug("Attempting to extract structured memory data")
            memory_data = extract_json(memory)
            if isinstance(memory_data, dict) and "memory" in memory_data:
                # We have structured memory data, format it for decision
                memory_obj = memory_data["memory"]
                logger.info("Using structured memory data for decision")
                
                context = f"""
                Memory has found a matching previous calculation: {memory}
                
                Extract ONLY the previous calculation result and return it in this JSON structure:
                {{
                    "task": "from_memory",
                    "input": [extracted input values],
                    "output": extracted result value
                }}
                
                Note: Do NOT modify or add explanations or descriptions. Return strictly ONLY the JSON object.
                """
            else:
                # Fallback if memory has unexpected structure
                logger.warning("Memory has unexpected structure, using fallback")
                context = f"""
                Facts: {facts}
                Memory: {memory}
                Available Tools: {tools_description}
                
                Based on the memory information provided, extract:
                1. The exact operation that was performed
                2. The exact input values that were used
                3. The exact result that was calculated
                
                Return this information in strict JSON format:
                {{
                    "task": "from_memory",
                    "function_call": "add(1, 2)",
                    "function_call_params": [1, 2], 
                    "output": extracted result value
                }}
                
                Note: Do NOT modify or add explanations or descriptions. Return strictly ONLY the JSON object.
                """
        except Exception as e:
            # Fallback if JSON extraction fails
            logger.warning(f"Failed to extract JSON from memory: {e}")
            context = f"""
            Facts: {facts}
            Memory: {memory}
            Available Tools: {tools_description}
            
            Based on the memory information provided, extract:
            1. The exact operation that was performed
            2. The exact input values that were used
            3. The exact result that was calculated
            
            Return this information in strict JSON format:
            {{
                "task": "from_memory",
                "function_call": "add(1, 2)",
                "function_call_params": [1, 2], 
                "output": extracted result value
            }}
            
            Note: Do NOT modify or add explanations or descriptions. Return strictly ONLY the JSON object.
            """
    
    
    logger.debug(f"Decision Prompt: {context}")
    
    # Use the provided connection or get the singleton
    if connection is None:
        logger.debug("No connection provided, getting singleton")
        connection = get_connection()
        
    logger.info("Sending decision prompt to LLM")
    response = await call_llm(context, connection=connection)
    logger.info("Decision response received")
    logger.debug(f"Decision Response: {response}")
    
    # Attempt to clean the response if it's not pure JSON
    if not response.strip().startswith("{"):
        logger.info("Response doesn't appear to be pure JSON, attempting to extract")
        try:
            # Extract JSON if response contains explanatory text
            extracted = extract_json(response)
            if extracted:
                logger.info("Successfully extracted JSON from response")
                return json.dumps(extracted)
        except Exception as e:
            logger.warning(f"Failed to extract JSON from response: {e}")
            
    return response


