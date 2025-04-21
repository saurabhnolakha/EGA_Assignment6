from llm import call_llm, LLMConnection, get_connection
from utils import get_logger

# Get a logger for this module
logger = get_logger(__name__)

async def perception(user_input: str, connection=None):
    prompt = f"""
Extract key facts from the user's input:
{user_input}

REASONING INSTRUCTIONS:
- First, carefully analyze what mathematical operation is being requested
- Break down the input to identify all numeric values present
- Map the operation to the appropriate function name
- Verify that all required parameters have been extracted
- Double-check that extracted values are actually numbers, not descriptions

IMPORTANT: For mathematical operations, always extract SPECIFIC NUMERIC VALUES from the input, not generic descriptions.

For example:
- Instead of "multiply two large numbers", extract the actual numbers like "multiply(786897676, 89696986)"
- Instead of "add some numbers", extract the specific values like "add(5, 7)"

VERIFICATION STEPS:
- Confirm all parameters are specific numeric values
- Ensure the mathematical operation matches the intended function
- Check that parameter count matches the operation requirements
- Verify no implicit conversions are needed

REASONING TYPES:
- [OPERATION IDENTIFICATION] - Recognizing the mathematical function
- [VALUE EXTRACTION] - Isolating specific numeric parameters
- [PARAMETER VALIDATION] - Ensuring all values are numeric and complete

If the user doesn't provide specific numbers, ask them to provide the exact values before proceeding.

OUTPUT FORMAT:
Return the result in strictly the json format with exactly these fields:
{{
    "task": "the specific operation to perform (add, multiply, etc.)",
    "function_call": "the function to call with specific parameters",
    "function_call_params": {{
        "param1": numeric_value1,
        "param2": numeric_value2
    }}
}}

For mathematical operations:
1. Always use the exact numbers from the input
2. Never use descriptions like "large numbers" or "small values" as parameters
3. Ensure parameters are numeric, not text descriptions
4. If specific numbers aren't provided, set task to "need_clarification" 

ERROR HANDLING:
- If numbers are ambiguous, choose the most reasonable interpretation
- If operation is unclear but numbers are present, use "unknown_operation"
- If no numbers are found, use "need_clarification"

Example for "What is 5 plus 7?":
{{
    "task": "add",
    "function_call": "add(5, 7)",
    "function_call_params": {{
        "param1": 5,
        "param2": 7
    }}

Note: 
1. You should not provide a list of sources/bibliography at the end of the response.
2. Do not add any explanations or descriptions. Return strictly ONLY the JSON object.
3. Do not add any other text or comments for any sections mentioned above. Return strictly ONLY the JSON object.
}}
"""
    
    logger.info("Perception Module Initialized...")
    logger.debug(f"Perception Prompt: {prompt}")
    
    # Use the provided connection or get the singleton
    if connection is None:
        logger.debug("No connection provided, getting singleton")
        connection = get_connection()
        
    logger.info(f"Processing user input: '{user_input}'")
    perception_result = await call_llm(prompt, connection=connection)
    logger.info("Perception processing completed")
    logger.debug(f"Perception Result: {perception_result}")
    return perception_result

                   
