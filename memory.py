import json
import os
from typing import List, Dict, Any
from llm import call_llm, LLMConnection, get_connection
from utils import combine_json, extract_json, get_logger

# Get a logger for this module
logger = get_logger(__name__)

class Memory:
    def __init__(self, file_path: str = "memory_data.json", connection=None):
        """
        Initialize Memory with optional LLM connection.
        
        Args:
            file_path: Path to the JSON file for persistence
            connection: Optional LLMConnection instance
        """
        self.file_path = file_path
        # Store connection or use None (will get singleton when needed)
        self.connection = connection
        
        logger.info(f"Initializing Memory with file path: {file_path}")
        
        # Load existing memory from file or initialize empty list
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as file:
                    self.memory = json.load(file)
            except json.JSONDecodeError:
                self.memory = []
        else:
            logger.info(f"Memory file {file_path} not found, initializing empty memory")
            self.memory = []

    async def add(self, fact: Dict[str, Any]):
        """
        Add a fact to memory and save to disk.
        
        Args:
            fact: Dictionary with at least 'user_input' and 'result' keys
        """
        # Ensure the fact has the required format
        if isinstance(fact, dict) and 'user_input' in fact:
            # Ensure it has simple user_input and result keys
            simplified_fact = {
                'user_input': fact.get('user_input', ''),
                'result': fact.get('result', None)
            }
            self.memory.append(simplified_fact)
            # Save to disk after each update
            self._save_to_disk()
        else:
            logger.warning(f"Skipping memory entry with invalid format: {fact}")

    def _save_to_disk(self):
        """Save memory to JSON file on disk"""
        logger.debug(f"Saving memory to disk: {self.file_path}")
        
        # Custom encoder to handle non-serializable types
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                # Handle objects with dict method
                if hasattr(obj, 'dict') and callable(obj.dict):
                    return obj.dict()
                # Handle nested objects with result attribute
                if hasattr(obj, 'result') and not callable(obj.result):
                    return obj.result
                # Default fallback
                return str(obj)
                
        try:
            with open(self.file_path, 'w') as file:
                json.dump(self.memory, file, indent=2, cls=CustomEncoder)
            logger.info(f"Successfully saved {len(self.memory)} memory entries to {self.file_path}")
        except Exception as e:
            logger.error(f"Error saving memory to disk: {e}")

    async def recall(self, query: str, connection=None):
        """
        Query memory for relevant information.
        
        Args:
            query: The query to search for in memory
            connection: Optional LLMConnection instance
            
        Returns:
            Response from LLM about relevant memory
        """
        # Use provided connection, instance connection, or get singleton
        if connection is not None:
            llm_connection = connection
        elif self.connection is not None:
            llm_connection = self.connection
        else:
            llm_connection = get_connection()
            
        # Improved prompt for better matching
        prompt = f""""Given the memory entries below, find any entries that are EXACTLY relevant to the query: "{query}"

Memory entries:
{json.dumps(self.memory, indent=2)}

REASONING INSTRUCTIONS:
- Reason step-by-step through each memory entry to assess relevance
- First identify the core components of the query (task, parameters, numbers, operations)
- For each memory entry, systematically compare these components to find matches
- For calculator operations, require EXACT matches on ALL parameters
- Assign a relevance score to each potential match
- Verify your reasoning before concluding

MATCHING CRITERIA:
Specifically, look for:
1. For mathematical operations (add, multiply, etc.), ALL numeric parameters MUST be an EXACT match
2. The operation type MUST be the same (e.g., "add" matches "add", "multiply" matches "multiply")
3. The order of parameters matters for non-commutative operations
4. For semantic matching, only apply this to the operation type, NOT to the parameter values

CRITICAL RULE: For calculator operations, if ANY parameter value is different, it is NOT a match.

The query may be in JSON format with "task" and "function_call_params" fields. If so, focus on matching the task and parameters.

REASONING TYPES:
- [QUERY ANALYSIS] - Breaking down the query components
- [PARAMETER EXTRACTION] - Identifying the exact numeric values in the query
- [MEMORY COMPARISON] - Comparing query elements to memory entries
- [RELEVANCE SCORING] - Determining match confidence
- [VERIFICATION] - Confirming exact matches for all parameters and the operation type

CONVERSATION CONTEXT:
- This system is for a calculator implementation
- Exact parameter matching is REQUIRED for calculator operations
- Previous results should be used to inform future queries ONLY when ALL parameters match exactly and the operation type is the same.

SELF-VERIFICATION:
- Double-check that function names match exactly
- Verify that ALL parameter values are EXACTLY identical, not just similar
- If ANY single parameter is different, reject the match
- If the operation type is different, reject the match.

EXAMPLES:
1. Query: "Add 1 and 2?"
   Memory: {{"user_input": "add 1 and 2", "result": 3}}
   Result: {{"memory": {{"user_input": "add 1 and 2", "result": 3}}}}

2. Query: "Add 1 and 3?"
   Memory: {{"user_input": "add(1, 2)", "result": 3}}
   Result: "No information found"  <-- DIFFERENT PARAMETERS, NOT A MATCH

3. Query: "Multiply 3 and 4?"
   Memory: {{"user_input": "multiply(1, 2)", "result": 2}}
   Result: "No information found"  <-- DIFFERENT PARAMETERS, NOT A MATCH

   Similarly for the following queries, these matching rules apply:
   - "Addition of 1 and 2?" matches "add(1, 2)"
   - "Multiply 1 and 2?" matches "multiply(1, 2)"  
   - "1 plus 2?" matches "add(1, 2)"
   - "1 * 2?" matches "multiply(1, 2)"

   While for the following queries, and with memory entry as Memory: {{"user_input": "add(1, 2)", "result": 3}}, the matching should fail and return "No information found":
   1) "Add 1 and 2 and 3?"
   2) "Sum of 1, 2 and 3?"
   3) "1 plus 2 plus 3?"
   4) "1 + 2 + 3?"

OUTPUT FORMAT:
Return the result in strictly the json format as shown below:
{{
    "memory": {{
        "user_input": "the matched input",
        "result": the matched result value
    }}
}}

or if multiple matches exist, return one only and only the closest match:
{{
    "memory": {{
        "user_input": "first matched input",
        "result": first matched result value
    }}
}}

ERROR HANDLING:
If answer is not found in the memory, return "No information found" in plain text not in JSON format

Note: 
1. You should not provide a list of sources/bibliography at the end of the response.
2. Do not add any explanations or descriptions. Return strictly ONLY the JSON object.
3. Do not add any other text or comments for any sections mentioned above. Return strictly ONLY the JSON object.

"""
        logger.debug("Sending memory recall prompt to LLM")
        response = await call_llm(prompt, connection=llm_connection)
        logger.info("Memory Module Response: ", response)
        return response

    