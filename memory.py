import json
import os
from typing import List, Dict, Any
from llm import call_llm, LLMConnection, get_connection, extract_json


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
        
        # Load existing memory from file or initialize empty list
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as file:
                    self.memory = json.load(file)
            except json.JSONDecodeError:
                self.memory = []
        else:
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
            print(f"Warning: Skipping memory entry with invalid format: {fact}")

    def _save_to_disk(self):
        """Save memory to JSON file on disk"""
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
                
        with open(self.file_path, 'w') as file:
            json.dump(self.memory, file, indent=2, cls=CustomEncoder)

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
        prompt = f"""Given the memory entries below, find any entries that are relevant to the query: "{query}"

Memory entries:
{json.dumps(self.memory, indent=2)}

Specifically, look for:
1. Entries where the task or operation matches the query
2. Entries that involve the same numbers or parameters as the query
3. Entries where the user input is semantically similar to the query

The query may be in JSON format with "task" and "function_call_params" fields. If so, focus on matching the task and parameters.

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

If answer is not found in the memory, return "No information found" in plain text not in JSON format
"""
        response = await call_llm(prompt, connection=llm_connection)
        # print("Memory Module Response: ", response)
        return response

    