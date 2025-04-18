import json
import os
from typing import List
from llm import call_llm, LLMConnection, get_connection


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

    async def add(self, fact: str):
        """Add a fact to memory and save to disk"""
        self.memory.append(fact)
        # Save to disk after each update
        self._save_to_disk()

    def _save_to_disk(self):
        """Save memory to JSON file on disk"""
        with open(self.file_path, 'w') as file:
            json.dump(self.memory, file, indent=2)

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
            
        prompt = f"""Given the memory: {self.memory}, extract the relevant information from the memory and answer the following query: {query}
        Return the result in strictly the json format as shown below:
        Example:
        {{
            "memory": [
                "fact1",
                "fact2",
                "fact3"
            ]
        }}

        If answer is not found in the memory, return "No information found" in plain text not in JSON format
        """
        response = await call_llm(prompt, connection=llm_connection)
        print("Memory Module Response: ", response)
        return response

    