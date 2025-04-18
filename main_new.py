import os
from dotenv import load_dotenv
from google import genai
import asyncio
from concurrent.futures import TimeoutError
from functools import partial
from llm import call_llm
from memory import Memory
from perception import perception
from action import action
# Load environment variables from .env file

async def main():
    memory = Memory()
    user_input = input("Enter your input: ")

    
   
    perception_result = await perception(user_input)
   
    print("\nPerception Result: ", perception_result)
    
    memory_result = await memory.recall(perception_result)
    print("\nMemory Result: ", memory_result)
    
    decision = await call_llm(memory_result)
    print("\nDecision: ", decision)
   
    action_result = await action(decision)
    print("\nAction Result: ", action_result)
    
    await memory.add(action_result)


if __name__ == "__main__":
    asyncio.run(main())
