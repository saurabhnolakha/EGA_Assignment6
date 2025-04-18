from llm import call_llm, LLMConnection, get_connection


async def perception(user_input: str, connection=None):
    prompt = f"""
Extract key facts from the user's input:
{user_input}
Return the result in strictly the json format with exactly these fields:
{{
    "task": "the task to perform",
    "function_call": "the function to call",
    "function_call_params": {{
        "param1": "value1",
        "param2": "value2"
    }}
}}
"""
    
    print("Perception Module Initialized...")
    print("Perception Prompt: ", prompt)
    
    # Use the provided connection or get the singleton
    if connection is None:
        connection = get_connection()
        
    perception_result = await call_llm(prompt, connection=connection)
    print("Perception Result: ", perception_result)
    return perception_result

                   
