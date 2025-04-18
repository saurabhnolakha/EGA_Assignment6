from llm import call_llm


async def perception(user_input: str):
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
    Perception_result = await call_llm(prompt)
    print("Perception Result: ", Perception_result)
    return Perception_result

                   
