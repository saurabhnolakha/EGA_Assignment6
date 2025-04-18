from llm import call_llm, LLMConnection, get_connection

async def decision(facts, memory=None, connection=None):
    """
    Make a decision based on facts and optional memory.
    
    Args:
        facts: The perceived facts
        memory: Optional memory information
        connection: Optional LLMConnection instance
        
    Returns:
        Decision response from LLM
    """
    if memory is None:
        context = f"""Facts: {facts}\n what should the agent do next?
        Respond next action strictly in JSON format as shown below:     
        {{
            "task": "add numbers",
            "input": [1, 2]
        }}
        """
    else:
        context = f"Facts: {facts}\nMemory: {memory} \n what should the agent do next?"
    
    print("Decision Prompt: ", context)
    
    # Use the provided connection or get the singleton
    if connection is None:
        connection = get_connection()
        
    response = await call_llm(context, connection=connection)
    print("Decision Response: ", response)
    return response


