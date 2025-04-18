from llm import call_llm

def decision(facts, memory=None):
    if memory is None:
        context=f"Facts: {facts}\n what should the agent do next?"
    else:
        context=f"Facts: {facts}\nMemory: {memory} \n what should the agent do next?"
    print("Decision Prompt: ", context)
    response = call_llm(context)
    print("Decision Response: ", response)
    return response


