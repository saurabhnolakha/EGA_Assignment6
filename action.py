from llm import call_llm, extract_structured_json
from calculator import add, multiply

async def action(decision_text: str):
    try:
        # First, extract the task and parameters from the decision
        from models import PerceptionOutput
        decision = extract_structured_json(decision_text, PerceptionOutput)
        
        if decision.task == "add numbers":
            # Extract parameters for addition
            from models import AddInput
            add_params = AddInput.parse_obj(decision.function_call_params)
            result = add(add_params.a, add_params.b)
            return {"result": result}
        
        elif decision.task == "multiply numbers":
            # Extract parameters for multiplication
            from models import MultiplyInput
            multiply_params = MultiplyInput.parse_obj(decision.function_call_params)
            result = multiply(multiply_params.a, multiply_params.b)
            return {"result": result}
        
        else:
            return await call_llm(decision_text)
    
    except ValueError as e:
        print(f"Error extracting structured data: {e}")
        return await call_llm(decision_text)