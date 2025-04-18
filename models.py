from pydantic import BaseModel
from typing import List, Optional

# Input/Output models for tools

class AddInput(BaseModel):
    a: int
    b: int

class AddOutput(BaseModel):
    result: int

class MultiplyInput(BaseModel):
    a: int
    b: int

class MultiplyOutput(BaseModel):
    result: int

class SqrtInput(BaseModel):
    a: int

class SqrtOutput(BaseModel):
    result: float

class StringsToIntsInput(BaseModel):
    string: str

class StringsToIntsOutput(BaseModel):
    ascii_values: List[int]

class ExpSumInput(BaseModel):
    int_list: List[int]

class ExpSumOutput(BaseModel):
    result: float

class PerceptionInput(BaseModel):
    user_input: str

class PerceptionOutput(BaseModel):
    task: str
    function_call: Optional[str] = None
    function_call_params: Optional[dict] = None
    
    # Optional: add model configuration to allow extra fields
    class Config:
        extra = "allow"