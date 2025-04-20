# LLM-Powered Mathematical Calculator

## Overview
This system is an advanced LLM-based mathematical operator capable of understanding and solving a wide range of mathematical queries in natural language. It leverages the power of Large Language Models (LLMs) to interpret user requests, extract mathematical operations, and execute them accurately.

The system combines the reasoning capabilities of LLMs with the precision of dedicated calculation functions, creating a flexible calculator that understands human language while maintaining computational accuracy.

## Core Features
- Natural language processing of mathematical queries
- Support for various operations (addition, multiplication, square root, etc.)
- Memory system to recall previous calculations
- Structured data extraction and validation using Pydantic models
- Error handling and graceful fallbacks

## Code Flow

The system follows this general flow:

1. **User Input Processing**: 
   - User inputs a query like "What is 5 plus 7?"
   - The query is processed by `perception()` to understand the intent

2. **Decision Making**:
   - `decision()` analyzes the perception output and determines what operation to perform
   - Memory is checked for previous similar calculations
   - A structured decision object is created with task, function_call, and parameters

3. **Action Execution**:
   - `action()` receives the decision and dynamically calls the appropriate function
   - Parameters are validated using Pydantic models
   - Calculations are performed using dedicated functions in `calculator.py`

4. **Memory Storage**:
   - Results are stored in the memory system for future reference
   - Memory is persisted to disk in JSON format

5. **Response Generation**:
   - Results are formatted into a user-friendly response
   - The final answer is returned to the user

## Critical Scenarios & Challenges

### JSON Extraction and Validation
One of the most critical challenges is extracting structured data from LLM outputs. The `extract_structured_json()` function handles this by:
- Converting various LLM output formats (text, JSON, etc.) into consistent data structures
- Mapping diverse parameter naming conventions to standardized formats (param1→a, integer1→a, etc.)
- Validating data against Pydantic models to ensure type safety
- Providing graceful fallbacks when validation fails

### Parameter Handling
The system must handle varied parameter formats:
- Converting list parameters to dictionaries with alphabetical keys
- Normalizing parameter names from different conventions
- Converting string numbers to integers when appropriate
- Extracting parameters from function call strings when needed

### Dynamic Function Execution
The `execute_function()` helper allows for dynamic execution of any calculator function by:
- Loading functions and models dynamically
- Converting function names to model names using naming conventions
- Handling parameter validation and type conversion
- Providing informative error messages

### Memory Management
The memory system faces challenges in:
- Finding exact function and parameter matches
- Distinguishing between semantically similar but different operations
- Persisting complex data structures to disk
- Efficiently recalling relevant information from memory

## Key LLM Prompts

### Perception Prompt
Used to interpret the user's intent and extract the mathematical operation:

```
Extract key facts from the user's input:
{user_input}

REASONING INSTRUCTIONS:
- First, carefully analyze what mathematical operation is being requested
- Break down the input to identify all numeric values present
- Map the operation to the appropriate function name
- Verify that all required parameters have been extracted
- Double-check that extracted values are actually numbers, not descriptions

IMPORTANT: For mathematical operations, always extract SPECIFIC NUMERIC VALUES from the input, not generic descriptions.

For example:
- Instead of "multiply two large numbers", extract the actual numbers like "multiply(786897676, 89696986)"
- Instead of "add some numbers", extract the specific values like "add(5, 7)"

VERIFICATION STEPS:
- Confirm all parameters are specific numeric values
- Ensure the mathematical operation matches the intended function
- Check that parameter count matches the operation requirements
- Verify no implicit conversions are needed

REASONING TYPES:
- [OPERATION IDENTIFICATION] - Recognizing the mathematical function
- [VALUE EXTRACTION] - Isolating specific numeric parameters
- [PARAMETER VALIDATION] - Ensuring all values are numeric and complete

If the user doesn't provide specific numbers, ask them to provide the exact values before proceeding.

OUTPUT FORMAT:
Return the result in strictly the json format with exactly these fields:
{{
    "task": "the specific operation to perform (add, multiply, etc.)",
    "function_call": "the function to call with specific parameters",
    "function_call_params": {{
        "param1": numeric_value1,
        "param2": numeric_value2
    }}
}}

For mathematical operations:
1. Always use the exact numbers from the input
2. Never use descriptions like "large numbers" or "small values" as parameters
3. Ensure parameters are numeric, not text descriptions
4. If specific numbers aren't provided, set task to "need_clarification" 

ERROR HANDLING:
- If numbers are ambiguous, choose the most reasonable interpretation
- If operation is unclear but numbers are present, use "unknown_operation"
- If no numbers are found, use "need_clarification"

Example for "What is 5 plus 7?":
{{
    "task": "add",
    "function_call": "add(5, 7)",
    "function_call_params": {{
        "param1": 5,
        "param2": 7
    }}

Note: 
1. You should not provide a list of sources/bibliography at the end of the response.
2. Do not add any explanations or descriptions. Return strictly ONLY the JSON object.
3. Do not add any other text or comments for any sections mentioned above. Return strictly ONLY the JSON object.
}}
```

### Memory Recall Prompt
Used to find relevant previous calculations in memory:

```
Given the memory entries below, find any entries that are EXACTLY relevant to the query: "{query}"

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

```

## Utility Functions

The `util.py` file contains several key functions that support the system:

1. **Parameter Normalization**: Functions that standardize parameter formats and names
2. **Type Conversion**: Utilities for safely converting between different data types
3. **String Processing**: Helpers for extracting data from text and parsing function calls
4. **Error Handling**: Utilities for handling and reporting errors gracefully
5. **Configuration Management**: Functions to manage system settings and defaults

## Getting Started

To use this LLM-based calculator:

1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Configure the LLM connection in `config.py` or using environment variables
3. Run the main application: `python main.py`
4. Enter mathematical queries in natural language
5. Review calculation results and memory operations

## Extending the System

New mathematical functions can be added by:

1. Creating function implementation in `calculator.py`
2. Adding corresponding input/output models in `models.py`
3. The system will automatically discover and use the new function

No modifications to `action.py` are needed thanks to the dynamic function execution system.
