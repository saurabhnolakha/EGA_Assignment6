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
Given the user input: "{user_input}"

Analyze this input and extract:
1. The mathematical operation or task the user wants to perform
2. The specific parameters or values needed for this operation

Return your analysis in the following JSON format:
{
    "task": "The specific task or operation",
    "function_call": "function_name(param1, param2, ...)",
    "function_call_params": {
        "param1": value1,
        "param2": value2
    }
}

For example, if the user asks "What is 5 plus 7?", return:
{
    "task": "add",
    "function_call": "add(5, 7)",
    "function_call_params": {
        "a": 5,
        "b": 7
    }
}
```

### Memory Recall Prompt
Used to find relevant previous calculations in memory:

```
Given the memory entries below, find any entries that are EXACTLY relevant to the query: "{query}"

Memory entries:
{memory_entries}

REASONING INSTRUCTIONS:
- Reason step-by-step through each memory entry to assess relevance
- First identify the core components of the query (task, parameters, numbers, operations)
- For each memory entry, systematically compare these components to find matches
- For calculator operations, require EXACT matches on ALL parameters
- Assign a relevance score to each potential match
- Verify your reasoning before concluding

CRITICAL RULE: For calculator operations, if ANY parameter value is different, it is NOT a match.
```

### Decision-Making Prompt
Used to determine the correct action based on the perception and memory:

```
Facts: {facts}
Memory: {memory}
Available Tools: {tools_description}

Based on the information provided, identify the mathematical operation being requested and the specific parameters needed.

Return your decision in the following JSON format:
{
    "task": "The specific task or operation (e.g., add, multiply)",
    "function_call": "function_name(param1, param2, ...)",
    "function_call_params": {
        "a": value1,
        "b": value2
    }
}
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
