import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import asyncio
from google import genai
from concurrent.futures import TimeoutError
from functools import partial
import json
import re
from perception import perception
from memory import memory
from action import action
from decision import decision

# Load environment variables from .env file
load_dotenv()

# Access your API key and initialize Gemini client correctly
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

max_iterations = 15
last_response = None
iteration = 0
iteration_response = []

async def generate_with_timeout(client, prompt, timeout=100):
    """Generate content with a timeout"""
    print("Starting LLM generation...")
    try:
        # Convert the synchronous generate_content call to run in a thread
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None, 
                lambda: client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
            ),
            timeout=timeout
        )
        print("LLM generation completed")
        return response
    except TimeoutError:
        print("LLM generation timed out!")
        raise
    except Exception as e:
        print(f"Error in LLM generation: {e}")
        raise

def reset_state():
    """Reset all global variables to their initial state"""
    global last_response, iteration, iteration_response
    last_response = None
    iteration = 0
    iteration_response = []

async def validateJSON(jsonData):
    """Validate if a string is valid JSON"""
    try:
        json.loads(jsonData)
        return True
    except json.JSONDecodeError:
        return False

def correct_JSON(json_str):
    """Correct common JSON formatting errors from LLM outputs"""
    # Add missing quotes around property names
    json_str = re.sub(r'(\w+):', r'"\1":', json_str)
    
    # Add quotes around string values that aren't already quoted
    # Pattern looks for ": followed by alphanumeric text until comma, closing brace, or end of string
    json_str = re.sub(r'": ?([^",{}\s][^",{}]*?)([,}]|$)', r'": "\1"\2', json_str)
    
    # Fix unclosed quotes or brackets
    brackets_stack = []
    quote_open = False
    last_char = None
    
    # Check for unclosed brackets or quotes
    for char in json_str:
        if char in '{[' and not quote_open:
            brackets_stack.append(char)
        elif char in '}]' and not quote_open:
            if not brackets_stack:
                # Too many closing brackets, can't fix properly
                break
            opening = brackets_stack.pop()
            if (opening == '{' and char != '}') or (opening == '[' and char != ']'):
                # Mismatched brackets, can't fix properly
                break
        elif char == '"' and last_char != '\\':
            quote_open = not quote_open
        last_char = char
    
    # Close any unclosed brackets
    if brackets_stack:
        for bracket in reversed(brackets_stack):
            if bracket == '{':
                json_str += '}'
            elif bracket == '[':
                json_str += ']'
    
    # Close any unclosed quotes
    if quote_open:
        json_str += '"'
    
    return json_str

async def main():
    reset_state()  # Reset at the start of main
    print("Starting main execution...")
    user_input = input("How can I help you today? ")
    try:
        # Create a single MCP server connection
        print("Establishing connection to MCP server...")
        server_params = StdioServerParameters(
            command="python3",
            args=["calculator.py"],
            timeout=50000 
        )

        async with stdio_client(server_params) as (read, write):
            print("Connection established, creating session...")
            async with ClientSession(read, write) as session:
                print("Session created, initializing...")
                await session.initialize()
                
                # Get available tools
                print("Requesting tool list...")
                tools_result = await session.list_tools()
                tools = tools_result.tools
                print(f"Successfully retrieved {len(tools)} tools")

                # Create system prompt with available tools
                print("Creating system prompt...")
                print(f"Number of tools: {len(tools)}")
                
                try:
                    # First, let's inspect what a tool object looks like
                    # if tools:
                    #     print(f"First tool properties: {dir(tools[0])}")
                    #     print(f"First tool example: {tools[0]}")
                    
                    tools_description = []
                    for i, tool in enumerate(tools):
                        try:
                            # Get tool properties
                            params = tool.inputSchema
                            desc = getattr(tool, 'description', 'No description available')
                            name = getattr(tool, 'name', f'tool_{i}')
                            
                            # Format the input schema in a more readable way
                            if 'properties' in params:
                                param_details = []
                                for param_name, param_info in params['properties'].items():
                                    param_type = param_info.get('type', 'unknown')
                                    param_details.append(f"{param_name}: {param_type}")
                                params_str = ', '.join(param_details)
                            else:
                                params_str = 'no parameters'

                            tool_desc = f"{i+1}. {name}({params_str}) - {desc}"
                            tools_description.append(tool_desc)
                            print(f"Added description for tool: {tool_desc}")
                        except Exception as e:
                            print(f"Error processing tool {i}: {e}")
                            tools_description.append(f"{i+1}. Error processing tool")
                    
                    tools_description = "\n".join(tools_description)
                    print("Successfully created tools description")
                except Exception as e:
                    print(f"Error creating tools description: {e}")
                    tools_description = "Error loading tools"
                
                print("Created system prompt...")
                
                system_prompt = f"""You are a super mathematical agent solving any problems in iterations. You have access to various tools as listed below:
Available tools:
{tools_description}
                
TASK:
1. Solve the problem using the available tools
2. Iterate until the problem is solved
3. Follow the order of operations strictly as below- 
    - Perception - Perceive the problem using the defined perception() function
    - Memory - Recall the problem using the defined memory() function
    - Decision - Decide the next step using the defined decision() function
    - Action - Perform the next step using the defined action() function

REASONING INSTRUCTIONS:
- Reason step-by-step through each calculation
- Tag your reasoning type (e.g., [ADDITION], [SUBTRACTION], [MULTIPLICATION], [DIVISION], [EXPONENTIAL CALCULATION])
- Verify each intermediate result before proceeding
- If uncertain about any step, explain your uncertainty and attempt an alternative approach

CONVERSATION LOOP:
- Each function call represents one iteration in our solution process
- After each function call, analyze the result before proceeding
- Update your approach based on previous results

OUTPUT FORMAT:
Use EXACTLY ONE line in these formats:
1. For function calls with perception, memory, decision and action:
   PERCEPTION_RESULT: [Perception result]
   FUNCTION_CALL: {{"name":"tool_name","args":{{"value1","value2"}}}}
   
2. For final answers:
   REASONING: [Final verification and confirmation]
   FINAL_ANSWER: [Success/Failure with explanation]

Examples:
- REASONING: [ASCII CONVERSION] Converting "A" to ASCII gives 65. Verified against ASCII table.
  FUNCTION_CALL: {{"name":"strings_to_chars_to_int","args":{{"INDIA"}}}}
  
- REASONING: [TOOL USAGE] Now that calculations are complete, opening paint to visualize result.
  FUNCTION_CALL: {{"name":"open_paint","args":null}}

- REASONING: [VERIFICATION] All steps completed successfully. The sum was correctly displayed in the rectangle.
  FINAL_ANSWER: [Success - All calculations correct and result displayed in paint]

ERROR HANDLING:
- If a function fails, try an alternative approach
- If calculations yield unexpected results, double-check your work
- If the output is not in the correct format, try again with the correct format
- If paint operations fail, retry with simplified parameters

"""

                query = """Find the ASCII values of characters in INDIA and then return sum of exponentials of those values. Open paint, draw a rectangle And print the result inside a rectangle in paint. """
                print("Starting iteration loop...")
                
                # Use global iteration variables
                global iteration, last_response
                
                while iteration < max_iterations:
                    print(f"\n--- Iteration {iteration + 1} ---")
                    if last_response is None:
                        current_query = query
                    else:
                        current_query = current_query + "\n\n" + " ".join(iteration_response)
                        current_query = current_query + "  What should I do next?"

                    # Get model's response with timeout
                    print("Preparing to generate LLM response...")
                    prompt = f"{system_prompt}\n\nQuery: {current_query}"
                    try:
                        response = await generate_with_timeout(client, prompt)
                        response_text = response.text.strip()
                        print(f"LLM Response: {response_text}")
                        
                        # Find the FUNCTION_CALL line in the response
                        for line in response_text.split('\n'):
                            line = line.strip()
                            if line.startswith("FUNCTION_CALL:"):
                                response_text = line
                                break
                        
                    except Exception as e:
                        print(f"Failed to get LLM response: {e}")
                        break


                    if response_text.startswith("FUNCTION_CALL:"):
                        _, function_info = response_text.split(":", 1)
                        
                        # Extract the JSON-like content with regex
                        json_match = re.search(r'\{.*\}', function_info)
                        if not json_match:
                            raise ValueError(f"Could not extract JSON from function call: {function_info}")
                            
                        # Replace double braces with single braces for proper JSON
                        json_str = json_match.group(0).replace('{{', '{').replace('}}', '}')
                        
                        # Check if JSON is valid, if not, try to correct it
                        is_valid = await validateJSON(json_str)
                        if not is_valid:
                            print(f"DEBUG: Invalid JSON detected: {json_str}")
                            corrected_json = correct_JSON(json_str)
                            print(f"DEBUG: Corrected JSON: {corrected_json}")
                            is_valid = await validateJSON(corrected_json)
                            if is_valid:
                                json_str = corrected_json
                            else:
                                print(f"DEBUG: Could not correct JSON: {corrected_json}")
                        
                        # Parse the JSON
                        try:
                            func_data = json.loads(json_str)
                            func_name = func_data.get('name')
                            args = func_data.get('args', {})
                            
                            print(f"\nDEBUG: Parsed function name: {func_name}")
                            print(f"DEBUG: Parsed arguments: {args}")
                            
                        except json.JSONDecodeError as e:
                            print(f"DEBUG: JSON decode error: {e}")
                            print(f"DEBUG: Attempted to parse: {json_str}")
                            
                            # Last resort: try to extract function name and args with regex
                            try:
                                # Extract function name with regex
                                name_match = re.search(r'"name"\s*:\s*"([^"]+)"', json_str)
                                if name_match:
                                    func_name = name_match.group(1)
                                    print(f"DEBUG: Extracted function name with regex: {func_name}")
                                    
                                    # Extract arguments
                                    args = {}
                                    
                                    # Special handling for int_list_to_exponential_sum function
                                    if func_name == "int_list_to_exponential_sum":
                                        # Try to extract array values with a specific regex for arrays
                                        array_match = re.search(r'"int_list"\s*:\s*\[(.*?)\]', json_str)
                                        if array_match:
                                            array_str = array_match.group(1)
                                            # Split by commas and convert to integers
                                            try:
                                                int_list = [int(x.strip()) for x in array_str.split(',')]
                                                args["int_list"] = int_list
                                                print(f"DEBUG: Extracted int_list with regex: {int_list}")
                                            except ValueError:
                                                print(f"DEBUG: Failed to parse int_list values: {array_str}")
                                    
                                    # For other arguments or as fallback
                                    if "int_list" not in args:
                                        # Look for other arguments
                                        args_matches = re.findall(r'"([^"]+)"\s*:\s*"?([^",}\[\]]+)"?', json_str)
                                        for key, value in args_matches:
                                            if key != "name" and key != "args" and key != "int_list":
                                                args[key] = value
                                    
                                    print(f"DEBUG: Extracted arguments with regex: {args}")
                                else:
                                    raise ValueError(f"Could not extract function information from: {json_str}")
                            except Exception as ex:
                                print(f"DEBUG: Regex extraction failed: {ex}")
                                raise ValueError(f"Invalid JSON in function call: {e}")
                        
                        try:
                            # Find the matching tool to get its input schema
                            tool = next((t for t in tools if t.name == func_name), None)
                            if not tool:
                                print(f"DEBUG: Available tools: {[t.name for t in tools]}")
                                raise ValueError(f"Unknown tool: {func_name}")

                            print(f"DEBUG: Found tool: {tool.name}")
                            print(f"DEBUG: Tool schema: {tool.inputSchema}")

                            # Prepare arguments according to the tool's input schema
                            arguments = {}
                            schema_properties = tool.inputSchema.get('properties', {})
                            print(f"DEBUG: Schema properties: {schema_properties}")

                            # Map the parsed arguments to the expected schema
                            for param_name, param_info in schema_properties.items():
                                # Find the corresponding argument in the parsed args
                                # Look for both the exact name and common prefixes (params_1, param1, etc.)
                                value = None
                                for arg_name, arg_value in args.items():
                                    if arg_name == param_name or f"params_{param_name}" == arg_name:
                                        value = arg_value
                                        break
                                
                                if value is None:
                                    raise ValueError(f"Required parameter {param_name} not found in arguments")
                                
                                param_type = param_info.get('type', 'string')
                                print(f"DEBUG: Converting parameter {param_name} with value {value} to type {param_type}")
                                
                                # Convert the value to the correct type based on the schema
                                if param_type == 'integer':
                                    arguments[param_name] = int(value)
                                elif param_type == 'number':
                                    arguments[param_name] = float(value)
                                elif param_type == 'array':
                                    # Make sure we're not converting a list to a single value
                                    if isinstance(value, list):
                                        arguments[param_name] = value
                                    elif isinstance(value, str):
                                        value = value.strip('[]').split(',')
                                        arguments[param_name] = [int(x.strip()) if x.strip().isdigit() else x.strip() for x in value]
                                    else:
                                        arguments[param_name] = [value]
                                else:
                                    arguments[param_name] = str(value)

                            print(f"DEBUG: Final arguments: {arguments}")
                            print(f"DEBUG: Calling tool {func_name}")
                            
                            result = await session.call_tool(func_name, arguments=arguments)
                            print(f"DEBUG: Raw result: {result}")
                            
                            # Get the full result content
                            if hasattr(result, 'content'):
                                print(f"DEBUG: Result has content attribute")
                                # Handle multiple content items
                                if isinstance(result.content, list):
                                    iteration_result = [
                                        item.text if hasattr(item, 'text') else str(item)
                                        for item in result.content
                                    ]
                                else:
                                    iteration_result = str(result.content)
                            else:
                                print(f"DEBUG: Result has no content attribute")
                                iteration_result = str(result)
                                
                            print(f"DEBUG: Final iteration result: {iteration_result}")
                            
                            # Format the response based on result type
                            if isinstance(iteration_result, list):
                                result_str = f"[{', '.join(iteration_result)}]"
                            else:
                                result_str = str(iteration_result)
                            
                            iteration_response.append(
                                f"In the {iteration + 1} iteration you called {func_name} with {arguments} parameters, "
                                f"and the function returned {result_str}."
                            )
                            last_response = iteration_result

                        except Exception as e:
                            print(f"DEBUG: Error details: {str(e)}")
                            print(f"DEBUG: Error type: {type(e)}")
                            import traceback
                            traceback.print_exc()
                            iteration_response.append(f"Error in iteration {iteration + 1}: {str(e)}")
                            break

                    elif response_text.startswith("FINAL_ANSWER:"):
                        print("FINAL_ANSWER:"+response_text)
                        print("\n=== Agent Execution Complete ===")
                        break
                    iteration += 1

    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        reset_state()  # Reset at the end of main

if __name__ == "__main__":
    asyncio.run(main())
    
    
