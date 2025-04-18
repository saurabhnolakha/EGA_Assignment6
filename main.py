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
from decision import decision
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

# Load environment variables from .env file

async def main():
    memory = Memory()
    #reset_state()  # Reset at the start of main
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

# Main loop starts here 
                user_input = input("How can I help you today? ")
            
                perception_result = await perception(user_input)
            
                print("\nPerception Result: ", perception_result)
                
                memory_result = await memory.recall(perception_result)
                print("\nMemory Result: ", memory_result)
                
                if "No information found" in memory_result:
                    print("No information found in memory, calling LLM for decision")
                    decision_result = await decision(perception_result)
                else:
                    print("Information found in memory, calling LLM for action")
                    decision_result = await call_llm(memory_result)
                print("\nDecision: ", decision_result)
            
                action_result = await action(decision_result)
                print("\nAction Result: ", action_result)
                
                await memory.add(action_result)

    except Exception as e:
            print(f"Error in main execution: {e}")
            import traceback
            traceback.print_exc()
    finally:
            #reset_state()  # Reset at the end of main
            print("Resetting state")


if __name__ == "__main__":
    asyncio.run(main())
