import os
from dotenv import load_dotenv
from google import genai
import asyncio
from concurrent.futures import TimeoutError
from functools import partial
from llm import LLMConnection
from memory import Memory
from perception import perception
from action import action
from decision import decision
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import logging
from utils import to_json, combine_json, extract_json, configure_logging, get_logger

# Configure logging with the custom formatter from utils.py
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "calculator.log")
configure_logging(log_level=logging.INFO, log_file=log_file, use_emojis=True)

# Configure logging to only show warnings and errors
logging.basicConfig(level=logging.WARNING)
# If needed, you can specifically configure the google.generativeai logger
logging.getLogger("google.generativeai").setLevel(logging.WARNING)

# Get a logger for this module
logger = get_logger(__name__)

# Load environment variables from .env file
load_dotenv()

async def main():
    # Initialize the LLM connection at the start
    logger.info("Initializing LLM connection...")
    llm_connection = LLMConnection.get_instance()
    
    # Initialize memory with the connection
    memory = Memory(connection=llm_connection)
    logger.info("Starting main execution...")
    try:
        # Create a single MCP server connection
        logger.info("Establishing connection to MCP server...")
        server_params = StdioServerParameters(
            command="python3",
            args=["calculator.py"],
            timeout=50000 
        )

        async with stdio_client(server_params) as (read, write):
            logger.info("Connection established, creating session...")
            async with ClientSession(read, write) as session:
                logger.info("Session created, initializing...")
                await session.initialize()
                
                # Get available tools
                logger.info("Requesting tool list...")
                tools_result = await session.list_tools()
                tools = tools_result.tools
                logger.info(f"Successfully retrieved {len(tools)} tools")

                # Create system prompt with available tools
                logger.info("Creating system prompt...")
                logger.info(f"Number of tools: {len(tools)}")
                
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
                            logger.debug(f"Added description for tool: {tool_desc}")
                        except Exception as e:
                            logger.warning(f"Error processing tool {i}: {e}")
                            tools_description.append(f"{i+1}. Error processing tool")
                    
                    tools_description = "\n".join(tools_description)
                    logger.info("Successfully created tools description")
                except Exception as e:
                    logger.error(f"Error creating tools description: {e}")
                    tools_description = "Error loading tools"

                # Main loop starts here 
                user_input = input("How can I help you today? ")
                logger.info(f"Received user query: {user_input}")
            
                # Pass the connection to all functions that need it
                perception_result = await perception(user_input, connection=llm_connection)
            
                logger.info("Perception completed")
                logger.debug(f"Perception Result: {perception_result}")
                
                memory_result = await memory.recall(perception_result, connection=llm_connection)
                logger.info("Memory lookup completed")
                logger.debug(f"Memory Result: {memory_result}")
                
                if "No information found" in memory_result:
                    logger.info("No information found in memory, calling LLM for decision")
                    # No memory found - pass perception_result as facts, memory=None (default)
                    decision_result = await decision(perception_result, tools_description, connection=llm_connection)
                else:
                    logger.info("Information found in memory, calling LLM for action")
                    # Memory found - pass perception_result as facts, memory_result as memory
                    decision_result = await decision(perception_result, tools_description, memory=memory_result, connection=llm_connection)
                logger.info("Decision process completed")
                logger.debug(f"Decision: {decision_result}")
            
                try:
                    decision_obj = extract_json(decision_result)
                    if decision_obj.get('output') not in (None, "null", "") :
                    # action_result = await action(decision_result, connection=llm_connection)
                        logger.info("Decision Tree: Output found in memory....")
                        logger.info(f"Action Result: {decision_obj.get('output')}")
                    # await memory.add(decision_result.output) -- Skip adding to memory as it is already in memory   
                    else:
                        logger.info("Decision Tree: Output not found in memory, calling LLM for action")
                        action_result = await action(decision_obj, connection=llm_connection)
                        logger.info("Action execution completed")
                        logger.info(f"Action Result: {action_result}")
                        
                        logger.info("Adding result to memory")
                        await memory.add(to_json(user_input, action_result))
                except ValueError as e:
                    logger.error(f"Error in extracting JSON from decision result: {e}")
    except Exception as e:
            logger.critical(f"Error in main execution: {e}")
            import traceback
            logger.critical(f"Stack trace: {traceback.format_exc()}")
    finally:
            logger.info("Execution complete")


if __name__ == "__main__":
    asyncio.run(main())
