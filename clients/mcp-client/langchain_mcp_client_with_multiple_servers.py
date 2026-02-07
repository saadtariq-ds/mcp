import asyncio
import os
import sys
import json
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()


class CustomEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle non-serializable objects returned by LangChain.
    If the object has a 'content' attribute (such as HumanMessage or ToolMessage), serialize it accordingly.
    """
    def default(self, o):
        # Check if the object has a 'content' attribute
        if hasattr(o, "content"):
            # Return a dictionary containing the type and content of the object
            return {"type": o.__class__.__name__, "content": o.content}
        # Otherwise, use the default serialization
        return super().default(o)


def read_config_json():
    """
    Reads the MCP server configuration JSON.

    Priority:
      1. Try to read the path from the THEAILANGUAGE_CONFIG environment variable.
      2. If not set, fallback to a default file 'theailanguage_config.json' in the same directory.

    Returns:
        dict: Parsed JSON content with MCP server definitions.
    """
    # Attempt to get the config file path from the environment variable
    config_path = os.getenv("CONFIG")

    if not config_path:
        # If environment variable is not set, use a default config file in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        print(f"⚠️ CONFIG not set. Falling back to: {config_path}")

    try:
        # Open and read the JSON config file
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        # If reading fails, print an error and exit the program
        print(f"❌ Failed to read config file at '{config_path}': {e}")
        sys.exit(1)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_retries=2,
    google_api_key=os.getenv("GEMINI_API_KEY")
)


async def run_agent():
    """
    Connects to all MCP servers defined in the configuration, loads their tools, creates a unified React agent,
    and starts an interactive loop to query the agent.
    """
    config = read_config_json()
    mcp_servers = config.get("mcpServers", {})
    if not mcp_servers:
        print("❌ No MCP servers found in the configuration.")
        return

    tools = [] 

    # Use AsyncExitStack to manage and cleanly close multiple asynchronous resources
    async with AsyncExitStack() as stack:
        # Iterate over each MCP server defined in the configuration
        for server_name, server_info in mcp_servers.items():
            print(f"\n🔗 Connecting to MCP Server: {server_name}...")

            # Create StdioServerParameters using the command and arguments specified for the server
            server_params = StdioServerParameters(
                command=server_info["command"],
                args=server_info["args"]
            )

            try:
                # Establish a stdio connection to the server using the server parameters
                read, write = await stack.enter_async_context(stdio_client(server_params))

                # Create a client session using the read and write streams from the connection
                session = await stack.enter_async_context(ClientSession(read, write))

                # Initialize the session (e.g., perform handshake or setup operations)
                await session.initialize()

                # Load the MCP tools from the connected server using the adapter function
                server_tools = await load_mcp_tools(session)

                # Iterate over each tool and add it to the aggregated tools list
                for tool in server_tools:
                    print(f"\n🔧 Loaded tool: {tool.name}")
                    tools.append(tool)

                print(f"\n✅ {len(server_tools)} tools loaded from {server_name}.")
            except Exception as e:
                # Handle any errors that occur during connection or tool loading for the server
                print(f"❌ Failed to connect to server {server_name}: {e}")

        # If no tools were loaded from any server, exit the function
        if not tools:
            print("❌ No tools loaded from any server. Exiting.")
            return

        # Create a React agent using the Google Gemini LLM and the list of aggregated tools
        agent = create_react_agent(llm, tools)

        # Start the interactive chat loop
        print("\n🚀 MCP Client Ready! Type 'quit' to exit.")
        while True:
            # Prompt the user to enter a query
            query = input("\nQuery: ").strip()
            if query.lower() == "quit":
                break

            # Invoke the agent asynchronously with the query as the input message
            response = await agent.ainvoke({"messages": query})

            # Format and print the agent's response as nicely formatted JSON
            print("\nResponse:")
            try:
                formatted = json.dumps(response, indent=2, cls=CustomEncoder)
                print(formatted)
            except Exception:
                # If JSON formatting fails, simply print the raw response
                print(str(response))


if __name__ == "__main__":
    asyncio.run(run_agent())
