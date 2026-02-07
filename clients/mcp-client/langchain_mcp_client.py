import asyncio 
import os 
import sys
import json
from contextlib import AsyncExitStack 
from typing import Optional, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# Custom JSON Encoder
# ---------------------------
class CustomEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles objects with a 'content' attribute.
    
    If an object has a 'content' attribute, it returns a dictionary with the object's type and its content.
    Otherwise, it falls back to the default encoding.
    """
    def default(self, o):
        if hasattr(o, "content"):
            return {"type": o.__class__.__name__, "content": o.content}
        return super().default(o)

# ---------------------------
# LLM Instantiation
# --------------------------- 
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_retries=2,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# ---------------------------
# MCP Server Script Argument
# ---------------------------
if len(sys.argv) < 2:
    print("Usage: python client_langchain_google_genai_bind_tools.py <path_to_server_script>")
    sys.exit(1)
server_script = sys.argv[1]


# ---------------------------
# MCP Server Parameters
# ---------------------------
# Configure parameters to launch the MCP server.
server_params = StdioServerParameters(
    command="python" if server_script.endswith(".py") else "node",
    args=[server_script],
)

mcp_client = None

async def run_agent():
    """
    Connect to the MCP server, load MCP tools, create a React agent, and run an interactive chat loop.
    
    Steps:
      1. Open a stdio connection to the MCP server.
      2. Create and initialize an MCP session.
      3. Store the session in a global holder (mcp_client) for tool access.
      4. Load MCP tools using load_mcp_tools.
      5. Create a React agent using create_react_agent with the LLM and loaded tools.
      6. Enter an interactive loop: for each user query, invoke the agent asynchronously using ainvoke,
         then print the response as formatted JSON using our custom encoder.
    """
    global mcp_client
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()  # Initialize MCP session

            # Set global mcp_client to a simple object holding the session.
            mcp_client = type("MCPClientHolder", (), {"session": session})()

            # Load MCP tools using the adapter; this handles awaiting and conversion.
            tools = await load_mcp_tools(session)

            # Create a React agent using the LLM and the loaded tools.
            agent = create_react_agent(llm, tools)

            print("MCP Client Started! Type 'quit' to exit.")
            while True:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break

                # The agent expects input as a dict with key "messages".
                response = await agent.ainvoke({"messages": query})

                # Format the response as JSON using the custom encoder.
                try:
                    formatted = json.dumps(response, indent=2, cls=CustomEncoder)
                except Exception as e:
                    formatted = str(response)
                print("\nResponse:")
                print(formatted)
    return


if __name__ == "__main__":
    asyncio.run(run_agent())