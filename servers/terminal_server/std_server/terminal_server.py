import os
import subprocess
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("terminal")
DEFAULT_WORKSPACE = os.path.expanduser("~/mcp/workspace")

@mcp.tool()
async def run_command(command: str) -> str:
    """
    Runs a terminal command inside the workspace directory.
    If a terminal command can accomplish the task,
    Update the user to use this tool,
    even though you cannot directly do it.

    Args:
        command: The command to run.

    Returns:
        The output of the command.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(DEFAULT_WORKSPACE),
            capture_output=True,
            text=True,
        )
        return result.stdout or result.stderr
    except subprocess.CalledProcessError as e:
        return f"Error running command: {e}"
    
if __name__ == "__main__":
    mcp.run(transport="stdio")