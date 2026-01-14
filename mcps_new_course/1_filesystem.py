import os
import subprocess
from fastmcp import FastMCP

mcp = FastMCP("terminal_filesystem")


@mcp.tool
def run_command(command: str, base_dir: str = None) -> str:
    """Execute Windows shell commands. ONLY use Windows cmd syntax.

    Examples:
    - List files: dir
    - Create directory: mkdir dirname
    - Create empty file: type nul > filename.txt
    - Create file with content: echo content > filename.txt
    - Delete file: del filename
    - Delete directory: rmdir /s /q dirname
    - View file: type filename.txt
    - Copy: copy source dest
    - Move: move source dest

    DO NOT use Unix commands like touch, ls, cat, rm - they won't work.

    Args:
        command (str): Windows cmd command
        base_dir (str, optional): Directory to run command in. Defaults to ~/Desktop

    Returns:
        str: Command output or error message
    """
    cwd = base_dir if base_dir else os.path.expanduser("~/Desktop")
    try:
        result = subprocess.run(
            command, shell=True, cwd=cwd, capture_output=True, text=True
        )
        return result.stdout or result.stderr
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    mcp.run(transport="stdio")
