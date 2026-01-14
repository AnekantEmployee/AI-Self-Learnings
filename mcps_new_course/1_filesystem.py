import os
import subprocess
from fastmcp import FastMCP

mcp = FastMCP("terminal_filesystem")


@mcp.tool
def run_unix_command(command: str, base_dir: str = None) -> str:
    """Run Unix-style commands using Git Bash with proper PATH"""
    cwd = base_dir if base_dir else os.path.expanduser("~/Desktop")

    git_bash_path = r"C:\Users\anekant.jain\AppData\Local\Programs\Git\usr\bin\bash.exe"
    git_bin_dir = r"C:\Users\anekant.jain\AppData\Local\Programs\Git\usr\bin"

    if not os.path.exists(git_bash_path):
        return "Error: Git Bash not found"

    try:
        # Set up environment with proper PATH for Unix commands
        env = os.environ.copy()
        env["PATH"] = f"{git_bin_dir};{env.get('PATH', '')}"

        result = subprocess.run(
            [git_bash_path, "-c", command],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
            shell=False,
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return f"Error (exit {result.returncode}): {error_msg}"

        return result.stdout if result.stdout else "Command executed successfully"

    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after 10 seconds"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool
def run_shell_command(command: str, base_dir: str = None) -> str:
    """Run shell commands using (supports commands: mkdir, type, del etc.)

    Args:
        command (str): Shell command
        base_dir (str, optional): Base directory to run command in. Defaults to ~/Desktop

    Returns:
        str: command output or error message
    """
    cwd = base_dir if base_dir else os.path.expanduser("~/Desktop")
    try:
        result = subprocess.run(
            command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=30
        )
        return result.stdout or result.stderr
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    mcp.run(transport="stdio")
