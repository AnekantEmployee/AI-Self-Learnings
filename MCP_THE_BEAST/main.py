import random
from fastmcp import FastMCP

mcp = FastMCP(name="Demo Server")


@mcp.tool
def roll_dice(n_dice: int = 1) -> list[int]:
    """Roll n_dice 6 dices side and returns hte results"""
    return [random.randint(1, 6) for _ in range(n_dice)]


@mcp.tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers are float and returns the outupt"""
    return a + b


if __name__ == "__main__":
    mcp.run()


# uv init
# uv add fastmcp
# uv run fastmcp dev main.py --test
# uv run fastmcp run main.py --run
# uv run fastmcp install claude-desktop main.py --claude installation