from fastmcp import FastMCP
import os
import json
import aiosqlite
import tempfile
import asyncio
import aiofiles

mcp = FastMCP("ExpenseTracker")

DB_PATH = os.path.join(tempfile.gettempdir(), "expenses.db")
CATEGORIES_PATH = os.path.join(os.path.dirname(__file__), "categories.json")


async def init_db():
    """Initialize the expenses database with required table."""
    async with aiosqlite.connect(DB_PATH) as c:
        await c.execute(
            """
            CREATE TABLE IF NOT EXISTS expenses(
                id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, 
                date TEXT NOT NULL,
                amount REAL NOT NULL,
                category TEXT NOT NULL,
                subcategory TEXT DEFAULT '',
                note TEXT DEFAULT ''
            )
        """
        )
        await c.commit()


@mcp.tool
async def add_expense(
    date: str, amount: float, category: str, subcategory: str = "", note: str = ""
):
    """Add a new expense to the database."""
    await init_db()
    async with aiosqlite.connect(DB_PATH) as c:
        cur = await c.execute(
            f"INSERT INTO expenses (date, amount, category, subcategory, note) VALUES (?, ?, ?, ?, ?)",
            (date, amount, category, subcategory, note),
        )
        await c.commit()
        return {"status": "ok", "id": cur.lastrowid}


@mcp.tool
async def list_expenses(start_date: str = "", end_date: str = "", category: str = "", subcategory: str = "", note: str = ""):
    """List expenses, optionally filtered by date range, category, subcategory, and note."""
    await init_db()
    async with aiosqlite.connect(DB_PATH) as c:
        query = "SELECT id, date, amount, category, subcategory, note FROM expenses"
        params = []
        conditions = []
        
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)
        if category:
            conditions.append("category = ?")
            params.append(category)
        if subcategory:
            conditions.append("subcategory = ?")
            params.append(subcategory)
        if note:
            conditions.append("note LIKE ?")
            params.append(f"%{note}%")
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY id ASC"
        
        cur = await c.execute(query, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in await cur.fetchall()]


@mcp.tool
async def delete_expense(expense_id: int):
    """Delete an expense by ID."""
    await init_db()
    async with aiosqlite.connect(DB_PATH) as c:
        cur = await c.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
        await c.commit()
        return {"status": "ok", "deleted": cur.rowcount > 0}


@mcp.tool
async def summarize_expenses(
    start_date: str = "", end_date: str = "", category: str = ""
):
    """Summarize expenses by category with optional date and category filters."""
    await init_db()
    async with aiosqlite.connect(DB_PATH) as c:
        query = "SELECT category, SUM(amount) as total, COUNT(*) as count FROM expenses"
        params = []
        conditions = []

        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)
        if category:
            conditions.append("category = ?")
            params.append(category)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " GROUP BY category ORDER BY total DESC"

        cur = await c.execute(query, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in await cur.fetchall()]


@mcp.resource("expense://categories", mime_type="application/json")
async def categories():
    await init_db()
    async with aiofiles.open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        return await f.read()


@mcp.resource("info://server")
def server_info() -> str:
    """get information about the server"""
    info = {
        "name": "Expense Tracker",
        "version": "1.0.0",
        "description": "Basic Expense Tracker Tool",
        "tools": [
            "add_expense",
            "list_expenses",
            "delete_expense",
            "summarize_expenses",
        ],
        "author": "Anekant",
    }
    return json.dumps(info, indent=2)


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
