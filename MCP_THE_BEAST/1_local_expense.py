from fastmcp import FastMCP
import os
import sqlite3

mcp = FastMCP("ExpenseTracker")

DB_PATH = os.path.join(os.path.dirname(__file__), "expenses.db")
CATEGORIES_PATH = os.path.join(os.path.dirname(__file__), "categories.json")


def init_db():
    """Initialize the expenses database with required table."""
    with sqlite3.connect(DB_PATH) as c:
        c.execute(
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


init_db()


@mcp.tool
def add_expense(date, amount, category, subcategory="", note=""):
    """Add a new expense to the database."""
    with sqlite3.connect(DB_PATH) as c:
        cur = c.execute(
            f"INSERT INTO expenses (date, amount, category, subcategory, note) VALUES (?, ?, ?, ?, ?)",
            (date, amount, category, subcategory, note),
        )
        return {"status": "ok", "id": cur.lastrowid}


@mcp.tool
def list_expenses(start_date="", end_date=""):
    """List expenses, optionally filtered by date range."""
    with sqlite3.connect(DB_PATH) as c:
        if start_date and end_date:
            cur = c.execute(
                """SELECT id, date, amount, category, subcategory, note 
                FROM expenses 
                WHERE date >= ? AND date <= ?
                ORDER BY id ASC""",
                (start_date, end_date),
            )
        elif start_date:
            cur = c.execute(
                """SELECT id, date, amount, category, subcategory, note 
                FROM expenses 
                WHERE date >= ?
                ORDER BY id ASC""",
                (start_date,),
            )
        elif end_date:
            cur = c.execute(
                """SELECT id, date, amount, category, subcategory, note 
                FROM expenses 
                WHERE date <= ?
                ORDER BY id ASC""",
                (end_date,),
            )
        else:
            cur = c.execute(
                """SELECT id, date, amount, category, subcategory, note 
                FROM expenses 
                ORDER BY id ASC"""
            )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


@mcp.tool
def delete_expense(expense_id):
    """Delete an expense by ID."""
    with sqlite3.connect(DB_PATH) as c:
        cur = c.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
        return {"status": "ok", "deleted": cur.rowcount > 0}


@mcp.tool
def summarize_expenses(start_date="", end_date="", category=""):
    """Summarize expenses by category with optional date and category filters."""
    with sqlite3.connect(DB_PATH) as c:
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

        cur = c.execute(query, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


@mcp.resource("expense://categories", mime_type="application/json")
def categories():
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    mcp.run()
