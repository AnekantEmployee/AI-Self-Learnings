import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from .async_utils import run_async

# PATCH: Add is_alive method to aiosqlite.Connection
if not hasattr(aiosqlite.Connection, "is_alive"):

    def is_alive(self):
        """Check if the connection is alive"""
        return self._conn is not None

    aiosqlite.Connection.is_alive = is_alive


async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot_mcp.db")
    return AsyncSqliteSaver(conn)


def get_checkpointer():
    return run_async(_init_checkpointer())


async def _alist_threads(checkpointer):
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads(checkpointer):
    return run_async(_alist_threads(checkpointer))