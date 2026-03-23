"""Entry point: python -m vector_os_nano.mcp

Runs the MCP server with stdio transport so Claude Code can communicate
with the Vector OS Nano simulated robot.

Examples:
    python -m vector_os_nano.mcp                 # headless sim (default)
    python -m vector_os_nano.mcp --sim            # sim with viewer
    python -m vector_os_nano.mcp --sim-headless   # headless sim (explicit)
"""

import asyncio

from vector_os_nano.mcp.server import main

asyncio.run(main())
