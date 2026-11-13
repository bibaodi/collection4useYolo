# stdio transport (for Claude Desktop, etc.)
python mcp_server.py

# OR with SSE transport (for remote access)
python -c "
from mcp_server import mcp
mcp.run(transport='sse', host='127.0.0.1', port=8000)
"

