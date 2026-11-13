# file: comparison_demo.py
"""
Side-by-side comparison: MCP Server vs Skills
"""

# ============== SKILLS APPROACH ==============
print("=" * 50)
print("SKILLS APPROACH")
print("=" * 50)

def greet_skill(name: str) -> str:
    return f"Hello, {name}!"

# Direct execution - no server needed
result = greet_skill(name="World")
print(f"Direct call result: {result}")

# Skills are just functions - simple, fast, same process
print("✓ No server process needed")
print("✓ Direct function calls")
print("✗ Not reusable by other clients")
print("✗ No standard protocol")

# ============== MCP APPROACH ==============
print("\n" + "=" * 50)
print("MCP APPROACH")
print("=" * 50)

# MCP requires:
# 1. A server process running
# 2. Client connects via protocol (stdio/SSE)
# 3. Tools discovered dynamically

print("""
To use the same 'greet' function via MCP:

1. Server (mcp_server_greet.py):
   from mcp.server.fastmcp import FastMCP
   mcp = FastMCP("greet_server")
   
   @mcp.tool()
   def greet(name: str) -> str:
       return f"Hello, {name}!"
   
   mcp.run()

2. Any MCP client (Claude, custom, etc.) can:
   - Discover 'greet' tool automatically
   - Call it via JSON-RPC protocol
   - Works across languages, processes

✓ Standard protocol
✓ Any MCP client can use it
✓ Separate process (isolation)
✗ More complex setup
✗ Overhead for simple use cases
""")

# ============== WHEN TO USE WHAT ==============
print("\n" + "=" * 50)
print("DECISION GUIDE")
print("=" * 50)
print("""
USE SKILLS WHEN:
  • Single agent, single script
  • Quick prototyping
  • Internal tools only
  • Simple function wrapping

USE MCP WHEN:
  • Multiple AI clients need same tools
  • Building reusable tool ecosystem
  • Need process isolation/security
  • Production deployments
  • Cross-language tool sharing
""")

