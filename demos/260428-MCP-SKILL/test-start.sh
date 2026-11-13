# Create and run all examples
#cd ~
mkdir -p agent-demo && cd agent-demo

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install mcp openai

# Create all files from above, then test:
python skills_simple.py
python skills_class.py
python comparison_demo.py

# Test MCP server
python mcp_server.py &            # Run in background
python mcp_client_test.py         # Test it
kill %1                           # Stop server

