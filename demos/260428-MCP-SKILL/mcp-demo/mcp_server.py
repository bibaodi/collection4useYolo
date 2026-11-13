# file: mcp_server.py
from mcp.server.fastmcp import FastMCP

# Create MCP server instance
mcp = FastMCP("my_tools_server")

# Define a TOOL - agent can call this
@mcp.tool()
def get_weather(city: str) -> str:
    """Get weather for a city"""
    # Simulated weather data
    weather_db = {
        "berlin": {"temp": 15, "condition": "Cloudy"},
        "paris": {"temp": 18, "condition": "Sunny"},
        "tokyo": {"temp": 22, "condition": "Rain"},
    }
    data = weather_db.get(city.lower(), {"temp": "N/A", "condition": "Unknown"})
    return f"Weather in {city}: {data['temp']}°C, {data['condition']}"

# Define another TOOL
@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a math expression safely"""
    try:
        # Only allow safe operations
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters"
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

# Define a RESOURCE - agent can read this context
@mcp.resource("config://app")
def get_config() -> str:
    """Application configuration"""
    return """{
        "version": "1.0",
        "default_city": "Berlin",
        "units": "metric"
    }"""

# Define a PROMPT TEMPLATE
@mcp.prompt()
def weather_prompt(city: str) -> str:
    """Generate a prompt for weather analysis"""
    return f"Analyze the weather in {city} and suggest activities."

# Start the server (stdio transport - default for local use)
if __name__ == "__main__":
    mcp.run()

