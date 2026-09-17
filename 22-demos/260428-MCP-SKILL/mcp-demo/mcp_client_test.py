# file: mcp_client_test.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_server():
    # Configure server connection
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize connection
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
            
            # Call a tool
            result = await session.call_tool("get_weather", arguments={"city": "Berlin"})
            print(f"\nWeather result: {result.content[0].text}")
            
            # Call another tool
            result = await session.call_tool("calculate", arguments={"expression": "15 * 3 + 7"})
            print(f"Calculation result: {result.content[0].text}")
            
            # Read a resource
            resource = await session.read_resource("config://app")
            print(f"\nConfig resource: {resource.contents[0].text}")

if __name__ == "__main__":
    asyncio.run(test_mcp_server())

