import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["c:/code/agenticai/6_mcp/6_0_database_mcp_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Add a couple of users
            result = await session.call_tool("add_user", {
                "name": "Alice",
                "email": "alice@example.com",
            })
            print(result.content[0].text)

            result = await session.call_tool("add_user", {
                "name": "Bob",
                "email": "bob@example.com",
            })
            print(result.content[0].text)

            # List all users
            result = await session.call_tool("list_users", {})
            print(result.content[0].text)

if __name__ == "__main__":
    asyncio.run(main())
