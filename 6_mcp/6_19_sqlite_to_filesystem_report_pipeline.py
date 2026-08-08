# pip install mcp-server-sqlite (already in requirements.txt)
# Needs Node.js/npm (npx) for the filesystem server - no extra pip install for that one.
#
# A small report pipeline chaining TWO official MCP servers in one script,
# no LLM involved - this is plain orchestration, not an agent:
#   1. Anthropic's official mcp-server-sqlite (also used in 6_13) holds the
#      "source data" - a small sales table.
#   2. A read_query aggregates that data into a report.
#   3. The official @modelcontextprotocol/server-filesystem (also used in
#      6_14) writes the formatted report out as a real file, sandboxed to
#      a reports/ directory - the two servers never talk to each other
#      directly, this script is the thing that reads from one and writes
#      to the other.
import asyncio
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "sales_report_data.db")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

SALES_DATA = [
    ("Wireless Mouse", "Electronics", 420, 8399.80),
    ("Mechanical Keyboard", "Electronics", 180, 16199.10),
    ("Standing Desk", "Furniture", 65, 22749.35),
    ("Office Chair", "Furniture", 140, 27999.60),
    ("Notebook Pack", "Stationery", 950, 4749.50),
    ("Desk Lamp", "Electronics", 310, 6199.70),
]


async def main():
    sqlite_params = StdioServerParameters(
        command="mcp-server-sqlite",
        args=["--db-path", DB_PATH],
    )
    filesystem_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", REPORTS_DIR],
    )

    async with stdio_client(sqlite_params) as (sql_read, sql_write):
        async with ClientSession(sql_read, sql_write) as sql:
            await sql.initialize()

            print("=== [SQLite] create_table + seed sales data ===")
            await sql.call_tool("create_table", {"query": """
                CREATE TABLE IF NOT EXISTS sales (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product TEXT NOT NULL,
                    category TEXT NOT NULL,
                    units_sold INTEGER NOT NULL,
                    revenue REAL NOT NULL
                )
            """})
            await sql.call_tool("write_query", {"query": "DELETE FROM sales"})
            for product, category, units, revenue in SALES_DATA:
                await sql.call_tool("write_query", {
                    "query": f"INSERT INTO sales (product, category, units_sold, revenue) "
                             f"VALUES ('{product}', '{category}', {units}, {revenue})"
                })
            print(f"Seeded {len(SALES_DATA)} rows")

            print("\n=== [SQLite] read_query: revenue by category ===")
            by_category = await sql.call_tool("read_query", {"query": """
                SELECT category, SUM(units_sold) AS total_units, ROUND(SUM(revenue), 2) AS total_revenue
                FROM sales GROUP BY category ORDER BY total_revenue DESC
            """})
            print(by_category.content[0].text)

            print("\n=== [SQLite] read_query: top product by revenue ===")
            top_product = await sql.call_tool("read_query", {"query": """
                SELECT product, category, units_sold, revenue
                FROM sales ORDER BY revenue DESC LIMIT 1
            """})
            print(top_product.content[0].text)

    # Build the report text from the two query results above.
    report = f"""# Sales Report

## Revenue by Category
{by_category.content[0].text}

## Top Product by Revenue
{top_product.content[0].text}
"""

    async with stdio_client(filesystem_params) as (fs_read, fs_write):
        async with ClientSession(fs_read, fs_write) as fs:
            await fs.initialize()

            print("\n=== [Filesystem] write_file: sales_report.md ===")
            result = await fs.call_tool("write_file", {"path": "sales_report.md", "content": report})
            print(result.content[0].text)

            print("\n=== [Filesystem] read_text_file: confirm it was written ===")
            result = await fs.call_tool("read_text_file", {"path": "sales_report.md"})
            print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
