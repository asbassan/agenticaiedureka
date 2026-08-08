import asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

load_dotenv(override=True)

llm = OpenAI()


async def draft_with_llm(session: ClientSession, prompt_name: str, arguments: dict) -> str:
    # Ask the SERVER to render its prompt template with these arguments -
    # the client never sees or maintains the wording itself, just the
    # rendered result.
    rendered = await session.get_prompt(prompt_name, arguments)
    messages = [{"role": m.role, "content": m.content.text} for m in rendered.messages]

    response = llm.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content


async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["c:/code/agenticai/6_mcp/6_11_prompt_template_mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover which prompt templates this server offers, and what
            # arguments each one needs - a client shouldn't have to hardcode
            # that knowledge in advance.
            available = await session.list_prompts()
            print("Available prompt templates:")
            for p in available.prompts:
                arg_names = [a.name for a in (p.arguments or [])]
                print(f"  - {p.name}({', '.join(arg_names)}): {p.description}")

            print("\n=== Refund confirmation ===")
            refund_email = await draft_with_llm(
                session,
                "draft_refund_reply",
                {"customer_name": "Priya Nair", "order_id": "ORD-48213", "reason": "item arrived damaged"},
            )
            print(refund_email)

            print("\n=== Shipping delay apology ===")
            delay_email = await draft_with_llm(
                session,
                "draft_delay_apology",
                {"customer_name": "Rahul Mehta", "order_id": "ORD-59021", "days_late": "4"},
            )
            print(delay_email)


if __name__ == "__main__":
    asyncio.run(main())
