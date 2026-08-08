from mcp.server.fastmcp import FastMCP

mcp = FastMCP("SupportPrompts")

# =====================================================================
# MCP has three kinds of building blocks: tools (functions the LLM can
# call), resources (data the client can read), and prompts (reusable,
# parameterized message templates the SERVER owns and the CLIENT fills
# in). This demo is about that third kind - a support team centrally
# maintains its "how do we phrase a refund confirmation" and "how do we
# apologize for a shipping delay" wording as versioned prompt templates
# on the MCP server, so every client (a Flask app, a Slack bot, a CLI)
# renders the exact same reviewed wording instead of each one hardcoding
# its own copy of the phrasing.
# =====================================================================

@mcp.prompt()
def draft_refund_reply(customer_name: str, order_id: str, reason: str) -> str:
    """Prompt template for drafting a refund confirmation email"""
    return f"""Write a short, professional refund confirmation email to {customer_name} \
regarding order {order_id}. Reason for refund: {reason}. Confirm the refund will be processed \
within 5-7 business days to the original payment method. Keep it under 100 words and sign off as \
"Customer Support Team"."""


@mcp.prompt()
def draft_delay_apology(customer_name: str, order_id: str, days_late: int) -> str:
    """Prompt template for drafting a shipping delay apology email"""
    return f"""Write a short, empathetic email to {customer_name} apologizing that order {order_id} \
is running {days_late} days behind its original delivery estimate. Offer a 10% discount code \
SORRY10 on their next order as a goodwill gesture. Keep it under 100 words and sign off as \
"Customer Support Team"."""


if __name__ == "__main__":
    mcp.run()
