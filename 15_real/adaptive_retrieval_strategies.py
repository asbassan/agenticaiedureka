# pip install openai-agents tavily-python python-dotenv

import os
import sys
from agents import Agent, Runner, function_tool
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, os.path.dirname(__file__))
from support_ticketing_agent_hitl import retrieve  # reuse the same FAQ vector search

MODEL = "gpt-4o-mini"
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# =====================================================================
# One agent, several retrieval strategies - which one gets used on any
# given request is the AGENT's own decision (via which tool(s) it picks
# and how many times), not a branch written in this file. This covers
# 5 of the 6 strategies from the slide with real tools (skips a literal
# SQL-database example - the same "pick the right source" idea is
# already shown by web vs. knowledge-base vs. calculator):
#   - Skip retrieval    -> answers with no tool call at all
#   - Single retrieval  -> one search_knowledge_base call
#   - Iterative/reformulated retrieval -> searches again, reworded,
#     when the first search wasn't good enough
#   - Multi-source retrieval -> combines search_knowledge_base + search_web
#   - Tool-based retrieval   -> picks calculate instead of a search tool
# Each test case below prints the actual tool calls made, so the
# strategy is verified from the real trace, not just claimed.
# =====================================================================


@function_tool
def search_knowledge_base(query: str) -> str:
    """Search the internal FAQ knowledge base for a routine support question."""
    faqs, similarity = retrieve(query)
    if similarity < 0.3:
        return "No closely related FAQ entry found - try a different query, or a different tool."
    return "\n".join(f"- Q: {f['question']} / A: {f['answer']}" for f in faqs)


@function_tool
def search_web(query: str) -> str:
    """Search the live web - use for anything the internal FAQ can't know, e.g. current status/news."""
    response = tavily_client.search(query=query, max_results=3)
    results = response.get("results", [])
    if not results:
        return f"No web results found for '{query}'."
    return "\n".join(f"- {r['title']} ({r['url']}): {r['content'][:200]}" for r in results)


@function_tool
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression, e.g. '12.99 * 8'."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Invalid expression: {e}"


retrieval_agent = Agent(
    name="AdaptiveRetrievalAssistant",
    model=MODEL,
    tools=[search_knowledge_base, search_web, calculate],
    instructions="""
You are a support assistant that adapts its retrieval strategy per request instead of
always following the same steps:
- Skip retrieval entirely ONLY for general-knowledge questions unrelated to this
  product/company. Never skip retrieval for a question about this app's own behavior,
  features, or policies, even if you think you already know the answer.
- Use ONE search_knowledge_base call for a simple, single-topic support question.
- If a request has multiple parts, or your first search doesn't return a good match,
  search again - reword the query if the first phrasing didn't match well.
- If the request needs both company policy AND live/external information, call both
  search_knowledge_base and search_web and combine what they return.
- If the request is a calculation, not a lookup, call calculate instead of searching.
Only call the tools you actually need - don't search when you don't have to.
""",
)


def run_and_trace(label: str, query: str):
    print("=" * 70)
    print(f"[{label}] {query}\n")

    result = Runner.run_sync(retrieval_agent, query)

    calls = [
        f"{item.raw_item.name}({item.raw_item.arguments})"
        for item in result.new_items
        if type(item).__name__ == "ToolCallItem"
    ]
    print(f"Tool calls: {calls or '(none - answered directly)'}")
    print(f"Answer: {result.final_output}\n")


if __name__ == "__main__":
    run_and_trace(
        "Skip retrieval",
        "What does the acronym 'FAQ' stand for?",
    )
    run_and_trace(
        "Single retrieval",
        "How do I reset my password?",
    )
    run_and_trace(
        "Iterative / reformulated retrieval",
        "I need help with two things: how do I clear the app's cache, and separately, how do I cancel my subscription?",
    )
    run_and_trace(
        "Multi-source retrieval",
        "How do I reset my password, and is there a known login outage happening right now?",
    )
    run_and_trace(
        "Tool-based retrieval",
        "My subscription costs $12.99 a month - how much will I pay over 8 months?",
    )
