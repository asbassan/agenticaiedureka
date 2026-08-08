# pip install openai-agents chromadb tavily-python python-dotenv

import os
import sys
import csv
import chromadb
from agents import Agent, Runner, function_tool
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv(override=True)
sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.join(os.path.dirname(__file__), "market_research_data")
MODEL = "gpt-4o-mini"
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# =====================================================================
# Market research assistant built on RAG over an internal knowledge
# base (competitor profiles, industry trend report, our own positioning
# brief - see market_research_data/), PLUS a structured pricing table
# and live web search for anything current the static docs can't know.
#
# The competitors are REAL (Salesforce, HubSpot, Zoho CRM, Pipedrive),
# researched via live web search/fetch and cited per-section in the
# markdown docs - not invented. "NimbusCRM" (our own product) is the
# one fictional entity, clearly labeled as a hypothetical for this
# exercise. Pricing was compiled in August 2026 from vendor pages where
# fetchable, and from aggregated third-party sources where the vendor's
# own site blocked direct fetching (Salesforce, Pipedrive) - real
# pricing changes often, so treat these as directionally accurate for
# a research exercise, not a live quote to act on without verifying.
# =====================================================================


def load_markdown_chunks(filename: str) -> list[dict]:
    """Split a markdown file into chunks by its '## ' section headings."""
    text = open(os.path.join(DATA_DIR, filename), encoding="utf-8").read()
    raw_sections = text.split("\n## ")
    sections = [raw_sections[0]] + [f"## {s}" for s in raw_sections[1:]]
    return [{"id": f"{filename}-{i}", "text": s.strip()} for i, s in enumerate(sections)]


def build_knowledge_base():
    collection = chromadb.Client().create_collection("market_research")
    chunks = [
        chunk
        for filename in ["competitor_profiles.md", "industry_trends_2026.md", "our_product_positioning.md"]
        for chunk in load_markdown_chunks(filename)
    ]
    collection.add(ids=[c["id"] for c in chunks], documents=[c["text"] for c in chunks])
    return collection


KNOWLEDGE_BASE = build_knowledge_base()

with open(os.path.join(DATA_DIR, "pricing_benchmarks.csv"), newline="", encoding="utf-8") as f:
    PRICING_BENCHMARKS = list(csv.DictReader(f))


@function_tool
def search_market_knowledge_base(query: str) -> str:
    """Search our internal market research docs: competitor profiles, industry trends, and our own positioning."""
    results = KNOWLEDGE_BASE.query(query_texts=[query], n_results=4)
    return "\n\n---\n\n".join(results["documents"][0])


@function_tool
def get_pricing_benchmark(competitor: str = "") -> str:
    """Look up structured pricing rows from the pricing benchmark table, optionally filtered by competitor name."""
    rows = [r for r in PRICING_BENCHMARKS if competitor.lower() in r["competitor"].lower()] if competitor else PRICING_BENCHMARKS
    if not rows:
        return f"No pricing data found for '{competitor}'."
    return "\n".join(
        f"- {r['competitor']} {r['tier']}: ${r['price_per_user_per_month_usd']}/user/month, "
        f"targets {r['target_segment']}, AI features included: {r['ai_features_included']}"
        for r in rows
    )


@function_tool
def search_web(query: str) -> str:
    """Search the live web for current market news, funding, or pricing changes not in our internal docs."""
    response = tavily_client.search(query=query, max_results=5)
    results = response.get("results", [])
    if not results:
        return f"No web results found for '{query}'."
    # 200 chars was too short to carry real substance, and low-quality pages (nav
    # boilerplate, error text) with an empty snippet were diluting genuinely useful
    # hits - drop those and give the model more to actually work with.
    usable = [r for r in results if len(r.get("content", "").strip()) > 40]
    if not usable:
        return f"Web results for '{query}' were too low-quality to use - try a more specific query."
    return "\n\n".join(f"- {r['title']} ({r['url']})\n  {r['content'][:600]}" for r in usable)


market_research_agent = Agent(
    name="MarketResearchAssistant",
    model=MODEL,
    tools=[search_market_knowledge_base, get_pricing_benchmark, search_web],
    instructions="""
You are a market research analyst assistant for NimbusCRM, a CRM SaaS company.
- Use search_market_knowledge_base for competitor positioning, strengths/weaknesses,
  industry trends, and our own product's positioning.
- Use get_pricing_benchmark for exact price comparisons across competitors. When a
  question compares "our" pricing against a competitor's, always also look up
  NimbusCRM's own pricing (get_pricing_benchmark("NimbusCRM")) - it's the same table.
- Use search_web for anything the internal docs can't know: news/pricing changes more
  recent than August 2026, or verifying a specific figure before it's used in a
  high-stakes recommendation. Before deciding a web result isn't new information, first
  check search_market_knowledge_base so you actually know what the internal research
  already says - don't guess at that baseline.
- Add disambiguating context to web searches (e.g. "CRM software company") since some
  competitor names are generic words, and if results clearly describe an unrelated
  company or product, say so explicitly instead of presenting it as relevant.
- If search_web returns real information, report the specifics (what changed, when, by
  how much) - never fall back to a vague "no significant changes found" if the results
  actually contained something concrete. Only report "nothing new" if the results
  genuinely had nothing usable.
- Structure your final answer as a short Market Research Brief: a 1-2 sentence
  summary, then bulleted key findings, then a clear recommendation if the question
  asks for one. Note which findings came from internal research vs. the live web.
""",
)


def run_query(label: str, query: str):
    print("=" * 70)
    print(f"[{label}] {query}\n")

    result = Runner.run_sync(market_research_agent, query)

    calls = [item.raw_item.name for item in result.new_items if type(item).__name__ == "ToolCallItem"]
    print(f"Tools used: {calls}\n")
    print(result.final_output)
    print()


if __name__ == "__main__":
    run_query(
        "Internal RAG + structured pricing",
        "How does our pricing compare to HubSpot and Pipedrive, and what should our "
        "positioning be for small-business customers?",
    )
    run_query(
        "Falls back to live web search",
        "What's changed about Salesforce's Agentforce pricing in 2026?",
    )
