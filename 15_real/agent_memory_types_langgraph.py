import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import json
from typing import Annotated, TypedDict
import chromadb
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(override=True)
sys.stdout.reconfigure(encoding="utf-8")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# =====================================================================
# Four kinds of agent memory, each backed by real storage - not just a
# printed label - matching the short-term / semantic / episodic /
# procedural split from LangGraph's own memory docs (and the CoALA
# paper they cite), not something invented for this file:
#
#   SHORT-TERM (working) memory - the running message list for ONE
#   conversation thread, kept by LangGraph's checkpointer. Gone once a
#   new thread_id starts, unless something from it got written down.
#
#   LONG-TERM SEMANTIC memory - durable FACTS about the user (name,
#   preferences, constraints), stored per user_id in a JSON file and
#   loaded fresh at the start of every session, regardless of thread_id.
#
#   LONG-TERM EPISODIC memory - durable records of specific past
#   situations and how they were resolved, stored in a ChromaDB
#   collection and retrieved by semantic similarity to the CURRENT
#   message, so a new-but-similar situation can recall precedent.
#
#   LONG-TERM PROCEDURAL memory - durable rules about HOW the agent
#   should behave, taught by explicit user feedback, stored per user_id
#   and folded into every future session's system prompt until changed.
#
# The proof this isn't a dummy example: the demo below runs a SECOND,
# brand-new thread_id (a new session) and shows the agent already
# knowing things it was only told in the FIRST thread - that only
# works if the long-term stores are real and actually being read.
#
# Scenario: an insurance claims assistant. Filing a claim genuinely
# spans multiple sessions (file -> gather documents -> follow up) and
# genuinely needs all four memory types: the policy/claim facts already
# on file (semantic), what proof a SIMILAR past claim required (episodic
# precedent), an accessibility/communication preference the claimant
# stated once (procedural), and mid-conversation recall of what was
# just said (short-term).
# =====================================================================

MEMORY_DIR = os.path.join(os.path.dirname(__file__), "agent_memory")
os.makedirs(MEMORY_DIR, exist_ok=True)
SEMANTIC_PATH = os.path.join(MEMORY_DIR, "semantic_memory.json")
PROCEDURAL_PATH = os.path.join(MEMORY_DIR, "procedural_memory.json")

episodic_collection = chromadb.PersistentClient(
    path=os.path.join(MEMORY_DIR, "episodic_db")
).get_or_create_collection("episodes")


# ---------------------------------------------------------------------
# Long-term store helpers - plain JSON files + a persistent vector DB,
# all keyed by user_id, all read fresh at the start of every session.
# ---------------------------------------------------------------------
def _load_json_store(path: str) -> dict:
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


def _save_json_store(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_semantic_facts(user_id: str) -> list[str]:
    return _load_json_store(SEMANTIC_PATH).get(user_id, [])


def save_semantic_facts(user_id: str, new_facts: list[str]):
    store = _load_json_store(SEMANTIC_PATH)
    existing = store.get(user_id, [])
    store[user_id] = existing + [f for f in new_facts if f not in existing]
    _save_json_store(SEMANTIC_PATH, store)


def load_procedural_rules(user_id: str) -> list[str]:
    return _load_json_store(PROCEDURAL_PATH).get(user_id, [])


def save_procedural_rule(user_id: str, rule: str):
    store = _load_json_store(PROCEDURAL_PATH)
    existing = store.get(user_id, [])
    if rule not in existing:
        store[user_id] = existing + [rule]
    _save_json_store(PROCEDURAL_PATH, store)


def retrieve_episodic_memories(user_id: str, current_message: str, top_k: int = 2) -> list[str]:
    if episodic_collection.count() == 0:
        return []
    results = episodic_collection.query(query_texts=[current_message], n_results=top_k, where={"user_id": user_id})
    return results["documents"][0] if results["documents"] else []


def save_episode(user_id: str, summary: str):
    episodic_collection.add(
        ids=[f"{user_id}-{episodic_collection.count()}"], documents=[summary], metadatas=[{"user_id": user_id}]
    )


# ---------------------------------------------------------------------
# Memory extraction: after each exchange, decide what (if anything) is
# actually worth persisting long-term. Most turns won't produce all
# three - a "what's the weather" aside shouldn't become a fact, rule,
# or episode, and the LLM is told that explicitly.
# ---------------------------------------------------------------------
class MemoryExtraction(BaseModel):
    new_semantic_facts: list[str]
    new_procedural_rule: str | None
    episode_summary: str | None


extractor_llm = llm.with_structured_output(MemoryExtraction)


class AgentState(TypedDict):
    user_id: str
    messages: Annotated[list, add_messages]
    semantic_facts: list[str]
    episodic_context: list[str]
    procedural_rules: list[str]


# ---- Node 1: pull all long-term memory for this user before responding ----
def load_long_term_memory(state: AgentState) -> AgentState:
    user_id = state["user_id"]
    last_message = state["messages"][-1].content

    semantic_facts = load_semantic_facts(user_id)
    procedural_rules = load_procedural_rules(user_id)
    episodic_context = retrieve_episodic_memories(user_id, last_message)

    print(f"  [loaded] semantic={semantic_facts or 'none'} | procedural={procedural_rules or 'none'} | "
          f"episodic={episodic_context or 'none'}")

    return {"semantic_facts": semantic_facts, "procedural_rules": procedural_rules, "episodic_context": episodic_context}


# ---- Node 2: respond, grounded in short-term (messages) + all long-term memory ----
def generate_response(state: AgentState) -> AgentState:
    parts = ["You are a helpful insurance claims assistant. Help policyholders understand what "
             "documents, proof, or verification steps their claim needs."]
    if state["procedural_rules"]:
        parts.append("Behavior rules you MUST follow:\n" + "\n".join(f"- {r}" for r in state["procedural_rules"]))
    if state["semantic_facts"]:
        parts.append("Known facts about this policyholder:\n" + "\n".join(f"- {f}" for f in state["semantic_facts"]))
    if state["episodic_context"]:
        parts.append("Similar past claims from this policyholder and what was required to resolve them:\n"
                      + "\n".join(f"- {e}" for e in state["episodic_context"]))

    response = llm.invoke([SystemMessage(content="\n\n".join(parts))] + state["messages"])
    return {"messages": [response]}


# ---- Node 3: decide what from this exchange is worth remembering long-term ----
def extract_and_save_memories(state: AgentState) -> AgentState:
    user_id = state["user_id"]
    user_turn, ai_turn = state["messages"][-2].content, state["messages"][-1].content

    extraction = extractor_llm.invoke([
        SystemMessage(content=(
            "From this exchange, extract: (1) new_semantic_facts - durable facts about "
            "the user worth remembering long-term (empty list if none), (2) "
            "new_procedural_rule - a new behavior rule ONLY if the user explicitly gave "
            "feedback about how the assistant should behave going forward (null "
            "otherwise), (3) episode_summary - a short summary of this exchange as a "
            "resolved episode ONLY if it's a concrete task/situation worth recalling if "
            "something similar comes up later (null for small talk or simple facts)."
        )),
        HumanMessage(content=f"User: {user_turn}\nAssistant: {ai_turn}"),
    ])

    if extraction.new_semantic_facts:
        save_semantic_facts(user_id, extraction.new_semantic_facts)
        print(f"  [semantic memory saved] {extraction.new_semantic_facts}")
    if extraction.new_procedural_rule:
        save_procedural_rule(user_id, extraction.new_procedural_rule)
        print(f"  [procedural memory saved] {extraction.new_procedural_rule!r}")
    if extraction.episode_summary:
        save_episode(user_id, extraction.episode_summary)
        print(f"  [episodic memory saved] {extraction.episode_summary!r}")

    return {}


# Graph: START -> load_long_term_memory -> generate_response -> extract_and_save_memories -> END
graph = StateGraph(AgentState)
graph.add_node("load_long_term_memory", load_long_term_memory)
graph.add_node("generate_response", generate_response)
graph.add_node("extract_and_save_memories", extract_and_save_memories)
graph.add_edge(START, "load_long_term_memory")
graph.add_edge("load_long_term_memory", "generate_response")
graph.add_edge("generate_response", "extract_and_save_memories")
graph.add_edge("extract_and_save_memories", END)

# The checkpointer is what makes messages behave as SHORT-TERM (thread-scoped)
# memory - a new thread_id starts with an empty message list no matter what.
app = graph.compile(checkpointer=MemorySaver())


def chat(user_id: str, thread_id: str, message: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"user_id": user_id, "messages": [HumanMessage(content=message)]}, config)
    return result["messages"][-1].content


if __name__ == "__main__":
    print("=" * 70)
    print("SESSION 1 (thread='session-1') - policyholder filing a claim for the first time")
    print("=" * 70)

    print("\nUser: Hi, my car got hit in a parking lot yesterday - minor bumper damage, no "
          "injuries. My policy number is AUTO-4471-B. What do I need to submit to file a claim?")
    print(f"Assistant: {chat('priya', 'session-1', 'Hi, my car got hit in a parking lot yesterday - minor bumper damage, no injuries. My policy number is AUTO-4471-B. What do I need to submit to file a claim?')}")

    print("\nUser: Sorry, what documents did you just say I need again?")
    print(f"Assistant: {chat('priya', 'session-1', 'Sorry, what documents did you just say I need again?')}")
    print("(^ answered from SHORT-TERM memory - same thread, no long-term lookup needed for this)")

    print("\nUser: One more thing - I have a hearing impairment, so please always send me "
          "written checklists instead of asking me to call a phone number.")
    print(f"Assistant: {chat('priya', 'session-1', 'One more thing - I have a hearing impairment, so please always send me written checklists instead of asking me to call a phone number.')}")

    print("\n" + "=" * 70)
    print("SESSION 2 (thread='session-2', NEW thread - policyholder returns weeks later for a different claim)")
    print("=" * 70)

    print("\nUser: Hi, I need to file another claim - my car window was smashed in a different "
          "parking lot. Do you have my policy info on file?")
    print(f"Assistant: {chat('priya', 'session-2', 'Hi, I need to file another claim - my car window was smashed in a different parking lot. Do you have my policy info on file?')}")
    print("(^ this is a NEW thread with an empty message list - correct policy-number recall here came from "
          "SEMANTIC memory, and a written checklist (not 'please call...') proves PROCEDURAL memory carried over)")

    print("\nUser: What kind of proof did I need to provide last time for something similar?")
    print(f"Assistant: {chat('priya', 'session-2', 'What kind of proof did I need to provide last time for something similar?')}")
    print("(^ should draw on EPISODIC memory of the earlier bumper-damage claim's document requirements)")
