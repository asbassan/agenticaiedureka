import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_tavily import TavilySearch
from langchain_experimental.tools import PythonREPLTool

load_dotenv(override=True)

# Tool output can contain characters outside Windows' default console
# codepage (cp1252) - reconfigure stdout to UTF-8 so printing doesn't crash.
sys.stdout.reconfigure(encoding="utf-8")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -----------------------------
# Executor: the same ReAct agent + tools as
# react_agent_web_search_python_repl.py. Plan-and-Execute doesn't replace
# ReAct - it wraps it. Each planned step below still gets carried out by
# a small ReAct loop that can call these tools as needed.
# -----------------------------
tools = [
    TavilySearch(max_results=3),
    PythonREPLTool(),
]

REACT_PROMPT = PromptTemplate.from_template("""\
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

agent = create_react_agent(llm, tools, REACT_PROMPT)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=6)

# =====================================================================
# PLAN-AND-EXECUTE
# ReAct decides what to do ONE step at a time, re-thinking after every
# observation, with no commitment to what comes next. Plan-and-Execute
# instead commits to a full ordered list of steps UP FRONT, then works
# through that list - closer to writing a to-do list before starting,
# rather than deciding the next move only after seeing the last result.
# This trades ReAct's ability to course-correct mid-task for a plan
# that's visible and auditable before any tool ever gets called.
# =====================================================================

def make_plan(question: str) -> list[str]:
    prompt = (
        "Break the following question down into a short ordered list of concrete "
        "steps needed to answer it. Some steps may need a web search, some may "
        "need a calculation. Return ONLY a JSON array of step strings, nothing else "
        "- no markdown code fences, no explanation.\n\n"
        f"Question: {question}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip().strip("`")
    if text.lower().startswith("json"):
        text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [text]

def run_plan_and_execute(question: str) -> str:
    plan = make_plan(question)

    print("=" * 70)
    print("PLAN")
    print("=" * 70)
    for i, step in enumerate(plan, start=1):
        print(f"{i}. {step}")

    findings = []
    for i, step in enumerate(plan, start=1):
        print(f"\n{'=' * 70}")
        print(f"EXECUTING STEP {i}: {step}")
        print("=" * 70)

        step_input = (
            f"Overall question: {question}\n"
            f"Full plan: {plan}\n"
            f"Findings so far: {findings}\n"
            f"Your ONLY job right now is this single step: {step}\n"
            "Do not attempt any other step."
        )
        result = executor.invoke({"input": step_input})
        findings.append({"step": step, "result": result["output"]})

    print(f"\n{'=' * 70}")
    print("SYNTHESIZING FINAL ANSWER FROM ALL FINDINGS")
    print("=" * 70)

    synthesis_prompt = (
        f"Question: {question}\n"
        f"Findings from executing each planned step:\n{json.dumps(findings, indent=2)}\n\n"
        "Using ONLY these findings, give the final answer."
    )
    final = llm.invoke([HumanMessage(content=synthesis_prompt)])
    return final.content

if __name__ == "__main__":
    question = (
        "Find Japan's current total population and its number of prefectures, "
        "calculate the average population per prefecture, and state whether that "
        "average is higher or lower than 2 million."
    )

    answer = run_plan_and_execute(question)

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(answer)
