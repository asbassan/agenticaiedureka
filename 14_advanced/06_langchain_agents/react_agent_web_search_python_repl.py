import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
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
# Tools
# A ReAct agent doesn't have any fixed capability of its own beyond
# reasoning in a loop - everything it can actually DO comes from these
# tools. Two very different kinds here on purpose: one that reaches out
# to the real world (search), one that computes (code execution).
# -----------------------------
tools = [
    TavilySearch(max_results=3),
    # REPL is a code execution tool, R = Read, E = Evaluate, P = Print, L = Loop
    # Now the agent can execute Python code when it needs to
    PythonREPLTool(), 
]

# -----------------------------
# ReAct prompt, written out explicitly (rather than pulled from LangChain
# Hub via hub.pull("hwchase17/react")) so the actual reasoning loop the
# agent is instructed to follow is visible here, not hidden behind a
# hub lookup. {tools} and {tool_names} get filled in automatically by
# create_react_agent from the tools list above; {agent_scratchpad} is
# where the agent's own prior Thought/Action/Observation steps get
# replayed back to it on each iteration.
# -----------------------------
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

# verbose=True prints every Thought/Action/Observation as it happens -
# this IS the point of the demo: watching the agent decide which tool to
# reach for, and when it's done reasoning and ready to answer.
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=6,
)

if __name__ == "__main__":
    # Deliberately needs BOTH tools in sequence: a live fact from search,
    # then arithmetic on that fact that the LLM shouldn't just guess at.
    question = (
        "Search the web for Japan's current population. Then use Python to divide "
        "that number by 47 (Japan's number of prefectures) and tell me the average "
        "population per prefecture, rounded to the nearest whole number."
    )

    result = agent_executor.invoke({"input": question})

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(result["output"])
