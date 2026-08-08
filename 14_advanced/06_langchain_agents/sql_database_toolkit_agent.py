import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

load_dotenv(override=True)

# Tool output can contain characters outside Windows' default console
# codepage (cp1252) - reconfigure stdout to UTF-8 so printing doesn't crash.
sys.stdout.reconfigure(encoding="utf-8")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -----------------------------
# SQLDatabase - reuses the sales.db already created by
# 14_advanced/03_llm_examples/create_sales_db.py (a "sales" table with
# product_name, sale_date, quantity, revenue).
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "03_llm_examples", "sales.db")
db = SQLDatabase.from_uri(f"sqlite:///{os.path.abspath(DB_PATH)}")
print(f"Connected to: {db.get_usable_table_names()}")

# -----------------------------
# SQLDatabaseToolkit - unlike 03_llm_examples/sales_db_agent.py's single
# hand-rolled run_sql tool (which blindly executes whatever SQL the LLM
# writes, with zero schema awareness), this toolkit gives the agent FOUR
# separate tools so it can discover the schema itself before querying it:
#   sql_db_list_tables   - what tables exist
#   sql_db_schema        - column names/types for specific tables
#   sql_db_query_checker  - asks the LLM to sanity-check a query before running it
#   sql_db_query          - actually executes a query
# The agent decides which of these to call and in what order - it isn't
# told the schema upfront, it has to look it up.
# -----------------------------
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
print("Tools:", ", ".join(t.name for t in tools))

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

Always look up the table schema before writing a query - never guess
column names. Use sql_db_query_checker before sql_db_query.

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

agent = create_react_agent(llm, tools, REACT_PROMPT)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=8,
)

if __name__ == "__main__":
    question = "Which product generated the highest total revenue, and what was that total?"
    result = agent_executor.invoke({"input": question})

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(result["output"])
