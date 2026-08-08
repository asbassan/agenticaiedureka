# Basic langchain example showing RunnableLambda - wraps a plain Python
# function so it can be dropped into an LCEL chain like any other Runnable
# uv pip install langchain-openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template(
    "Tell me an interesting fact about {topic}."
)


# 1. A plain function used to pre-process the input before it hits the prompt
def clean_topic(input_dict: dict) -> dict:
    return {"topic": input_dict["topic"].strip().title()}


# 2. A plain function used to post-process the LLM's text output
def add_word_count(fact: str) -> str:
    word_count = len(fact.split())
    return f"{fact}\n(word count: {word_count})"


# 3. Wrap both functions as RunnableLambda so they can be piped with `|`
clean_topic_runnable = RunnableLambda(clean_topic)
add_word_count_runnable = RunnableLambda(add_word_count)

# 4. Chain everything together using LCEL
chain = clean_topic_runnable | prompt | llm | StrOutputParser() | add_word_count_runnable

# 5. Run the chain
response = chain.invoke({"topic": "  einstein  "})
print(response)
