# Basic langchain example showing the most common output parsers:
# str, json, pydantic and csv (comma-separated list)
# uv pip install langchain-openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    CommaSeparatedListOutputParser,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")

# ---------------------------------------------------------------------------
# 1. StrOutputParser - just extracts the plain text content from the AIMessage
# ---------------------------------------------------------------------------
str_prompt = ChatPromptTemplate.from_template(
    "Tell me an interesting fact about {topic}."
)
str_chain = str_prompt | llm | StrOutputParser()
str_result = str_chain.invoke({"topic": "einstein"})
print("STR OUTPUT:")
print(str_result)
print(type(str_result))

# ---------------------------------------------------------------------------
# 2. JsonOutputParser - asks the model to reply with JSON and parses it into
#    a plain Python dict
# ---------------------------------------------------------------------------
json_parser = JsonOutputParser()
json_prompt = ChatPromptTemplate.from_template(
    "Give an interesting fact about {topic}.\n"
    "{format_instructions}\n"
    "Reply with a JSON object with keys 'topic' and 'fact'."
).partial(format_instructions=json_parser.get_format_instructions())
json_chain = json_prompt | llm | json_parser
json_result = json_chain.invoke({"topic": "einstein"})
print("\nJSON OUTPUT:")
print(json_result)
print(type(json_result))


# ---------------------------------------------------------------------------
# 3. PydanticOutputParser - same idea as JSON, but parsed straight into a
#    validated Pydantic model
# ---------------------------------------------------------------------------
class Fact(BaseModel):
    topic: str = Field(description="the topic the fact is about")
    fact: str = Field(description="the interesting fact itself")


pydantic_parser = PydanticOutputParser(pydantic_object=Fact)
pydantic_prompt = ChatPromptTemplate.from_template(
    "Give an interesting fact about {topic}.\n{format_instructions}"
).partial(format_instructions=pydantic_parser.get_format_instructions())
pydantic_chain = pydantic_prompt | llm | pydantic_parser
pydantic_result = pydantic_chain.invoke({"topic": "einstein"})
print("\nPYDANTIC OUTPUT:")
print(pydantic_result)
print(type(pydantic_result))

# ---------------------------------------------------------------------------
# 4. CommaSeparatedListOutputParser - asks the model for a CSV-style list and
#    parses it into a Python list of strings
# ---------------------------------------------------------------------------
csv_parser = CommaSeparatedListOutputParser()
csv_prompt = ChatPromptTemplate.from_template(
    "List 5 interesting facts about {topic}.\n{format_instructions}"
).partial(format_instructions=csv_parser.get_format_instructions())
csv_chain = csv_prompt | llm | csv_parser
csv_result = csv_chain.invoke({"topic": "einstein"})
print("\nCSV OUTPUT:")
print(csv_result)
print(type(csv_result))
