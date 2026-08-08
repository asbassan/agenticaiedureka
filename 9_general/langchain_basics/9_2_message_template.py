# Basic langchain example using individual message prompt templates
# uv pip install langchain-openai
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 1. Build each message template separately...
system_message_template = SystemMessagePromptTemplate.from_template(
    "You are a helpful assistant who answers in a single short sentence."
)
human_message_template = HumanMessagePromptTemplate.from_template(
    "Tell me an interesting fact about {topic}."
)

# ...then combine them into a chat prompt template
chat_prompt_template = ChatPromptTemplate.from_messages(
    [system_message_template, human_message_template]
)

# 2. Instantiate GPT-4o mini
llm = ChatOpenAI(model="gpt-4o-mini")

# 3. Chain them together using LCEL (LangChain Expression Language)
chain = chat_prompt_template | llm

# 4. Run the chain
response = chain.invoke({"topic": "einstein"})
print(response.content)
