# Basic langchain example using a chat prompt template
# uv pip install langchain-openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 1. Instantiate a chat prompt template made up of a system and a human turn
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant who answers in a single short sentence."),
        ("human", "Tell me an interesting fact about {topic}."),
    ]
)

# 2. Instantiate GPT-4o mini
llm = ChatOpenAI(model="gpt-4o-mini")

# 3. Chain them together using LCEL (LangChain Expression Language)
chain = chat_prompt_template | llm

# 4. Run the chain
response = chain.invoke({"topic": "einstein"})
print(response.content)
