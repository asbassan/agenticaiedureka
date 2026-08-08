# Basic langchain example
# uv pip install langchain-openai
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 1. Instantiate the template with a placeholder
prompt_template = PromptTemplate.from_template(
    "Tell me an interesting fact about {topic}."
)

# 2. Instantiate GPT-4o mini
llm = ChatOpenAI(model="gpt-4o-mini")

# 3. Chain them together using LCEL (LangChain Expression Language)
chain = prompt_template | llm

# 4. Run the chain
response = chain.invoke({"topic": "einstein"})
print(response.content)
