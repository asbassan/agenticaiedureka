# Basic langchain example
# uv pip install langchain-google-genai
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

# 1. Instantiate the template with a placeholder
prompt_template = PromptTemplate.from_template(
    "Tell me an interesting fact about {topic}."
)

# 2. Instantiate Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash") 

# 3. Chain them together using LCEL (LangChain Expression Language)
chain = prompt_template | llm

# 4. Run the chain
response = chain.invoke({"topic": "einstein"})
print(response.content)