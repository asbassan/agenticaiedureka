import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    VectorStoreRetrieverMemory,
)
from langchain_core._api.deprecation import (
    LangChainDeprecationWarning,
    LangChainPendingDeprecationWarning,
)

# langchain_core re-enables its own deprecation warnings on import (it calls
# warnings.filterwarnings("default", ...) internally), which runs AFTER our
# filters above and so overrides them. Re-apply "ignore" now that the import
# has happened, so ours is the one that wins.
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

load_dotenv(override=True)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# -----------------------------
# 1. ConversationBufferMemory
# Stores every turn verbatim, in order. Simple and lossless, but grows
# unbounded - every past message gets resent to the LLM on every call,
# so token cost climbs with conversation length.
# -----------------------------
print("=" * 70)
print("1. ConversationBufferMemory - stores every turn verbatim")
print("=" * 70)

buffer_memory = ConversationBufferMemory()
buffer_memory.save_context({"input": "My name is Atul."}, {"output": "Nice to meet you, Atul!"})
buffer_memory.save_context({"input": "I live in Pune."}, {"output": "Pune is a great city!"})

print(buffer_memory.load_memory_variables({})["history"])

# -----------------------------
# 2. ConversationSummaryMemory
# Instead of storing raw turns, asks an LLM to progressively rewrite a
# running summary after each turn. Size stays roughly constant regardless
# of conversation length, at the cost of an LLM call per turn and losing
# verbatim wording/detail.
# -----------------------------
print("\n" + "=" * 70)
print("2. ConversationSummaryMemory - condenses history via an LLM")
print("=" * 70)

summary_memory = ConversationSummaryMemory(llm=llm)
summary_memory.save_context({"input": "My name is Atul."}, {"output": "Nice to meet you, Atul!"})
summary_memory.save_context({"input": "I live in Pune."}, {"output": "Pune is a great city!"})
summary_memory.save_context({"input": "I work as an AI Trainer now."}, {"output": "That's a great field to be in!"})

print(summary_memory.load_memory_variables({})["history"])

# -----------------------------
# 3. VectorStoreRetrieverMemory
# Stores each turn as an embedding in a vector store and, given a new
# input, retrieves only the top-k most SIMILAR past turns - not the most
# recent ones. Good for long-term recall across a long history where most
# of it is irrelevant to the current question.
# -----------------------------
print("\n" + "=" * 70)
print("3. VectorStoreRetrieverMemory - retrieves relevant past turns by similarity")
print("=" * 70)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# FAISS needs at least one text to initialize the index
vectorstore = FAISS.from_texts(["memory store initialized"], embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
vector_memory = VectorStoreRetrieverMemory(retriever=retriever)

vector_memory.save_context({"input": "My favorite color is blue."}, {"output": "Blue is a calm, cool color."})
vector_memory.save_context({"input": "I enjoy playing chess on weekends."}, {"output": "Chess is a great strategic game."})
vector_memory.save_context({"input": "My favorite food is fruit salad."}, {"output": "Fruit salad is delicious!"})

# Ask about only ONE of the three saved turns - notice that similarity,
# not recency, decides what comes back
query = "What game do I like to play?"
print(f"Query: '{query}'")
print(vector_memory.load_memory_variables({"prompt": query})["history"])
