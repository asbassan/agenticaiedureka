# Basic end-to-end LangChain RAG example:
# load a web page -> split -> embed with a local HuggingFace model ->
# store in a persistent Chroma vector store -> retrieve + answer with an LLM
# uv pip install langchain-community langchain-text-splitters langchain-huggingface langchain-chroma langchain-openai beautifulsoup4 sentence-transformers
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ.setdefault("USER_AGENT", "agenticai-course-demo/1.0")

URL = "https://www.gutenberg.org/cache/epub/1661/pg1661-images.html"
PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db_sherlock")
COLLECTION_NAME = "sherlock_holmes"

# 1. Hugging Face embedding model (runs locally, no API key needed)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Load and embed the book into a persistent Chroma store (only once)
if os.path.exists(PERSIST_DIR):
    print(f"Loading existing Chroma store from {PERSIST_DIR}")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
else:
    print(f"Loading page: {URL}")
    docs = WebBaseLoader(URL).load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    print(f"Embedding chunks and persisting to {PERSIST_DIR}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 3. RAG prompt + LLM
llm = ChatOpenAI(model="gpt-4o-mini")

rag_prompt = ChatPromptTemplate.from_template(
    "Answer the question using ONLY the following context.\n"
    "If the answer isn't in the context, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def format_docs(retrieved_docs) -> str:
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


# 4. Wire retriever + prompt + llm together with LCEL
rag_chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# 5. Test with 3 queries
test_queries = [
    "What does the book say about Ku Klux Klan?",
    "Where does Irene Adler live?",
    "What is Sherlock Holmes wearing and doing when Watson visits him in the morning?",
]

for query in test_queries:
    print(f"\n{'='*70}")
    print(f"Question: {query}")
    print(f"{'='*70}")
    answer = rag_chain.invoke(query)
    print(f"\nAnswer:\n{answer}")
