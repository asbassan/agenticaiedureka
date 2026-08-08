# -------------------------------
# 1. Install dependencies if needed
# -------------------------------
# pip install llama-index openai pypdf

# -------------------------------
# 2. Imports
# -------------------------------
import sys
import os

# PDF text extraction can produce stray characters outside Windows' default
# console codepage (cp1252) - reconfigure stdout to UTF-8 so printing
# retrieved chunks doesn't crash on them.
sys.stdout.reconfigure(encoding="utf-8")

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)

# -------------------------------
# 3. Configure LLM
# -------------------------------
# LLM is used for Answer synthesis, Context compression, Reasoning over retrieved chunks
Settings.llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "Principles-of-Data-Science.pdf")
STORAGE_DIR = os.path.join(BASE_DIR, "llamaindex_storage")

# -------------------------------
# 4. Build (or reuse) the Vector Index
# -------------------------------
# Real usage never re-embeds the whole book on every run - that's slow and
# costs an embedding API call per chunk. Build the index once and persist it;
# later runs just load the saved vectors from disk.
if os.path.exists(STORAGE_DIR):
    print(f"Loading previously built index from {STORAGE_DIR} ...")
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context)
else:
    print(f"No saved index found - building one from {PDF_PATH} (first run only)...")

    # SimpleDirectoryReader reads the PDF and wraps each page in a Document object.
    # No chunking yet - that happens below, during indexing.
    documents = SimpleDirectoryReader(input_files=[PDF_PATH]).load_data()

    # Internally, LlamaIndex:
    # Step 1: Splits the book into smaller chunks (~512 tokens by default), with overlap
    #   e.g. Chunk 1 -> Chapter 1 paragraphs, Chunk 2 -> Next section
    # Step 2: Embeds each chunk -> vector, e.g. [0.023, -0.91, 0.44, ...]
    # Step 3: Stores all embeddings in an in-memory vector store
    index = VectorStoreIndex.from_documents(documents)

    index.storage_context.persist(persist_dir=STORAGE_DIR)
    print(f"Index built and persisted to {STORAGE_DIR} - future runs will reuse it.")

# Create a query engine: builds a retrieval + reasoning (synthesis) chain.
# This is RAG, although we have not manually implemented it:
# User question -> Embed question -> Similarity search (top-k) -> Send chunks to LLM -> Generate final answer
# similarity_top_k controls how many chunks get retrieved and sent to the LLM per query.
query_engine = index.as_query_engine(similarity_top_k=3)

# -------------------------------
# 5. Query Examples
# -------------------------------
def ask(question: str) -> None:
    # Per query: (1) question is embedded, (2) similar chunks from the textbook
    # are retrieved, (3) the LLM receives those chunks plus the question, and
    # (4) it synthesizes a final answer grounded in the book.
    response = query_engine.query(question)

    print(f"\nQ: {question}")
    print(f"A: {response}")

    # Printing the retrieved chunks makes the "retrieval" half of RAG visible -
    # otherwise it's easy to mistake this for the LLM just answering from memory.
    print("Grounded in these retrieved chunks:")
    for i, node in enumerate(response.source_nodes, start=1):
        preview = node.node.get_text().strip().replace("\n", " ")[:150]
        print(f"  [{i}] score={node.score:.3f}  \"{preview}...\"")

ask("Summarize the main ideas of the book.")
ask("What is a p-value?")
