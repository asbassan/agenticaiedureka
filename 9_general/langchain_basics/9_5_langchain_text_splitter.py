# Basic langchain example showing a text splitter - loads a PDF and splits
# it into overlapping character-based chunks ready for embedding
# uv pip install langchain-community langchain-text-splitters pypdf
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "rag", "Principles-of-Data-Science.pdf")

# 1. Load the PDF - one Document per page
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
print(f"Loaded {len(pages)} pages from {PDF_PATH}")

# 2. Split the pages into overlapping chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
)
chunks = text_splitter.split_documents(pages)

# 3. Inspect the result
print(f"Split into {len(chunks)} chunks")
print("\nHundredth chunk:")
print("-" * 70)
print(chunks[99].page_content)
print("-" * 70)
print(f"Metadata: {chunks[99].metadata}")
