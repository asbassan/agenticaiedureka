import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os

# WebBaseLoader warns if this isn't set, since some sites use it to identify
# well-behaved scrapers vs. bots.
os.environ.setdefault("USER_AGENT", "agenticai-course-demo/1.0")

from typing import Iterator, List
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    CSVLoader,
    NotionDirectoryLoader,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def show(label: str, docs: List[Document], preview_chars: int = 150) -> None:
    print(f"\n{'=' * 70}")
    print(f"{label} - {len(docs)} document(s) loaded")
    print("=" * 70)
    for i, doc in enumerate(docs[:3], start=1):
        preview = doc.page_content.strip().replace("\n", " ")[:preview_chars]
        print(f"{i}. metadata={doc.metadata}")
        print(f"   content: {preview}...")

# -----------------------------
# 1. PDF Loader
# Splits a PDF into one Document per page, with page number in metadata.
# Reuses the Principles-of-Data-Science.pdf already used in 9_general/rag.
# -----------------------------
pdf_path = os.path.join(BASE_DIR, "..", "..", "9_general", "rag", "Principles-of-Data-Science.pdf")
pdf_docs = PyPDFLoader(pdf_path).load()
show("1. PyPDFLoader", pdf_docs)

# -----------------------------
# 2. Web Loader
# Fetches a live web page and strips it down to its text content via
# BeautifulSoup. One Document per URL by default.
# -----------------------------
web_docs = WebBaseLoader(["https://en.wikipedia.org/wiki/Artificial_intelligence"]).load()
show("2. WebBaseLoader", web_docs)

# -----------------------------
# 3. CSV Loader
# One Document per ROW, with each column rendered as "column: value" text
# and the row's column values also kept in metadata.
# Reuses the documents.csv already used by the 04_rag demos.
# -----------------------------
csv_path = os.path.join(BASE_DIR, "..", "04_rag", "documents.csv")
csv_docs = CSVLoader(csv_path).load()
show("3. CSVLoader", csv_docs)

# -----------------------------
# 4. Notion Loader
# Reads every .md file out of a directory (this is the format Notion's
# "Export as Markdown" produces) - one Document per page.
# -----------------------------
notion_path = os.path.join(BASE_DIR, "sample_notion_export")
notion_docs = NotionDirectoryLoader(notion_path).load()
show("4. NotionDirectoryLoader", notion_docs)

# -----------------------------
# 5. Custom Loader
# There's no built-in loader for every possible data source (an internal
# API, a queue, a proprietary DB). Writing one is just: subclass BaseLoader
# and implement lazy_load() to yield Document objects. Everything downstream
# (text splitters, vector stores, retrievers) works with any loader the
# same way, because they only ever depend on the Document interface.
# -----------------------------
class InMemoryRecordLoader(BaseLoader):
    """Loads Documents out of an arbitrary in-memory list of dicts -
    stands in for records you'd normally pull from an API or database."""

    def __init__(self, records: List[dict], text_field: str):
        self.records = records
        self.text_field = text_field

    def lazy_load(self) -> Iterator[Document]:
        for record in self.records:
            metadata = {k: v for k, v in record.items() if k != self.text_field}
            yield Document(page_content=record[self.text_field], metadata=metadata)

support_tickets = [
    {"id": 101, "priority": "high", "text": "Login page throws a 500 error on submit."},
    {"id": 102, "priority": "low", "text": "Dark mode toggle doesn't persist after refresh."},
    {"id": 103, "priority": "medium", "text": "Export to CSV is missing the last column."},
]

custom_docs = InMemoryRecordLoader(support_tickets, text_field="text").load()
show("5. Custom InMemoryRecordLoader", custom_docs)
