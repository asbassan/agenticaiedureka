# pip install llama-index openai python-dotenv

import os
import sys
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
sys.stdout.reconfigure(encoding="utf-8")

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)

DATA_DIR = os.path.join(os.path.dirname(__file__), "loan_data")

# =====================================================================
# Grounded document-based generation with LlamaIndex's CitationQueryEngine.
#
# 9_general/rag/rag_7_llamaindex.py already shows a plain query engine
# (retrieve chunks -> generate an answer -> print the source_nodes
# separately, so you have to trust the answer actually used them).
# CitationQueryEngine goes a step further: it splits retrieved context
# into small, numbered source chunks and instructs the LLM to cite them
# inline - [1], [2], etc. - for every claim, so each individual sentence
# of the answer can be traced back to an exact passage, not just "this
# whole answer is grounded in these 3 chunks somewhere."
#
# Reuses the real bank policy + RBI documents from
# loan_eligibility_rag_reasoning.py's loan_data/ rather than a new corpus -
# dense, numbered policy rules are exactly where precise citation matters.
# =====================================================================

documents = SimpleDirectoryReader(
    input_files=[
        os.path.join(DATA_DIR, "loan_eligibility_policy.md"),
        os.path.join(DATA_DIR, "rbi_guidelines.md"),
    ]
).load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=5,
    citation_chunk_size=256,  # smaller chunks = more precise, less "which sentence in this chunk?"
)


def ask(question: str):
    response = query_engine.query(question)

    print(f"\nQ: {question}")
    print(f"A: {response}\n")

    print("Cited sources:")
    for i, node in enumerate(response.source_nodes, start=1):
        preview = node.node.get_text().strip().replace("\n", " ")[:150]
        print(f"  [{i}] (score={node.score:.3f}) \"{preview}...\"")


if __name__ == "__main__":
    ask(
        "What is the maximum DTI ratio allowed for a salaried home loan applicant, "
        "and how is the maximum eligible loan amount calculated?"
    )
    ask(
        "What is the maximum loan-to-value ratio for a ₹50 lakh home loan, and what "
        "happens to an application if the applicant's credit score is below 650?"
    )
