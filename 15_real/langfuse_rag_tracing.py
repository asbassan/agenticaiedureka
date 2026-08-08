import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import csv
import sys
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from langfuse import get_client

load_dotenv(override=True)
sys.stdout.reconfigure(encoding="utf-8")

# =====================================================================
# Tracing a small RAG pipeline with Langfuse.
#
# The existing Langfuse scripts in this repo (9_general/observability/
# langfuse_local, 9_general/observability/langfuse_internet,
# 9_general/prod/langfuse) all trace ONE flat LLM call as a single
# "generation" observation. This one traces a multi-step pipeline -
# retrieve -> generate - as a proper trace TREE: a parent span for the
# whole query, with a "retriever" observation and a "generation"
# observation nested underneath it. That nesting is the actual point:
# opening the trace in the Langfuse UI shows retrieval and generation as
# separate timed steps, each with their own input/output, instead of one
# opaque call.
#
# Reuses the same FAQ dataset and in-memory ChromaDB setup as
# support_ticketing_agent_hitl.py (kept deliberately simple here - no
# LangGraph, no HITL - since the point of this file is the tracing, not
# the RAG logic itself).
#
# After running, open the trace list at:
#   https://us.cloud.langfuse.com  ->  your project  ->  Tracing
# =====================================================================

CSV_PATH = os.path.join(os.path.dirname(__file__), "customer_support_qa_500.csv")
MODEL = "gpt-4o-mini"
TOP_K = 3

openai_client = OpenAI()
langfuse = get_client()


def load_faq_index(csv_path: str):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    collection = chromadb.Client().create_collection("support_faq_traced", metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[row["id"] for row in rows],
        documents=[row["question"] for row in rows],
        metadatas=[{"answer": row["answer"]} for row in rows],
    )
    return collection


FAQ_INDEX = load_faq_index(CSV_PATH)


def answer_question(question: str) -> str:
    """One traced RAG query: a parent span containing a retriever observation
    and a generation observation, so the two steps show up separately in Langfuse."""
    with langfuse.start_as_current_observation(name="support-query", as_type="span", input=question) as query_span:

        with query_span.start_as_current_observation(
            name="retrieve-faqs", as_type="retriever", input=question
        ) as retrieval:
            result = FAQ_INDEX.query(query_texts=[question], n_results=TOP_K)
            faqs = [
                {"question": q, "answer": meta["answer"]}
                for q, meta in zip(result["documents"][0], result["metadatas"][0])
            ]
            top_similarity = 1 - result["distances"][0][0]  # cosine distance -> similarity
            retrieval.update(
                output=faqs,
                metadata={"top_k": TOP_K, "top_similarity": round(top_similarity, 4)},
            )

        context = "\n\n".join(f"Q: {faq['question']}\nA: {faq['answer']}" for faq in faqs)
        system_prompt = (
            "You are a customer support assistant. Answer using ONLY the FAQ context below - "
            "do not invent policies. Keep it short."
        )

        with query_span.start_as_current_observation(
            name="generate-answer", as_type="generation", model=MODEL,
            input={"system": system_prompt, "context": context, "question": question},
        ) as generation:
            response = openai_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"FAQ context:\n{context}\n\nCustomer question: {question}"},
                ],
            )
            answer = response.choices[0].message.content
            usage = response.usage
            generation.update(
                output=answer,
                usage_details={
                    "input": usage.prompt_tokens if usage else 0,
                    "output": usage.completion_tokens if usage else 0,
                    "total": usage.total_tokens if usage else 0,
                },
            )

        query_span.update(output=answer)
        return answer


if __name__ == "__main__":
    if not langfuse.auth_check():
        raise SystemExit("Langfuse authentication failed - check LANGFUSE_SECRET_KEY/PUBLIC_KEY/BASE_URL in .env.")
    print("Langfuse client authenticated.\n")

    questions = [
        "How do I reset my password?",
        "What is your return policy for damaged items?",
        "How do I enable two-factor authentication?",
    ]

    for question in questions:
        print(f"Q: {question}")
        print(f"A: {answer_question(question)}\n")

    langfuse.flush()  # short-lived script - make sure everything is sent before exit
    print("Traces sent. View them at https://us.cloud.langfuse.com -> your project -> Tracing")
