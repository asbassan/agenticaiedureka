# pip install cohere python-dotenv

import os
import csv
import cohere
from dotenv import load_dotenv

load_dotenv(override=True)

co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
CSV_PATH = os.path.join(os.path.dirname(__file__), "customer_support_qa_500.csv")

# =====================================================================
# A tour of Cohere's own API surface (not OpenAI's), covering the 3
# pieces of a RAG pipeline it provides natively:
#   1. embed()  - turn text into vectors
#   2. rerank() - given a query and a batch of candidate documents,
#      score them by relevance - the "narrow down to what matters" step
#   3. chat()   - generate an answer, with documents= passed straight
#      in for grounded generation - Cohere's chat endpoint handles the
#      "stuff retrieved context into the prompt" step for you
# Reuses the same FAQ CSV as the rest of 15_real, so this is a real
# side-by-side alternative to the OpenAI + ChromaDB pipeline used
# elsewhere in this directory, not a new toy dataset.
# =====================================================================


def load_faqs(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def show_embeddings(texts: list[str]):
    response = co.embed(model="embed-v4.0", input_type="search_document", texts=texts, embedding_types=["float"])
    for text, vector in zip(texts, response.embeddings.float_):
        print(f"  {text!r} -> {len(vector)}-dim vector, first 5 values: {[round(v, 3) for v in vector[:5]]}")


def rerank_faqs(query: str, faqs: list[dict], top_n: int = 3) -> list[dict]:
    documents = [faq["question"] for faq in faqs]
    response = co.rerank(model="rerank-v3.5", query=query, documents=documents, top_n=top_n)
    return [{**faqs[result.index], "relevance_score": result.relevance_score} for result in response.results]


def generate_grounded_answer(query: str, top_faqs: list[dict]) -> str:
    documents = [f"Q: {faq['question']}\nA: {faq['answer']}" for faq in top_faqs]
    response = co.chat(model="command-a-03-2025", messages=[{"role": "user", "content": query}], documents=documents)
    return response.message.content[0].text


if __name__ == "__main__":
    faqs = load_faqs(CSV_PATH)
    query = "How do I get my money back for an order?"

    print("=== 1. Embed: turning text into vectors ===")
    show_embeddings([query, "How do I reset my password?"])

    print(f"\n=== 2. Rerank: narrowing {len(faqs)} FAQs down to the most relevant ===")
    top_faqs = rerank_faqs(query, faqs)
    for faq in top_faqs:
        print(f"  ({faq['relevance_score']:.3f}) {faq['question']}")

    print("\n=== 3. Chat: generating a grounded answer from those top FAQs ===")
    print(generate_grounded_answer(query, top_faqs))
