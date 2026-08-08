import chromadb
from chromadb.utils import embedding_functions
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
import os

# --------------------------------
# CONFIGURE OLLAMA ENDPOINT
# --------------------------------
os.environ["OLLAMA_HOST"] = "http://localhost:11434"

# --------------------------------
# INITIALIZE CHROMADB
# --------------------------------
client = chromadb.PersistentClient(path=r"c:/code/agenticai/9_general/rag/chromadb")

# Hugging Face embedding function
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create / load collection with embedding function
collection = client.get_or_create_collection(
    name="my_collection_hf_with_metadata_search",
    embedding_function=embedding_fn
)

# --------------------------------
# KNOWLEDGE BASE
# Question -> Answer + Metadata
# --------------------------------
knowledge_base = {
    "What is your shipping time?": {
        "answer": "Our standard shipping time is 3-5 business days.",
        "metadata": {"category": "shipping", "priority": "high"}
    },

    "What is your return policy?": {
        "answer": "You can return any product within 30 days of delivery.",
        "metadata": {"category": "returns", "priority": "medium"}
    },

    "What warranty do you provide?": {
        "answer": "All products come with a one-year warranty covering manufacturing defects.",
        "metadata": {"category": "warranty", "priority": "medium"}
    },

    "What payment methods do you accept?": {
        "answer": "We accept credit cards, debit cards, and PayPal.",
        "metadata": {"category": "payment", "priority": "low"}
    },

    "How can I contact customer support?": {
        "answer": "You can reach our support team 24/7 via email or chat.",
        "metadata": {"category": "support", "priority": "high"}
    }
}

# --------------------------------
# LOAD INTO CHROMADB (only once)
# --------------------------------
if collection.count() == 0:

    metadatas = []

    for item in knowledge_base.values():
        md = item["metadata"].copy()
        md["answer"] = item["answer"]
        metadatas.append(md)

    collection.add(
        documents=list(knowledge_base.keys()),      # Embed QUESTIONS
        ids=[str(i) for i in range(len(knowledge_base))],
        metadatas=metadatas
    )

    print("Knowledge base loaded into ChromaDB")

else:
    print(f"ChromaDB already contains {collection.count()} documents")

# --------------------------------
# INITIALIZE LANGCHAIN WITH OLLAMA
# --------------------------------
model = Ollama(
    model="llama3.2:latest",
    base_url="http://localhost:11434"
)

# --------------------------------
# CREATE RAG PROMPT TEMPLATE
# --------------------------------
rag_template = """You are a helpful customer support assistant.

Use ONLY the following context to answer the user's question.

If the answer is not present in the context, reply:
"I don't have that information in our knowledge base."

Context:
{context}

User Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(rag_template)

# --------------------------------
# RAG QUERY FUNCTION
# --------------------------------
def query_with_rag(user_question, category_filter=None, n_results=3):

    print(f"\n{'='*70}")
    print(f"User Question: {user_question}")
    print(f"{'='*70}")

    query_params = {
        "query_texts": [user_question],
        "n_results": n_results
    }

    if category_filter:
        query_params["where"] = category_filter
        print(f"Filter: {category_filter}")

    results = collection.query(**query_params)

    print(f"\nRetrieved {len(results['documents'][0])} relevant documents:")
    print("-" * 70)

    retrieved_answers = []

    for i, (question, metadata, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ),
        1
    ):

        print(f"\n{i}. Question : {question}")
        print(f"   Answer   : {metadata['answer']}")
        print(f"   Category : {metadata['category']}")
        print(f"   Priority : {metadata['priority']}")
        print(f"   Similarity : {1-distance:.3f}")

        retrieved_answers.append(metadata["answer"])

    # Use answers as LLM context
    context = "\n\n".join(retrieved_answers)

    print(f"\n{'='*70}")
    print("Generating Answer with LLM...")
    print(f"{'='*70}")

    formatted_prompt = prompt.format(
        context=context,
        question=user_question
    )

    response = model.invoke(formatted_prompt)

    print("\nAnswer:\n")
    print(response)

    print(f"\n{'='*70}\n")

    return {
        "question": user_question,
        "retrieved_questions": results["documents"][0],
        "retrieved_answers": retrieved_answers,
        "answer": response,
        "metadata": results["metadatas"][0]
    }

# --------------------------------
# EXAMPLE QUERIES
# --------------------------------
if __name__ == "__main__":

    query_with_rag("How long does shipping take?")

    query_with_rag(
        "What payment options do you have?",
        category_filter={"category": "payment"}
    )

    query_with_rag("Can I return a product if I don't like it?")

    query_with_rag("How do I contact customer support?")

    query_with_rag(
        "Tell me about your policies",
        n_results=5
    )

    print("\n" + "="*70)
    print("Interactive Mode - Ask questions (type 'quit' to exit)")
    print("="*70)

    while True:

        user_input = input("\nYour question: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input:
            query_with_rag(user_input)