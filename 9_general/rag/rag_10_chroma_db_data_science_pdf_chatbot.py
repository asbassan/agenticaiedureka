# pip install chromadb pypdf langchain-google-genai python-dotenv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import chromadb
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv(override=True)

# --------------------------------
# CONFIGURE GEMINI (used to synthesize an answer from the retrieved chunks)
# --------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_retries=2,
)

def get_text(resp):
    # In some langchain-core versions, .text is a method, not a string - skip it then.
    text = getattr(resp, "text", None)
    if callable(text):
        text = None
    return text or resp.content

# --------------------------------
# INITIALIZE CHROMADB
# Chroma will embed the text on its own using its own model
# --------------------------------
client = chromadb.PersistentClient(path=r"c:/code/agenticai/9_general/rag/chromadb")

collection = client.get_or_create_collection(name="data_science_book")

PDF_PATH = r"c:/code/agenticai/9_general/rag/Principles-of-Data-Science.pdf"

CHUNK_SIZE_WORDS = 500
CHUNK_OVERLAP_WORDS = 100

# Below this similarity (1 - distance), the best chunk isn't actually about the
# question. Empirically, on-topic questions ("What is a p-value?", "What is
# linear regression?") score anywhere from -0.04 to 0.34 against this corpus,
# while off-topic questions ("capital of France?") score below -0.68 - so
# -0.3 sits safely in the gap between the two.
SIMILARITY_THRESHOLD = -0.3

# --------------------------------
# CHUNK TEXT INTO OVERLAPPING WORD WINDOWS
# --------------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):

    words = text.split()
    step = chunk_size - overlap

    chunks = []

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break

    return chunks

# --------------------------------
# LOAD PDF, CHUNK, AND EMBED (only once)
# --------------------------------
if collection.count() == 0:

    print(f"Reading PDF: {PDF_PATH}")

    reader = PdfReader(PDF_PATH)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    chunks = chunk_text(full_text)

    collection.add(
        documents=chunks,
        ids=[str(i) for i in range(len(chunks))],
        metadatas=[{"chunk_index": i, "source": "Principles-of-Data-Science.pdf"} for i in range(len(chunks))]
    )

    print(f"Stored {len(chunks)} chunks ({CHUNK_SIZE_WORDS} words each, {CHUNK_OVERLAP_WORDS} word overlap)")

else:
    print(f"ChromaDB already contains {collection.count()} chunks")

# --------------------------------
# RETRIEVAL FUNCTION
# --------------------------------
def query_data_science_book(user_question, n_results=3):

    print(f"\n{'='*70}")
    print(f"User Question: {user_question}")
    print(f"{'='*70}")

    results = collection.query(
        query_texts=[user_question],      # Chroma embeds this automatically
        n_results=n_results
    )

    print(f"\nTop {len(results['documents'][0])} matching excerpts:")
    print("-" * 70)

    for i, (chunk, metadata, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ),
        1
    ):

        print(f"\n{i}. Chunk #{metadata['chunk_index']} (similarity: {1-distance:.3f})")
        print(f"   {chunk}")

    # --------------------------------
    # GENERATION: only trust ChromaDB's chunks if the best match clears the
    # similarity threshold - otherwise the book simply doesn't cover this
    # question, so let Gemini answer from its own general knowledge instead.
    # --------------------------------
    best_similarity = 1 - results["distances"][0][0]

    if best_similarity >= SIMILARITY_THRESHOLD:
        print(f"\nBest match similarity {best_similarity:.3f} >= {SIMILARITY_THRESHOLD} -> answering from the book")
        context = "\n\n".join(results["documents"][0])
        prompt = (
            "Answer the question using the context below, which is excerpted "
            "from the OpenStax textbook 'Principles of Data Science'.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {user_question}"
        )
    else:
        print(f"\nBest match similarity {best_similarity:.3f} < {SIMILARITY_THRESHOLD} -> book has nothing relevant, answering from general knowledge")
        prompt = (
            "This question is not covered by the textbook 'Principles of Data Science'. "
            f"Answer it from your own general knowledge instead.\n\nQuestion: {user_question}"
        )

    resp = llm.invoke([HumanMessage(content=prompt)])

    print(f"\n{'-'*70}")
    print("Answer (Gemini):")
    print(get_text(resp))
    print(f"\n{'='*70}\n")

# --------------------------------
# TERMINAL CHATBOT
# --------------------------------
if __name__ == "__main__":

    print("\n" + "="*70)
    print("Data Science Book Chatbot - Ask questions (type 'quit' to exit)")
    print("="*70)

    while True:

        user_input = input("\nYour question: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input:
            query_data_science_book(user_input)
