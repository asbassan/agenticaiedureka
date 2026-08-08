import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import csv
from typing import Annotated, TypedDict
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(override=True)
sys.stdout.reconfigure(encoding="utf-8")

# =====================================================================
# Voice-driven, contextual customer-support RAG assistant.
#
# Combines two things the other 15_real scripts each show separately:
#   - RAG over the support FAQs (support_ticketing_agent_hitl.py), but
#     multi-turn instead of one-shot-per-ticket - a LangGraph checkpointer
#     carries conversation history across turns, and a "contextualize"
#     node rewrites follow-ups ("and how do I turn it back off?") into a
#     standalone query BEFORE retrieval, so vague follow-ups still hit the
#     right FAQ. That's the "contextual" half of the brief.
#   - Voice I/O (voice_tool_calling_stock_price.py's record/transcribe/
#     speak pattern) - the "audio input/output" half.
#
# Embeddings are explicit HuggingFace sentence-transformers, run locally
# (no embedding API calls), via the same all-MiniLM-L6-v2 model used in
# 9_general/rag/rag_3_chroma_db_hugging_face_embeddings.py. Note this
# isn't a night-and-day change from support_ticketing_agent_hitl.py -
# chromadb's bundled default embedding function is ALSO a local MiniLM
# model under the hood - the real difference is that here the model is
# pinned explicitly via the actual sentence-transformers library, which
# opens the door to swapping in any embedding model on the HF hub
# (bigger, multilingual, domain-specific, ...) instead of being stuck
# with whatever chromadb bundles.
#
# "Real-time" here means the same thing it does in the other 15_real
# voice scripts: turn-based (record -> transcribe -> respond -> speak),
# not full-duplex streaming audio - true low-latency streaming would need
# a websocket audio pipeline with partial transcription, well beyond what
# any other script in this repo does.
#
# Two ways to run:
#   python customer_support_voice_rag.py
#       Automated demo, no microphone needed. Two scripted turns are
#       synthesized into real audio via TTS first - standing in for a
#       live mic recording - so the full pipeline still runs audio in ->
#       Whisper -> contextualize -> retrieve -> generate -> TTS -> audio
#       out. Turn 2 is deliberately a vague follow-up ("and how do I turn
#       it back off again?") that only resolves correctly because of the
#       conversation history from turn 1.
#   python customer_support_voice_rag.py mic
#       Real usage: records you talking through your actual microphone
#       for each turn instead of using the scripted lines.
# =====================================================================

CSV_PATH = os.path.join(os.path.dirname(__file__), "customer_support_qa_500.csv")
MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
MATCH_THRESHOLD = 0.4  # min cosine similarity to trust a retrieval

openai_client = OpenAI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "customer_support_voice_audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

RECORD_SECONDS = 6
SAMPLE_RATE = 16000  # matches Whisper's native sample rate


def load_faq_index(csv_path: str):
    """Read the FAQ CSV and embed it into an in-memory ChromaDB collection using a
    real HuggingFace sentence-transformers model, downloaded and run locally rather
    than called via an API."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = chromadb.Client().create_collection(
        "support_faq_voice", embedding_function=embedding_fn, metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[row["id"] for row in rows],
        documents=[row["question"] for row in rows],
        metadatas=[{"answer": row["answer"]} for row in rows],
    )
    return collection


FAQ_INDEX = load_faq_index(CSV_PATH)

CONTEXTUALIZE_INSTRUCTIONS = (
    "Given the recent conversation and a new user message, rewrite the new message into "
    "a standalone question that can be understood without the conversation history (resolve "
    "things like 'it', 'that', 'the second one'). If the new message is already standalone, "
    "return it unchanged. Output ONLY the rewritten question - do not answer it."
)

GENERATE_INSTRUCTIONS = (
    "You are a voice-based customer support assistant. Answer using ONLY the FAQ context "
    "below - do not invent policies. This answer will be read aloud by text-to-speech, so "
    "respond in one or two short, natural spoken sentences - no markdown, no bullet points. "
    "If the FAQ context doesn't actually answer the question, say so plainly and that a "
    "human agent will follow up."
)


class SupportState(TypedDict):
    messages: Annotated[list, add_messages]  # conversation history, carried by the checkpointer
    query: str
    standalone_query: str
    retrieved_evidence: list[str]
    confidence: float
    response: str


# ---- Node 1: rewrite a possibly-vague follow-up into a standalone query ----
def contextualize(state: SupportState) -> SupportState:
    history = state["messages"]
    if not history:
        return {"standalone_query": state["query"]}

    recent = history[-4:]  # last couple of exchanges is enough to resolve a follow-up
    convo = "\n".join(f"{m.type}: {m.content}" for m in recent)
    response = openai_client.responses.create(
        model=MODEL,
        instructions=CONTEXTUALIZE_INSTRUCTIONS,
        input=f"Conversation so far:\n{convo}\n\nNew message: {state['query']}",
    )
    return {"standalone_query": response.output_text.strip()}


# ---- Node 2: semantic search against the FAQ index ----
def retrieve(state: SupportState) -> SupportState:
    result = FAQ_INDEX.query(query_texts=[state["standalone_query"]], n_results=TOP_K)
    matches = [
        {"question": q, "answer": meta["answer"]}
        for q, meta in zip(result["documents"][0], result["metadatas"][0])
    ]
    top_similarity = 1 - result["distances"][0][0]  # cosine distance -> similarity
    return {
        "retrieved_evidence": [f"Q: {m['question']}  A: {m['answer']}" for m in matches],
        "confidence": top_similarity,
    }


# ---- Node 3: answer grounded in the retrieved FAQs, phrased for TTS ----
def generate(state: SupportState) -> SupportState:
    context = "\n\n".join(state["retrieved_evidence"]) if state["retrieved_evidence"] else "(no relevant FAQ found)"
    response = openai_client.responses.create(
        model=MODEL,
        instructions=GENERATE_INSTRUCTIONS,
        input=(
            f"FAQ context:\n{context}\n\n"
            f"Customer's original message: {state['query']}\n"
            f"Standalone version of the question: {state['standalone_query']}"
        ),
    )
    answer = response.output_text
    return {"response": answer, "messages": [HumanMessage(state["query"]), AIMessage(answer)]}


# Graph: START -> contextualize -> retrieve -> generate -> END
graph = StateGraph(SupportState)
graph.add_node("contextualize", contextualize)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.add_edge(START, "contextualize")
graph.add_edge("contextualize", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

# MemorySaver keeps "messages" (and the rest of the state) alive across turns that
# share the same thread_id - that's what makes contextualize's follow-up rewriting
# possible. Only persists in-process; a real desk would use a DB-backed checkpointer.
app = graph.compile(checkpointer=MemorySaver())


def record_from_microphone(path: str, seconds: int = RECORD_SECONDS) -> None:
    """Real mic capture - only used in 'mic' mode, so sounddevice/soundfile are
    imported lazily here rather than at module load."""
    import sounddevice as sd
    import soundfile as sf

    print(f"Recording for {seconds}s... speak now.")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    sf.write(path, audio, SAMPLE_RATE)
    print("Done recording.")


def synthesize_sample_user_turn(text: str, path: str) -> None:
    """Stand-in for a live mic recording: turns a scripted line into real audio via
    TTS, so the automated demo still exercises actual speech-to-text, not just text."""
    response = openai_client.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input=text)
    response.write_to_file(path)


def transcribe(path: str) -> str:
    """Whisper: speech -> text."""
    with open(path, "rb") as audio_file:
        transcript = openai_client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return transcript.text


def speak(text: str, path: str) -> None:
    """TTS: text -> speech, so the answer is an actual voice file, not just text."""
    response = openai_client.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input=text)
    response.write_to_file(path)


def run_turn(turn_num: int, config: dict, use_mic: bool, scripted_line: str = "") -> None:
    user_audio_path = os.path.join(AUDIO_DIR, f"turn{turn_num}_user.wav")

    if use_mic:
        record_from_microphone(user_audio_path)
    else:
        synthesize_sample_user_turn(scripted_line, user_audio_path)
        print(f'(simulated microphone input, synthesized via TTS): "{scripted_line}"')

    user_text = transcribe(user_audio_path)
    print(f"You said: {user_text}")

    result = app.invoke({"query": user_text}, config)
    print(f"  [standalone query] {result['standalone_query']}")
    print(f"  [retrieval confidence] {result['confidence']:.0%}")

    reply = result["response"]
    print(f"Assistant: {reply}")

    reply_audio_path = os.path.join(AUDIO_DIR, f"turn{turn_num}_assistant.mp3")
    speak(reply, reply_audio_path)
    print(f"(assistant reply spoken to {reply_audio_path})\n")
    os.startfile(reply_audio_path)  # auto-play in the default media app (Windows only)


if __name__ == "__main__":
    use_mic = len(sys.argv) > 1 and sys.argv[1] == "mic"
    config = {"configurable": {"thread_id": "support-session-1"}}

    if use_mic:
        print("=== VOICE CUSTOMER SUPPORT ASSISTANT (live microphone) ===")
        print("Two turns - the second is a vague follow-up. Speak when prompted.\n")
        run_turn(1, config, use_mic=True)
        run_turn(2, config, use_mic=True)
    else:
        print("=== VOICE CUSTOMER SUPPORT ASSISTANT (automated demo - no microphone needed) ===")
        print("Run with 'mic' as an argument to use your real microphone instead.\n")
        run_turn(1, config, use_mic=False, scripted_line="How do I enable two-factor authentication?")
        run_turn(2, config, use_mic=False, scripted_line="And how do I turn it back off again?")
