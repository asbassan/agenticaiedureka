import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import base64
import csv
import os
import sys
from typing import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

load_dotenv(override=True)

# LLM output can contain characters outside Windows' default console
# codepage (cp1252) - reconfigure stdout to UTF-8 so printing doesn't crash.
sys.stdout.reconfigure(encoding="utf-8")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATIENT_DATA_PATH = os.path.join(BASE_DIR, "patient_data.csv")
LAB_RESULTS_PATH = os.path.join(BASE_DIR, "lab_results.csv")
XRAY_PATH = os.path.join(BASE_DIR, "Chest-x-ray.jpg")

# =====================================================================
# Medical diagnosis pipeline built from SUBGRAPHS.
#
# medical_history and lab_analysis are each their own compiled
# StateGraph, with their OWN narrow state schema (HistoryState,
# LabState) rather than sharing the parent's MedicalState. A thin
# adapter function (run_medical_history/run_lab_analysis) invokes each
# subgraph and maps its result back onto the specific MedicalState
# keys it owns. Each subgraph is independently testable and could be
# swapped into another graph entirely.
#
# Why not just add the compiled subgraph directly as a node (which
# works when parent and subgraph share one schema)? Because these two
# subgraphs run in PARALLEL with analyze_xray (see the fan-out below),
# and a schema-sharing subgraph re-emits every key it saw - including
# ones it never touched - as part of its own completion. Two sibling
# nodes both "writing" the same key in one superstep (one for real, one
# an unchanged passthrough) is exactly what triggers LangGraph's
# InvalidUpdateError ("Can receive only one value per step") - hit
# this for real while building this file, on the xray_analysis key.
# Giving each subgraph its own schema avoids the collision entirely.
#
# All three of medical_history, lab_analysis, and analyze_xray fan out
# from START in parallel (three independent edges from the same
# source run concurrently) and converge on diagnosis - a draft an LLM
# writes but does NOT get to act on. doctor_review is a real
# interrupt() gate (same pattern as human_approval_gate.py in this
# directory) - the graph genuinely pauses for a physician's real
# decision before prescription ever runs.
#
# IMPORTANT: this is a teaching demo of LangGraph mechanics (subgraphs,
# parallel fan-in, vision input, human-in-the-loop), NOT a real
# diagnostic tool. Nothing here should inform an actual medical
# decision.
# =====================================================================


class MedicalState(TypedDict):
    patient_id: str
    patient_data: str
    history_summary: str
    lab_data: str
    abnormal_flags: list[str]
    lab_summary: str
    xray_analysis: str
    diagnosis: str
    doctor_decision: str
    doctor_notes: str
    prescription: str


# =====================================================================
# SUBGRAPH 1: medical_history
#
# Deliberately its OWN narrow state (HistoryState), not MedicalState.
# Sharing the parent's exact schema would work fine if this subgraph
# ran alone, but it runs in PARALLEL with lab_analysis and analyze_xray
# (see the fan-out below) - a subgraph that shares the full parent
# schema re-emits every key it saw, including untouched ones, as part
# of its own completion. Two sibling nodes both "writing" the same
# key (one for real, one as an unchanged passthrough) in the same
# superstep is exactly what LangGraph's InvalidUpdateError guards
# against ("Can receive only one value per step"). Giving the subgraph
# its own schema - so it only ever touches patient_data/history_summary
# - avoids the collision entirely, and is also the officially
# documented pattern for subgraphs whose state differs from the
# parent's.
# =====================================================================
class HistoryState(TypedDict):
    patient_data: str
    history_summary: str


def load_patient_data(state: HistoryState) -> HistoryState:
    with open(PATIENT_DATA_PATH, encoding="utf-8") as f:
        data = f.read()
    print(f"[load_patient_data] loaded {PATIENT_DATA_PATH}")
    return {"patient_data": data}


def summarize_history(state: HistoryState) -> HistoryState:
    print("[summarize_history] summarizing patient history...")
    prompt = (
        "Summarize this patient's relevant medical history, current symptoms, and vitals for a "
        f"treating physician, in a few short sentences:\n\n{state['patient_data']}"
    )
    response = llm.invoke(prompt)
    return {"history_summary": response.content}


history_builder = StateGraph(HistoryState)
history_builder.add_node("load_patient_data", load_patient_data)
history_builder.add_node("summarize_history", summarize_history)
history_builder.add_edge(START, "load_patient_data")
history_builder.add_edge("load_patient_data", "summarize_history")
history_builder.add_edge("summarize_history", END)
medical_history_subgraph = history_builder.compile()


def run_medical_history(state: MedicalState) -> MedicalState:
    # Adapter: this subgraph needs no input from the parent (the data
    # path is fixed), so it starts from empty and only its OUTPUT keys
    # get merged back into MedicalState.
    result = medical_history_subgraph.invoke({"patient_data": "", "history_summary": ""})
    return {"patient_data": result["patient_data"], "history_summary": result["history_summary"]}


# =====================================================================
# SUBGRAPH 2: lab_analysis - same reasoning, its own LabState schema.
# =====================================================================
class LabState(TypedDict):
    lab_data: str
    abnormal_flags: list[str]
    lab_summary: str


def load_lab_results(state: LabState) -> LabState:
    with open(LAB_RESULTS_PATH, encoding="utf-8") as f:
        data = f.read()
    print(f"[load_lab_results] loaded {LAB_RESULTS_PATH}")
    return {"lab_data": data}


def flag_abnormal_results(state: LabState) -> LabState:
    # Fixed logic, no LLM - the CSV already has normal-range bounds, so
    # comparing to them is a plain numeric check, not something to ask
    # an LLM to (possibly mis-)judge.
    flags = []
    with open(LAB_RESULTS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result = float(row["Result"])
            low = float(row["Normal Range Low"])
            high = float(row["Normal Range High"])
            if result < low or result > high:
                direction = "high" if result > high else "low"
                flags.append(
                    f"{row['Test']}: {row['Result']} {row['Unit']} ({direction} - normal range {low}-{high})"
                )
    print(f"[flag_abnormal_results] {len(flags)} abnormal result(s) found")
    return {"abnormal_flags": flags}


def summarize_labs(state: LabState) -> LabState:
    print("[summarize_labs] interpreting lab panel...")
    flags_text = "\n".join(state["abnormal_flags"]) if state["abnormal_flags"] else "All values within normal range."
    prompt = (
        f"Interpret this lab panel for a treating physician, focusing on the abnormal findings "
        f"below:\n\nAbnormal findings:\n{flags_text}\n\nFull panel:\n{state['lab_data']}"
    )
    response = llm.invoke(prompt)
    return {"lab_summary": response.content}


lab_builder = StateGraph(LabState)
lab_builder.add_node("load_lab_results", load_lab_results)
lab_builder.add_node("flag_abnormal_results", flag_abnormal_results)
lab_builder.add_node("summarize_labs", summarize_labs)
lab_builder.add_edge(START, "load_lab_results")
lab_builder.add_edge("load_lab_results", "flag_abnormal_results")
lab_builder.add_edge("flag_abnormal_results", "summarize_labs")
lab_builder.add_edge("summarize_labs", END)
lab_analysis_subgraph = lab_builder.compile()


def run_lab_analysis(state: MedicalState) -> MedicalState:
    result = lab_analysis_subgraph.invoke({"lab_data": "", "abnormal_flags": [], "lab_summary": ""})
    return {
        "lab_data": result["lab_data"],
        "abnormal_flags": result["abnormal_flags"],
        "lab_summary": result["lab_summary"],
    }


# =====================================================================
# Parallel branch (not a subgraph): read the chest X-ray and send it
# straight to the vision-capable LLM.
# =====================================================================
def analyze_xray(state: MedicalState) -> MedicalState:
    if not os.path.exists(XRAY_PATH):
        print(f"[analyze_xray] no file at {XRAY_PATH} - skipping")
        return {"xray_analysis": "(no chest X-ray image was available for this encounter)"}

    print(f"[analyze_xray] reading {XRAY_PATH} and sending to vision LLM...")
    with open(XRAY_PATH, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    message = HumanMessage(content=[
        {
            "type": "text",
            "text": (
                "You are assisting a radiologist with a preliminary read. Describe any notable "
                "findings in this chest X-ray - look for signs of infection, consolidation, "
                "effusion, or other abnormalities. Be measured and explicit that this is a "
                "preliminary AI-assisted read only, not a final radiological diagnosis."
            ),
        },
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
    ])
    response = llm.invoke([message])
    return {"xray_analysis": response.content}


# =====================================================================
# diagnosis - fan-in point for medical_history, lab_analysis, and
# analyze_xray. Explicitly framed as a DRAFT, since a real decision
# only happens at doctor_review.
# =====================================================================
def diagnosis(state: MedicalState) -> MedicalState:
    print("[diagnosis] drafting preliminary diagnosis...")
    prompt = f"""You are an AI diagnostic assistant. Based on the following, propose a preliminary \
diagnosis and brief reasoning. This is a DRAFT for physician review, NOT a final medical decision.

Medical history summary:
{state['history_summary']}

Lab analysis summary:
{state['lab_summary']}

Chest X-ray analysis (preliminary AI read):
{state['xray_analysis']}"""
    response = llm.invoke(prompt)
    return {"diagnosis": response.content}


# =====================================================================
# doctor_review - a real interrupt() gate, same pattern as
# human_approval_gate.py: the graph genuinely pauses here and waits
# for actual console input from a human reviewer.
# =====================================================================
def doctor_review(state: MedicalState) -> MedicalState:
    pending = interrupt({
        "message": "Draft diagnosis ready for physician review.",
        "patient_id": state["patient_id"],
        "diagnosis_draft": state["diagnosis"],
    })
    return {"doctor_decision": pending["decision"], "doctor_notes": pending.get("notes", "")}


def ask_doctor_for_decision(pending: dict) -> dict:
    print("\nPAUSED - waiting for physician review:")
    print(f"  patient_id: {pending['patient_id']}")
    print(f"  draft diagnosis:\n{pending['diagnosis_draft']}\n")

    while True:
        raw = input("Approve or reject this draft diagnosis? [approve/reject]: ").strip().lower()
        if raw in ("approve", "reject", "approved", "rejected"):
            break
        print("Please type 'approve' or 'reject'.")

    notes = input("Physician notes (optional): ").strip()
    decision = "approved" if raw.startswith("approve") else "rejected"
    return {"decision": decision, "notes": notes}


# =====================================================================
# prescription - only writes a real treatment plan if the physician
# approved; otherwise records the rejection instead of prescribing
# anything.
# =====================================================================
def prescription(state: MedicalState) -> MedicalState:
    if state["doctor_decision"] != "approved":
        print("[prescription] diagnosis was rejected - no prescription issued")
        return {"prescription": f"No prescription issued - diagnosis rejected. Physician notes: {state['doctor_notes']}"}

    print("[prescription] writing treatment plan...")
    prompt = f"""Write a brief treatment plan based on this physician-approved diagnosis. Note the \
patient's existing medication and check for interactions before suggesting anything new.

Diagnosis:
{state['diagnosis']}

Physician notes: {state['doctor_notes']}

Patient's current medication (from medical history):
{state['patient_data']}"""
    response = llm.invoke(prompt)
    return {"prescription": response.content}


# =====================================================================
# Parent graph: three parallel branches (2 subgraphs + 1 plain node)
# fan into diagnosis, then a real human gate, then prescription.
# =====================================================================
graph = StateGraph(MedicalState)
graph.add_node("medical_history", run_medical_history)
graph.add_node("lab_analysis", run_lab_analysis)
graph.add_node("analyze_xray", analyze_xray)
graph.add_node("diagnosis", diagnosis)
graph.add_node("doctor_review", doctor_review)
graph.add_node("prescription", prescription)

graph.add_edge(START, "medical_history")
graph.add_edge(START, "lab_analysis")
graph.add_edge(START, "analyze_xray")
graph.add_edge("medical_history", "diagnosis")
graph.add_edge("lab_analysis", "diagnosis")
graph.add_edge("analyze_xray", "diagnosis")
graph.add_edge("diagnosis", "doctor_review")
graph.add_edge("doctor_review", "prescription")
graph.add_edge("prescription", END)

# A checkpointer is required for interrupt()/Command(resume=...).
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "patient-P10045"}}
    initial: MedicalState = {
        "patient_id": "P10045",
        "patient_data": "",
        "history_summary": "",
        "lab_data": "",
        "abnormal_flags": [],
        "lab_summary": "",
        "xray_analysis": "",
        "diagnosis": "",
        "doctor_decision": "",
        "doctor_notes": "",
        "prescription": "",
    }

    result = app.invoke(initial, config)

    if "__interrupt__" in result:
        pending = result["__interrupt__"][0].value
        human_response = ask_doctor_for_decision(pending)
        result = app.invoke(Command(resume=human_response), config)

    print("\n" + "=" * 70)
    print("FINAL PRESCRIPTION / TREATMENT PLAN")
    print("=" * 70)
    print(result["prescription"])
