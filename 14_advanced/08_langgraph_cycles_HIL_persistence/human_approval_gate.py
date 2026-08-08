import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

# LLM output can contain characters outside Windows' default console
# codepage (cp1252) - reconfigure stdout to UTF-8 so printing doesn't crash.
sys.stdout.reconfigure(encoding="utf-8")

# =====================================================================
# Human-approval gate via LangGraph's interrupt()/Command(resume=...).
#
# evaluate_application applies fixed, deterministic rules. Clear-cut
# cases (strong or very weak applicants) are auto-decided and never
# involve a human. Borderline cases route to human_approval, which
# calls interrupt(...) - this PAUSES the graph and returns control to
# whoever called .invoke(), along with the exact payload passed to
# interrupt(). The graph's state at that point is durably checkpointed
# (via MemorySaver here; a real app would use a database-backed
# checkpointer), so the process can exit entirely and the pending
# application will still be there whenever a human resumes it with
# app.invoke(Command(resume=<their decision>), config) - re-entering
# human_approval as if interrupt() had just returned that value. This
# demo asks for that decision via real console input() rather than
# simulating one, so the pause is genuine, not just illustrated.
# =====================================================================


class LoanState(TypedDict):
    applicant_name: str
    age: int
    credit_score: int
    annual_income: float
    loan_amount: float
    auto_decision: str
    auto_reason: str
    human_decision: str
    human_notes: str
    final_status: str


# ---------------------------------------------------------------------
# Node 1: evaluate_application - fixed, rule-based logic. No LLM, no
# human - just thresholds a real underwriting rule set would use.
# ---------------------------------------------------------------------
def evaluate_application(state: LoanState) -> LoanState:
    debt_to_loan_ratio = state["loan_amount"] / state["annual_income"]

    if state["age"] < 18:
        return {"auto_decision": "rejected", "auto_reason": "Applicant is under 18."}

    if state["credit_score"] >= 720 and debt_to_loan_ratio <= 0.3:
        reason = (
            f"Strong credit score ({state['credit_score']}) and low loan-to-income "
            f"ratio ({debt_to_loan_ratio:.2f})."
        )
        return {"auto_decision": "approved", "auto_reason": reason}

    if state["credit_score"] < 550:
        reason = f"Credit score ({state['credit_score']}) is below the minimum threshold of 550."
        return {"auto_decision": "rejected", "auto_reason": reason}

    reason = (
        f"Credit score ({state['credit_score']}) and loan-to-income ratio "
        f"({debt_to_loan_ratio:.2f}) are borderline - requires manual review."
    )
    return {"auto_decision": "needs_review", "auto_reason": reason}


def route_after_evaluation(state: LoanState) -> str:
    return "human_approval" if state["auto_decision"] == "needs_review" else "finalize"


# ---------------------------------------------------------------------
# Node 2: human_approval - the interrupt point. Only reached for
# borderline applications.
# ---------------------------------------------------------------------
def human_approval(state: LoanState) -> LoanState:
    decision = interrupt({
        "message": "Manual review required for this loan application.",
        "applicant_name": state["applicant_name"],
        "age": state["age"],
        "credit_score": state["credit_score"],
        "annual_income": state["annual_income"],
        "loan_amount": state["loan_amount"],
        "system_reason": state["auto_reason"],
    })
    return {"human_decision": decision["decision"], "human_notes": decision.get("notes", "")}


# ---------------------------------------------------------------------
# Node 3: finalize - reads whichever decision actually settled this
# application, automatic or human.
# ---------------------------------------------------------------------
def finalize(state: LoanState) -> LoanState:
    if state["auto_decision"] in ("approved", "rejected"):
        status = state["auto_decision"]
        print(f"[finalize] automatic decision: {status.upper()} - {state['auto_reason']}")
    else:
        status = state["human_decision"]
        print(f"[finalize] human decision: {status.upper()} - {state['human_notes']}")
    return {"final_status": status}


# ---------------------------------------------------------------------
# Graph: START -> evaluate_application --needs_review--> human_approval -> finalize -> END
#                        \_______________auto-decided_______________/^
# ---------------------------------------------------------------------
graph = StateGraph(LoanState)
graph.add_node("evaluate_application", evaluate_application)
graph.add_node("human_approval", human_approval)
graph.add_node("finalize", finalize)

graph.add_edge(START, "evaluate_application")
graph.add_conditional_edges(
    "evaluate_application",
    route_after_evaluation,
    {"human_approval": "human_approval", "finalize": "finalize"},
)
graph.add_edge("human_approval", "finalize")
graph.add_edge("finalize", END)

# A checkpointer is REQUIRED for interrupt()/Command(resume=...) to work -
# it's what lets the graph "remember" where it paused. MemorySaver only
# persists in this process's memory; a real app would use a Postgres/
# SQLite checkpointer so a pending approval survives a restart.
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)


def ask_human_for_decision(pending: dict) -> dict:
    print("PAUSED - waiting for human approval:")
    for key, value in pending.items():
        print(f"  {key}: {value}")

    while True:
        raw = input("Approve or reject this loan? [approve/reject]: ").strip().lower()
        if raw in ("approve", "reject", "approved", "rejected"):
            break
        print("Please type 'approve' or 'reject'.")

    notes = input("Notes (optional): ").strip()
    decision = "approved" if raw.startswith("approve") else "rejected"
    return {"decision": decision, "notes": notes}


def run_application(thread_id: str, application: dict):
    config = {"configurable": {"thread_id": thread_id}}
    initial: LoanState = {
        "auto_decision": "",
        "auto_reason": "",
        "human_decision": "",
        "human_notes": "",
        "final_status": "",
        **application,
    }

    result = app.invoke(initial, config)

    if "__interrupt__" in result:
        pending = result["__interrupt__"][0].value
        human_response = ask_human_for_decision(pending)
        result = app.invoke(Command(resume=human_response), config)

    return result


if __name__ == "__main__":
    applications = [
        (
            "loan-1",
            {
                "applicant_name": "Priya Sharma",
                "age": 34,
                "credit_score": 780,
                "annual_income": 120_000,
                "loan_amount": 20_000,
            },
            # strong applicant - auto-approved, no human involved
        ),
        (
            "loan-2",
            {
                "applicant_name": "Sam Reyes",
                "age": 22,
                "credit_score": 480,
                "annual_income": 30_000,
                "loan_amount": 15_000,
            },
            # weak applicant - auto-rejected, no human involved
        ),
        (
            "loan-3",
            {
                "applicant_name": "Jordan Lee",
                "age": 41,
                "credit_score": 640,
                "annual_income": 60_000,
                "loan_amount": 25_000,
            },
            # borderline - will pause and prompt for a real decision
        ),
    ]

    for thread_id, application in applications:
        print("=" * 70)
        print(f"Processing {thread_id}: {application['applicant_name']}")
        result = run_application(thread_id, application)
        print(f"Final status: {result['final_status'].upper()}")
        print()
