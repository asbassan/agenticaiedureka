import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import subprocess
import sys
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# LLM output can contain characters outside Windows' default console
# codepage (cp1252) - reconfigure stdout to UTF-8 so printing doesn't crash.
# line_buffering=True matters here specifically: os._exit() (used below
# to simulate a hard crash) skips Python's normal stdout flush, so any
# buffered print() output would otherwise vanish silently whenever this
# script's output is redirected to a file instead of a live terminal.
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# =====================================================================
# SQLite checkpointing and crash recovery.
#
# The sibling files in this directory (order_processing_state_machine.py,
# human_approval_gate.py) use MemorySaver, which only persists while the
# Python process is alive. This file shows the harder, more valuable
# guarantee: a multi-step pipeline that survives the PROCESS ITSELF
# dying mid-execution, and resumes without re-running steps that already
# completed - the realistic pain point being double-charging a customer
# on restart because a naive retry re-ran everything from scratch.
#
# Resuming after a crash uses the exact same call LangGraph uses to
# resume after an interrupt(): app.invoke(None, config). There is no
# separate "crashed" vs "interrupted" concept - LangGraph only checks
# whether a thread has unfinished checkpoint state. Passing a FRESH,
# non-None input on a thread with incomplete history instead DISCARDS
# that pending progress and starts over from START - so resuming
# correctly means passing None, not the original order details again.
#
# durability="sync" is passed explicitly on every invoke() below. The
# default, durability="async", persists a completed node's checkpoint
# WHILE the next step is already starting - there's a real window where
# a node has returned but its write hasn't reached SQLite yet if the
# process dies right then, which would make that step needlessly
# re-run. durability="sync" persists before the next step begins,
# which is what makes "a completed step never re-runs" a guarantee
# here rather than a race.
# =====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "order_pipeline_checkpoints.sqlite")
THREAD_ID = "order-42"

# --crash-before and --worker are parsed once at process startup into
# plain module-level variables - they deliberately do NOT go into graph
# state, so they can't themselves get checkpointed or replayed on
# resume. --worker marks "just run one pass, don't orchestrate" - it's
# how run_auto_demo() below invokes this same script as a subprocess
# without recursing back into run_auto_demo() itself.
CRASH_BEFORE = None
if "--crash-before" in sys.argv:
    CRASH_BEFORE = sys.argv[sys.argv.index("--crash-before") + 1]

WORKER_MODE = "--worker" in sys.argv


def maybe_crash(node_name: str) -> None:
    if CRASH_BEFORE == node_name:
        print(f"\n*** SIMULATING A HARD CRASH before '{node_name}' (os._exit, no cleanup) ***\n")
        os._exit(1)


class OrderState(TypedDict):
    order_id: str
    item: str
    quantity: int
    completed_steps: list[str]


# ---------------------------------------------------------------------
# Four linear nodes. maybe_crash() runs FIRST in every node - checking
# at the START of a node (not the end of the previous one) guarantees
# any simulated crash lands cleanly between two supersteps, so the
# prior node's completion is already durably checkpointed.
# ---------------------------------------------------------------------
def receive_order(state: OrderState) -> OrderState:
    maybe_crash("receive_order")
    print(f"[receive_order] order {state['order_id']} received: {state['quantity']}x {state['item']}")
    return {"completed_steps": state["completed_steps"] + ["receive_order"]}


def reserve_inventory(state: OrderState) -> OrderState:
    maybe_crash("reserve_inventory")
    print(f"[reserve_inventory] reserved {state['quantity']}x {state['item']}")
    return {"completed_steps": state["completed_steps"] + ["reserve_inventory"]}


def charge_payment(state: OrderState) -> OrderState:
    maybe_crash("charge_payment")
    print(f"[charge_payment] charged payment for order {state['order_id']}")
    return {"completed_steps": state["completed_steps"] + ["charge_payment"]}


def send_confirmation(state: OrderState) -> OrderState:
    maybe_crash("send_confirmation")
    print(f"[send_confirmation] confirmation sent for order {state['order_id']}")
    return {"completed_steps": state["completed_steps"] + ["send_confirmation"]}


graph = StateGraph(OrderState)
graph.add_node("receive_order", receive_order)
graph.add_node("reserve_inventory", reserve_inventory)
graph.add_node("charge_payment", charge_payment)
graph.add_node("send_confirmation", send_confirmation)

graph.add_edge(START, "receive_order")
graph.add_edge("receive_order", "reserve_inventory")
graph.add_edge("reserve_inventory", "charge_payment")
graph.add_edge("charge_payment", "send_confirmation")
graph.add_edge("send_confirmation", END)


def run_pipeline_once() -> None:
    """One pass: start the order fresh, resume it, or report it's already done -
    whichever applies given what's currently checkpointed in DB_PATH."""
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        app = graph.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": THREAD_ID}}

        existing = app.get_state(config).values
        completed_so_far = existing.get("completed_steps", []) if existing else []

        if completed_so_far and len(completed_so_far) == 4:
            print(f"Order {THREAD_ID} was already fully processed: {completed_so_far}")
        elif completed_so_far:
            print(f"Resuming order {THREAD_ID} - already completed: {completed_so_far}")
            result = app.invoke(None, config, durability="sync")
            print(f"\nFinal completed steps: {result['completed_steps']}")
        else:
            print(f"Starting new order {THREAD_ID}")
            initial: OrderState = {
                "order_id": THREAD_ID,
                "item": "Wireless Mouse",
                "quantity": 2,
                "completed_steps": [],
            }
            result = app.invoke(initial, config, durability="sync")
            print(f"\nFinal completed steps: {result['completed_steps']}")


def run_auto_demo() -> None:
    """Self-orchestrating demo: resets the checkpoint DB, then runs this same
    script as TWO separate OS processes - one that genuinely crashes partway
    through, one that resumes it - so the crash is real, not just described,
    and running the file with no arguments is enough to see the whole point."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    print("=" * 70)
    print("PHASE 1: run the pipeline in a subprocess that crashes before 'charge_payment'")
    print("=" * 70)
    phase1 = subprocess.run([sys.executable, __file__, "--worker", "--crash-before", "charge_payment"])
    print(f"\n(subprocess exited with code {phase1.returncode} - a real crash, not simulated in-process)\n")

    print("=" * 70)
    print("PHASE 2: run the SAME script again as a NEW subprocess, no crash flag")
    print("It should resume exactly where phase 1 left off, not restart from scratch.")
    print("=" * 70)
    phase2 = subprocess.run([sys.executable, __file__, "--worker"])
    print(f"\n(subprocess exited with code {phase2.returncode})")


if __name__ == "__main__":
    if WORKER_MODE:
        run_pipeline_once()
    else:
        run_auto_demo()
