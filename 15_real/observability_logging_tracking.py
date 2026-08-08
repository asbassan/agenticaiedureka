# pip install openai-agents python-dotenv

import os
import json
import time
import uuid
import logging
from datetime import datetime, timezone
from agents import Agent, Runner, RunHooks, function_tool
from dotenv import load_dotenv

load_dotenv(override=True)

# =====================================================================
# Observability for an agent, built from two plain, from-scratch pieces
# (no LangSmith/Langfuse here - see 14_advanced/09_langsmith and
# 9_general/observability for those) so the underlying idea is visible:
#
# 1. Structured (JSON-line) LOGGING - every lifecycle event becomes one
#    JSON object on one line, so logs are grep/parse-friendly instead
#    of free text.
# 2. Persistent TASK TRACKING - each agent run gets a task_id and is
#    saved as a record (status, duration, every tool call) to a JSON
#    file, so runs can be reviewed after the fact, not just watched
#    live in the console.
#
# Both hang off agents.RunHooks - the SDK's own lifecycle callbacks
# (on_agent_start/on_tool_start/on_tool_end/on_agent_end) - rather than
# manually wrapping every tool function.
# =====================================================================

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
TASKS_PATH = os.path.join(LOGS_DIR, "tasks.json")


# ---- 1. Structured logging: one JSON object per log line ----
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "task_id": getattr(record, "task_id", None),
            "event": record.getMessage(),
            "details": getattr(record, "details", {}),
        }
        return json.dumps(payload)


logger = logging.getLogger("agent_observability")
logger.setLevel(logging.INFO)
logger.propagate = False  # don't also run through the root logger's default handler
for handler in (logging.FileHandler(os.path.join(LOGS_DIR, "agent.log"), encoding="utf-8"), logging.StreamHandler()):
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)


def log_event(task_id: str, event: str, **details):
    logger.info(event, extra={"task_id": task_id, "details": details})


# ---- 2. Task tracking: an append-only, file-backed run history ----
def save_task_record(record: dict):
    history = json.load(open(TASKS_PATH, encoding="utf-8")) if os.path.exists(TASKS_PATH) else []
    history.append(record)
    with open(TASKS_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


class ObservabilityHooks(RunHooks):
    """Logs and records every step of one agent run via the SDK's own lifecycle events."""

    def __init__(self):
        self.task_id = str(uuid.uuid4())[:8]
        self.started_at = None
        self.steps: list[dict] = []

    async def on_agent_start(self, context, agent):
        self.started_at = time.time()
        log_event(self.task_id, "agent_start", agent=agent.name)

    async def on_tool_start(self, context, agent, tool):
        self._tool_started_at = time.time()
        log_event(self.task_id, "tool_start", tool=tool.name)

    async def on_tool_end(self, context, agent, tool, result):
        duration_sec = round(time.time() - self._tool_started_at, 3)
        self.steps.append({"tool": tool.name, "duration_sec": duration_sec})
        log_event(self.task_id, "tool_end", tool=tool.name, duration_sec=duration_sec)

    async def on_agent_end(self, context, agent, output):
        duration_sec = round(time.time() - self.started_at, 3)
        log_event(self.task_id, "agent_end", duration_sec=duration_sec, tool_calls=len(self.steps))
        save_task_record({
            "task_id": self.task_id,
            "agent": agent.name,
            "status": "completed",
            "duration_sec": duration_sec,
            "steps": self.steps,
        })


def run_task(agent: Agent, query: str) -> str:
    """Runs one agent task under observability, tracking failures as well as successes."""
    hooks = ObservabilityHooks()
    log_event(hooks.task_id, "task_received", query=query)

    try:
        result = Runner.run_sync(agent, query, hooks=hooks)
        return result.final_output
    except Exception as e:
        log_event(hooks.task_id, "task_failed", error=str(e))
        save_task_record({
            "task_id": hooks.task_id, "agent": agent.name, "status": "failed",
            "duration_sec": round(time.time() - (hooks.started_at or time.time()), 3),
            "steps": hooks.steps, "error": str(e),
        })
        raise


# ---- A small demo agent - the domain here doesn't matter, it's just something to observe ----
@function_tool
def get_current_date() -> str:
    """Return today's date."""
    return datetime.now().date().isoformat()


@function_tool
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression, e.g. '12 * 3 + 1'."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Invalid expression: {e}"


@function_tool
def get_mock_weather(city: str) -> str:
    """Return a mock current temperature for a city."""
    temps = {"new york": "72F", "london": "65F", "tokyo": "78F", "mumbai": "85F"}
    return temps.get(city.lower(), "70F")


assistant = Agent(
    name="ObservedAssistant",
    model="gpt-4o-mini",
    tools=[get_current_date, calculate, get_mock_weather],
    instructions="You are a helpful assistant. Use tools when needed. Be concise.",
)


def print_task_dashboard():
    if not os.path.exists(TASKS_PATH):
        print("No tracked tasks yet.")
        return

    history = json.load(open(TASKS_PATH, encoding="utf-8"))
    print(f"\n{'Task ID':<10}{'Status':<12}{'Duration':<10}{'Tool calls':<12}")
    for record in history:
        print(f"{record['task_id']:<10}{record['status']:<12}{record['duration_sec']:<10}{len(record['steps']):<12}")


if __name__ == "__main__":
    queries = [
        "What's the weather in Tokyo, and what's today's date?",
        "Calculate 45 * 3 + 12, then tell me the weather in Mumbai.",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print(f"Answer: {run_task(assistant, query)}")

    print_task_dashboard()
    print(f"\nFull logs: {os.path.join(LOGS_DIR, 'agent.log')}")
    print(f"Task history: {TASKS_PATH}")
