import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# =====================================================================
# Minimal subgraph illustration - no LLM, no real data, just mock
# one-line returns so the MECHANISM (not the content) is what stands
# out: each subgraph is its own tiny, independently compiled
# StateGraph with its OWN state schema, wired into a parent graph via
# a thin adapter function. See medical_diagnosis_subgraphs.py in this
# same directory for a full real-world version of this exact pattern.
# =====================================================================


# --- Subgraph 1: weather check, one node, mock return ---
class WeatherState(TypedDict):
    weather_report: str


def check_weather(state: WeatherState) -> WeatherState:
    return {"weather_report": "30 degrees celsius and sunny"}


weather_builder = StateGraph(WeatherState)
weather_builder.add_node("check_weather", check_weather)
weather_builder.add_edge(START, "check_weather")
weather_builder.add_edge("check_weather", END)
weather_subgraph = weather_builder.compile()


# --- Subgraph 2: traffic check, same shape, different mock node ---
class TrafficState(TypedDict):
    traffic_report: str


def check_traffic(state: TrafficState) -> TrafficState:
    return {"traffic_report": "15 minutes, light traffic"}


traffic_builder = StateGraph(TrafficState)
traffic_builder.add_node("check_traffic", check_traffic)
traffic_builder.add_edge(START, "check_traffic")
traffic_builder.add_edge("check_traffic", END)
traffic_subgraph = traffic_builder.compile()


# --- Parent graph: an adapter per subgraph, then combine ---
class CommuteState(TypedDict):
    weather_report: str
    traffic_report: str
    commute_advisory: str


def run_weather_subgraph(state: CommuteState) -> CommuteState:
    result = weather_subgraph.invoke({"weather_report": ""})
    return {"weather_report": result["weather_report"]}


def run_traffic_subgraph(state: CommuteState) -> CommuteState:
    result = traffic_subgraph.invoke({"traffic_report": ""})
    return {"traffic_report": result["traffic_report"]}


def combine(state: CommuteState) -> CommuteState:
    return {"commute_advisory": f"Weather: {state['weather_report']} | Traffic: {state['traffic_report']}"}


graph = StateGraph(CommuteState)
graph.add_node("weather_subgraph", run_weather_subgraph)
graph.add_node("traffic_subgraph", run_traffic_subgraph)
graph.add_node("combine", combine)

graph.add_edge(START, "weather_subgraph")
graph.add_edge(START, "traffic_subgraph")
graph.add_edge("weather_subgraph", "combine")
graph.add_edge("traffic_subgraph", "combine")
graph.add_edge("combine", END)

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({"weather_report": "", "traffic_report": "", "commute_advisory": ""})
    print(result)
