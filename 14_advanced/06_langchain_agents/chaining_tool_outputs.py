import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import re
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor

load_dotenv(override=True)

# Tool output can contain characters outside Windows' default console
# codepage (cp1252) - reconfigure stdout to UTF-8 so printing doesn't crash.
sys.stdout.reconfigure(encoding="utf-8")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# =====================================================================
# Three tools, each one's OUTPUT format matching the next one's expected
# INPUT format on purpose - geocode_city's "lat,lon" string is exactly
# what get_current_weather's docstring says it wants, and
# get_current_weather's "temperature=..C, wind_speed=..km/h" string is
# exactly what recommend_clothing's docstring says it wants. Free
# Open-Meteo API, no key required.
# =====================================================================

def _clean(text: str) -> str:
    # ReAct agents often wrap Action Input in quotes like a Python string
    # literal (e.g. 'Tokyo' instead of Tokyo) - the parser passes that
    # raw text straight through, so strip it here rather than let every
    # tool call fail on a literal quote character.
    return text.strip().strip("'\"").strip()

@tool
def geocode_city(city: str) -> str:
    """Look up a city's coordinates. Input: a city name, e.g. Tokyo.
    Returns a string 'latitude,longitude' - feed this directly into
    get_current_weather."""
    city = _clean(city)
    resp = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    ).json()
    results = resp.get("results", [])
    if not results:
        return f"Could not find coordinates for {city}"
    loc = results[0]
    return f"{loc['latitude']},{loc['longitude']}"

@tool
def get_current_weather(lat_lon: str) -> str:
    """Get current temperature and wind speed for a location.
    Input: 'latitude,longitude', exactly as returned by geocode_city
    (e.g. '35.6895,139.6917'). Returns a string like
    'temperature=18.2C, wind_speed=12.4km/h' - feed this directly into
    recommend_clothing."""
    lat_str, lon_str = _clean(lat_lon).split(",")[:2]
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat_str.strip(),
            "longitude": lon_str.strip(),
            "current": "temperature_2m,wind_speed_10m",
            "timezone": "auto",
        },
        timeout=10,
    ).json()
    current = resp.get("current", {})
    return f"temperature={current.get('temperature_2m')}C, wind_speed={current.get('wind_speed_10m')}km/h"

@tool
def recommend_clothing(weather_summary: str) -> str:
    """Recommend what to wear given a weather summary string, exactly as
    returned by get_current_weather (e.g. 'temperature=18.2C,
    wind_speed=12.4km/h'). Pure logic, no external call."""
    temp_match = re.search(r"temperature=(-?\d+\.?\d*)", weather_summary)
    wind_match = re.search(r"wind_speed=(-?\d+\.?\d*)", weather_summary)
    temp = float(temp_match.group(1)) if temp_match else None
    wind = float(wind_match.group(1)) if wind_match else None

    if temp is None:
        return "Could not parse a temperature from that weather summary."

    if temp < 10:
        outfit = "a warm coat and gloves"
    elif temp < 18:
        outfit = "a light jacket"
    elif temp < 27:
        outfit = "a t-shirt, no jacket needed"
    else:
        outfit = "light, breathable clothing"

    if wind is not None and wind > 25:
        outfit += " (and something windproof - it's quite windy)"

    return f"Recommended outfit: {outfit}."

tools = [geocode_city, get_current_weather, recommend_clothing]
question = "What should I wear outside today in Tokyo?"

# =====================================================================
# 1. AGENT-DRIVEN chaining
# The ReAct agent itself reads each Observation (plain text) and decides
# what string to write as the next Action Input - there is no code
# anywhere that says "take geocode_city's output and pass it to
# get_current_weather". The LLM infers that from the tool descriptions
# and the shape of what came back. Flexible, but costs an LLM call per
# step and can misparse an observation if its format is ever ambiguous.
# =====================================================================
print("=" * 70)
print("1. AGENT-DRIVEN chaining (ReAct)")
print("=" * 70)

REACT_PROMPT = PromptTemplate.from_template("""\
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

agent = create_react_agent(llm, tools, REACT_PROMPT)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=6)
agent_result = agent_executor.invoke({"input": question})

# =====================================================================
# 2. DETERMINISTIC chaining
# The EXACT same three tools, the EXACT same data flow - but hardcoded by
# the developer instead of decided by an LLM. No reasoning, no tokens
# spent deciding "what's next", nothing that can misinterpret an
# observation. The tradeoff: this only works because we already know the
# pipeline shape in advance. It can't handle a question that needs a
# different sequence of tools, or a different number of steps.
# =====================================================================
print("\n" + "=" * 70)
print("2. DETERMINISTIC chaining (hardcoded, no agent)")
print("=" * 70)

coords = geocode_city.invoke({"city": "Tokyo"})
print(f"geocode_city output:        {coords}")

weather = get_current_weather.invoke({"lat_lon": coords})
print(f"get_current_weather output: {weather}")

recommendation = recommend_clothing.invoke({"weather_summary": weather})
print(f"recommend_clothing output:  {recommendation}")

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Agent-driven:  {agent_result['output']}")
print(f"Deterministic: {recommendation}")
