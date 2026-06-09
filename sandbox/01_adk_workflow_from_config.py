"""Sandbox: Build ADK WorkflowAgent from a config dict.

FINDING: ADK 2.0 Workflow can be constructed programmatically with nodes + edges.
This proves the YAML-to-ADK-Workflow path is feasible.
"""
import os, asyncio
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "alanblount-demo")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

from google.adk.agents import LlmAgent
from google.adk.workflow import Workflow
from google.adk.workflow._base_node import START
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

config = {
    "name": "math_workflow",
    "nodes": [
        {"id": "classify", "type": "agent", "model": "gemini-3.1-flash-lite",
         "instruction": "Classify the math operation. Reply: addition/subtraction/multiplication/division."},
        {"id": "solve", "type": "agent", "model": "gemini-3.1-flash-lite",
         "instruction": "Solve the math problem. Reply with just the number."},
    ],
    "edges": [
        {"from": "START", "to": "classify"},
        {"from": "classify", "to": "solve"},
    ],
}

nodes = {}
for n in config["nodes"]:
    nodes[n["id"]] = LlmAgent(name=n["id"], model=n["model"], instruction=n["instruction"])

edges = [(START if e["from"] == "START" else nodes[e["from"]], nodes[e["to"]]) for e in config["edges"]]
workflow = Workflow(name=config["name"], nodes=list(nodes.values()), edges=edges)
print(f"Workflow built from config: {config['name']} ✅")

session_service = InMemorySessionService()
runner = Runner(agent=workflow, app_name="test", session_service=session_service)

async def run():
    session = await session_service.create_session(app_name="test", user_id="u1")
    async for event in runner.run_async(
        user_id="u1", session_id=session.id,
        new_message=types.Content(parts=[types.Part(text="What is 15 * 23?")], role="user"),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            print("Result:", event.content.parts[0].text)
            break

if __name__ == "__main__":
    asyncio.run(run())
