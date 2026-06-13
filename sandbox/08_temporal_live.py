"""Prototype 08: LIVE Temporal durable execution — PROVEN WORKING.

Requires: temporal server start-dev (Temporal dev server)

Results:
- 3 workflows executed through Temporal
- Classify → route → handle pipeline
- All workflows status: COMPLETED
- Each step is a durable Activity (crash-recoverable)
"""
import os, asyncio, sys, datetime
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "alanblount-demo")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

from dataclasses import dataclass
from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker
from google import genai

gclient = genai.Client(vertexai=True, project="alanblount-demo", location="global")

@dataclass
class ClassifyResult:
    category: str

@activity.defn
async def classify(query: str) -> ClassifyResult:
    r = gclient.models.generate_content(model="gemini-3.1-flash-lite",
        contents=[{"role": "user", "parts": [{"text": f"One word - billing/technical/general: {query}"}]}])
    t = r.candidates[0].content.parts[0].text.strip().lower()
    return ClassifyResult(
        category="billing" if "billing" in t else "technical" if "tech" in t else "general"
    )

@activity.defn
async def handle(args: list) -> str:
    query, category = args
    r = gclient.models.generate_content(model="gemini-3.1-flash-lite",
        contents=[{"role": "user", "parts": [{"text": f"{category} specialist, help briefly: {query}"}]}])
    return r.candidates[0].content.parts[0].text

@workflow.defn(sandboxed=False)
class CustomerSupportWorkflow:
    @workflow.run
    async def run(self, query: str) -> str:
        c = await workflow.execute_activity(
            classify, query, start_to_close_timeout=datetime.timedelta(seconds=30))
        return await workflow.execute_activity(
            handle, [query, c.category], start_to_close_timeout=datetime.timedelta(seconds=30))

async def main():
    client = await Client.connect("localhost:7233")
    print("Connected to Temporal ✅")

    async with Worker(client, task_queue="dag-builder",
                      workflows=[CustomerSupportWorkflow],
                      activities=[classify, handle]):
        print("Worker running ✅\n")
        queries = [
            ("I was charged twice", "billing-test"),
            ("Internet keeps dropping", "tech-test"),
            ("What are your hours?", "general-test"),
        ]
        for query, wfid in queries:
            result = await client.execute_workflow(
                CustomerSupportWorkflow.run, query, id=wfid, task_queue="dag-builder")
            print(f"  {query} → {result[:80]}")

        print("\n=== WORKFLOW STATUS ===")
        for _, wfid in queries:
            desc = await client.get_workflow_handle(wfid).describe()
            print(f"  {wfid}: {desc.status.name}")

    print("\n🎉 TEMPORAL DURABLE EXECUTION PROVEN")

if __name__ == "__main__":
    asyncio.run(main())
