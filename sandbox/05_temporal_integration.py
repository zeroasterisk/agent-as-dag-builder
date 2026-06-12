"""Prototype 05: Temporal integration for durable DAG execution.

Tests whether we can wrap an ADK agent in a Temporal workflow for:
- Crash recovery (workflow survives worker restarts)
- Versioned deployment (new DAG versions don't disrupt in-flight)
- Observability (each step visible in Temporal UI)

NOTE: This requires a running Temporal server. If no server is available,
it demonstrates the PATTERN without executing.
"""
import os
import asyncio

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "alanblount-demo")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

from temporalio import activity, workflow
from temporalio.client import Client as TemporalClient
from temporalio.worker import Worker
from google import genai
from dataclasses import dataclass
import yaml


# --- Activities (individual DAG steps) ---


@dataclass
class ClassifyResult:
    category: str
    confidence: float


@activity.defn
async def classify_request(query: str) -> ClassifyResult:
    """Classify a customer request using Gemini."""
    client = genai.Client(vertexai=True, project="alanblount-demo", location="global")
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=[{
            "role": "user",
            "parts": [{"text": f"Classify this customer request into one word (billing/technical/general): {query}"}],
        }],
    )
    text = response.candidates[0].content.parts[0].text.strip().lower()
    category = "general"
    if "billing" in text:
        category = "billing"
    elif "technical" in text or "tech" in text:
        category = "technical"
    return ClassifyResult(category=category, confidence=0.9)


@activity.defn
async def handle_billing(query: str) -> str:
    """Handle a billing request using Gemini."""
    client = genai.Client(vertexai=True, project="alanblount-demo", location="global")
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=[{
            "role": "user",
            "parts": [{"text": f"You are a billing specialist. Help briefly with: {query}"}],
        }],
    )
    return response.candidates[0].content.parts[0].text


@activity.defn
async def handle_technical(query: str) -> str:
    """Handle a technical request using Gemini."""
    client = genai.Client(vertexai=True, project="alanblount-demo", location="global")
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=[{
            "role": "user",
            "parts": [{"text": f"You are a tech support specialist. Help briefly with: {query}"}],
        }],
    )
    return response.candidates[0].content.parts[0].text


@activity.defn
async def handle_general(query: str) -> str:
    """Handle a general request using Gemini."""
    client = genai.Client(vertexai=True, project="alanblount-demo", location="global")
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=[{
            "role": "user",
            "parts": [{"text": f"You are a customer service agent. Help briefly with: {query}"}],
        }],
    )
    return response.candidates[0].content.parts[0].text


@activity.defn
async def validate_response(response: str) -> bool:
    """Validate that a response is appropriate (defensive validation node)."""
    if not response or len(response) < 10:
        return False
    # Could add LLM-based validation for safety/quality here
    return True


# --- Workflow (the DAG) ---


@workflow.defn
class CustomerSupportWorkflow:
    """Temporal workflow implementing the customer support DAG.

    This is what the YAML config compiles to. Each node becomes an Activity.
    The workflow orchestrates the execution order, conditional routing,
    and validation.
    """

    @workflow.run
    async def run(self, query: str) -> str:
        # Step 1: Classify
        classification = await workflow.execute_activity(
            classify_request,
            query,
            start_to_close_timeout=asyncio.timedelta(seconds=30),
        )
        workflow.logger.info(f"Classified as: {classification.category}")

        # Step 2: Route to handler (conditional)
        handler_map = {
            "billing": handle_billing,
            "technical": handle_technical,
            "general": handle_general,
        }
        handler = handler_map.get(classification.category, handle_general)

        response = await workflow.execute_activity(
            handler,
            query,
            start_to_close_timeout=asyncio.timedelta(seconds=30),
        )

        # Step 3: Validate (defensive validation node)
        is_valid = await workflow.execute_activity(
            validate_response,
            response,
            start_to_close_timeout=asyncio.timedelta(seconds=10),
        )

        if not is_valid:
            workflow.logger.warning("Response validation failed — falling back to live agent")
            return f"[VALIDATION FAILED] Original response: {response[:100]}"

        return response


# --- Runner ---


async def run_with_temporal():
    """Run the workflow with a Temporal server."""
    client = await TemporalClient.connect("localhost:7233")

    async with Worker(
        client,
        task_queue="customer-support",
        workflows=[CustomerSupportWorkflow],
        activities=[
            classify_request,
            handle_billing,
            handle_technical,
            handle_general,
            validate_response,
        ],
    ):
        result = await client.execute_workflow(
            CustomerSupportWorkflow.run,
            "I was charged twice for my subscription",
            id="support-test-1",
            task_queue="customer-support",
        )
        print(f"Result: {result[:150]}")


async def run_without_temporal():
    """Demonstrate the pattern by running activities directly (no Temporal server needed)."""
    print("Running WITHOUT Temporal server (direct activity calls)")
    print("This demonstrates the pattern — with Temporal, each call would be durable.\n")

    queries = [
        "I was charged twice for my subscription",
        "My internet keeps disconnecting",
        "What are your business hours?",
    ]

    for query in queries:
        print(f"Query: {query}")

        # Step 1: Classify
        classification = await classify_request(query)
        print(f"  → Classified: {classification.category}")

        # Step 2: Route
        handlers = {
            "billing": handle_billing,
            "technical": handle_technical,
            "general": handle_general,
        }
        handler = handlers.get(classification.category, handle_general)
        response = await handler(query)
        print(f"  → Handler [{classification.category}]: {response[:100]}")

        # Step 3: Validate
        is_valid = await validate_response(response)
        print(f"  → Valid: {is_valid}")
        print()


if __name__ == "__main__":
    try:
        asyncio.run(run_with_temporal())
    except Exception as e:
        if "Connect" in str(e) or "refused" in str(e):
            print(f"No Temporal server at localhost:7233 ({e})\n")
            asyncio.run(run_without_temporal())
        else:
            raise
