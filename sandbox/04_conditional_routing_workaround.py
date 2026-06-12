"""Prototype 04: Conditional routing workaround.

FINDING: ADK 2.0 Workflow's routing map syntax (node, {value: target})
accepts the edges but doesn't dispatch to the target nodes. This appears
to be a bug or unimplemented feature in the current ADK 2.0 release.

WORKAROUND: Use a SequentialAgent with transfer_to_agent for routing,
or use a single LlmAgent with sub_agents for delegation.
"""
import os
import asyncio
import sys

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "alanblount-demo")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


async def test_sub_agent_routing():
    """Use LlmAgent with sub_agents for conditional routing.

    The parent agent classifies and delegates to the right sub-agent
    using ADK's built-in transfer_to_agent tool.
    """
    billing = LlmAgent(
        name="billing_handler",
        model="gemini-3.1-flash-lite",
        instruction="You are a billing specialist. Help with the billing issue. Be concise.",
    )

    technical = LlmAgent(
        name="technical_handler",
        model="gemini-3.1-flash-lite",
        instruction="You are a tech support specialist. Help with the technical issue. Be concise.",
    )

    general = LlmAgent(
        name="general_handler",
        model="gemini-3.1-flash-lite",
        instruction="You are a general support agent. Help with the request. Be concise.",
    )

    router = LlmAgent(
        name="router",
        model="gemini-3.1-flash-lite",
        instruction="""You are a customer service router. Classify the customer's request and
transfer to the appropriate handler:
- billing_handler: for billing, payment, charges, subscription issues
- technical_handler: for technical problems, bugs, connectivity, errors
- general_handler: for everything else

Use the transfer_to_agent tool to route the request.""",
        sub_agents=[billing, technical, general],
    )

    session_service = InMemorySessionService()
    runner = Runner(
        agent=router, app_name="routing_test", session_service=session_service
    )

    test_queries = [
        "I was charged twice for my subscription",
        "My internet keeps disconnecting",
        "What are your business hours?",
    ]

    for query in test_queries:
        session = await session_service.create_session(
            app_name="routing_test", user_id="user1"
        )
        print(f"\nQuery: {query}")
        async for event in runner.run_async(
            user_id="user1",
            session_id=session.id,
            new_message=types.Content(
                parts=[types.Part(text=query)], role="user"
            ),
        ):
            if event.content and event.content.parts:
                for p in event.content.parts:
                    if hasattr(p, "text") and p.text:
                        print(f"  [{event.author}]: {p.text[:120]}")
            if event.is_final_response():
                break


if __name__ == "__main__":
    asyncio.run(test_sub_agent_routing())
