"""Prototype 07: Agent learning loop — watch interactions, propose DAG updates.

The agent:
1. Handles requests live (no DAG match)
2. Tracks patterns (taxonomy)
3. Proposes DAG nodes when a pattern is frequent enough
4. Generates YAML config updates

This is the "agent-as-DAG-builder" core concept.
"""
import os
import asyncio
import json
import yaml
from dataclasses import dataclass, field
from collections import Counter

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "alanblount-demo")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

from google import genai


@dataclass
class InteractionRecord:
    query: str
    category: str
    response: str
    success: bool
    latency_ms: float


@dataclass
class TaxonomyTracker:
    """Tracks patterns in live agent interactions."""
    records: list[InteractionRecord] = field(default_factory=list)
    category_counts: Counter = field(default_factory=Counter)
    dag_creation_threshold: int = 3  # Lower for demo (would be 10+ in production)

    def record(self, interaction: InteractionRecord):
        self.records.append(interaction)
        self.category_counts[interaction.category] += 1

    def get_dag_candidates(self) -> list[dict]:
        """Return categories that have enough data for DAG creation."""
        candidates = []
        for category, count in self.category_counts.items():
            if count >= self.dag_creation_threshold:
                # Calculate success rate
                cat_records = [r for r in self.records if r.category == category]
                success_rate = sum(1 for r in cat_records if r.success) / len(cat_records)
                avg_latency = sum(r.latency_ms for r in cat_records) / len(cat_records)
                candidates.append({
                    "category": category,
                    "count": count,
                    "success_rate": success_rate,
                    "avg_latency_ms": avg_latency,
                })
        return candidates


class AgentWithLearning:
    """An agent that handles requests live and learns to build DAGs."""

    def __init__(self):
        self.client = genai.Client(
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION"),
        )
        self.model = "gemini-3.1-flash-lite"
        self.tracker = TaxonomyTracker()
        self.existing_dags = {}  # category → YAML config

    async def handle_live(self, query: str) -> str:
        """Handle a request using live LLM reasoning."""
        import time
        start = time.time()

        # Step 1: Classify
        classify_resp = self.client.models.generate_content(
            model=self.model,
            contents=[{
                "role": "user",
                "parts": [{"text": f"Classify this request in one word (billing/technical/general/returns): {query}"}],
            }],
        )
        category = classify_resp.candidates[0].content.parts[0].text.strip().lower()
        for c in ["billing", "technical", "returns", "general"]:
            if c in category:
                category = c
                break
        else:
            category = "general"

        # Step 2: Handle
        response = self.client.models.generate_content(
            model=self.model,
            contents=[{
                "role": "user",
                "parts": [{"text": f"You are a {category} specialist. Help briefly with: {query}"}],
            }],
        )
        result = response.candidates[0].content.parts[0].text

        latency = (time.time() - start) * 1000

        # Record the interaction
        self.tracker.record(InteractionRecord(
            query=query,
            category=category,
            response=result[:100],
            success=True,  # Simplified — would check actual outcome
            latency_ms=latency,
        ))

        return result, category, latency

    async def propose_dag_update(self, category: str) -> dict:
        """Use LLM to propose a DAG config for a frequently-seen category."""
        # Get example interactions for this category
        examples = [r for r in self.tracker.records if r.category == category][:5]
        example_text = "\n".join(
            f"- Query: {e.query}\n  Response: {e.response[:80]}"
            for e in examples
        )

        prompt = f"""You are a workflow designer. Based on these customer service interactions
in the '{category}' category, design a workflow DAG as YAML.

Example interactions:
{example_text}

Generate a YAML workflow with:
1. A classification/validation step
2. The main handling step with a focused instruction
3. A response validation step

Output ONLY valid YAML, no explanation. Use this format:
nodes:
  - id: validate_input
    type: agent
    model: gemini-3.1-flash-lite
    instruction: "..."
  - id: handle_{category}
    type: agent
    model: gemini-3.1-flash-lite
    instruction: "..."
  - id: validate_response
    type: agent
    model: gemini-3.1-flash-lite
    instruction: "..."
edges:
  - from: START
    to: validate_input
  - from: validate_input
    to: handle_{category}
  - from: handle_{category}
    to: validate_response"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
        )
        yaml_text = response.candidates[0].content.parts[0].text

        # Try to parse the YAML
        # Strip markdown code blocks if present
        if "```" in yaml_text:
            lines = yaml_text.split("\n")
            in_block = False
            block_lines = []
            for line in lines:
                if line.strip().startswith("```"):
                    if in_block:
                        break
                    in_block = True
                    continue
                if in_block:
                    block_lines.append(line)
            if block_lines:
                yaml_text = "\n".join(block_lines)

        try:
            dag_config = yaml.safe_load(yaml_text)
            return {
                "category": category,
                "config": dag_config,
                "yaml": yaml_text,
                "based_on_examples": len(examples),
            }
        except yaml.YAMLError as e:
            return {
                "category": category,
                "error": f"Failed to parse generated YAML: {e}",
                "raw": yaml_text[:500],
            }

    async def check_and_propose(self) -> list[dict]:
        """Check taxonomy and propose DAGs for frequent categories."""
        candidates = self.tracker.get_dag_candidates()
        proposals = []
        for candidate in candidates:
            if candidate["category"] not in self.existing_dags:
                print(f"\n  📊 Category '{candidate['category']}' seen {candidate['count']} times "
                      f"(success: {candidate['success_rate']:.0%}, "
                      f"avg latency: {candidate['avg_latency_ms']:.0f}ms)")
                print(f"  🔧 Proposing DAG for '{candidate['category']}'...")

                proposal = await self.propose_dag_update(candidate["category"])
                proposals.append(proposal)

                if "config" in proposal:
                    print(f"  ✅ DAG proposed with {len(proposal['config'].get('nodes', []))} nodes")
                else:
                    print(f"  ❌ Failed: {proposal.get('error', 'unknown')[:80]}")
        return proposals


async def main():
    agent = AgentWithLearning()

    # Simulate a stream of customer requests
    queries = [
        # Billing cluster
        "I was charged twice for my subscription",
        "Can I get a refund for last month?",
        "Why is my bill higher than usual?",
        "I need to update my payment method",
        # Technical cluster
        "My internet keeps disconnecting",
        "The app crashes when I open settings",
        "I can't log into my account",
        # General
        "What are your business hours?",
        "How do I contact support?",
    ]

    print("=== LIVE AGENT HANDLING ===\n")
    for query in queries:
        result, category, latency = await agent.handle_live(query)
        print(f"[{category}] ({latency:.0f}ms) {query}")
        print(f"  → {result[:80]}\n")

    # Check taxonomy and propose DAGs
    print("\n=== TAXONOMY CHECK ===")
    print(f"Categories seen: {dict(agent.tracker.category_counts)}")

    proposals = await agent.check_and_propose()

    if proposals:
        print("\n=== PROPOSED DAGS ===\n")
        for p in proposals:
            if "config" in p:
                print(f"Category: {p['category']}")
                print(f"Based on {p['based_on_examples']} examples")
                print(f"YAML:\n{p['yaml'][:500]}")
                print()
    else:
        print("\nNo categories met the DAG creation threshold yet.")


if __name__ == "__main__":
    asyncio.run(main())
