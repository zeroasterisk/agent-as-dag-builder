"""Prototype 06: YAML config → Temporal Workflow compiler.

Reads a YAML DAG config and generates a Temporal workflow that:
- Creates an activity for each node
- Routes conditionally based on edge conditions
- Includes validation nodes before/after each step
- Supports feature flags via session state
"""
import os
import asyncio
import yaml

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "alanblount-demo")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

from google import genai
from temporalio import activity
from dataclasses import dataclass


# --- Generic LLM activity ---

@dataclass
class LLMNodeInput:
    node_id: str
    instruction: str
    model: str
    user_input: str
    context: dict


@dataclass
class LLMNodeOutput:
    node_id: str
    response: str
    success: bool


@activity.defn
async def run_llm_node(input: LLMNodeInput) -> LLMNodeOutput:
    """Generic activity that runs any LLM node from the DAG config."""
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "alanblount-demo"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
    )

    prompt = f"{input.instruction}\n\nUser request: {input.user_input}"
    if input.context:
        prompt += f"\n\nContext: {input.context}"

    try:
        response = client.models.generate_content(
            model=input.model,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
        )
        text = ""
        for p in response.candidates[0].content.parts:
            if hasattr(p, "text") and p.text:
                text += p.text
        return LLMNodeOutput(node_id=input.node_id, response=text, success=True)
    except Exception as e:
        return LLMNodeOutput(
            node_id=input.node_id,
            response=f"Error: {str(e)}",
            success=False,
        )


@dataclass
class ValidationInput:
    node_id: str
    output: str
    expected_type: str  # "classification", "response", "any"


@activity.defn
async def validate_output(input: ValidationInput) -> bool:
    """Validate a node's output meets basic requirements."""
    if not input.output or len(input.output.strip()) < 2:
        return False
    if input.expected_type == "classification":
        valid_classes = {"billing", "technical", "general"}
        return input.output.strip().lower() in valid_classes
    return True


# --- DAG Interpreter ---

class DAGInterpreter:
    """Interprets a YAML DAG config and executes it step by step.

    This is what would become a Temporal workflow. Each step is an activity.
    With Temporal, the workflow engine handles durability, retries, and
    versioning. Without Temporal, this runs as a simple sequential executor.
    """

    def __init__(self, config: dict, feature_flags: dict = None):
        self.config = config
        self.flags = feature_flags or {}
        self.nodes = {n["id"]: n for n in config["nodes"]}
        self.edges = config["edges"]
        self.outputs = {}  # node_id → output

    def get_next_nodes(self, current_id: str) -> list[str]:
        """Get the next node(s) to execute based on edges and conditions."""
        next_nodes = []
        for edge in self.edges:
            if edge["from"] != current_id:
                continue

            # Check feature flag
            flag = edge.get("flag")
            if flag and not self.flags.get(flag, True):
                continue

            # Check condition
            condition = edge.get("condition")
            if condition:
                current_output = self.outputs.get(current_id, "")
                if condition.lower() not in current_output.lower():
                    continue

            next_nodes.append(edge["to"])
        return next_nodes

    async def execute(self, user_input: str) -> dict:
        """Execute the DAG from START to completion."""
        results = []

        # Find the start node
        start_edges = [e for e in self.edges if e["from"] == "START"]
        current_nodes = [e["to"] for e in start_edges]

        visited = set()
        max_steps = 10  # Safety limit

        for step in range(max_steps):
            if not current_nodes:
                break

            next_round = []
            for node_id in current_nodes:
                if node_id in visited:
                    continue
                visited.add(node_id)

                node = self.nodes[node_id]
                print(f"  Step {step}: executing '{node_id}'")

                # Run the node as an activity
                output = await run_llm_node(LLMNodeInput(
                    node_id=node_id,
                    instruction=node["instruction"],
                    model=node["model"],
                    user_input=user_input,
                    context={"previous_outputs": self.outputs},
                ))

                self.outputs[node_id] = output.response
                results.append({
                    "node": node_id,
                    "output": output.response[:100],
                    "success": output.success,
                })

                # Validate
                is_valid = await validate_output(ValidationInput(
                    node_id=node_id,
                    output=output.response,
                    expected_type="classification" if "classify" in node_id else "response",
                ))
                print(f"    → output: {output.response[:60]}... valid={is_valid}")

                if not is_valid:
                    print(f"    ⚠️ Validation failed for {node_id}")
                    results[-1]["valid"] = False
                    continue

                results[-1]["valid"] = True

                # Get next nodes
                next_ids = self.get_next_nodes(node_id)
                next_round.extend(next_ids)

            current_nodes = next_round

        return {
            "dag": self.config["name"],
            "version": self.config["version"],
            "steps": results,
            "feature_flags": self.flags,
        }


async def main():
    config_path = os.path.join(os.path.dirname(__file__), "customer_support.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)["dag"]

    print(f"=== DAG: {config['name']} v{config['version']} ===\n")

    queries = [
        "I was charged twice for my subscription",
        "My internet keeps disconnecting",
        "What are your business hours?",
    ]

    for query in queries:
        print(f"Query: {query}")
        interpreter = DAGInterpreter(config)
        result = await interpreter.execute(query)
        print(f"  Result: {len(result['steps'])} steps executed")
        for step in result["steps"]:
            print(f"    [{step['node']}] valid={step.get('valid')} → {step['output'][:50]}")
        print()

    # Test with feature flags
    print("=== WITH FEATURE FLAGS ===\n")
    print("Query: billing question with billing_v2=True")
    interpreter = DAGInterpreter(config, feature_flags={"billing_v2": True})
    result = await interpreter.execute("I was charged twice")
    for step in result["steps"]:
        print(f"  [{step['node']}] → {step['output'][:50]}")


if __name__ == "__main__":
    asyncio.run(main())
