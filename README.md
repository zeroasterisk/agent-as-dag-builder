# Agent as DAG Builder

An experiment in agents that build and continuously refine their own workflow DAGs.

## Core Idea

Instead of agents operating purely through LLM reasoning (expensive, slow, non-deterministic), agents learn from interactions and encode their knowledge as executable workflow DAGs. These DAGs handle ~70% of known tasks deterministically, while the live agent handles the ~30% of novel/exceptional cases — and then updates the DAG with what it learned.

## Status

Research and design phase.
