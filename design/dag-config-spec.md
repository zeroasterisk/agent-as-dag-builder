# DAG Config Specification for Agent-as-DAG-Builder

**Version:** 0.1.0-draft
**Date:** 2026-06-08
**Status:** Design Proposal

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Survey of Existing Workflow Config Languages](#2-survey-of-existing-workflow-config-languages)
3. [Recommendation](#3-recommendation)
4. [DAG Config Specification](#4-dag-config-specification)
5. [Agent-DAG Interaction Loop](#5-agent-dag-interaction-loop)
6. [Runtime Architecture](#6-runtime-architecture)
7. [Appendix: Sources](#7-appendix-sources)

---

## 1. Executive Summary

This document specifies a YAML-based DAG configuration language for the Agent-as-DAG-Builder project. The system allows AI agents to:

1. Execute workflows defined as directed acyclic graphs (DAGs) in YAML config
2. Learn from live interactions and encode patterns as new DAG nodes/edges
3. Propose, validate, canary-test, and promote DAG updates autonomously
4. Gate workflow paths on external feature flags for safe progressive rollout

After surveying 15+ existing workflow config systems, we recommend a **custom YAML format** that draws heavily from the CNCF Serverless Workflow DSL, Google ADK Agent Config, and Argo Workflows DAG templates. No single existing system meets all requirements — particularly the combination of explicit edge-level feature flag conditions, agent-invocation nodes, LLM editability, and sub-DAG hierarchy. Our format is designed to be:

- **LLM-native**: Flat, regular YAML that LLMs can reliably generate and edit
- **Explicit graph model**: Nodes AND edges are first-class, not implicit
- **Feature-flag-aware**: Edges carry optional conditions referencing external flags
- **Runtime-portable**: Config is separate from any specific execution engine
- **Schema-validatable**: Full JSON Schema for pre-execution validation

---

## 2. Survey of Existing Workflow Config Languages

### 2.1 Evaluation Criteria

Each system was evaluated against:

| # | Criterion | Weight |
|---|-----------|--------|
| C1 | Can an LLM reliably generate/edit it? | Critical |
| C2 | Supports conditionals, loops, fan-out/fan-in? | High |
| C3 | Feature flags natively or via conditional nodes? | High |
| C4 | Diffable/versionable in git? | High |
| C5 | Validatable before execution (schema, dry-run)? | High |
| C6 | Nodes can represent: functions, agents, sub-DAGs, validators? | Critical |
| C7 | Existing runtime can execute it? | Medium |
| C8 | Maturity and adoption? | Medium |

### 2.2 Detailed Evaluations

#### 2.2.1 Argo Workflows (YAML)

**Format:** Kubernetes CRD YAML
**Graph model:** DAG template with tasks listing `dependencies` (implicit edges)

```yaml
# Argo DAG example (simplified)
apiVersion: argoproj.io/v1alpha1
kind: Workflow
spec:
  templates:
  - name: my-dag
    dag:
      tasks:
      - name: classify
        template: classify-intent
      - name: handle-billing
        dependencies: [classify]
        when: "{{tasks.classify.outputs.parameters.intent}} == billing"
        template: billing-handler
      - name: handle-returns
        dependencies: [classify]
        when: "{{tasks.classify.outputs.parameters.intent}} == returns"
        template: returns-handler
```

| Criterion | Score | Notes |
|-----------|-------|-------|
| C1: LLM editability | **Good** | Regular YAML structure, but K8s CRD boilerplate adds noise |
| C2: Conditionals/loops | **Excellent** | `when` clauses, loops via `withItems`/`withParam`, DAG fan-out |
| C3: Feature flags | **Poor** | No native support; would need custom `when` expression |
| C4: Diffable | **Good** | Standard YAML, git-friendly |
| C5: Validatable | **Excellent** | CRD schema validation, `argo lint` |
| C6: Node types | **Limited** | Container-centric; agents/validators need custom templates |
| C7: Runtime | **Excellent** | Mature K8s-native runtime |
| C8: Maturity | **Excellent** | CNCF graduated, wide adoption |

**Verdict:** Strong DAG model and mature runtime, but Kubernetes-specific and container-centric. Would require significant extension for agent/function/validator node types and feature flags. The implicit edge model (via `dependencies` + `when` on nodes) doesn't cleanly separate edge conditions from node logic.

#### 2.2.2 AWS Step Functions — Amazon States Language (ASL)

**Format:** JSON (`.asl.json`)
**Graph model:** State machine with explicit transitions via `Next` or `Choice`

```json
{
  "StartAt": "ClassifyIntent",
  "States": {
    "ClassifyIntent": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:classify",
      "Next": "RouteByIntent"
    },
    "RouteByIntent": {
      "Type": "Choice",
      "Choices": [
        {
          "Variable": "$.intent",
          "StringEquals": "billing",
          "Next": "HandleBilling"
        },
        {
          "Variable": "$.intent",
          "StringEquals": "returns",
          "Next": "HandleReturns"
        }
      ],
      "Default": "HandleGeneral"
    }
  }
}
```

| Criterion | Score | Notes |
|-----------|-------|-------|
| C1: LLM editability | **Fair** | JSON is verbose and bracket-heavy; LLMs generate it less reliably than YAML. State names as keys means renaming breaks references. |
| C2: Conditionals/loops | **Excellent** | Choice, Parallel, Map (fan-out), Wait |
| C3: Feature flags | **Poor** | No native support; could abuse Choice + external lookup |
| C4: Diffable | **Fair** | JSON diffs are noisy (brackets, commas) |
| C5: Validatable | **Excellent** | Full JSON Schema, `aws stepfunctions validate` |
| C6: Node types | **Good** | Task (Lambda, ECS, etc.), but no native agent concept |
| C7: Runtime | **Excellent** | Fully managed AWS service |
| C8: Maturity | **Excellent** | 8+ years in production, massive scale |

**Verdict:** Excellent execution model but JSON format is suboptimal for LLM generation. AWS-specific. The state machine model (vs. DAG) forces sequential thinking and explicit routing states. No agent/validator/sub-DAG primitives.

#### 2.2.3 GitHub Actions (YAML)

**Format:** YAML
**Graph model:** Jobs with `needs` dependencies (implicit edges), `if` conditions

| Criterion | Score | Notes |
|-----------|-------|-------|
| C1: LLM editability | **Good** | Simple YAML, widely in LLM training data |
| C2: Conditionals/loops | **Fair** | `if` conditions, `matrix` for fan-out, no loops |
| C3: Feature flags | **Poor** | `if` can reference vars, but no flag integration |
| C4: Diffable | **Good** | Standard YAML |
| C5: Validatable | **Good** | Schema exists, linters available |
| C6: Node types | **Poor** | CI/CD-specific (run commands, use actions) |
| C7: Runtime | **Good** | GitHub-hosted, self-hosted runners |
| C8: Maturity | **Excellent** | Ubiquitous in CI/CD |

**Verdict:** Too CI/CD-specific. The `needs` + `if` pattern is simple but insufficient for complex agent orchestration. No sub-DAGs, no agent invocation, no validators.

#### 2.2.4 Tekton Pipelines (YAML)

**Format:** Kubernetes CRD YAML
**Graph model:** Pipeline with tasks, `runAfter` for ordering, `when` for conditions

| Criterion | Score | Notes |
|-----------|-------|-------|
| C1: LLM editability | **Fair** | K8s CRD boilerplate, but regular structure |
| C2: Conditionals/loops | **Fair** | `when` expressions (in/notin), CEL (alpha), no loops |
| C3: Feature flags | **Poor** | Would need custom `when` + parameter injection |
| C4: Diffable | **Good** | Standard YAML |
| C5: Validatable | **Good** | CRD validation, `tkn` CLI |
| C6: Node types | **Limited** | Container-centric (like Argo) |
| C7: Runtime | **Good** | K8s-native, but less adopted than Argo |
| C8: Maturity | **Good** | CNCF, but smaller community than Argo |

**Verdict:** Similar to Argo but less feature-rich. K8s-specific, container-centric. Not a good fit for agent orchestration.

#### 2.2.5 Temporal (Code-first + YAML layers)

**Format:** Primarily code (Go, Java, Python, TypeScript); YAML via Temporal DSL / Zigflow
**Graph model:** Workflow-as-code with activities, signals, queries

| Criterion | Score | Notes |
|-----------|-------|-------|
| C1: LLM editability | **Poor (code)** / **Good (YAML DSL)** | Code-first is hard for LLMs to safely mutate; YAML layers are simpler |
| C2: Conditionals/loops | **Excellent** | Full programming language + durable execution |
| C3: Feature flags | **Fair** | Can integrate via activity calls, but not declarative |
| C4: Diffable | **Varies** | Code diffs well; YAML DSL diffs well |
| C5: Validatable | **Good** | Type checking (code), schema (YAML DSL) |
| C6: Node types | **Good** | Activities can wrap anything; Zigflow adds declarative tasks |
| C7: Runtime | **Excellent** | Production-grade, battle-tested durable execution |
| C8: Maturity | **Excellent** | Major enterprises in production |

**Key insight — Zigflow:** An open-source project that lets you define Temporal workflows in YAML using the CNCF Serverless Workflow specification. Zigflow compiles YAML into fully-featured Temporal workflows with retries, state management, and deterministic execution. This is the bridge between declarative config and Temporal's runtime.

**Verdict:** Temporal's runtime is best-in-class for durable execution, but its code-first model is wrong for LLM-driven DAG mutation. Zigflow's YAML-to-Temporal bridge is the right pattern — use a declarative YAML config that compiles to a runtime-specific execution plan.

#### 2.2.6 Apache Airflow (Python + DAG Factory YAML)

**Format:** Python code; YAML via DAG Factory
**Graph model:** DAG of operators with dependencies

| Criterion | Score | Notes |
|-----------|-------|-------|
| C1: LLM editability | **Poor (Python)** / **Fair (DAG Factory YAML)** | DAG Factory YAML is simpler but limited |
| C2: Conditionals/loops | **Good** | BranchPythonOperator, ShortCircuit; loops via code |
| C3: Feature flags | **Poor** | No native support |
| C4: Diffable | **Varies** | Python diffs OK; YAML diffs well |
| C5: Validatable | **Fair** | DAG parsing validates structure, but runtime errors common |
| C6: Node types | **Good** | Rich operator ecosystem |
| C7: Runtime | **Excellent** | Mature scheduler, AWS MWAA Serverless adoption |
| C8: Maturity | **Excellent** | Industry standard for data orchestration |

**Verdict:** Data pipeline-focused. DAG Factory's YAML approach (now adopted by AWS MWAA Serverless) validates the DAG-as-config pattern but lacks agent/validator primitives and feature flag support.

#### 2.2.7 Dapr Workflows (Code-only)

**Format:** Code (Go, Python, .NET, Java, JS)
**Graph model:** Workflow-as-code with activities

| Criterion | Score | Notes |
|-----------|-------|-------|
| C1-C8 | **N/A** | Code-only; no declarative config format for workflow logic |

**Verdict:** Excluded — purely code-first with no declarative config option for workflow definitions. YAML is only for operational configuration (concurrency limits, etc.).

#### 2.2.8 Prefect (Python-first)

**Format:** Python code; deployment YAML
**Graph model:** Flow of tasks with dependencies

| Criterion | Score | Notes |
|-----------|-------|-------|
| C1-C8 | **Limited** | Python-first, deployment YAML is not workflow logic |

**Verdict:** Similar to Airflow — Python-first, data pipeline-focused. No declarative workflow config.

#### 2.2.9 CNCF Serverless Workflow DSL 1.0 (YAML/JSON)

**Format:** YAML or JSON
**Graph model:** Sequential task list with `switch`, `fork`, `for`, and `do` control flow

```yaml
document:
  dsl: '1.0.3'
  namespace: customer-support
  name: support-workflow
  version: '1.0.0'
do:
  - classifyIntent:
      call: http
      with:
        method: POST
        endpoint:
          uri: https://api.example.com/classify
  - routeByIntent:
      switch:
        - billing:
            when: .intent == "billing"
            then: handleBilling
        - returns:
            when: .intent == "returns"
            then: handleReturns
  - handleBilling:
      call: http
      with:
        method: POST
        endpoint:
          uri: https://api.example.com/billing
  - handleReturns:
      call: http
      with:
        method: POST
        endpoint:
          uri: https://api.example.com/returns
```

| Criterion | Score | Notes |
|-----------|-------|-------|
| C1: LLM editability | **Excellent** | Clean YAML, minimal boilerplate, fluent DSL |
| C2: Conditionals/loops | **Excellent** | `switch`, `for`, `fork` (parallel), `do` (sequential) |
| C3: Feature flags | **Poor** | No native feature flag concept; could be added to `when` |
| C4: Diffable | **Excellent** | Clean YAML, designed for versioning |
| C5: Validatable | **Excellent** | JSON Schema, CTK conformance tests, SDKs with validators |
| C6: Node types | **Good** | `call` (HTTP, gRPC, OpenAPI, AsyncAPI), `run` (container, script, workflow), custom extensions possible |
| C7: Runtime | **Fair** | Multiple implementations (Synapse, SonataFlow) but not as mature as Temporal/Argo |
| C8: Maturity | **Fair** | CNCF sandbox, v1.0 released Jan 2025, growing |

**Verdict:** Closest existing system to our needs. The DSL is designed to be fluent, minimal, and platform-neutral. However: (1) it uses implicit sequential ordering, not explicit edges; (2) no feature flag primitive; (3) no agent-invocation node type; (4) the sequential `do` model doesn't naturally express DAGs with complex dependency graphs. Would need extension for our use case.

#### 2.2.10 Google ADK Agent Config (YAML)

**Format:** YAML
**Graph model:** Agent hierarchy with `sub_agents`, type-based workflow (sequential/parallel/loop)

```yaml
name: support_pipeline
type: sequential
sub_agents:
  - name: classify_intent
    model: gemini-2.0-flash
    instruction: "Classify the customer's intent"
    output_key: intent_result
  - name: handle_request
    type: parallel
    sub_agents:
      - name: billing_agent
        model: gemini-2.0-flash
        instruction: "Handle billing requests"
        output_key: billing_result
      - name: returns_agent
        model: gemini-2.0-flash
        instruction: "Handle return requests"
        output_key: returns_result
  - name: verify_response
    model: gemini-2.0-flash
    instruction: "Verify the quality of the response"
    output_key: verified_result
```

| Criterion | Score | Notes |
|-----------|-------|-------|
| C1: LLM editability | **Excellent** | Minimal YAML, designed for non-developers |
| C2: Conditionals/loops | **Fair** | Sequential/parallel/loop workflow types, but conditional routing requires code or LLM-based transfer |
| C3: Feature flags | **Poor** | No native support |
| C4: Diffable | **Good** | Clean YAML |
| C5: Validatable | **Fair** | Experimental, limited validation tooling |
| C6: Node types | **Good** | LLM agents native; tools, sub-agents, functions |
| C7: Runtime | **Good** | ADK runtime, Vertex AI Agent Engine |
| C8: Maturity | **Fair** | v2.0 just released, experimental config feature |

**Verdict:** The agent-native YAML config is the right shape for AI agent orchestration. However: (1) no explicit edges or conditions — routing is LLM-driven or type-based; (2) no feature flags; (3) no validator node type; (4) limited to sequential/parallel/loop patterns — no arbitrary DAG topology; (5) Gemini-only for now.

#### 2.2.11 CrewAI (YAML + Python Flows)

**Format:** YAML for agents/tasks; Python for flows
**Graph model:** Crew (agent team) + Flow (multi-stage pipeline)

| Criterion | Score | Notes |
|-----------|-------|-------|
| C1: LLM editability | **Good** | Agent/task YAML is simple; flow logic requires Python |
| C2: Conditionals/loops | **Fair** | Python flows for conditionals; YAML is sequential only |
| C3: Feature flags | **Poor** | No native support |
| C4: Diffable | **Good** | YAML parts diff well |
| C5: Validatable | **Fair** | Pydantic validation in code |
| C6: Node types | **Good** | Agents, tasks, tools are first-class |
| C7: Runtime | **Good** | CrewAI runtime, async support |
| C8: Maturity | **Good** | Growing adoption, active development |

**Verdict:** Good agent/task YAML model, but workflow orchestration still requires Python code (Flows). No explicit DAG topology in YAML.

#### 2.2.12 LangGraph (Code-first)

**Format:** Python code (StateGraph API)
**Graph model:** Explicit directed graph with nodes, edges, conditional edges

| Criterion | Score | Notes |
|-----------|-------|-------|
| C1: LLM editability | **Poor** | Python graph construction code |
| C2: Conditionals/loops | **Excellent** | Conditional edges, cycles allowed (not DAG-only) |
| C3: Feature flags | **Fair** | Can integrate via state/config, not declarative |
| C4: Diffable | **Fair** | Code diffs |
| C5: Validatable | **Fair** | Python type checking |
| C6: Node types | **Good** | Functions, LLM calls, tool use |
| C7: Runtime | **Good** | LangGraph runtime, LangGraph Platform |
| C8: Maturity | **Good** | v1.0 Oct 2025, wide adoption |

**Verdict:** Best explicit graph model (nodes + edges + conditional edges), but code-first. The graph model is exactly what we want — it just needs to be expressed in YAML.

#### 2.2.13 Config Languages (CUE, Jsonnet, Dhall, Nickel)

| Language | LLM Editability | Validation | Workflow Support | Verdict |
|----------|-----------------|------------|-----------------|---------|
| **CUE** | Poor (high learning curve, no functions, constraint-based) | Excellent (types = values) | Built-in `_tool.cue` scripting | Too complex for LLM generation |
| **Jsonnet** | Fair (familiar JSON-like syntax) | Poor (dynamically typed) | External tooling needed | Good for templating, weak for validation |
| **Dhall** | Poor (verbose functional syntax) | Good (static types) | External tooling needed | Too verbose, performance issues |
| **Nickel** | Fair (CUE-like with functions) | Good (contracts/types) | External tooling needed | Promising but immature |

**Verdict:** These are config *generation* languages, not workflow *definition* languages. They solve a different problem (generating valid YAML/JSON from templates with type safety). CUE could be used as a schema/validation layer on top of our YAML format, but the workflow DSL itself should be plain YAML for LLM editability.

### 2.3 Comparative Summary

| System | Format | LLM Edit | Conditionals | Feature Flags | Agents | Sub-DAGs | Runtime | Maturity |
|--------|--------|----------|-------------|---------------|--------|----------|---------|----------|
| Argo Workflows | YAML | Good | Excellent | Poor | Poor | Good | Excellent | Excellent |
| AWS Step Functions | JSON | Fair | Excellent | Poor | Poor | Good | Excellent | Excellent |
| GitHub Actions | YAML | Good | Fair | Poor | Poor | Fair | Good | Excellent |
| Tekton | YAML | Fair | Fair | Poor | Poor | Fair | Good | Good |
| Temporal | Code | Poor | Excellent | Fair | Fair | Excellent | Excellent | Excellent |
| Zigflow (Temporal+SWF) | YAML | Good | Good | Poor | Fair | Good | Excellent | Fair |
| Airflow + DAG Factory | YAML | Fair | Fair | Poor | Poor | Fair | Excellent | Excellent |
| Serverless Workflow 1.0 | YAML | Excellent | Excellent | Poor | Fair | Good | Fair | Fair |
| Google ADK Config | YAML | Excellent | Fair | Poor | **Excellent** | Good | Good | Fair |
| CrewAI | YAML+Py | Good | Fair | Poor | Good | Fair | Good | Good |
| LangGraph | Code | Poor | Excellent | Fair | Good | Good | Good | Good |

**Key finding:** No existing system scores well across ALL criteria. In particular, **no system has native feature flag support on edges/conditions**, and **no system combines explicit edge modeling with agent-invocation nodes in a YAML config**.

---

## 3. Recommendation

### 3.1 Decision: Custom YAML Format

We recommend a **custom YAML format** that synthesizes the best ideas from existing systems:

| Borrowed From | What We Take |
|---------------|-------------|
| **LangGraph** | Explicit node + edge + conditional-edge graph model |
| **CNCF Serverless Workflow** | YAML fluency, task type system, `switch` semantics |
| **Google ADK Agent Config** | Agent-native node definitions, sub-agent hierarchy |
| **Argo Workflows** | DAG task structure, parameter passing between nodes |
| **CrewAI** | Agent role/goal/backstory pattern for agent nodes |
| **AWS Step Functions** | Input/output processing, error handling, retry policies |

### 3.2 Rationale

**Why not adopt an existing system directly?**

1. **No system models edges with feature flag conditions.** All existing systems put conditions on nodes (Argo's `when`, Tekton's `when`) or use routing states (ASL's `Choice`, SWF's `switch`). For our use case, conditions belong on *edges* because:
   - The same node should be reachable via multiple paths with different flag configurations
   - Feature flags gate *transitions*, not *nodes*
   - Edge-level conditions are more diffable — adding a flag-gated alternative path means adding an edge, not modifying a node

2. **No system has agent-invocation as a first-class node type with configuration.** ADK comes closest but its YAML config doesn't support conditional routing or arbitrary DAG topology.

3. **No system is designed for LLM-driven mutation.** While LLMs can generate YAML for any of these systems, none is optimized for the *edit* pattern — where an LLM reads an existing config, understands its semantics, and makes targeted modifications via structured tool calls.

4. **Feature flags as edge conditions is a novel requirement.** The A/B testing / canary pattern for workflow versions doesn't exist in any surveyed system.

**Why not extend an existing system?**

We considered extending CNCF Serverless Workflow or Argo Workflows. The issues:

- **Serverless Workflow** uses a sequential task list model that doesn't naturally express DAGs with complex dependency graphs. Converting to an explicit node+edge model would break compatibility with existing runtimes.
- **Argo Workflows** is Kubernetes-specific and container-centric. Extending it for agent orchestration would require fighting the K8s CRD model.

Instead, we **draw heavily from these systems** while creating a format optimized for our three unique requirements: LLM editability, feature flag edges, and agent-native nodes.

### 3.3 Runtime Strategy

The config format is **runtime-portable**. We recommend:

1. **Primary runtime:** Custom lightweight interpreter built on ADK 2.0's graph workflow engine
2. **Durable execution:** Compile to Temporal workflows (via Zigflow-like bridge) for production deployments requiring durability
3. **Validation:** JSON Schema + custom validator (simulates execution without side effects)

---

## 4. DAG Config Specification

### 4.1 Format Overview

```yaml
# Every DAG config file has this structure
dag:
  name: string              # Unique identifier (kebab-case)
  version: string           # Semantic version
  description: string       # Human & LLM readable description
  
  # Optional metadata
  metadata:
    owner: string
    created: datetime
    tags: [string]
  
  # Input/output schema
  input:
    schema: object           # JSON Schema for workflow input
  output:
    schema: object           # JSON Schema for workflow output
  
  # The graph
  nodes:
    - node definitions...
  
  edges:
    - edge definitions...
  
  # Optional: default retry/timeout policy
  defaults:
    retry: retry-policy
    timeout: duration
```

### 4.2 Node Types

Every node has a common header and a type-specific body:

```yaml
nodes:
  - id: string               # Unique within this DAG (kebab-case)
    type: function | agent | sub_dag | validator | router
    name: string             # Human-readable display name
    description: string      # What this node does (LLM-readable)
    
    # Type-specific configuration (see below)
    config: object
    
    # Optional overrides
    retry: retry-policy
    timeout: duration
    
    # Optional: metadata for observability
    metadata:
      metrics: [string]      # Metric names to collect
      tags: [string]
```

#### 4.2.1 Function Node

Invokes a deterministic function (HTTP endpoint, gRPC call, serverless function).

```yaml
- id: classify-intent
  type: function
  name: "Classify Customer Intent"
  description: "Analyzes customer message and returns intent classification"
  config:
    function:
      call: http              # http | grpc | openapi | script
      endpoint: "https://api.example.com/classify"
      method: POST
      headers:
        Content-Type: "application/json"
      input_mapping:
        body: "$.message"     # JSONPath from workflow input
      output_mapping:
        intent: "$.result.intent"
        confidence: "$.result.confidence"
```

#### 4.2.2 Agent Node

Invokes an LLM-powered agent. This is the primary node type for AI-driven steps.

```yaml
- id: handle-returns
  type: agent
  name: "Returns Handler Agent"
  description: "Handles product return requests using conversational AI"
  config:
    agent:
      model: "claude-sonnet-4-20250514"    # or "gemini-2.0-flash", etc.
      instruction: |
        You are a customer service agent specializing in product returns.
        Help the customer process their return request.
        Collect: order number, reason for return, preferred resolution.
        Output a structured return request.
      tools:
        - name: lookup_order
          type: function_ref
          ref: "orders-api.lookup"
        - name: create_return
          type: function_ref
          ref: "returns-api.create"
      output_schema:
        type: object
        properties:
          return_id: { type: string }
          status: { type: string, enum: [approved, denied, escalated] }
          resolution: { type: string }
        required: [return_id, status]
      max_turns: 10
      temperature: 0.3
```

#### 4.2.3 Sub-DAG Node

References another DAG config, enabling hierarchical composition.

```yaml
- id: handle-billing
  type: sub_dag
  name: "Billing Handler"
  description: "Delegates to the billing-specific workflow"
  config:
    sub_dag:
      ref: "billing/billing-workflow-v2"    # Path to another DAG config
      version: ">=2.0.0"                   # Semver constraint
      input_mapping:
        customer_id: "$.customer.id"
        issue: "$.classification.issue"
      output_mapping:
        resolution: "$.result"
```

#### 4.2.4 Validator Node

Checks output quality, safety, or correctness. Returns pass/fail + details.

```yaml
- id: verify-response
  type: validator
  name: "Response Quality Validator"
  description: "Ensures response meets quality and safety standards"
  config:
    validator:
      checks:
        - name: safety_check
          type: llm_judge
          model: "claude-sonnet-4-20250514"
          prompt: |
            Review this customer service response for:
            1. Accuracy of information
            2. Professional tone
            3. No PII leakage
            4. Complete resolution of the request
            Return pass/fail with reasoning.
          threshold: 0.8     # Minimum score to pass
          
        - name: schema_check
          type: json_schema
          schema:
            type: object
            required: [response_text, resolution_status]
            
        - name: toxicity_check
          type: function_ref
          ref: "safety-api.toxicity-score"
          threshold: 0.1     # Maximum toxicity score
      
      # What happens on validation failure
      on_failure: retry | escalate | fallback
      fallback_node: "live-agent-handoff"     # If on_failure is 'fallback'
      max_retries: 2
```

#### 4.2.5 Router Node

Evaluates conditions and determines which outgoing edge to take. This is a convenience node that makes conditional logic explicit in the graph (alternative to putting conditions on edges).

```yaml
- id: route-by-intent
  type: router
  name: "Intent Router"
  description: "Routes to the appropriate handler based on classified intent"
  config:
    router:
      input: "$.classification.intent"
      routes:
        - match: "billing"
          description: "Route billing requests"
        - match: "returns"
          description: "Route return requests"
        - match: "technical"
          description: "Route technical support requests"
      default: "general"
```

### 4.3 Edge Definitions

Edges are **first-class objects** with optional conditions. This is the key differentiator from existing systems.

```yaml
edges:
  - from: source-node-id
    to: target-node-id
    
    # Optional: condition for this edge to be traversed
    condition:
      # Feature flag condition (evaluated against external flag service)
      flag: string             # Flag name (e.g., "billing_v2")
      operator: is | is_not | in | not_in | gt | lt | gte | lte
      value: any               # Expected value
      
    # Alternative: expression-based condition (on workflow data)
    when: string               # Expression evaluated against current state
                               # e.g., "$.intent == 'billing'"
    
    # Optional: edge metadata
    priority: integer          # When multiple edges match, highest priority wins
    description: string        # Human-readable description of this transition
    
    # Optional: data transformation on this edge
    transform:
      input_mapping: object    # Map source output to target input
```

#### Edge Condition Types

```yaml
# 1. Unconditional edge (always traversed)
edges:
  - from: classify-intent
    to: handle-billing

# 2. Feature-flag-gated edge
edges:
  - from: classify-intent
    to: handle-billing-v2
    condition:
      flag: "billing_v2_enabled"
      operator: is
      value: true

# 3. Data-conditional edge (based on node output)
edges:
  - from: classify-intent
    to: handle-billing
    when: "$.intent == 'billing'"

# 4. Combined: flag + data condition (both must be true)
edges:
  - from: classify-intent
    to: handle-billing-v2
    condition:
      flag: "billing_v2_enabled"
      operator: is
      value: true
    when: "$.intent == 'billing'"

# 5. Priority-based edge selection (first match wins)
edges:
  - from: classify-intent
    to: handle-billing-v2
    condition:
      flag: "billing_v2_enabled"
      operator: is
      value: true
    when: "$.intent == 'billing'"
    priority: 10
    
  - from: classify-intent
    to: handle-billing-v1
    when: "$.intent == 'billing'"
    priority: 1               # Lower priority = fallback
```

#### Edge Resolution Rules

1. All edges from a node are evaluated when the node completes
2. `when` expressions are evaluated against the node's output data
3. `condition.flag` is evaluated against the external feature flag service
4. Both `when` AND `condition` must be true for the edge to activate
5. If multiple edges match, `priority` determines which is taken (highest wins)
6. If multiple edges have equal priority and all match, they execute in **parallel** (fan-out)
7. If no edges match, the workflow fails with an `EdgeResolutionError`

### 4.4 Feature Flag Integration

Feature flags are **external** to the DAG config. The config only *references* flag names; the A/B testing system manages flag values.

```yaml
# The DAG config references flags but does not define them
edges:
  - from: classify-intent
    to: handle-billing-v2
    condition:
      flag: "billing_v2_enabled"    # Resolved at runtime by flag service
      operator: is
      value: true
    when: "$.intent == 'billing'"
```

**Flag evaluation context** passed to the flag service at runtime:

```json
{
  "workflow_id": "customer-support-v3",
  "workflow_version": "3.2.0",
  "node_id": "classify-intent",
  "edge_to": "handle-billing-v2",
  "user_id": "user-123",
  "session_id": "sess-456",
  "environment": "production",
  "timestamp": "2026-06-08T10:30:00Z"
}
```

This context allows the flag service (LaunchDarkly, Unleash, Flagsmith, etc.) to make percentage-based rollout, user-targeting, or environment-based decisions.

### 4.5 Complete Example: Customer Support Workflow

```yaml
dag:
  name: customer-support
  version: "3.2.0"
  description: |
    Main customer support workflow. Classifies intent, routes to
    specialized handlers, validates response, and sends to customer.
  
  metadata:
    owner: support-platform-team
    tags: [customer-support, production]
  
  input:
    schema:
      type: object
      properties:
        customer_id:
          type: string
          description: "Customer identifier"
        message:
          type: string
          description: "Customer's message text"
        channel:
          type: string
          enum: [chat, email, phone]
      required: [customer_id, message, channel]
  
  output:
    schema:
      type: object
      properties:
        response_text:
          type: string
        resolution_status:
          type: string
          enum: [resolved, escalated, pending]
        handler_type:
          type: string
      required: [response_text, resolution_status]
  
  defaults:
    retry:
      max_attempts: 3
      backoff: exponential
      initial_interval: "1s"
    timeout: "120s"
  
  # ============================================================
  # NODES
  # ============================================================
  nodes:
    # --- Node 1: Classify the customer's intent ---
    - id: classify-intent
      type: function
      name: "Classify Intent"
      description: "Analyzes customer message to determine intent category"
      config:
        function:
          call: http
          endpoint: "https://api.internal/ml/classify-intent"
          method: POST
          input_mapping:
            body:
              text: "$.message"
              customer_id: "$.customer_id"
          output_mapping:
            intent: "$.prediction.intent"
            confidence: "$.prediction.confidence"
            entities: "$.prediction.entities"
      metadata:
        metrics: [intent_classification_latency, intent_confidence]
    
    # --- Node 2a: Handle billing (sub-DAG, v2 behind flag) ---
    - id: handle-billing-v2
      type: sub_dag
      name: "Billing Handler v2"
      description: "New billing workflow with automated refund capability"
      config:
        sub_dag:
          ref: "billing/billing-workflow-v2"
          version: ">=2.0.0"
          input_mapping:
            customer_id: "$.customer_id"
            issue_details: "$.entities"
            channel: "$.channel"
          output_mapping:
            response_text: "$.resolution.message"
            resolution_status: "$.resolution.status"
            handler_type: "'billing_v2'"
    
    # --- Node 2a-fallback: Handle billing (v1, default) ---
    - id: handle-billing-v1
      type: sub_dag
      name: "Billing Handler v1"
      description: "Stable billing workflow"
      config:
        sub_dag:
          ref: "billing/billing-workflow-v1"
          version: "^1.0.0"
          input_mapping:
            customer_id: "$.customer_id"
            issue_details: "$.entities"
          output_mapping:
            response_text: "$.resolution.message"
            resolution_status: "$.resolution.status"
            handler_type: "'billing_v1'"
    
    # --- Node 2b: Handle returns (agent call) ---
    - id: handle-returns
      type: agent
      name: "Returns Agent"
      description: "AI agent that handles product return requests conversationally"
      config:
        agent:
          model: "claude-sonnet-4-20250514"
          instruction: |
            You are a customer service specialist for product returns.
            
            Your task:
            1. Look up the customer's order using the order lookup tool
            2. Understand their reason for return
            3. Check return eligibility (within 30-day window)
            4. If eligible, create the return and provide shipping label
            5. If not eligible, explain why and offer alternatives
            
            Always be empathetic and solution-oriented.
          tools:
            - name: lookup_order
              type: function_ref
              ref: "orders-api.lookup"
            - name: check_eligibility
              type: function_ref
              ref: "returns-api.check-eligibility"
            - name: create_return
              type: function_ref
              ref: "returns-api.create"
          output_schema:
            type: object
            properties:
              response_text: { type: string }
              resolution_status:
                type: string
                enum: [resolved, escalated, pending]
              return_id: { type: string }
            required: [response_text, resolution_status]
          max_turns: 15
          temperature: 0.3
      metadata:
        metrics: [returns_resolution_rate, returns_agent_turns, returns_csat]
    
    # --- Node 2c: Handle general queries (agent call) ---
    - id: handle-general
      type: agent
      name: "General Support Agent"
      description: "Handles all other customer queries"
      config:
        agent:
          model: "claude-sonnet-4-20250514"
          instruction: |
            You are a general customer support agent.
            Help the customer with their query. If you cannot resolve it,
            escalate by setting resolution_status to "escalated".
          tools:
            - name: search_knowledge_base
              type: function_ref
              ref: "kb-api.search"
          output_schema:
            type: object
            properties:
              response_text: { type: string }
              resolution_status:
                type: string
                enum: [resolved, escalated, pending]
            required: [response_text, resolution_status]
          max_turns: 10
          temperature: 0.4
    
    # --- Node 3: Validate the response ---
    - id: verify-response
      type: validator
      name: "Response Validator"
      description: "Ensures response quality, safety, and completeness"
      config:
        validator:
          checks:
            - name: quality_check
              type: llm_judge
              model: "claude-haiku-4-20250514"
              prompt: |
                Rate this customer service response on:
                1. Accuracy (is the information correct?)
                2. Completeness (does it fully address the issue?)
                3. Tone (is it professional and empathetic?)
                4. Safety (no PII leakage, no harmful content?)
                Return a score 0.0-1.0 and brief reasoning.
              threshold: 0.7
            - name: pii_check
              type: function_ref
              ref: "safety-api.detect-pii"
              threshold: 0.0    # Zero tolerance for PII
          on_failure: retry
          max_retries: 2
      metadata:
        metrics: [response_quality_score, pii_detection_rate, validation_pass_rate]
    
    # --- Node 4: Send the response ---
    - id: send-response
      type: function
      name: "Send Response"
      description: "Delivers the validated response to the customer via their channel"
      config:
        function:
          call: http
          endpoint: "https://api.internal/messaging/send"
          method: POST
          input_mapping:
            body:
              customer_id: "$.customer_id"
              channel: "$.channel"
              message: "$.response_text"
              metadata:
                workflow_version: "$.dag.version"
                handler_type: "$.handler_type"
          output_mapping:
            delivery_status: "$.status"
            message_id: "$.message_id"
      metadata:
        metrics: [response_delivery_latency, delivery_success_rate]
  
  # ============================================================
  # EDGES
  # ============================================================
  edges:
    # --- From classify-intent: route by intent + feature flags ---
    
    # Billing with v2 flag (canary)
    - from: classify-intent
      to: handle-billing-v2
      condition:
        flag: "billing_v2"
        operator: is
        value: true
      when: "$.intent == 'billing'"
      priority: 10
      description: "Route billing to v2 handler (canary)"
    
    # Billing fallback to v1 (default)
    - from: classify-intent
      to: handle-billing-v1
      when: "$.intent == 'billing'"
      priority: 1
      description: "Route billing to v1 handler (stable)"
    
    # Returns
    - from: classify-intent
      to: handle-returns
      when: "$.intent == 'returns'"
      description: "Route return requests to returns agent"
    
    # General / catch-all
    - from: classify-intent
      to: handle-general
      when: "$.intent != 'billing' && $.intent != 'returns'"
      description: "Route all other intents to general agent"
    
    # --- All handlers flow to verification ---
    - from: handle-billing-v2
      to: verify-response
      description: "Verify billing v2 response"
    
    - from: handle-billing-v1
      to: verify-response
      description: "Verify billing v1 response"
    
    - from: handle-returns
      to: verify-response
      description: "Verify returns response"
    
    - from: handle-general
      to: verify-response
      description: "Verify general response"
    
    # --- Verification to send ---
    - from: verify-response
      to: send-response
      description: "Send validated response to customer"
```

### 4.6 JSON Schema for Validation

The full JSON Schema for validating DAG configs:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://agent-dag-builder.dev/schemas/dag-config/v0.1.0",
  "title": "DAG Configuration",
  "description": "Schema for Agent-as-DAG-Builder workflow configuration",
  "type": "object",
  "required": ["dag"],
  "properties": {
    "dag": {
      "type": "object",
      "required": ["name", "version", "nodes", "edges"],
      "properties": {
        "name": {
          "type": "string",
          "pattern": "^[a-z][a-z0-9-]*$",
          "description": "Unique identifier in kebab-case"
        },
        "version": {
          "type": "string",
          "pattern": "^\\d+\\.\\d+\\.\\d+",
          "description": "Semantic version"
        },
        "description": { "type": "string" },
        "metadata": {
          "type": "object",
          "properties": {
            "owner": { "type": "string" },
            "created": { "type": "string", "format": "date-time" },
            "tags": { "type": "array", "items": { "type": "string" } }
          }
        },
        "input": {
          "type": "object",
          "properties": {
            "schema": { "type": "object", "description": "JSON Schema for workflow input" }
          }
        },
        "output": {
          "type": "object",
          "properties": {
            "schema": { "type": "object", "description": "JSON Schema for workflow output" }
          }
        },
        "defaults": {
          "type": "object",
          "properties": {
            "retry": { "$ref": "#/$defs/retry-policy" },
            "timeout": { "type": "string", "description": "ISO 8601 duration" }
          }
        },
        "nodes": {
          "type": "array",
          "minItems": 1,
          "items": { "$ref": "#/$defs/node" }
        },
        "edges": {
          "type": "array",
          "minItems": 0,
          "items": { "$ref": "#/$defs/edge" }
        }
      }
    }
  },
  "$defs": {
    "node": {
      "type": "object",
      "required": ["id", "type", "name", "config"],
      "properties": {
        "id": {
          "type": "string",
          "pattern": "^[a-z][a-z0-9-]*$"
        },
        "type": {
          "type": "string",
          "enum": ["function", "agent", "sub_dag", "validator", "router"]
        },
        "name": { "type": "string" },
        "description": { "type": "string" },
        "config": { "type": "object" },
        "retry": { "$ref": "#/$defs/retry-policy" },
        "timeout": { "type": "string" },
        "metadata": {
          "type": "object",
          "properties": {
            "metrics": { "type": "array", "items": { "type": "string" } },
            "tags": { "type": "array", "items": { "type": "string" } }
          }
        }
      }
    },
    "edge": {
      "type": "object",
      "required": ["from", "to"],
      "properties": {
        "from": { "type": "string" },
        "to": { "type": "string" },
        "condition": {
          "type": "object",
          "required": ["flag", "operator", "value"],
          "properties": {
            "flag": { "type": "string" },
            "operator": {
              "type": "string",
              "enum": ["is", "is_not", "in", "not_in", "gt", "lt", "gte", "lte"]
            },
            "value": {}
          }
        },
        "when": { "type": "string" },
        "priority": { "type": "integer", "minimum": 0 },
        "description": { "type": "string" },
        "transform": {
          "type": "object",
          "properties": {
            "input_mapping": { "type": "object" }
          }
        }
      }
    },
    "retry-policy": {
      "type": "object",
      "properties": {
        "max_attempts": { "type": "integer", "minimum": 1 },
        "backoff": {
          "type": "string",
          "enum": ["constant", "linear", "exponential"]
        },
        "initial_interval": { "type": "string" },
        "max_interval": { "type": "string" }
      }
    }
  }
}
```

### 4.7 Design Principles for LLM Editability

The format is designed for reliable LLM generation and editing:

1. **Flat node/edge arrays** (not nested hierarchies): An LLM can add a node by appending to the `nodes` array and add edges by appending to the `edges` array. No need to understand deeply nested structures.

2. **Kebab-case IDs everywhere**: Consistent, simple naming convention. No camelCase/snake_case mixing.

3. **Explicit over implicit**: Edges are separate from nodes. No hidden dependencies via key ordering or nesting depth.

4. **String-based expressions**: `when` conditions are simple string expressions, not embedded code. JSONPath-like syntax (`$.field == 'value'`) is well-represented in LLM training data.

5. **No YAML advanced features**: No anchors (`&`/`*`), no merge keys (`<<`), no complex types. Plain YAML that any parser handles identically.

6. **Quoted strings for values**: All string values in `when` expressions and `condition.value` should be quoted, avoiding YAML's auto-typing pitfalls.

7. **Self-documenting**: Every node and edge has a `description` field that the LLM uses to understand intent.

### 4.8 Agent Tool Commands for DAG Mutation

The agent interacts with DAG configs through structured tool commands:

```yaml
# Tool: dag.add_node
# Adds a new node to the DAG
dag.add_node:
  dag_ref: "customer-support"
  node:
    id: handle-complaints
    type: agent
    name: "Complaints Handler"
    description: "Specialized agent for handling customer complaints"
    config:
      agent:
        model: "claude-sonnet-4-20250514"
        instruction: "Handle customer complaints with empathy..."
        output_schema:
          type: object
          properties:
            response_text: { type: string }
            resolution_status: { type: string }

# Tool: dag.add_edge
# Adds a new edge to the DAG
dag.add_edge:
  dag_ref: "customer-support"
  edge:
    from: classify-intent
    to: handle-complaints
    when: "$.intent == 'complaint'"
    description: "Route complaints to specialized handler"

# Tool: dag.remove_edge
# Removes an edge from the DAG
dag.remove_edge:
  dag_ref: "customer-support"
  from: classify-intent
  to: handle-general
  when: "$.intent == 'complaint'"    # Match criteria to identify the edge

# Tool: dag.update_node
# Updates an existing node's configuration
dag.update_node:
  dag_ref: "customer-support"
  node_id: handle-returns
  updates:
    config.agent.instruction: |
      Updated instruction with new return policy...
    config.agent.max_turns: 20

# Tool: dag.validate
# Validates a DAG config against the schema and checks graph properties
dag.validate:
  dag_ref: "customer-support"
  checks:
    - schema          # JSON Schema validation
    - acyclic         # Verify no cycles
    - reachable       # All nodes reachable from entry points
    - no_orphans      # No nodes without incoming or outgoing edges (except entry/exit)
    - edge_targets    # All edge from/to reference valid node IDs
    - flag_coverage   # Every flagged edge has a non-flagged fallback

# Tool: dag.diff
# Shows diff between two DAG versions
dag.diff:
  dag_ref: "customer-support"
  from_version: "3.1.0"
  to_version: "3.2.0"

# Tool: dag.simulate
# Dry-run execution with mock data
dag.simulate:
  dag_ref: "customer-support"
  input:
    customer_id: "test-123"
    message: "I want to return my order"
    channel: "chat"
  flags:
    billing_v2: false
  # Returns: execution trace showing which nodes/edges would activate
```

---

## 5. Agent-DAG Interaction Loop

### 5.1 Overview

The Agent-as-DAG-Builder operates in a continuous learning loop:

```
Live Interaction → Pattern Recognition → DAG Proposal → Validation → Canary → Promotion
```

### 5.2 Detailed Flow

#### Phase 1: Live Handling (No DAG Match)

```
┌─────────────────────────────────────────────────┐
│  Customer request arrives                       │
│  ↓                                              │
│  DAG Router: evaluate edges from entry node     │
│  ↓                                              │
│  No matching edge/node for this request type    │
│  ↓                                              │
│  FALLBACK: Route to Live Agent                  │
│  ↓                                              │
│  Live Agent handles request conversationally    │
│  ↓                                              │
│  Interaction logged to Trace Store:             │
│  - Full conversation transcript                 │
│  - Tools used (and in what order)               │
│  - Resolution outcome                           │
│  - Customer satisfaction signal                 │
│  - Time to resolution                           │
└─────────────────────────────────────────────────┘
```

The live agent fallback node is always present in every DAG:

```yaml
nodes:
  - id: live-agent-fallback
    type: agent
    name: "Live Agent Fallback"
    description: "Handles requests that don't match any specialized DAG path"
    config:
      agent:
        model: "claude-sonnet-4-20250514"
        instruction: |
          You are a general-purpose customer service agent.
          Handle this request to the best of your ability.
          Your interaction will be analyzed to potentially create
          a specialized workflow for similar future requests.
        tools: [all_available_tools]
        max_turns: 25
```

#### Phase 2: Pattern Recognition

A background analysis agent periodically scans the trace store:

```yaml
# Pattern Recognition Config
pattern_recognition:
  trigger:
    min_similar_traces: 5           # Minimum traces before considering a pattern
    similarity_threshold: 0.85      # Semantic similarity threshold
    time_window: "7d"               # Look-back window
    
  analysis:
    clustering_method: semantic      # Group similar interactions
    feature_extraction:
      - intent_category              # What the customer wanted
      - tool_sequence                # What tools were used (and in what order)
      - resolution_path              # Steps taken to resolve
      - success_rate                 # % of successful resolutions in cluster
      
  output:
    # For each recognized pattern, produce:
    pattern:
      id: string
      description: string           # Natural language description
      frequency: integer            # How often this pattern occurs
      avg_resolution_time: duration
      success_rate: float
      representative_traces: [trace_id]
      proposed_dag_fragment:         # Suggested nodes and edges
        nodes: [node]
        edges: [edge]
```

**Example pattern recognition output:**

```yaml
pattern:
  id: "warranty-extension-request"
  description: |
    Customers requesting warranty extensions on recently purchased products.
    Common resolution: check purchase date, verify product eligibility,
    either extend warranty or explain why not.
  frequency: 47
  avg_resolution_time: "4m30s"
  success_rate: 0.89
  representative_traces:
    - trace-2026-06-01-abc
    - trace-2026-06-03-def
    - trace-2026-06-05-ghi
  proposed_dag_fragment:
    nodes:
      - id: handle-warranty-extension
        type: agent
        name: "Warranty Extension Handler"
        description: "Handles warranty extension requests"
        config:
          agent:
            model: "claude-sonnet-4-20250514"
            instruction: |
              You handle warranty extension requests.
              Steps:
              1. Look up the customer's purchase using order lookup
              2. Check if the product is within the warranty extension window (90 days from purchase)
              3. If eligible, extend the warranty and confirm
              4. If not eligible, explain the policy and offer alternatives
            tools:
              - name: lookup_order
                type: function_ref
                ref: "orders-api.lookup"
              - name: extend_warranty
                type: function_ref
                ref: "warranty-api.extend"
            output_schema:
              type: object
              properties:
                response_text: { type: string }
                resolution_status: { type: string, enum: [resolved, escalated] }
                warranty_extended: { type: boolean }
    edges:
      - from: classify-intent
        to: handle-warranty-extension
        when: "$.intent == 'warranty_extension'"
```

#### Phase 3: DAG Update Proposal

The pattern recognition agent generates a formal DAG update proposal:

```yaml
dag_update_proposal:
  id: "proposal-2026-06-08-001"
  dag_ref: "customer-support"
  base_version: "3.2.0"
  proposed_version: "3.3.0-rc.1"
  
  # What changes
  changes:
    - type: add_node
      node:
        id: handle-warranty-extension
        # ... (full node definition as above)
    
    - type: add_edge
      edge:
        from: classify-intent
        to: handle-warranty-extension
        when: "$.intent == 'warranty_extension'"
        condition:
          flag: "warranty_extension_handler"
          operator: is
          value: true
        priority: 5
    
    - type: update_node
      node_id: classify-intent
      updates:
        description: "Updated to include warranty_extension intent category"
  
  # Evidence
  evidence:
    pattern_id: "warranty-extension-request"
    trace_count: 47
    success_rate: 0.89
    time_saved_estimate: "2m per interaction"
  
  # Feature flag for canary
  canary_flag:
    name: "warranty_extension_handler"
    initial_rollout: 0.05      # 5% of traffic
    target_metrics:
      - name: resolution_rate
        target: ">= 0.85"
      - name: csat_score
        target: ">= 4.0"
      - name: avg_resolution_time
        target: "<= 300s"
```

#### Phase 4: Validation

Before deploying, the proposed DAG undergoes automated validation:

```
┌─────────────────────────────────────────────────┐
│  1. Schema Validation                           │
│     - Validate against JSON Schema              │
│     - Check all node IDs are valid              │
│     - Check all edge references are valid       │
│     ↓                                           │
│  2. Graph Validation                            │
│     - Verify DAG is acyclic                     │
│     - Verify all nodes reachable from entries   │
│     - Verify no orphan nodes                    │
│     - Verify flag-gated edges have fallbacks    │
│     ↓                                           │
│  3. Simulation                                  │
│     - Dry-run with representative traces        │
│     - Verify the new path activates correctly   │
│     - Verify existing paths still work          │
│     - Check for edge resolution conflicts       │
│     ↓                                           │
│  4. Safety Review                               │
│     - LLM judge reviews agent instructions      │
│     - Check for prompt injection vulnerabilities │
│     - Verify tool permissions are appropriate   │
│     - Check output schemas are complete         │
│     ↓                                           │
│  5. Human Review (optional, configurable)       │
│     - Generate diff for human review            │
│     - Require approval for certain node types   │
│     ↓                                           │
│  PASS → Deploy with canary flag                 │
│  FAIL → Return to proposal with error details   │
└─────────────────────────────────────────────────┘
```

#### Phase 5: Canary Deployment

```
┌─────────────────────────────────────────────────┐
│  Deploy new DAG version with canary flag        │
│  ↓                                              │
│  Feature flag service: warranty_extension_       │
│  handler = true for 5% of traffic               │
│  ↓                                              │
│  Monitor metrics (configurable observation       │
│  window, default 48 hours):                     │
│  ↓                                              │
│  ┌─────────────────────────────────────────┐    │
│  │ Metrics Evaluator (runs every 15 min)   │    │
│  │                                         │    │
│  │ Resolution rate:  0.91 ✓ (>= 0.85)     │    │
│  │ CSAT score:       4.3  ✓ (>= 4.0)      │    │
│  │ Avg resolution:   180s ✓ (<= 300s)     │    │
│  │ Error rate:       0.02 ✓ (<= 0.05)     │    │
│  │                                         │    │
│  │ Status: ALL_PASSING                     │    │
│  └─────────────────────────────────────────┘    │
│  ↓                                              │
│  Progressive rollout:                           │
│  5% → 25% → 50% → 100%                         │
│  (auto-advance if metrics hold for 24h each)    │
│  ↓                                              │
│  At 100%: Remove canary flag from edge           │
│  condition (edge becomes unconditional)          │
│  ↓                                              │
│  DAG version promoted: 3.3.0 (from 3.3.0-rc.1) │
└─────────────────────────────────────────────────┘
```

**Rollback trigger:**

```yaml
rollback_policy:
  triggers:
    - metric: error_rate
      threshold: "> 0.10"
      window: "30m"
      action: immediate_rollback
      
    - metric: resolution_rate
      threshold: "< 0.70"
      window: "2h"
      action: pause_and_alert
      
    - metric: csat_score
      threshold: "< 3.5"
      window: "4h"
      action: pause_and_alert
  
  rollback_action:
    # Set canary flag to false (routes all traffic to fallback edges)
    set_flag:
      name: "warranty_extension_handler"
      value: false
    # Alert the team
    notify:
      channels: [slack, pagerduty]
      message: "DAG canary rollback: {proposal_id}"
```

### 5.3 Interaction Loop State Machine

```
                    ┌──────────────┐
                    │  MONITORING  │ ← Trace store collects interactions
                    └──────┬───────┘
                           │ pattern detected (N >= threshold)
                           ▼
                    ┌──────────────┐
                    │  PROPOSING   │ ← Agent generates DAG update proposal
                    └──────┬───────┘
                           │ proposal generated
                           ▼
                    ┌──────────────┐
              ┌─────│  VALIDATING  │ ← Schema + graph + simulation checks
              │     └──────┬───────┘
              │            │ validation passed
    fail      │            ▼
    (back to  │     ┌──────────────┐
    proposing │     │  REVIEWING   │ ← Optional human review
    with      │     └──────┬───────┘
    feedback) │            │ approved
              │            ▼
              │     ┌──────────────┐
              ├─────│   CANARY     │ ← Progressive rollout with metrics
              │     └──────┬───────┘
              │            │ metrics good at 100%
    rollback  │            ▼
              │     ┌──────────────┐
              └─────│  PROMOTED    │ ← Remove canary flag, update version
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  MONITORING  │ ← Continue monitoring for new patterns
                    └──────────────┘
```

---

## 6. Runtime Architecture

### 6.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Config Layer                          │
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐   │
│  │ DAG YAML │   │ JSON     │   │ Version Control  │   │
│  │ Config   │   │ Schema   │   │ (Git)            │   │
│  └────┬─────┘   └────┬─────┘   └────────┬─────────┘   │
│       │              │                    │             │
│       ▼              ▼                    ▼             │
│  ┌──────────────────────────────────────────────┐      │
│  │           Config Validator                    │      │
│  │  (schema check + graph analysis + simulate)  │      │
│  └───────────────────┬──────────────────────────┘      │
└──────────────────────┼──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  Runtime Layer                           │
│                                                         │
│  ┌──────────────────────────────────────────────┐      │
│  │           DAG Interpreter                     │      │
│  │                                               │      │
│  │  1. Load DAG config                           │      │
│  │  2. Resolve entry node(s)                     │      │
│  │  3. Execute node → evaluate outgoing edges    │      │
│  │  4. For each matching edge → execute target   │      │
│  │  5. Repeat until exit node(s) reached         │      │
│  └──────┬───────────────────────────┬────────────┘      │
│         │                           │                    │
│         ▼                           ▼                    │
│  ┌──────────────┐           ┌──────────────────┐       │
│  │ Node          │           │ Edge Evaluator   │       │
│  │ Executors     │           │                  │       │
│  │               │           │ - when expr      │       │
│  │ - FunctionExe │           │ - flag lookup    │       │
│  │ - AgentExec   │           │ - priority sort  │       │
│  │ - SubDAGExec  │           │ - transform      │       │
│  │ - ValidatorEx │           └────────┬─────────┘       │
│  │ - RouterExec  │                    │                  │
│  └──────┬────────┘                    │                  │
│         │                             │                  │
│         ▼                             ▼                  │
│  ┌──────────────────────────────────────────────┐      │
│  │              Integration Layer                │      │
│  │                                               │      │
│  │  ┌────────┐ ┌────────┐ ┌──────┐ ┌─────────┐ │      │
│  │  │Feature │ │Metrics │ │Trace │ │ Tool    │ │      │
│  │  │ Flag   │ │Collect.│ │Store │ │Registry │ │      │
│  │  │Service │ │        │ │      │ │         │ │      │
│  │  └────────┘ └────────┘ └──────┘ └─────────┘ │      │
│  └──────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

### 6.2 Runtime Recommendation

We recommend a **layered approach** with different runtimes for different deployment contexts:

#### Development & Testing: Custom Lightweight Interpreter

A custom Python/TypeScript interpreter that directly executes DAG configs:

- **Advantages:** Fast iteration, easy debugging, no infrastructure dependencies
- **Implementation:** ~2000 lines of code for the core interpreter
- **Used for:** Local development, CI testing, simulation, DAG validation

```python
# Pseudocode for core interpreter
class DAGInterpreter:
    def __init__(self, config: DAGConfig, flag_service, metrics_collector):
        self.graph = build_graph(config)
        self.flags = flag_service
        self.metrics = metrics_collector
    
    async def execute(self, input_data: dict) -> dict:
        entry_nodes = self.graph.entry_nodes()
        state = ExecutionState(input_data)
        
        for node in self.topological_order(entry_nodes):
            # Check if this node is reachable via any active edge
            if not self.has_active_incoming_edge(node, state):
                continue
            
            # Execute the node
            result = await self.execute_node(node, state)
            state.set_output(node.id, result)
            
            # Collect metrics
            self.metrics.record(node.id, result.duration, result.status)
        
        return state.final_output()
    
    def has_active_incoming_edge(self, node, state) -> bool:
        for edge in self.graph.incoming_edges(node):
            if self.evaluate_edge(edge, state):
                return True
        return False
    
    def evaluate_edge(self, edge, state) -> bool:
        # Check data condition
        if edge.when and not evaluate_expression(edge.when, state):
            return False
        # Check feature flag condition
        if edge.condition:
            flag_value = self.flags.evaluate(
                edge.condition.flag,
                context=state.flag_context()
            )
            if not compare(flag_value, edge.condition.operator, edge.condition.value):
                return False
        return True
```

#### Production: Temporal via Bridge Layer

For production deployments requiring durability, we compile DAG configs to Temporal workflows:

- **Why Temporal:** Battle-tested durable execution, automatic retries, workflow versioning, observability
- **Bridge approach:** Similar to Zigflow — a compiler that translates DAG YAML into Temporal workflow definitions
- **Used for:** Production, long-running workflows, workflows requiring durable execution guarantees

```
DAG YAML Config
     │
     ▼
┌─────────────┐
│ DAG-to-      │
│ Temporal     │
│ Compiler     │
└──────┬──────┘
       │ generates
       ▼
┌─────────────┐     ┌──────────────┐
│ Temporal     │────▶│ Temporal     │
│ Workflow     │     │ Server       │
│ Definition   │     │ (durable     │
│ (Go/Python)  │     │  execution)  │
└─────────────┘     └──────────────┘
```

**Mapping DAG concepts to Temporal:**

| DAG Concept | Temporal Equivalent |
|-------------|-------------------|
| Node (function) | Activity |
| Node (agent) | Activity wrapping LLM call |
| Node (sub_dag) | Child Workflow |
| Node (validator) | Activity with retry/compensation |
| Edge (unconditional) | Workflow transition |
| Edge (when condition) | Workflow conditional (if/switch) |
| Edge (flag condition) | Activity calling flag service + conditional |
| Fan-out (multiple matching edges) | Parallel activities |
| Fan-in (multiple edges to same node) | Await all / Promise.all |

#### Alternative Production Runtimes

The config format is runtime-portable. Other viable runtimes:

| Runtime | When to Use |
|---------|-------------|
| **ADK 2.0 Graph Workflows** | When deeply integrated with Google Cloud / Vertex AI |
| **Argo Workflows** | When running on Kubernetes with container-based node execution |
| **Custom on Cloud Run / Lambda** | For serverless, event-driven execution |
| **CNCF Serverless Workflow runtimes** | When adopting the Serverless Workflow ecosystem |

### 6.3 Node Executor Details

#### Function Executor

```
Input → HTTP/gRPC/Script call → Output
        ↓
    Apply input_mapping (extract from state)
        ↓
    Execute call (with retry policy)
        ↓
    Apply output_mapping (inject into state)
```

#### Agent Executor

```
Input → LLM Session Setup → Conversation Loop → Output
        ↓                     ↓
    Load instruction       Agent calls tools
    Load tools             (function_ref → Function Executor)
    Set temperature        ↓
    Set output_schema      Check max_turns
                           ↓
                       Force structured output via schema
                           ↓
                       Validate output against schema
```

#### Sub-DAG Executor

```
Input → Load referenced DAG config → Recursive DAG execution → Output
        ↓
    Resolve ref (file path or registry lookup)
        ↓
    Version constraint check
        ↓
    Apply input_mapping
        ↓
    Execute sub-DAG (new DAGInterpreter instance)
        ↓
    Apply output_mapping
```

#### Validator Executor

```
Input → Run all checks in parallel → Aggregate results → Pass/Fail
        ↓
    llm_judge: LLM call with judge prompt → score
    json_schema: JSON Schema validation → pass/fail
    function_ref: External API call → score
        ↓
    All checks must pass (AND logic)
        ↓
    If fail:
      on_failure=retry → re-execute preceding node (up to max_retries)
      on_failure=escalate → route to escalation path
      on_failure=fallback → route to fallback_node
```

### 6.4 Metrics Collection

Every node execution emits standardized metrics:

```yaml
metrics_schema:
  # Per-node metrics (automatic)
  node_execution:
    - node_id: string
    - dag_name: string
    - dag_version: string
    - status: success | failure | timeout | skipped
    - duration_ms: integer
    - retry_count: integer
    - error_type: string        # If failed
    
  # Per-edge metrics (automatic)
  edge_evaluation:
    - from_node: string
    - to_node: string
    - flag_name: string         # If flag-gated
    - flag_value: any
    - condition_result: boolean
    - edge_taken: boolean
    
  # Per-workflow metrics (automatic)
  workflow_execution:
    - dag_name: string
    - dag_version: string
    - total_duration_ms: integer
    - nodes_executed: integer
    - final_status: success | failure | timeout
    - entry_path: [string]      # Sequence of node IDs traversed
    
  # Custom metrics (per-node, defined in metadata.metrics)
  custom:
    - metric_name: string
    - value: number
    - labels: object
```

### 6.5 Feature Flag Service Interface

The runtime integrates with any feature flag service that implements this interface:

```typescript
interface FeatureFlagService {
  /**
   * Evaluate a feature flag in the context of a workflow execution.
   * @param flagName - The flag identifier referenced in edge conditions
   * @param context - Execution context for targeting/percentage rules
   * @returns The flag value (boolean, string, number, or object)
   */
  evaluate(flagName: string, context: FlagContext): Promise<any>;
}

interface FlagContext {
  workflow_id: string;
  workflow_version: string;
  node_id: string;
  edge_to: string;
  user_id?: string;
  session_id?: string;
  environment: string;
  timestamp: string;
  custom_attributes?: Record<string, any>;
}
```

**Supported flag services** (via adapters):
- LaunchDarkly
- Unleash (open-source)
- Flagsmith (open-source)
- Split
- Custom (HTTP endpoint)

---

## 7. Appendix: Sources

### 7.1 Workflow Systems Researched

| System | Primary Source |
|--------|--------------|
| Argo Workflows | [argo-workflows.readthedocs.io](https://argo-workflows.readthedocs.io/en/latest/walk-through/dag/), [GitHub examples](https://github.com/argoproj/argo-workflows/blob/main/examples/dag-conditional-parameters.yaml) |
| AWS Step Functions (ASL) | [AWS Docs](https://docs.aws.amazon.com/step-functions/latest/dg/concepts-amazon-states-language.html), [states-language.net](https://states-language.net/) |
| GitHub Actions | [GitHub Docs](https://docs.github.com/actions/using-workflows/workflow-syntax-for-github-actions) |
| Tekton Pipelines | [tekton.dev](https://tekton.dev/docs/pipelines/pipelines/), [When Expressions](https://www.mintlify.com/tektoncd/pipeline/api/types/when-expressions) |
| Temporal | [docs.temporal.io](https://docs.temporal.io/workflows), [Temporal DSL](https://temporal.io/code-exchange/temporal-dsl) |
| Zigflow | [zigflow.dev](https://zigflow.dev/articles/why-i-built-a-yaml-dsl-for-temporal-workflows/) |
| Apache Airflow | [airflow.apache.org](https://airflow.apache.org/docs/apache-airflow/2.5.2/core-concepts/dags.html) |
| DAG Factory | [Astronomer docs](https://www.astronomer.io/docs/learn/dag-factory), [AWS MWAA](https://aws.amazon.com/blogs/big-data/dynamic-dag-generation-with-yaml-and-dag-factory-in-amazon-mwaa/) |
| Dapr Workflows | [docs.dapr.io](https://docs.dapr.io/developing-applications/building-blocks/workflow/workflow-overview/) |
| Prefect | [docs.prefect.io](https://docs.prefect.io/) |
| CNCF Serverless Workflow | [serverlessworkflow.io](https://serverlessworkflow.io/), [GitHub spec](https://github.com/serverlessworkflow/specification/blob/main/dsl-reference.md) |
| Google ADK | [adk.dev](https://adk.dev/agents/config/), [Workflow Agents](https://google.github.io/adk-docs/agents/workflow-agents/) |
| CrewAI | [docs.crewai.com](https://docs.crewai.com/), [GitHub](https://github.com/crewaiinc/crewai) |
| LangGraph | [langchain.com/langgraph](https://www.langchain.com/langgraph), [GitHub](https://github.com/langchain-ai/langgraph) |
| CUE | [cuelang.org](https://cuelang.org/docs/concept/configuration-use-case/), [GitHub discussion](https://github.com/cue-lang/cue/discussions/669) |

### 7.2 Research Papers and Articles

| Topic | Source |
|-------|--------|
| Dynamic Runtime Graphs Survey | [arxiv.org/html/2603.22386v1](https://arxiv.org/html/2603.22386v1) |
| Automated Agentic Workflow Generation | [emergentmind.com](https://www.emergentmind.com/topics/automated-agentic-workflow-generation) |
| Self-Evolving AI Agents | [emergentmind.com](https://www.emergentmind.com/topics/self-evolving-ai-agent) |
| Evolving Excellence: Automated Optimization of LLM-based Agents | [arxiv.org/pdf/2512.09108](https://arxiv.org/pdf/2512.09108) |
| Optimizing Agentic Workflows using Meta-tools | [arxiv.org/html/2601.22037v2](https://arxiv.org/html/2601.22037v2) |
| LLM-format comparison (YAML vs JSON) | [medium.com/@michael.hannecke](https://medium.com/@michael.hannecke/beyond-json-picking-the-right-format-for-llm-pipelines-b65f15f77f7d) |
| BAML vs POML vs YAML vs JSON for LLM | [augmentcode.com](https://www.augmentcode.com/learn/baml-vs-poml-vs-yaml-vs-json-for-llm-prompts) |
| Feature Flags Best Practices | [octopus.com](https://octopus.com/devops/feature-flags/feature-flag-best-practices/) |
| Canary Releases with Feature Flags | [configcat.com](https://configcat.com/blog/how-to-implement-a-canary-release-with-feature-flags/) |
| CUE vs Jsonnet vs Dhall | [pv.wtf](https://pv.wtf/posts/taming-the-beast) |

### 7.3 Key Design Decisions Log

| Decision | Options Considered | Chosen | Rationale |
|----------|-------------------|--------|-----------|
| Config format | YAML, JSON, CUE, Jsonnet | **YAML** | Best LLM editability, most readable, good diff, standard tooling |
| Graph model | Implicit (deps lists), explicit (nodes+edges) | **Explicit nodes + edges** | Feature flag conditions belong on edges; explicit model is more diffable |
| Edge conditions | On nodes (Argo style), on edges, separate router nodes | **On edges + optional router nodes** | Edge conditions are the primary mechanism; routers are convenience sugar |
| Feature flag integration | Inline flag definitions, external reference | **External reference only** | Separation of concerns: DAG config doesn't own flag state |
| Sub-workflow model | Inline nesting, external reference | **External reference (via `ref`)** | Keeps files manageable, enables independent versioning |
| Runtime | Single runtime, portable config | **Portable config + multiple runtimes** | Custom interpreter for dev, Temporal for production |
| Validation | Runtime-only, pre-execution | **Pre-execution (schema + graph + simulation)** | Catch errors before deployment, enable CI integration |
| LLM mutation interface | Direct file editing, structured tool commands | **Structured tool commands** | Safer, auditable, validates atomically |

---

*End of DAG Config Specification v0.1.0-draft*
