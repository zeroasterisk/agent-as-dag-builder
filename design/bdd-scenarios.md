# Agent-as-DAG-Builder: BDD Scenarios & Exploration Plan

**Version:** 0.1.0-draft
**Date:** 2026-06-08
**Status:** Exploration — defining what we want before building

---

## Purpose

This document defines the full lifecycle of the DAG builder system through
concrete BDD scenarios. The goals are:

1. **Describe clearly** what the system does across any implementation
2. **Test against it** to discover problems before committing to code
3. **Build sandbox implementations** to validate specific paths

We use a benchmark agent harness (Tau2 customer service) as the running example
because it's a domain with repeated processes (classify → route → handle → respond)
that naturally lends itself to DAG encoding.

---

## Conventions

All DAGs follow these structural rules:
- **Validation nodes** before and after every step (schema validation, auth N/Z, guardrails)
- **Recovery nodes** for anticipated failure modes
- **Exception fallback** to live agent with context dump (task, last good state, current state, errors)
- **Metrics emission** at every node (latency, success, error type, user satisfaction)

---

## Scenario Group 1: Cold Start — No DAGs Exist

### Scenario 1.1: First user interaction, agent handles live

```gherkin
Feature: Cold start — live agent handles novel requests

  Scenario: User asks a question with no existing DAG
    Given the agent has no workflow DAGs defined
    And a user sends "Cancel my flight reservation AH3BDS"
    When the agent receives the request
    Then the agent handles the request live using LLM reasoning
    And the agent successfully cancels the reservation
    And the agent emits metrics: {task_type: "flight_cancellation", handled_by: "live_agent", success: true, latency_ms: 4200}
    And the agent does NOT attempt to write a DAG (single occurrence)

  Scenario: User asks a similar question for the Nth time
    Given the agent has handled 5 "flight cancellation" requests live
    And the success rate for live handling is 80%
    And the average latency is 4000ms
    When the agent receives a 6th "Cancel my flight reservation XY1234"
    Then the agent handles it live (still no DAG)
    And the agent's taxonomy tracker notes: {task_type: "flight_cancellation", count: 6, live_success_rate: 0.83}
    And the agent proposes: "I've handled 6 flight cancellations. Should I create a workflow for this?"

  Scenario: Agent decides to create a DAG after threshold
    Given the taxonomy tracker shows {task_type: "flight_cancellation", count: 10, live_success_rate: 0.85}
    When the threshold for DAG creation is met (count >= 10, success_rate >= 0.7)
    Then the agent generates a DAG config for "flight_cancellation"
    And the DAG includes nodes: [validate_input, lookup_reservation, check_cancellation_policy, execute_cancellation, validate_result, send_confirmation]
    And each node has pre/post validation nodes
    And the DAG includes a recovery node for "reservation not found"
    And the DAG includes an exception fallback to live agent
    And the agent submits the DAG for validation
```

### Scenario 1.2: DAG validation and deployment

```gherkin
Feature: DAG validation before deployment

  Scenario: Valid DAG passes schema and simulation
    Given the agent has proposed a new "flight_cancellation" DAG
    When the system validates the DAG
    Then schema validation passes (all nodes have types, edges have valid targets)
    And simulation runs against 3 historical inputs without errors
    And the DAG is deployed with feature flag "dag_flight_cancel_v1" = false (off by default)
    And the DAG version is recorded: {name: "flight_cancellation", version: "1.0.0", status: "deployed_inactive"}

  Scenario: Invalid DAG fails validation
    Given the agent has proposed a DAG with a node referencing a nonexistent tool
    When the system validates the DAG
    Then schema validation fails: "Node 'execute_cancellation' references tool 'cancel_flight' which is not registered"
    And the DAG is NOT deployed
    And the agent is notified: "DAG validation failed: [error details]"
    And the agent can fix and resubmit

  Scenario: DAG with security violation fails guardrail check
    Given the agent has proposed a DAG that skips auth validation before accessing customer data
    When the system validates the DAG
    Then guardrail check fails: "Node 'lookup_reservation' accesses customer data without preceding auth_validation node"
    And the DAG is NOT deployed
```

---

## Scenario Group 2: DAG Execution — Happy Path

### Scenario 2.1: Request matches a DAG

```gherkin
Feature: DAG handles known requests

  Scenario: Incoming request matches an active DAG
    Given the "flight_cancellation" DAG is deployed and flag "dag_flight_cancel_v1" = true
    And a user sends "Cancel my flight reservation QR5678"
    When the intent classifier matches to "flight_cancellation" with confidence 0.95
    And confidence >= threshold (0.8)
    Then the system executes the "flight_cancellation" DAG (not the live agent)
    And the DAG executes: validate_input → lookup_reservation → check_policy → execute_cancellation → validate_result → send_confirmation
    And each validation node passes
    And metrics are emitted: {handled_by: "dag", dag_name: "flight_cancellation", version: "1.0.0", success: true, latency_ms: 800}

  Scenario: DAG execution is faster and cheaper than live agent
    Given the "flight_cancellation" DAG has handled 50 requests
    And the live agent previously handled this in ~4000ms with ~$0.05 LLM cost per request
    When comparing DAG vs live metrics
    Then DAG average latency is ~800ms (5x faster)
    And DAG LLM cost is ~$0.002 per request (validation nodes only, 25x cheaper)
    And DAG success rate is 92% (vs live 85%)
```

### Scenario 2.2: Validation nodes catch issues

```gherkin
Feature: Validation nodes enforce correctness

  Scenario: Pre-validation catches invalid input
    Given the "flight_cancellation" DAG is executing
    When the validate_input node checks the reservation ID format
    And the reservation ID "INVALID!!!" does not match pattern [A-Z0-9]{6}
    Then the validation node fails: {valid: false, error: "Invalid reservation ID format"}
    And the DAG does NOT proceed to lookup_reservation
    And the DAG responds to the user: "I couldn't process that reservation ID. Could you double-check it?"
    And metrics are emitted: {handled_by: "dag", success: false, failure_node: "validate_input", failure_type: "input_validation"}

  Scenario: Post-validation catches unexpected state
    Given the "flight_cancellation" DAG has executed execute_cancellation
    When the post-validation node checks the cancellation result
    And the result state is {status: "partially_cancelled", refund: null}
    Then the post-validation node flags unexpected state
    And the exception is NOT in the recovery nodes
    Then the DAG falls back to the live agent with context:
      | Field | Value |
      | task | "Cancel flight reservation QR5678" |
      | last_good_state | {node: "execute_cancellation", output: {status: "partially_cancelled"}} |
      | current_state | {status: "partially_cancelled", refund: null} |
      | errors | ["Post-validation failed: refund is null for cancelled reservation"] |
    And the live agent takes over with full context
```

---

## Scenario Group 3: DAG Failure & Recovery

### Scenario 3.1: Known failure with recovery node

```gherkin
Feature: Recovery nodes handle anticipated failures

  Scenario: Reservation not found — recovery node handles it
    Given the "flight_cancellation" DAG is executing
    When lookup_reservation returns {error: "reservation_not_found"}
    And this error type has a recovery node defined
    Then the recovery node executes: search_by_customer_email
    And if the alternative lookup succeeds, the DAG continues from check_policy
    And metrics: {recovery_triggered: true, recovery_type: "alternative_lookup", recovery_success: true}
```

### Scenario 3.2: Unknown failure — fallback to live agent

```gherkin
Feature: Unexpected errors fall back to live agent

  Scenario: Tool returns unexpected error
    Given the "flight_cancellation" DAG is executing
    When execute_cancellation raises an unhandled exception: "ServiceUnavailable: booking system down"
    And no recovery node matches this error type
    Then the DAG execution halts
    And the system falls back to the live agent with:
      | task | "Cancel flight reservation QR5678" |
      | last_good_state | {node: "check_policy", output: {cancellable: true, fee: 0}} |
      | current_state | {node: "execute_cancellation", error: "ServiceUnavailable"} |
      | errors | ["ServiceUnavailable: booking system down"] |
      | truncated | false |
    And the live agent sees this context and can decide to retry, wait, or handle differently
    And metrics: {handled_by: "dag_then_live", fallback_reason: "unhandled_exception", exception: "ServiceUnavailable"}
```

---

## Scenario Group 4: DAG Mutation — Agent Updates the DAG

### Scenario 4.1: Agent learns a new path from live handling

```gherkin
Feature: Agent encodes new patterns as DAG updates

  Scenario: Live agent handles a variant and proposes DAG update
    Given the "flight_cancellation" DAG exists (v1.0.0)
    And a user sends "Cancel my flight and rebook for next week"
    When the intent classifier matches "flight_cancellation" with confidence 0.6 (below threshold)
    Then the live agent handles the combined cancel+rebook request
    And the agent succeeds
    And the agent recognizes: "This is a cancel+rebook pattern, not just cancel"
    And the agent proposes a DAG update: add a "rebook" branch after cancellation
    And the proposed update is: add nodes [check_rebooking_options, execute_rebook, validate_rebook] with edges from execute_cancellation

  Scenario: Proposed DAG update goes through validation and canary
    Given the agent has proposed adding a "rebook" branch to flight_cancellation
    When the system validates the updated DAG (v1.1.0)
    Then schema validation passes
    And simulation passes against historical cancel+rebook inputs
    And the updated DAG is deployed with flag "dag_flight_cancel_v1.1_rebook" = false
    And the system schedules a canary rollout: flag = true for 5% of users
    And metrics collection begins for the canary population
```

### Scenario 4.2: Canary succeeds → promote, canary fails → rollback

```gherkin
Feature: Safe progressive rollout of DAG updates

  Scenario: Canary succeeds and DAG is promoted
    Given DAG v1.1.0 is running for 5% of users (canary)
    And after 100 canary executions:
      | Metric | v1.0.0 (control) | v1.1.0 (canary) |
      | success_rate | 92% | 94% |
      | latency_p50 | 800ms | 850ms |
      | fallback_rate | 8% | 4% |
    When the promotion criteria are met (canary success >= control success)
    Then the flag "dag_flight_cancel_v1.1_rebook" is set to true for 100% of users
    And v1.0.0 is retained as rollback target
    And v1.1.0 becomes the default

  Scenario: Canary fails and DAG is rolled back
    Given DAG v1.1.0 is running for 5% of users (canary)
    And after 50 canary executions, success_rate is 60% (vs control 92%)
    When the rollback criteria are met (canary success < control success - 10%)
    Then the flag "dag_flight_cancel_v1.1_rebook" is set to false for all users
    And v1.1.0 is marked as "rolled_back"
    And the agent is notified: "DAG v1.1.0 rolled back. Canary success 60% vs control 92%."
    And the agent can analyze failures and propose a revised v1.2.0
```

---

## Scenario Group 5: Infrastructure — Hot Reload & Persistence

### Scenario 5.1: DAG config update lifecycle

```gherkin
Feature: Safe DAG config updates via Temporal

  Scenario: Agent updates DAG config while system is running
    Given the system is running with DAG v1.0.0 in Temporal
    And 3 conversations are in-flight using v1.0.0
    When the agent commits DAG v1.1.0 to the config store
    And CI/CD deploys a new Temporal worker with v1.1.0
    Then the 3 in-flight conversations continue on v1.0.0 (Temporal versioned workers)
    And new conversations start on v1.1.0 (with flag = false, so DAG is inactive)
    And no conversations are disrupted

  Scenario: Temporal worker crashes mid-execution
    Given a conversation is executing the "flight_cancellation" DAG at node "execute_cancellation"
    When the Temporal worker process crashes
    Then Temporal assigns the workflow to a new worker
    And the new worker replays the event history
    And execution resumes at "execute_cancellation" (activities before this are skipped via replay)
    And the user experiences a brief delay but no data loss
    And metrics: {recovery: true, recovery_type: "temporal_replay", resumed_at: "execute_cancellation"}
```

### Scenario 5.2: Feature flag lifecycle

```gherkin
Feature: Feature flags control DAG activation

  Scenario: Deploy new DAG version with flag off
    Given DAG v1.1.0 is committed and deployed
    And flag "dag_flight_cancel_v1.1" = false (in external flag service)
    When a user sends a flight cancellation request
    Then the system evaluates the flag for this user → false
    And the system uses v1.0.0 (previous active version)

  Scenario: Enable flag for canary population
    Given flag "dag_flight_cancel_v1.1" is set to true for user_group="canary_5pct"
    When a canary user sends a flight cancellation request
    Then the system evaluates the flag → true
    And the system uses v1.1.0

  Scenario: Flag service is unreachable
    Given the external flag service is down
    When the system needs to evaluate a flag
    Then the system uses the default value (false = use previous stable version)
    And metrics: {flag_evaluation: "default", flag_service: "unreachable"}
```

---

## Scenario Group 6: Taxonomy & Metrics

### Scenario 6.1: System learns when to create DAGs

```gherkin
Feature: Taxonomy-driven DAG creation decisions

  Scenario: High-frequency task type triggers DAG suggestion
    Given the taxonomy tracker records:
      | task_type | count | live_success_rate | avg_latency | has_dag |
      | flight_cancel | 45 | 0.88 | 3800ms | yes |
      | flight_rebook | 12 | 0.75 | 5200ms | no |
      | baggage_claim | 3 | 0.67 | 6100ms | no |
    When the system evaluates DAG creation opportunities
    Then "flight_rebook" is flagged: {count: 12 (>= 10), success_rate: 0.75 (>= 0.7)} → recommend DAG
    And "baggage_claim" is NOT flagged: {count: 3 (< 10)} → insufficient data
    And the agent is notified: "Consider creating a DAG for 'flight_rebook' — 12 occurrences, 75% success rate live"

  Scenario: Low-frequency task type stays live
    Given task_type "vip_escalation" has count: 2 over 30 days
    When the system evaluates DAG creation
    Then "vip_escalation" is not recommended for DAG creation
    And it remains handled by the live agent
```

---

## Exploration Plan

### Phase 1: Define (THIS DOCUMENT)
- [x] Write BDD scenarios covering the full lifecycle
- [ ] Review with stakeholders
- [ ] Refine based on discovered gaps

### Phase 2: Sandbox — Infrastructure
Build minimal sandbox to validate:
- [ ] ADK WorkflowAgent from YAML config (load, execute, validate)
- [ ] Temporal integration (deploy, hot reload, crash recovery, versioned workers)
- [ ] Feature flag evaluation in DAG conditions (inject flags, branch on them)
- [ ] Config persistence (commit YAML, trigger CI/CD, deploy new worker)

### Phase 3: Sandbox — Agent Loop
Build minimal sandbox to validate:
- [ ] Intent classification → DAG routing vs live agent
- [ ] Validation nodes (pre/post, schema check, guardrails)
- [ ] Exception fallback with context dump to live agent
- [ ] Taxonomy tracker (count tasks, suggest DAG creation)
- [ ] Agent proposes DAG → validation → canary deployment

### Phase 4: Refine PRD & Design
- [ ] Update DAG config spec based on sandbox learnings
- [ ] Update architecture based on Temporal integration learnings
- [ ] Finalize runtime choices
- [ ] Estimate implementation effort for production version
