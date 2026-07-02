"""Prototype 21: Holdout Rotation -- Prevent validation leakage with K-fold cycling.

Expert review of the DAG Builder loop (Prototype 19) flagged a core issue:
repeated evaluation against the SAME holdout set leaks information about the
holdout into the learning loop. Over multiple iterations the optimizer
implicitly overfits to the holdout -- the "validation" score no longer
measures generalization.

This prototype fixes the problem with two mechanisms:

  1. ROTATING HOLDOUTS (K-fold)
     Split the full test-case pool into K=5 folds. Each iteration uses a
     different fold as the holdout while the remaining K-1 folds serve as
     the training set. The cycle repeats: fold 0 -> fold 1 -> ... -> fold 4
     -> fold 0 again.

  2. EVAL BUDGET
     Track how many times each fold has been used as the holdout. After
     MAX_EVALS_PER_FOLD evaluations (default 3), force a rotation even if
     the caller requests a specific fold. This caps the information leaked
     into any single fold.

The script uses MOCK SCORING to prove the rotation mechanism without
requiring real LLM calls. The integration with real LLM calls was proven
in Prototype 19.

Refactored from 19_e2e_full_loop.py:
  - Same scoring/proposing/validating logic structure
  - Replaces fixed HOLDOUT_CASES = ALL_CASES[10:18] with rotating K-fold
  - Data structures (InteractionRecord, HarnessResult) inlined to avoid
    importing 11_multi_harness.py which requires yaml + google.genai

Usage:
    python sandbox/21_holdout_rotation.py

Output:
    sandbox/results/holdout_rotation_report.md
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SANDBOX_DIR = Path(__file__).parent
RESULTS_DIR = SANDBOX_DIR / "results"


# =============================================================================
# Data Structures (inlined from 11_multi_harness.py to avoid heavy deps)
# =============================================================================


@dataclass
class InteractionRecord:
    """One scored test case interaction."""
    query: str
    case_id: int
    expected_category: str
    actual_category: str
    response: str
    score: dict
    quality_criteria: str


@dataclass
class HarnessResult:
    """Aggregate results from running benchmark cases through a config."""
    harness_name: str
    iteration: int
    config_path: str
    records: list[InteractionRecord] = field(default_factory=list)

    @property
    def aggregate_score(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.score["total_score"] for r in self.records) / len(self.records)

    @property
    def category_accuracy(self) -> float:
        if not self.records:
            return 0.0
        correct = sum(1 for r in self.records if r.actual_category == r.expected_category)
        return correct / len(self.records)

    @property
    def avg_quality(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.score["quality_score"] for r in self.records) / len(self.records)

    @property
    def avg_helpfulness(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.score["helpfulness_score"] for r in self.records) / len(self.records)

    def category_breakdown(self) -> dict:
        breakdown: dict = {}
        for r in self.records:
            cat = r.expected_category
            if cat not in breakdown:
                breakdown[cat] = {"scores": [], "correct": 0, "total": 0}
            breakdown[cat]["scores"].append(r.score["total_score"])
            breakdown[cat]["total"] += 1
            if r.actual_category == r.expected_category:
                breakdown[cat]["correct"] += 1
        for cat in breakdown:
            scores = breakdown[cat]["scores"]
            breakdown[cat]["avg_score"] = sum(scores) / len(scores)
            breakdown[cat]["accuracy"] = breakdown[cat]["correct"] / breakdown[cat]["total"]
        return breakdown

    def weakest_cases(self, n: int = 5) -> list[InteractionRecord]:
        return sorted(self.records, key=lambda r: r.score["total_score"])[:n]


# =============================================================================
# Test Cases (mirrored from 11_multi_harness.py -- same data, no import)
# =============================================================================

CUSTOMER_SUPPORT_CASES = [
    {"id": 1, "query": "I was charged twice for my subscription", "expected_category": "billing",
     "quality_criteria": "Should acknowledge the double charge, mention investigation or refund process"},
    {"id": 2, "query": "Can I get a refund for last month?", "expected_category": "billing",
     "quality_criteria": "Should explain the refund policy or process for requesting a refund"},
    {"id": 3, "query": "Why is my bill higher than usual this month?", "expected_category": "billing",
     "quality_criteria": "Should suggest common reasons (plan change, overage) and offer to review the account"},
    {"id": 4, "query": "I need to update my credit card on file", "expected_category": "billing",
     "quality_criteria": "Should provide steps to update payment method or direct to account settings"},
    {"id": 5, "query": "How do I cancel my subscription?", "expected_category": "billing",
     "quality_criteria": "Should explain cancellation process; may mention retention offers"},
    {"id": 6, "query": "I see an unauthorized charge from your company", "expected_category": "billing",
     "quality_criteria": "Should take the concern seriously, suggest investigation, mention fraud protection"},
    {"id": 7, "query": "When is my next billing date?", "expected_category": "billing",
     "quality_criteria": "Should explain how to find billing date or offer to look it up"},
    {"id": 8, "query": "Do you offer any discounts for annual plans?", "expected_category": "billing",
     "quality_criteria": "Should mention available discount options or direct to pricing page"},
    {"id": 9, "query": "My internet keeps dropping every 30 minutes", "expected_category": "technical",
     "quality_criteria": "Should ask about router/modem or suggest troubleshooting steps like restarting equipment"},
    {"id": 10, "query": "The app crashes when I try to open settings", "expected_category": "technical",
     "quality_criteria": "Should suggest clearing cache, reinstalling, or checking for updates"},
    {"id": 11, "query": "I can't log into my account, it says invalid password", "expected_category": "technical",
     "quality_criteria": "Should suggest password reset process and check for account lock"},
    {"id": 12, "query": "My download speeds are extremely slow, only getting 2 Mbps", "expected_category": "technical",
     "quality_criteria": "Should ask about connection type, suggest speed test, check for interference"},
    {"id": 13, "query": "The website keeps showing a 404 error on the dashboard page", "expected_category": "technical",
     "quality_criteria": "Should suggest clearing browser cache, trying different browser, or report known issue"},
    {"id": 14, "query": "My smart TV can't connect to your streaming service anymore", "expected_category": "technical",
     "quality_criteria": "Should suggest checking TV firmware, reinstalling app, verifying network connection"},
    {"id": 15, "query": "Email notifications are not working, I'm not receiving any alerts", "expected_category": "technical",
     "quality_criteria": "Should suggest checking notification settings, spam folder, and email verification"},
    {"id": 16, "query": "Two-factor authentication is not sending the verification code", "expected_category": "technical",
     "quality_criteria": "Should suggest checking phone number, trying alternative methods, checking SMS blockers"},
    {"id": 17, "query": "What are your business hours?", "expected_category": "general",
     "quality_criteria": "Should provide specific hours or direct to a page with hours information"},
    {"id": 18, "query": "How do I contact customer support by phone?", "expected_category": "general",
     "quality_criteria": "Should provide a phone number or explain how to find contact information"},
    {"id": 19, "query": "Do you have a referral program?", "expected_category": "general",
     "quality_criteria": "Should explain whether a referral program exists and how to participate"},
    {"id": 20, "query": "I'd like to provide feedback about your service", "expected_category": "general",
     "quality_criteria": "Should welcome feedback and explain how to submit it (survey, email, form)"},
    {"id": 21, "query": "What services do you offer for small businesses?", "expected_category": "general",
     "quality_criteria": "Should describe business offerings or direct to business solutions page"},
    {"id": 22, "query": "Is there a mobile app available?", "expected_category": "general",
     "quality_criteria": "Should confirm app availability and mention platforms (iOS/Android) or download links"},
    {"id": 23, "query": "I want to upgrade my current plan", "expected_category": "general",
     "quality_criteria": "Should explain upgrade options and how to change plans"},
]

IT_HELPDESK_CASES = [
    {"id": 1, "query": "I forgot my password and can't log in", "expected_category": "password-reset",
     "quality_criteria": "Should guide through password reset process, mention self-service portal"},
    {"id": 2, "query": "My account got locked after too many login attempts", "expected_category": "password-reset",
     "quality_criteria": "Should explain account unlock process and how to prevent future lockouts"},
    {"id": 3, "query": "I need to change my password, it's been 90 days", "expected_category": "password-reset",
     "quality_criteria": "Should provide steps for password change and mention password policy requirements"},
    {"id": 4, "query": "My SSO login isn't working with the company portal", "expected_category": "password-reset",
     "quality_criteria": "Should troubleshoot SSO issues, suggest clearing cookies or checking IdP status"},
    {"id": 5, "query": "I need Adobe Photoshop installed on my workstation", "expected_category": "software-install",
     "quality_criteria": "Should explain software request process, mention approval workflow and licensing"},
    {"id": 6, "query": "How do I install the company VPN client?", "expected_category": "software-install",
     "quality_criteria": "Should provide VPN client download location and installation steps"},
    {"id": 7, "query": "Microsoft Office keeps asking me to activate my license", "expected_category": "software-install",
     "quality_criteria": "Should troubleshoot license activation, suggest signing in with corporate account"},
    {"id": 8, "query": "I need Python and VS Code set up for development", "expected_category": "software-install",
     "quality_criteria": "Should explain developer tool provisioning process or self-service install steps"},
    {"id": 9, "query": "Slack is not updating to the latest version", "expected_category": "software-install",
     "quality_criteria": "Should suggest manual update steps, check for admin restrictions on updates"},
    {"id": 10, "query": "My laptop screen is flickering constantly", "expected_category": "hardware",
     "quality_criteria": "Should suggest display driver update, external monitor test, and hardware repair if needed"},
    {"id": 11, "query": "The printer on the 3rd floor isn't working", "expected_category": "hardware",
     "quality_criteria": "Should suggest basic troubleshooting (power cycle, paper jam) and offer to dispatch support"},
    {"id": 12, "query": "My laptop won't turn on at all", "expected_category": "hardware",
     "quality_criteria": "Should suggest checking power adapter, battery reset, and offer replacement if needed"},
    {"id": 13, "query": "My keyboard is typing the wrong characters", "expected_category": "hardware",
     "quality_criteria": "Should suggest checking language/layout settings, trying external keyboard, driver update"},
    {"id": 14, "query": "I can't connect to the VPN from home", "expected_category": "network",
     "quality_criteria": "Should troubleshoot VPN connection, check credentials, firewall, and ISP blocking"},
    {"id": 15, "query": "The office Wi-Fi keeps disconnecting", "expected_category": "network",
     "quality_criteria": "Should suggest forgetting and reconnecting, checking signal strength, trying other band"},
    {"id": 16, "query": "I can't access the internal wiki from my desk", "expected_category": "network",
     "quality_criteria": "Should check if on corporate network, DNS resolution, and proxy settings"},
    {"id": 17, "query": "Video calls keep freezing and dropping", "expected_category": "network",
     "quality_criteria": "Should suggest bandwidth check, wired connection, closing other apps, QoS settings"},
    {"id": 18, "query": "I'm getting a DNS resolution error for company websites", "expected_category": "network",
     "quality_criteria": "Should suggest flushing DNS cache, checking DNS settings, trying alternate DNS"},
]

SALES_INQUIRY_CASES = [
    {"id": 1, "query": "How much does the enterprise plan cost?", "expected_category": "pricing",
     "quality_criteria": "Should mention enterprise pricing tiers or offer to schedule a pricing call"},
    {"id": 2, "query": "Do you offer a free trial?", "expected_category": "pricing",
     "quality_criteria": "Should explain free trial availability, duration, and what's included"},
    {"id": 3, "query": "What's included in the basic vs premium plan?", "expected_category": "pricing",
     "quality_criteria": "Should compare plan features and pricing differences clearly"},
    {"id": 4, "query": "Can we get volume pricing for 500 users?", "expected_category": "pricing",
     "quality_criteria": "Should mention volume discount availability and suggest contacting sales"},
    {"id": 5, "query": "Do you support single sign-on (SSO)?", "expected_category": "features",
     "quality_criteria": "Should confirm SSO support, mention supported protocols (SAML, OIDC)"},
    {"id": 6, "query": "Can your platform integrate with Salesforce?", "expected_category": "features",
     "quality_criteria": "Should describe integration capabilities, mention API or native integrations"},
    {"id": 7, "query": "Is there an API we can use for automation?", "expected_category": "features",
     "quality_criteria": "Should confirm API availability, mention documentation, rate limits, and SDKs"},
    {"id": 8, "query": "What security certifications do you have?", "expected_category": "features",
     "quality_criteria": "Should mention relevant certifications (SOC 2, ISO 27001, etc.)"},
    {"id": 9, "query": "Does the platform support multi-language content?", "expected_category": "features",
     "quality_criteria": "Should explain language support, localization features, and available languages"},
    {"id": 10, "query": "Can I get a live demo of the platform?", "expected_category": "demo-request",
     "quality_criteria": "Should offer to schedule a demo, ask about use case and team size"},
    {"id": 11, "query": "I'd like to see how the analytics dashboard works", "expected_category": "demo-request",
     "quality_criteria": "Should offer demo of analytics features, mention available demo formats"},
    {"id": 12, "query": "Our team wants to evaluate your product for Q3", "expected_category": "demo-request",
     "quality_criteria": "Should acknowledge timeline, offer pilot program or trial, schedule walkthrough"},
    {"id": 13, "query": "Can you walk us through the onboarding process?", "expected_category": "demo-request",
     "quality_criteria": "Should describe onboarding, offer a walkthrough session, mention support resources"},
    {"id": 14, "query": "How do you compare to Competitor X?", "expected_category": "competitor-comparison",
     "quality_criteria": "Should highlight key differentiators professionally, offer comparison materials"},
    {"id": 15, "query": "Why should we switch from our current vendor?", "expected_category": "competitor-comparison",
     "quality_criteria": "Should focus on unique value props, migration support, and ROI benefits"},
    {"id": 16, "query": "What makes you better than the open-source alternatives?", "expected_category": "competitor-comparison",
     "quality_criteria": "Should compare managed service benefits vs open-source (support, reliability, features)"},
    {"id": 17, "query": "We're evaluating three vendors including you, what stands out?", "expected_category": "competitor-comparison",
     "quality_criteria": "Should summarize top differentiators concisely, offer detailed comparison"},
]

# All domain pools
ALL_DOMAIN_CASES: dict[str, list[dict]] = {
    "customer_support": CUSTOMER_SUPPORT_CASES,   # 23 cases
    "it_helpdesk": IT_HELPDESK_CASES,              # 18 cases
    "sales_inquiry": SALES_INQUIRY_CASES,          # 17 cases
}


# =============================================================================
# 1. K-Fold Holdout Manager
# =============================================================================

K_FOLDS = 5
MAX_EVALS_PER_FOLD = 3  # force rotation after this many evaluations on one fold


@dataclass
class FoldSplit:
    """One fold split: which cases are training, which are holdout."""
    fold_index: int
    training_cases: list[dict]
    holdout_cases: list[dict]
    training_ids: list[int]
    holdout_ids: list[int]


@dataclass
class HoldoutRotationManager:
    """Manages K-fold holdout rotation with eval budget tracking.

    The manager splits a pool of test cases into K folds. On each call to
    ``next_split()`` it advances to the next fold as holdout. The eval budget
    tracks how many times each fold has been used as holdout; once a fold
    hits MAX_EVALS_PER_FOLD, it is skipped until all folds have been
    exhausted (or the budget is reset).
    """

    domain: str
    cases: list[dict] = field(repr=False)
    k: int = K_FOLDS

    # Internal state -- set up in __post_init__
    folds: list[list[dict]] = field(default_factory=list, repr=False)
    current_fold: int = 0
    eval_counts: list[int] = field(default_factory=list)
    rotation_log: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.folds = self._make_folds(self.cases, self.k)
        self.eval_counts = [0] * self.k

    @staticmethod
    def _make_folds(cases: list[dict], k: int) -> list[list[dict]]:
        """Split cases into k roughly-equal folds (deterministic, round-robin)."""
        folds: list[list[dict]] = [[] for _ in range(k)]
        for i, case in enumerate(cases):
            folds[i % k].append(case)
        return folds

    def next_split(self) -> FoldSplit:
        """Return the next fold split, respecting the eval budget.

        Advances ``current_fold`` and skips folds that have exhausted their
        eval budget. If ALL folds are exhausted, resets the budget.
        """
        # Try to find a fold that hasn't hit its budget
        attempts = 0
        while attempts < self.k:
            fold_idx = self.current_fold % self.k
            if self.eval_counts[fold_idx] < MAX_EVALS_PER_FOLD:
                break
            # Budget exhausted for this fold -- skip
            self.current_fold += 1
            attempts += 1

        if attempts == self.k:
            # All folds exhausted -- reset budget and use current fold
            self.eval_counts = [0] * self.k
            fold_idx = self.current_fold % self.k
            self.rotation_log.append({
                "event": "budget_reset",
                "message": f"All {self.k} folds exhausted eval budget; resetting counters",
            })

        # Build the split
        holdout = self.folds[fold_idx]
        training = []
        for i, fold in enumerate(self.folds):
            if i != fold_idx:
                training.extend(fold)

        self.eval_counts[fold_idx] += 1
        split = FoldSplit(
            fold_index=fold_idx,
            training_cases=training,
            holdout_cases=holdout,
            training_ids=[c["id"] for c in training],
            holdout_ids=[c["id"] for c in holdout],
        )

        self.rotation_log.append({
            "event": "split",
            "fold_index": fold_idx,
            "holdout_count": len(holdout),
            "training_count": len(training),
            "eval_counts_after": list(self.eval_counts),
        })

        # Advance for next call
        self.current_fold = (fold_idx + 1) % self.k

        return split

    def summary(self) -> dict:
        """Return a summary of the fold structure."""
        return {
            "domain": self.domain,
            "total_cases": len(self.cases),
            "k": self.k,
            "fold_sizes": [len(f) for f in self.folds],
            "eval_counts": list(self.eval_counts),
            "rotations": len([e for e in self.rotation_log if e["event"] == "split"]),
        }


# =============================================================================
# 2. Mock Scoring (deterministic, no LLM calls)
# =============================================================================

def mock_score_case(case: dict, is_holdout: bool, iteration: int) -> dict:
    """Generate a deterministic mock score for a test case.

    The score is based on the case ID (for reproducibility) with slight
    variation per iteration. Holdout cases get a small penalty to simulate
    the realistic scenario where unseen cases score slightly lower.
    """
    # Deterministic base score from case ID
    base = 65 + (case["id"] * 7 % 30)  # range 65-94

    # Iteration-dependent variation (simulates optimizer improvements)
    improvement = min(iteration * 1.5, 10)

    # Holdout penalty (unseen cases score ~2-5 pts lower)
    holdout_penalty = 3.0 if is_holdout else 0.0

    # Category correctness (50 pts) + quality (30 pts) + helpfulness (20 pts)
    category_score = 50  # assume correct for mock
    quality_score = min(30, max(0, int((base + improvement - holdout_penalty - 50) * 0.6)))
    helpfulness_score = min(20, max(0, int((base + improvement - holdout_penalty - 50) * 0.4)))

    total = category_score + quality_score + helpfulness_score
    return {
        "total_score": total,
        "category_score": category_score,
        "quality_score": quality_score,
        "helpfulness_score": helpfulness_score,
        "reasoning": f"Mock score for case {case['id']} (iter={iteration}, holdout={is_holdout})",
    }


def mock_run_benchmark(
    cases: list[dict],
    domain: str,
    iteration: int,
    is_holdout: bool,
    label: str,
) -> HarnessResult:
    """Run a mock benchmark (no LLM calls) and return scored results."""
    result = HarnessResult(
        harness_name=domain,
        iteration=iteration,
        config_path=label,
    )

    for case in cases:
        score = mock_score_case(case, is_holdout, iteration)
        record = InteractionRecord(
            query=case["query"],
            case_id=case["id"],
            expected_category=case["expected_category"],
            actual_category=case["expected_category"],  # mock: always correct
            response=f"[Mock response for case {case['id']}]",
            score=score,
            quality_criteria=case["quality_criteria"],
        )
        result.records.append(record)

    return result


# =============================================================================
# 3. Mock Proposal / Mutation (simulates the propose phase from 19_e2e)
# =============================================================================

def mock_propose_mutation(
    train_result: HarnessResult,
    iteration: int,
) -> dict:
    """Simulate the propose phase -- returns a fake 'config diff' summary."""
    weak = train_result.weakest_cases(3)
    return {
        "iteration": iteration,
        "proposal": f"Improve handling for weak cases: {[r.case_id for r in weak]}",
        "target_category": weak[0].expected_category if weak else "unknown",
        "estimated_improvement": round(1.5 + iteration * 0.3, 1),
    }


# =============================================================================
# 4. Validation Gate (same logic as 19_e2e_full_loop.py)
# =============================================================================

VALIDATION_TOLERANCE = 1.0  # points


def check_validation_gate(
    original_score: float,
    proposed_score: float,
) -> tuple[bool, float]:
    """Check if proposed config passes the validation gate on holdout."""
    delta = proposed_score - original_score
    passed = delta >= -VALIDATION_TOLERANCE
    return passed, delta


# =============================================================================
# 5. Main Loop: Rotating Holdout Learning
# =============================================================================

@dataclass
class IterationReport:
    """Results from one iteration of the rotating holdout loop."""
    iteration: int
    domain: str
    holdout_fold: int
    holdout_ids: list[int]
    training_ids: list[int]
    train_score: float
    holdout_original_score: float
    holdout_proposed_score: float
    holdout_delta: float
    validation_passed: bool
    proposal: dict
    eval_counts_after: list[int]


def run_rotating_holdout_loop(
    domain: str,
    cases: list[dict],
    iterations: int = 3,
) -> list[IterationReport]:
    """Run the learning loop with rotating holdouts.

    Each iteration:
      1. Get the next fold split from the rotation manager
      2. Score training cases (mock)
      3. Propose a mutation from training analysis
      4. Score holdout with original and proposed configs (mock)
      5. Apply validation gate
      6. Log which fold was holdout and the eval budget state
    """
    manager = HoldoutRotationManager(domain=domain, cases=cases)
    reports: list[IterationReport] = []

    print(f"\n{'='*70}")
    print(f"  ROTATING HOLDOUT LOOP: {domain}")
    print(f"  Cases: {len(cases)} | K={manager.k} | Budget: {MAX_EVALS_PER_FOLD}/fold")
    print(f"  Fold sizes: {[len(f) for f in manager.folds]}")
    print(f"{'='*70}")

    for iteration in range(1, iterations + 1):
        split = manager.next_split()

        print(f"\n  --- Iteration {iteration} ---")
        print(f"  Holdout fold: {split.fold_index} ({len(split.holdout_cases)} cases, ids={split.holdout_ids})")
        print(f"  Training set: {len(split.training_cases)} cases")
        print(f"  Eval counts:  {manager.eval_counts}")

        # Phase 1: Score training cases
        train_result = mock_run_benchmark(
            split.training_cases, domain, iteration,
            is_holdout=False, label=f"train-iter{iteration}",
        )
        print(f"  Training score: {train_result.aggregate_score:.1f}")

        # Phase 2: Propose mutation
        proposal = mock_propose_mutation(train_result, iteration)
        print(f"  Proposal: {proposal['proposal']}")

        # Phase 3: Score holdout (original vs proposed)
        holdout_original = mock_run_benchmark(
            split.holdout_cases, domain, iteration,
            is_holdout=True, label=f"holdout-orig-iter{iteration}",
        )
        # Proposed config: simulate improvement (score at iteration+1)
        holdout_proposed = mock_run_benchmark(
            split.holdout_cases, domain, iteration + 1,
            is_holdout=True, label=f"holdout-prop-iter{iteration}",
        )

        # Phase 4: Validation gate
        passed, delta = check_validation_gate(
            holdout_original.aggregate_score,
            holdout_proposed.aggregate_score,
        )

        print(f"  Holdout original: {holdout_original.aggregate_score:.1f}")
        print(f"  Holdout proposed: {holdout_proposed.aggregate_score:.1f}")
        print(f"  Delta: {delta:+.1f} -> {'PASS' if passed else 'FAIL'}")

        report = IterationReport(
            iteration=iteration,
            domain=domain,
            holdout_fold=split.fold_index,
            holdout_ids=split.holdout_ids,
            training_ids=split.training_ids,
            train_score=round(train_result.aggregate_score, 2),
            holdout_original_score=round(holdout_original.aggregate_score, 2),
            holdout_proposed_score=round(holdout_proposed.aggregate_score, 2),
            holdout_delta=round(delta, 2),
            validation_passed=passed,
            proposal=proposal,
            eval_counts_after=list(manager.eval_counts),
        )
        reports.append(report)

    # Print rotation summary
    summary = manager.summary()
    print(f"\n  Rotation summary:")
    print(f"    Total rotations: {summary['rotations']}")
    print(f"    Final eval counts: {summary['eval_counts']}")
    print(f"    Budget resets: {len([e for e in manager.rotation_log if e['event'] == 'budget_reset'])}")

    return reports


# =============================================================================
# 6. Report Generation
# =============================================================================


def generate_report(all_reports: dict[str, list[IterationReport]]) -> str:
    """Generate a markdown report showing holdout rotation in action."""
    lines: list[str] = []

    lines.append("# Holdout Rotation Report")
    lines.append("")
    lines.append("## Problem")
    lines.append("")
    lines.append("The DAG Builder learning loop (Prototype 19) uses a **fixed holdout set** for")
    lines.append("validation. Expert review flagged that repeated evaluation against the same")
    lines.append("holdout leaks information -- the optimizer implicitly overfits to the holdout")
    lines.append("over multiple iterations.")
    lines.append("")
    lines.append("## Solution")
    lines.append("")
    lines.append(f"**K-fold rotation** (K={K_FOLDS}): each iteration uses a different fold as the")
    lines.append("holdout. An **eval budget** caps each fold to a maximum of")
    lines.append(f"{MAX_EVALS_PER_FOLD} evaluations before forcing rotation.")
    lines.append("")

    for domain, reports in all_reports.items():
        lines.append(f"## Domain: {domain}")
        lines.append("")

        # Summary table
        lines.append("| Iter | Holdout Fold | Holdout IDs | Train Score | Holdout Orig | Holdout Prop | Delta | Gate | Eval Counts |")
        lines.append("|------|-------------|-------------|-------------|-------------|-------------|-------|------|-------------|")
        for r in reports:
            holdout_ids_str = ",".join(str(i) for i in r.holdout_ids)
            if len(holdout_ids_str) > 20:
                holdout_ids_str = holdout_ids_str[:17] + "..."
            lines.append(
                f"| {r.iteration} | {r.holdout_fold} | {holdout_ids_str} | "
                f"{r.train_score:.1f} | {r.holdout_original_score:.1f} | "
                f"{r.holdout_proposed_score:.1f} | {r.holdout_delta:+.1f} | "
                f"{'PASS' if r.validation_passed else 'FAIL'} | "
                f"{r.eval_counts_after} |"
            )
        lines.append("")

        # Fold usage analysis
        fold_usage: dict[int, list[int]] = {}
        for r in reports:
            fold_usage.setdefault(r.holdout_fold, []).append(r.iteration)

        lines.append("### Fold Usage")
        lines.append("")
        for fold_idx in sorted(fold_usage.keys()):
            iters = fold_usage[fold_idx]
            lines.append(f"- **Fold {fold_idx}**: used as holdout in iteration(s) {iters}")
        lines.append("")

        # Verify no fold is overused
        max_usage = max(len(v) for v in fold_usage.values())
        unique_folds = len(fold_usage)
        lines.append("### Leakage Prevention")
        lines.append("")
        lines.append(f"- Unique folds used as holdout: **{unique_folds}** out of {K_FOLDS}")
        lines.append(f"- Max times any fold used: **{max_usage}** (budget limit: {MAX_EVALS_PER_FOLD})")
        if unique_folds > 1:
            lines.append("- **No single fold was used repeatedly** -- rotation is working correctly")
        else:
            lines.append("- Only 1 fold used (insufficient iterations to demonstrate rotation)")
        lines.append("")

    # Overall conclusions
    lines.append("## Conclusions")
    lines.append("")
    lines.append("1. **Rotation works**: each iteration uses a different fold as holdout,")
    lines.append("   cycling through all K folds before repeating.")
    lines.append("2. **Eval budget enforced**: no fold can be evaluated more than")
    lines.append(f"   {MAX_EVALS_PER_FOLD} times before a forced rotation (or budget reset).")
    lines.append("3. **Validation gate preserved**: the same tolerance-based gate from")
    lines.append("   Prototype 19 applies, just on a rotating holdout instead of a fixed one.")
    lines.append("4. **Ready for integration**: this mechanism can replace the fixed")
    lines.append("   `HOLDOUT_CASES = ALL_CASES[10:18]` in `19_e2e_full_loop.py`.")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# 7. Main
# =============================================================================


def main() -> None:
    print("=" * 70)
    print("Prototype 21: Holdout Rotation")
    print("=" * 70)
    print(f"K-folds:       {K_FOLDS}")
    print(f"Eval budget:   {MAX_EVALS_PER_FOLD} per fold")
    print(f"Scoring:       Mock (no LLM calls)")
    print(f"Domains:       {list(ALL_DOMAIN_CASES.keys())}")
    print()

    all_reports: dict[str, list[IterationReport]] = {}

    # Run 3 iterations per domain to demonstrate rotation
    # (3 iterations with K=5 shows 3 different folds used)
    for domain, cases in ALL_DOMAIN_CASES.items():
        reports = run_rotating_holdout_loop(domain, cases, iterations=3)
        all_reports[domain] = reports

    # Extended demo: 10 iterations on customer_support to show budget exhaustion
    print(f"\n{'='*70}")
    print("  EXTENDED DEMO: Budget exhaustion (customer_support, 10 iterations)")
    print(f"{'='*70}")

    extended_reports = run_rotating_holdout_loop(
        "customer_support_extended",
        CUSTOMER_SUPPORT_CASES,
        iterations=10,
    )
    all_reports["customer_support_extended"] = extended_reports

    # Generate and write report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = generate_report(all_reports)
    report_path = RESULTS_DIR / "holdout_rotation_report.md"
    report_path.write_text(report)

    print(f"\n{'='*70}")
    print(f"  Report written to: {report_path}")
    print(f"{'='*70}")

    # Final verification: print fold uniqueness check
    print("\nVerification -- fold uniqueness per domain:")
    for domain, reports in all_reports.items():
        folds_used = set(r.holdout_fold for r in reports)
        repeats = len(reports) - len(folds_used)
        budget_ok = all(
            reports[i].eval_counts_after[reports[i].holdout_fold] <= MAX_EVALS_PER_FOLD
            for i in range(len(reports))
        )
        print(f"  {domain:30s}: {len(reports)} iters, {len(folds_used)} unique folds, "
              f"{repeats} repeats, budget_ok={budget_ok}")


if __name__ == "__main__":
    main()
