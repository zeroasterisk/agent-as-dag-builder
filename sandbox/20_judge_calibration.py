"""Prototype 20: Judge Calibration — measure LLM judge agreement with human labels.

Validates our LLM-as-judge scoring by comparing its pass/fail decisions against
human-labeled ground truth. Uses a manually-curated calibration set of 20
customer-support interactions with clear pass/fail labels, modeled after the
Tau-Bench evaluation style (correct tool use, policy adherence, helpful resolution).

Tau-Bench (tau2) provides human-labeled ground truth for customer service tasks
in retail, airline, and telecom domains. Since tau2 is not installed in this
environment, we use a hand-labeled calibration set that mirrors tau2's structure:
each case has a user query, an agent response, and a human-assigned label.

If tau2 becomes available, this script can be extended to load real tau2 data
via exgentic's TAU2Benchmark adapter (see /workspace/exgentic/src/exgentic/
benchmarks/tau2/).

===========================================================================
WHAT THIS SCRIPT DOES
===========================================================================

Phase 1 -- LOAD CALIBRATION DATA
    Load 20 manually-labeled test cases. Each has:
    - A customer query
    - An agent response (some good, some bad)
    - A human label: pass (1) or fail (0)
    - A quality_criteria string for the judge
    - Expected category

Phase 2 -- RUN OUR JUDGE
    Score each case with the same LLM-as-judge rubric from 19_e2e_full_loop.py.
    Record: our_score (0-100), our_pass_fail (pass if >=60), human_label.

Phase 3 -- MEASURE AGREEMENT
    - Agreement rate: % where judge pass/fail matches human label
    - Cohen's kappa: chance-corrected agreement
    - Pearson/Spearman correlation between scores and labels
    - False positives (judge says pass, human says fail)
    - False negatives (judge says fail, human says pass)
    Target: >80% agreement.

Phase 4 -- LOG RESULTS
    Write calibration report to sandbox/results/judge_calibration_report.md.

===========================================================================
ENVIRONMENT
===========================================================================

    GOOGLE_GENAI_USE_VERTEXAI=1
    GOOGLE_CLOUD_PROJECT=alanblount-demo
    GOOGLE_CLOUD_LOCATION=global

===========================================================================
USAGE
===========================================================================

    python sandbox/20_judge_calibration.py
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# -- Environment defaults (Vertex AI) ----------------------------------------
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "alanblount-demo")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

from google import genai

# -- Import shared infrastructure from 11_multi_harness.py --------------------
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "multi_harness",
    str(Path(__file__).parent / "11_multi_harness.py"),
)
_mh = _ilu.module_from_spec(_spec)
sys.modules["multi_harness"] = _mh
_spec.loader.exec_module(_mh)

score_response = _mh.score_response
JUDGE_MODEL = _mh.JUDGE_MODEL
LLM_DELAY = _mh.LLM_DELAY
SANDBOX_DIR = _mh.SANDBOX_DIR

HARNESS_NAME = "customer_support"
RESULTS_DIR = SANDBOX_DIR / "results"
REPORT_FILE = RESULTS_DIR / "judge_calibration_report.md"
RESULTS_JSON = RESULTS_DIR / "judge_calibration_results.json"

# Pass/fail threshold: a case "passes" if total_score >= this value.
# 60/100 is a reasonable threshold: it requires correct category (50 pts) plus
# at least some quality/helpfulness (10+ out of 50).
PASS_THRESHOLD = 60


# =============================================================================
# 1. Calibration Dataset — 20 human-labeled cases
# =============================================================================
# Each case has a clear human label. "PASS" cases have correct, helpful responses.
# "FAIL" cases have responses that are wrong, unhelpful, or misrouted.
#
# Design principles:
# - 10 PASS cases: correct category, substantive help, addresses criteria
# - 10 FAIL cases: wrong category, vague/unhelpful, or policy-violating responses
# - Mix of billing, technical, and general categories
# - Responses are realistic (not strawmen), so borderline cases test judge quality

CALIBRATION_CASES = [
    # =========================================================================
    # PASS CASES (10) — correct routing, helpful, addresses quality criteria
    # =========================================================================
    {
        "case_id": 1,
        "query": "I was charged twice for my subscription this month",
        "expected_category": "billing",
        "actual_category": "billing",
        "quality_criteria": "Should acknowledge the double charge, mention investigation or refund process",
        "response": (
            "I'm sorry to hear about the double charge on your subscription. "
            "I can see this is frustrating, and I want to help resolve it right away. "
            "I'll initiate an investigation into the duplicate charge. Typically, "
            "if a double charge is confirmed, we process a refund within 3-5 business "
            "days back to your original payment method. In the meantime, I've flagged "
            "your account to prevent any further duplicate billing. Is there anything "
            "else I can help you with?"
        ),
        "human_label": 1,  # PASS — correct category, acknowledges issue, mentions refund
        "human_reasoning": "Correct routing, acknowledges double charge, explains refund process with timeline",
    },
    {
        "case_id": 2,
        "query": "Can I get a refund for last month?",
        "expected_category": "billing",
        "actual_category": "billing",
        "quality_criteria": "Should explain the refund policy or process for requesting a refund",
        "response": (
            "I'd be happy to help you with a refund request for last month. Our refund "
            "policy allows refunds within 30 days of the charge. To process your request, "
            "I'll need to verify your account details. Once confirmed, refunds typically "
            "take 5-7 business days to appear on your statement. Please note that if you "
            "used any premium features during that billing period, the refund may be "
            "prorated. Would you like me to go ahead and start the refund process?"
        ),
        "human_label": 1,  # PASS — explains policy, mentions timeline, offers to proceed
        "human_reasoning": "Clear refund policy explanation with timeline and next steps",
    },
    {
        "case_id": 3,
        "query": "My internet keeps dropping every 30 minutes",
        "expected_category": "technical",
        "actual_category": "technical",
        "quality_criteria": "Should ask about router/modem or suggest troubleshooting steps like restarting equipment",
        "response": (
            "I understand how frustrating intermittent connectivity can be. Let's "
            "troubleshoot this step by step:\n\n"
            "1. First, try restarting your modem and router — unplug them for 30 seconds, "
            "then plug the modem back in first, wait for it to fully connect, then plug "
            "in the router.\n"
            "2. Check if the issue happens on all devices or just one — this helps us "
            "determine if it's a device-specific issue.\n"
            "3. If you're on Wi-Fi, try connecting via ethernet cable to rule out "
            "wireless interference.\n"
            "4. Check if there are any firmware updates available for your router.\n\n"
            "If the issue persists after these steps, we may need to run a line "
            "diagnostic. Would you like me to schedule that?"
        ),
        "human_label": 1,  # PASS — structured troubleshooting, mentions router/modem
        "human_reasoning": "Excellent step-by-step troubleshooting, covers modem/router restart and diagnostic",
    },
    {
        "case_id": 4,
        "query": "The app crashes when I try to open settings",
        "expected_category": "technical",
        "actual_category": "technical",
        "quality_criteria": "Should suggest clearing cache, reinstalling, or checking for updates",
        "response": (
            "I'm sorry you're experiencing app crashes. Here are some steps that "
            "should help:\n\n"
            "1. Clear the app cache: Go to your phone's Settings > Apps > [Our App] > "
            "Storage > Clear Cache.\n"
            "2. Check for updates: Visit the App Store or Google Play to make sure "
            "you have the latest version installed.\n"
            "3. If the issue persists, try uninstalling and reinstalling the app.\n"
            "4. Make sure your device's operating system is up to date as well.\n\n"
            "If none of these steps resolve the issue, please let me know your device "
            "model and OS version so I can investigate further."
        ),
        "human_label": 1,  # PASS — covers cache, updates, reinstall
        "human_reasoning": "Addresses all three criteria: clear cache, check updates, reinstall",
    },
    {
        "case_id": 5,
        "query": "What are your business hours?",
        "expected_category": "general",
        "actual_category": "general",
        "quality_criteria": "Should provide specific hours or direct to a page with hours information",
        "response": (
            "Our customer support is available during the following hours:\n\n"
            "- Monday to Friday: 8:00 AM - 8:00 PM (EST)\n"
            "- Saturday: 9:00 AM - 5:00 PM (EST)\n"
            "- Sunday: Closed\n\n"
            "For 24/7 support, you can also visit our Help Center at "
            "help.example.com where you'll find answers to common questions. "
            "Is there anything else I can help you with?"
        ),
        "human_label": 1,  # PASS — specific hours provided
        "human_reasoning": "Provides specific hours with days and timezone, plus self-service option",
    },
    {
        "case_id": 6,
        "query": "I need to update my credit card on file",
        "expected_category": "billing",
        "actual_category": "billing",
        "quality_criteria": "Should provide steps to update payment method or direct to account settings",
        "response": (
            "You can update your credit card information through your account settings. "
            "Here's how:\n\n"
            "1. Log into your account at account.example.com\n"
            "2. Go to 'Account Settings' > 'Payment Methods'\n"
            "3. Click 'Edit' next to your current card or 'Add New Payment Method'\n"
            "4. Enter your new card details and save\n\n"
            "For security reasons, I'm unable to update payment information directly "
            "through this chat. If you're having trouble accessing your account, "
            "I can help you reset your login credentials."
        ),
        "human_label": 1,  # PASS — clear steps, mentions security
        "human_reasoning": "Step-by-step instructions for updating payment, appropriate security note",
    },
    {
        "case_id": 7,
        "query": "How do I contact customer support by phone?",
        "expected_category": "general",
        "actual_category": "general",
        "quality_criteria": "Should provide a phone number or explain how to find contact information",
        "response": (
            "You can reach our customer support team by phone at 1-800-555-0123. "
            "Our phone support hours are Monday through Friday, 8 AM to 8 PM EST. "
            "For faster service, have your account number ready when you call. "
            "You can also reach us through this chat 24/7 or email us at "
            "support@example.com."
        ),
        "human_label": 1,  # PASS — provides phone number and hours
        "human_reasoning": "Provides phone number, hours, and alternative contact methods",
    },
    {
        "case_id": 8,
        "query": "I see an unauthorized charge from your company",
        "expected_category": "billing",
        "actual_category": "billing",
        "quality_criteria": "Should take the concern seriously, suggest investigation, mention fraud protection",
        "response": (
            "I take unauthorized charges very seriously, and I want to help resolve "
            "this immediately. Here's what we'll do:\n\n"
            "1. I'll flag your account for an immediate fraud investigation.\n"
            "2. While we investigate, I recommend contacting your bank to dispute "
            "the charge and request a temporary credit.\n"
            "3. Our fraud protection team will review the charge within 24-48 hours.\n"
            "4. If the charge is confirmed as unauthorized, we'll issue a full refund "
            "and ensure your account is secured.\n\n"
            "Can you tell me the exact amount and date of the charge so I can look "
            "into it right away?"
        ),
        "human_label": 1,  # PASS — takes it seriously, investigation, fraud protection
        "human_reasoning": "Takes concern seriously, outlines investigation steps, mentions fraud protection",
    },
    {
        "case_id": 9,
        "query": "My download speeds are extremely slow, only getting 2 Mbps",
        "expected_category": "technical",
        "actual_category": "technical",
        "quality_criteria": "Should ask about connection type, suggest speed test, check for interference",
        "response": (
            "I'm sorry to hear about the slow speeds. Let's diagnose this:\n\n"
            "1. First, can you run a speed test at speedtest.net and share the results? "
            "This gives us an accurate baseline.\n"
            "2. Are you connected via Wi-Fi or ethernet? If Wi-Fi, try connecting "
            "directly via ethernet to rule out wireless interference.\n"
            "3. Check if other devices on your network are experiencing the same "
            "speeds — this helps determine if it's device-specific or network-wide.\n"
            "4. If on Wi-Fi, make sure you're connected to the 5GHz band rather than "
            "2.4GHz for better speeds.\n\n"
            "What speed plan are you subscribed to? That will help me determine if "
            "2 Mbps represents a significant drop from your expected speed."
        ),
        "human_label": 1,  # PASS — speed test, connection type, interference
        "human_reasoning": "Covers speed test, connection type, interference check, and asks about plan",
    },
    {
        "case_id": 10,
        "query": "Do you offer any discounts for annual plans?",
        "expected_category": "billing",
        "actual_category": "billing",
        "quality_criteria": "Should mention available discount options or direct to pricing page",
        "response": (
            "Yes, we do offer discounts for annual billing! When you switch to an "
            "annual plan, you typically save 15-20% compared to monthly billing. "
            "Here are your options:\n\n"
            "- Basic Annual: $8/month (billed annually) — save 20% vs monthly\n"
            "- Premium Annual: $15/month (billed annually) — save 17% vs monthly\n\n"
            "You can view all pricing options and switch to annual billing at "
            "example.com/pricing. Would you like me to help you switch your current "
            "plan to annual billing?"
        ),
        "human_label": 1,  # PASS — mentions discounts, provides specifics, directs to pricing
        "human_reasoning": "Confirms discounts, provides specific savings, links to pricing page",
    },

    # =========================================================================
    # FAIL CASES (10) — wrong routing, unhelpful, or policy-violating responses
    # =========================================================================
    {
        "case_id": 11,
        "query": "I was charged twice for my subscription this month",
        "expected_category": "billing",
        "actual_category": "technical",  # WRONG CATEGORY
        "quality_criteria": "Should acknowledge the double charge, mention investigation or refund process",
        "response": (
            "It sounds like you might be experiencing a technical glitch with your "
            "subscription service. Have you tried logging out and logging back in? "
            "Sometimes clearing your browser cache can resolve display issues with "
            "billing amounts. If the issue persists, try using a different browser."
        ),
        "human_label": 0,  # FAIL — wrong category, doesn't address billing issue at all
        "human_reasoning": "Misrouted to technical. Treats billing complaint as a display issue, no refund mention",
    },
    {
        "case_id": 12,
        "query": "Can I get a refund for last month?",
        "expected_category": "billing",
        "actual_category": "billing",
        "quality_criteria": "Should explain the refund policy or process for requesting a refund",
        "response": (
            "I understand you'd like a refund. Unfortunately, I'm not able to process "
            "refunds through this channel. You'll need to figure that out on your own. "
            "Good luck!"
        ),
        "human_label": 0,  # FAIL — dismissive, no policy explanation, no process
        "human_reasoning": "Correct category but completely unhelpful — no refund policy explained, dismissive tone",
    },
    {
        "case_id": 13,
        "query": "My internet keeps dropping every 30 minutes",
        "expected_category": "technical",
        "actual_category": "billing",  # WRONG CATEGORY
        "quality_criteria": "Should ask about router/modem or suggest troubleshooting steps like restarting equipment",
        "response": (
            "I'd be happy to review your billing statement for any internet-related "
            "charges. Would you like me to check if you're on the right plan for "
            "your usage needs? We have several internet packages available."
        ),
        "human_label": 0,  # FAIL — wrong category, treats connectivity issue as billing
        "human_reasoning": "Misrouted to billing. No troubleshooting steps, doesn't address connectivity problem",
    },
    {
        "case_id": 14,
        "query": "The app crashes when I try to open settings",
        "expected_category": "technical",
        "actual_category": "technical",
        "quality_criteria": "Should suggest clearing cache, reinstalling, or checking for updates",
        "response": (
            "Sorry about that. These things happen sometimes with apps. Technology "
            "can be unpredictable. Have you considered switching to a different app? "
            "There are many alternatives available in the app store."
        ),
        "human_label": 0,  # FAIL — vague, no troubleshooting, suggests switching products
        "human_reasoning": "Correct category but no actionable troubleshooting. Suggests competitor — anti-helpful",
    },
    {
        "case_id": 15,
        "query": "What are your business hours?",
        "expected_category": "general",
        "actual_category": "general",
        "quality_criteria": "Should provide specific hours or direct to a page with hours information",
        "response": (
            "We're open during normal business hours. Feel free to reach out "
            "whenever you need assistance and we'll get back to you as soon as "
            "possible."
        ),
        "human_label": 0,  # FAIL — no specific hours provided, completely vague
        "human_reasoning": "Correct category but fails criteria: 'normal business hours' is not specific",
    },
    {
        "case_id": 16,
        "query": "I need to update my credit card on file",
        "expected_category": "billing",
        "actual_category": "billing",
        "quality_criteria": "Should provide steps to update payment method or direct to account settings",
        "response": (
            "Sure, I can help with that! Please provide me with your full credit card "
            "number, expiration date, CVV, and billing address, and I'll update it "
            "in our system right away."
        ),
        "human_label": 0,  # FAIL — security violation, asking for full card details in chat
        "human_reasoning": "SECURITY VIOLATION: Asking customer for full credit card details via chat is a policy violation",
    },
    {
        "case_id": 17,
        "query": "How do I cancel my subscription?",
        "expected_category": "billing",
        "actual_category": "general",  # WRONG CATEGORY
        "quality_criteria": "Should explain cancellation process; may mention retention offers",
        "response": (
            "Thank you for your interest in our services! We offer a wide range of "
            "plans to suit your needs. Would you like to hear about our premium "
            "features? We also have a referral program where you can earn credits."
        ),
        "human_label": 0,  # FAIL — wrong category, completely ignores cancellation request
        "human_reasoning": "Misrouted to general. Ignores cancellation request entirely, tries to upsell instead",
    },
    {
        "case_id": 18,
        "query": "I can't log into my account, it says invalid password",
        "expected_category": "technical",
        "actual_category": "technical",
        "quality_criteria": "Should suggest password reset process and check for account lock",
        "response": (
            "I see you're having trouble logging in. Your account may have been "
            "compromised. I'm going to need to delete your account for security "
            "reasons. You can create a new one afterward."
        ),
        "human_label": 0,  # FAIL — extreme overreaction, doesn't suggest password reset
        "human_reasoning": "Extreme overreaction (account deletion). No password reset suggestion, no account lock check",
    },
    {
        "case_id": 19,
        "query": "Do you have a referral program?",
        "expected_category": "general",
        "actual_category": "general",
        "quality_criteria": "Should explain whether a referral program exists and how to participate",
        "response": (
            "Thanks for your question. I'm not sure if we have a referral program. "
            "You might want to check our website or maybe ask someone else. I can "
            "help you with billing questions though if you need that."
        ),
        "human_label": 0,  # FAIL — uncertain, deflects, doesn't actually answer
        "human_reasoning": "Fails to answer the question, admits ignorance instead of checking, deflects",
    },
    {
        "case_id": 20,
        "query": "Email notifications are not working, I'm not receiving any alerts",
        "expected_category": "technical",
        "actual_category": "technical",
        "quality_criteria": "Should suggest checking notification settings, spam folder, and email verification",
        "response": (
            "Email notifications can sometimes be unreliable. Have you tried using "
            "our mobile app instead? Push notifications are generally more reliable "
            "than email. You can download the app from the App Store or Google Play."
        ),
        "human_label": 0,  # FAIL — doesn't address the actual problem, just suggests workaround
        "human_reasoning": "Doesn't troubleshoot the issue (no spam check, no settings check). Just suggests a workaround",
    },
]


# =============================================================================
# 2. Data classes
# =============================================================================


@dataclass
class CalibrationRecord:
    case_id: int
    query: str
    expected_category: str
    actual_category: str
    response: str
    human_label: int  # 1=pass, 0=fail
    human_reasoning: str
    our_score: int = 0
    our_pass: int = 0  # 1=pass, 0=fail (based on threshold)
    quality_score: int = 0
    helpfulness_score: int = 0
    category_score: int = 0
    judge_reasoning: str = ""
    agrees: bool = False


# =============================================================================
# 3. Run judge on all calibration cases
# =============================================================================


async def run_judge_on_cases(
    client: genai.Client,
    cases: list[dict],
) -> list[CalibrationRecord]:
    """Score each calibration case with our LLM-as-judge."""
    records = []

    for case in cases:
        print(f"  Scoring case {case['case_id']:2d}: \"{case['query'][:50]}...\"")

        score = await score_response(
            client,
            query=case["query"],
            expected_category=case["expected_category"],
            actual_category=case["actual_category"],
            response=case["response"],
            quality_criteria=case["quality_criteria"],
            harness_name=HARNESS_NAME,
        )

        our_score = score["total_score"]
        our_pass = 1 if our_score >= PASS_THRESHOLD else 0
        agrees = our_pass == case["human_label"]

        record = CalibrationRecord(
            case_id=case["case_id"],
            query=case["query"],
            expected_category=case["expected_category"],
            actual_category=case["actual_category"],
            response=case["response"],
            human_label=case["human_label"],
            human_reasoning=case["human_reasoning"],
            our_score=our_score,
            our_pass=our_pass,
            quality_score=score["quality_score"],
            helpfulness_score=score["helpfulness_score"],
            category_score=score["category_score"],
            judge_reasoning=score.get("reasoning", ""),
            agrees=agrees,
        )
        records.append(record)

        status = "AGREE" if agrees else "DISAGREE"
        human_lbl = "PASS" if case["human_label"] == 1 else "FAIL"
        judge_lbl = "PASS" if our_pass == 1 else "FAIL"
        print(
            f"    Score={our_score:3d} (cat={score['category_score']}, "
            f"qual={score['quality_score']}, help={score['helpfulness_score']}) "
            f"| Judge={judge_lbl} | Human={human_lbl} | {status}"
        )

        await asyncio.sleep(LLM_DELAY)

    return records


# =============================================================================
# 4. Compute agreement metrics
# =============================================================================


def compute_agreement_metrics(records: list[CalibrationRecord]) -> dict:
    """Compute agreement rate, Cohen's kappa, and correlation metrics."""
    n = len(records)
    if n == 0:
        return {"agreement_rate": 0.0, "cohens_kappa": 0.0}

    # Agreement rate
    agreements = sum(1 for r in records if r.agrees)
    agreement_rate = agreements / n

    # Confusion matrix
    tp = sum(1 for r in records if r.our_pass == 1 and r.human_label == 1)
    tn = sum(1 for r in records if r.our_pass == 0 and r.human_label == 0)
    fp = sum(1 for r in records if r.our_pass == 1 and r.human_label == 0)
    fn = sum(1 for r in records if r.our_pass == 0 and r.human_label == 1)

    # Cohen's kappa
    p_o = (tp + tn) / n  # observed agreement
    p_yes = ((tp + fp) / n) * ((tp + fn) / n)
    p_no = ((tn + fn) / n) * ((tn + fp) / n)
    p_e = p_yes + p_no  # expected agreement by chance
    if p_e < 1.0:
        cohens_kappa = (p_o - p_e) / (1 - p_e)
    else:
        cohens_kappa = 1.0

    # Pearson correlation between our_score and human_label
    scores = [r.our_score for r in records]
    labels = [float(r.human_label) for r in records]
    pearson_r = _pearson_correlation(scores, labels)

    # Spearman rank correlation
    spearman_r = _spearman_correlation(scores, labels)

    # Precision, Recall, F1 for "pass" class
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Score distributions
    pass_scores = [r.our_score for r in records if r.human_label == 1]
    fail_scores = [r.our_score for r in records if r.human_label == 0]

    return {
        "agreement_rate": agreement_rate,
        "cohens_kappa": cohens_kappa,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "pass_score_mean": sum(pass_scores) / len(pass_scores) if pass_scores else 0,
        "pass_score_min": min(pass_scores) if pass_scores else 0,
        "pass_score_max": max(pass_scores) if pass_scores else 0,
        "fail_score_mean": sum(fail_scores) / len(fail_scores) if fail_scores else 0,
        "fail_score_min": min(fail_scores) if fail_scores else 0,
        "fail_score_max": max(fail_scores) if fail_scores else 0,
        "total_cases": n,
        "total_agreements": agreements,
    }


def _pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    if std_x == 0 or std_y == 0:
        return 0.0
    return cov_xy / (std_x * std_y)


def _spearman_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0

    def _rank(values: list[float]) -> list[float]:
        indexed = sorted(enumerate(values), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1  # 1-based average rank for ties
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rank_x = _rank(x)
    rank_y = _rank(y)
    return _pearson_correlation(rank_x, rank_y)


# =============================================================================
# 5. Generate report
# =============================================================================


def generate_report(
    records: list[CalibrationRecord],
    metrics: dict,
    elapsed: float,
) -> str:
    """Generate a markdown calibration report."""
    lines = []
    lines.append("# Judge Calibration Report")
    lines.append("")
    lines.append(f"**Date**: 2026-07-02")
    lines.append(f"**Judge Model**: {JUDGE_MODEL}")
    lines.append(f"**Domain**: {HARNESS_NAME}")
    lines.append(f"**Pass Threshold**: {PASS_THRESHOLD}/100")
    lines.append(f"**Calibration Cases**: {metrics['total_cases']}")
    lines.append(f"**Elapsed**: {elapsed:.1f}s")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    target_met = metrics["agreement_rate"] >= 0.80
    status = "PASSED" if target_met else "FAILED"
    lines.append(f"| Metric | Value | Target | Status |")
    lines.append(f"|--------|-------|--------|--------|")
    lines.append(f"| Agreement Rate | {metrics['agreement_rate']:.1%} | >= 80% | {status} |")
    lines.append(f"| Cohen's Kappa | {metrics['cohens_kappa']:.3f} | > 0.6 (substantial) | {'GOOD' if metrics['cohens_kappa'] > 0.6 else 'WEAK'} |")
    lines.append(f"| Pearson r | {metrics['pearson_r']:.3f} | > 0.7 | {'GOOD' if metrics['pearson_r'] > 0.7 else 'WEAK'} |")
    lines.append(f"| Spearman rho | {metrics['spearman_r']:.3f} | > 0.7 | {'GOOD' if metrics['spearman_r'] > 0.7 else 'WEAK'} |")
    lines.append(f"| Precision | {metrics['precision']:.3f} | - | - |")
    lines.append(f"| Recall | {metrics['recall']:.3f} | - | - |")
    lines.append(f"| F1 | {metrics['f1']:.3f} | - | - |")
    lines.append("")

    # Confusion matrix
    cm = metrics["confusion_matrix"]
    lines.append("## Confusion Matrix")
    lines.append("")
    lines.append("|  | Human PASS | Human FAIL |")
    lines.append("|--|-----------|-----------|")
    lines.append(f"| **Judge PASS** | {cm['tp']} (TP) | {cm['fp']} (FP) |")
    lines.append(f"| **Judge FAIL** | {cm['fn']} (FN) | {cm['tn']} (TN) |")
    lines.append("")

    # Score distributions
    lines.append("## Score Distributions")
    lines.append("")
    lines.append(f"- **Human PASS cases**: mean={metrics['pass_score_mean']:.1f}, "
                 f"min={metrics['pass_score_min']}, max={metrics['pass_score_max']}")
    lines.append(f"- **Human FAIL cases**: mean={metrics['fail_score_mean']:.1f}, "
                 f"min={metrics['fail_score_min']}, max={metrics['fail_score_max']}")
    separation = metrics["pass_score_mean"] - metrics["fail_score_mean"]
    lines.append(f"- **Score separation**: {separation:.1f} points (higher = better discrimination)")
    lines.append("")

    # All cases
    lines.append("## All Cases")
    lines.append("")
    lines.append("| Case | Query | Score | Judge | Human | Status |")
    lines.append("|------|-------|-------|-------|-------|--------|")
    for r in records:
        query_short = r.query[:40] + "..." if len(r.query) > 40 else r.query
        judge_lbl = "PASS" if r.our_pass == 1 else "FAIL"
        human_lbl = "PASS" if r.human_label == 1 else "FAIL"
        status = "AGREE" if r.agrees else "**DISAGREE**"
        lines.append(f"| {r.case_id} | {query_short} | {r.our_score} | {judge_lbl} | {human_lbl} | {status} |")
    lines.append("")

    # Disagreements
    disagreements = [r for r in records if not r.agrees]
    if disagreements:
        lines.append("## Disagreements")
        lines.append("")
        for r in disagreements:
            judge_lbl = "PASS" if r.our_pass == 1 else "FAIL"
            human_lbl = "PASS" if r.human_label == 1 else "FAIL"
            error_type = "FALSE POSITIVE" if r.our_pass == 1 and r.human_label == 0 else "FALSE NEGATIVE"
            lines.append(f"### Case {r.case_id} ({error_type})")
            lines.append(f"- **Query**: {r.query}")
            lines.append(f"- **Expected Category**: {r.expected_category}")
            lines.append(f"- **Actual Category**: {r.actual_category}")
            lines.append(f"- **Judge Score**: {r.our_score} (cat={r.category_score}, qual={r.quality_score}, help={r.helpfulness_score})")
            lines.append(f"- **Judge Verdict**: {judge_lbl}")
            lines.append(f"- **Human Verdict**: {human_lbl}")
            lines.append(f"- **Judge Reasoning**: {r.judge_reasoning}")
            lines.append(f"- **Human Reasoning**: {r.human_reasoning}")
            lines.append(f"- **Response**: {r.response[:200]}...")
            lines.append("")
    else:
        lines.append("## Disagreements")
        lines.append("")
        lines.append("*No disagreements — perfect agreement!*")
        lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")

    if metrics["agreement_rate"] >= 0.90:
        lines.append("- **Judge quality is excellent.** Agreement rate >= 90% indicates the judge is well-calibrated.")
    elif metrics["agreement_rate"] >= 0.80:
        lines.append("- **Judge quality is good.** Agreement rate >= 80% meets our target. Review disagreements for edge case patterns.")
    else:
        lines.append("- **Judge quality needs improvement.** Agreement rate < 80%. Consider:")
        lines.append("  - Adjusting the pass/fail threshold")
        lines.append("  - Adding more specific rubric criteria")
        lines.append("  - Increasing judge temperature diversity (more averaging passes)")

    if cm["fp"] > cm["fn"]:
        lines.append(f"- **More false positives ({cm['fp']}) than false negatives ({cm['fn']})**: "
                     "Judge is too lenient — consider stricter quality criteria or higher threshold.")
    elif cm["fn"] > cm["fp"]:
        lines.append(f"- **More false negatives ({cm['fn']}) than false positives ({cm['fp']})**: "
                     "Judge is too strict — consider looser quality criteria or lower threshold.")

    if separation < 20:
        lines.append(f"- **Low score separation ({separation:.1f} pts)**: Judge doesn't strongly distinguish "
                     "pass/fail cases. Consider refining the rubric.")
    elif separation > 40:
        lines.append(f"- **Good score separation ({separation:.1f} pts)**: Judge clearly distinguishes "
                     "pass/fail cases. Rubric is well-designed.")

    lines.append("")
    lines.append("## Methodology Notes")
    lines.append("")
    lines.append("- Calibration data is manually curated (not from Tau-Bench), with 10 PASS and 10 FAIL cases.")
    lines.append("- Tau-Bench (tau2) is the intended source for production calibration but requires installation.")
    lines.append("- To extend this with tau2 data: install tau2 via exgentic's setup.sh, use `load_tasks()` to")
    lines.append("  get retail/airline tasks with human-labeled rewards, and map reward >= 1.0 to PASS.")
    lines.append(f"- Judge uses 2x averaging at temperature=0 for variance reduction (same as production scoring).")
    lines.append(f"- Pass/fail threshold: {PASS_THRESHOLD}/100 (category=50 + quality+helpfulness >= {PASS_THRESHOLD - 50}).")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# 6. Main
# =============================================================================


async def run_judge_calibration() -> dict:
    """Run the full judge calibration pipeline."""
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "alanblount-demo"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
    )

    t_start = time.time()

    # Phase 1: Load calibration data
    print(f"\n{'='*70}")
    print("PHASE 1: LOAD CALIBRATION DATA")
    print(f"{'='*70}")
    print(f"  {len(CALIBRATION_CASES)} manually-labeled cases")
    pass_count = sum(1 for c in CALIBRATION_CASES if c["human_label"] == 1)
    fail_count = sum(1 for c in CALIBRATION_CASES if c["human_label"] == 0)
    print(f"  Human labels: {pass_count} PASS, {fail_count} FAIL")
    print(f"  Categories: {', '.join(sorted(set(c['expected_category'] for c in CALIBRATION_CASES)))}")
    print()
    print("  Note: Using manually-labeled calibration set. Tau-Bench (tau2) is not")
    print("  installed. To use tau2 data, run:")
    print("    cd /workspace/exgentic && bash src/exgentic/benchmarks/tau2/setup.sh")
    print("  Then extend this script to load tasks via tau2's load_tasks() API.")

    # Phase 2: Run judge
    print(f"\n{'='*70}")
    print("PHASE 2: RUN OUR JUDGE")
    print(f"{'='*70}")
    print(f"  Scoring all {len(CALIBRATION_CASES)} cases with {JUDGE_MODEL}...")
    print(f"  Pass/fail threshold: {PASS_THRESHOLD}/100\n")

    records = await run_judge_on_cases(client, CALIBRATION_CASES)

    # Phase 3: Measure agreement
    print(f"\n{'='*70}")
    print("PHASE 3: MEASURE AGREEMENT")
    print(f"{'='*70}")

    metrics = compute_agreement_metrics(records)

    cm = metrics["confusion_matrix"]
    print(f"\n  Agreement rate: {metrics['agreement_rate']:.1%} ({metrics['total_agreements']}/{metrics['total_cases']})")
    print(f"  Cohen's kappa:  {metrics['cohens_kappa']:.3f}")
    print(f"  Pearson r:      {metrics['pearson_r']:.3f}")
    print(f"  Spearman rho:   {metrics['spearman_r']:.3f}")
    print(f"\n  Confusion matrix:")
    print(f"                  Human PASS  Human FAIL")
    print(f"    Judge PASS       {cm['tp']:2d} (TP)     {cm['fp']:2d} (FP)")
    print(f"    Judge FAIL       {cm['fn']:2d} (FN)     {cm['tn']:2d} (TN)")
    print(f"\n  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1:        {metrics['f1']:.3f}")
    print(f"\n  Score distributions:")
    print(f"    Human PASS cases: mean={metrics['pass_score_mean']:.1f}, "
          f"range=[{metrics['pass_score_min']}, {metrics['pass_score_max']}]")
    print(f"    Human FAIL cases: mean={metrics['fail_score_mean']:.1f}, "
          f"range=[{metrics['fail_score_min']}, {metrics['fail_score_max']}]")
    separation = metrics["pass_score_mean"] - metrics["fail_score_mean"]
    print(f"    Separation: {separation:.1f} points")

    target_met = metrics["agreement_rate"] >= 0.80
    print(f"\n  TARGET (>= 80% agreement): {'MET' if target_met else 'NOT MET'}")

    # Disagreements
    disagreements = [r for r in records if not r.agrees]
    if disagreements:
        print(f"\n  Disagreements ({len(disagreements)}):")
        for r in disagreements:
            judge_lbl = "PASS" if r.our_pass == 1 else "FAIL"
            human_lbl = "PASS" if r.human_label == 1 else "FAIL"
            error_type = "FP" if r.our_pass == 1 and r.human_label == 0 else "FN"
            print(
                f"    Case {r.case_id:2d} ({error_type}): "
                f"score={r.our_score}, judge={judge_lbl}, human={human_lbl} "
                f"| \"{r.query[:40]}...\""
            )
            print(f"             Human: {r.human_reasoning[:80]}")
            print(f"             Judge: {r.judge_reasoning[:80]}")

    elapsed = time.time() - t_start

    # Phase 4: Write report
    print(f"\n{'='*70}")
    print("PHASE 4: LOG RESULTS")
    print(f"{'='*70}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report = generate_report(records, metrics, elapsed)
    REPORT_FILE.write_text(report)
    print(f"  Report written to: {REPORT_FILE}")

    # Also save raw results as JSON
    results_data = {
        "domain": HARNESS_NAME,
        "judge_model": JUDGE_MODEL,
        "pass_threshold": PASS_THRESHOLD,
        "calibration_source": "manual (20 cases)",
        "tau2_note": "tau2 not installed; using manually-labeled calibration set",
        "elapsed_seconds": round(elapsed, 1),
        "metrics": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in metrics.items()
        },
        "records": [
            {
                "case_id": r.case_id,
                "query": r.query,
                "expected_category": r.expected_category,
                "actual_category": r.actual_category,
                "human_label": r.human_label,
                "human_reasoning": r.human_reasoning,
                "our_score": r.our_score,
                "our_pass": r.our_pass,
                "category_score": r.category_score,
                "quality_score": r.quality_score,
                "helpfulness_score": r.helpfulness_score,
                "judge_reasoning": r.judge_reasoning,
                "agrees": r.agrees,
            }
            for r in records
        ],
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"  Results JSON written to: {RESULTS_JSON}")

    # Final summary
    print(f"\n{'='*70}")
    print("JUDGE CALIBRATION — FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Agreement:   {metrics['agreement_rate']:.1%} ({metrics['total_agreements']}/{metrics['total_cases']})")
    print(f"  Kappa:       {metrics['cohens_kappa']:.3f}")
    print(f"  Pearson:     {metrics['pearson_r']:.3f}")
    print(f"  Spearman:    {metrics['spearman_r']:.3f}")
    print(f"  Separation:  {separation:.1f} pts")
    print(f"  Target met:  {'YES' if target_met else 'NO'}")
    print(f"  Elapsed:     {elapsed:.1f}s")

    return results_data


async def main():
    print("=" * 70)
    print("Judge Calibration: measure LLM judge agreement with human labels")
    print("=" * 70)
    print(f"Judge model:  {JUDGE_MODEL}")
    print(f"Domain:       {HARNESS_NAME}")
    print(f"Cases:        {len(CALIBRATION_CASES)}")
    print(f"Threshold:    {PASS_THRESHOLD}/100")
    print(f"LLM delay:    {LLM_DELAY}s")
    print()
    print("Phases:")
    print("  1. LOAD    -- Load 20 manually-labeled calibration cases")
    print("  2. JUDGE   -- Score each case with our LLM-as-judge")
    print("  3. MEASURE -- Compute agreement, correlation, confusion matrix")
    print("  4. LOG     -- Write calibration report to sandbox/results/")

    await run_judge_calibration()


if __name__ == "__main__":
    asyncio.run(main())
