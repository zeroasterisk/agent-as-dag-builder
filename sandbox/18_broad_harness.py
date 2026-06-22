"""Prototype 18: Broad 6-Harness Benchmark with Tuned Promotion Threshold.

Extends Graph Gardener's hypothesis-tree learning loop (sandbox/17) with:

  A. 3 new harness domains:
     - Healthcare Triage (emergency, appointment, prescription, insurance, general-health)
     - E-commerce Support (order-status, returns, payment, product-info, shipping)
     - Legal Intake (personal-injury, family-law, business, criminal, real-estate)

  B. Expanded test cases: 30 per harness (180 total across 6 harnesses)

  C. Tuned promotion threshold:
     - OLD: promote only if holdout score IMPROVES over best-ever
     - NEW: promote if holdout score is within 1.0 pt of best-ever (lateral moves OK)
     - Holdout evaluated 2x and averaged (reduces noise)
     - Holdout split increased to 35% (more signal)

  D. 10 iterations targeting 3-5 promotions (not 0-1)

Usage:
    python sandbox/18_broad_harness.py                    # Run 10 iterations
    python sandbox/18_broad_harness.py --iterations 5     # Run 5 iterations
    python sandbox/18_broad_harness.py --reset            # Reset versioned YAMLs

Environment:
    GOOGLE_GENAI_USE_VERTEXAI=1
    GOOGLE_CLOUD_PROJECT=alanblount-demo
    GOOGLE_CLOUD_LOCATION=global
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# -- Environment defaults (Vertex AI) ----------------------------------------
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "alanblount-demo")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

from google import genai

# -- Imports from the base multi-harness system --------------------------------
import importlib.util as _ilu
import sys as _sys
_spec = _ilu.spec_from_file_location(
    "multi_harness",
    str(Path(__file__).parent / "11_multi_harness.py"),
)
_mh = _ilu.module_from_spec(_spec)
_sys.modules["multi_harness"] = _mh
_spec.loader.exec_module(_mh)

# Import infrastructure from base module
SANDBOX_DIR = _mh.SANDBOX_DIR
JUDGE_MODEL = _mh.JUDGE_MODEL
LLM_DELAY = _mh.LLM_DELAY
STABILITY_THRESHOLD = _mh.STABILITY_THRESHOLD
ROLLBACK_THRESHOLD = _mh.ROLLBACK_THRESHOLD
InteractionRecord = _mh.InteractionRecord
HarnessResult = _mh.HarnessResult
IterationResult = _mh.IterationResult
load_dag_config = _mh.load_dag_config
build_routing = _mh.build_routing
extract_categories_from_config = _mh.extract_categories_from_config
run_dag_query = _mh.run_dag_query
score_response = _mh.score_response
_single_judge_call = _mh._single_judge_call
_validate_dag_config = _mh._validate_dag_config
_build_agent_driven_context = _mh._build_agent_driven_context

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("broad_harness")

SCORES_FILE = SANDBOX_DIR / "scores_broad.json"

# Hypothesis strategies (same as sandbox/17)
STRATEGIES = ["templates", "knowledge", "clarity"]

# Tuned promotion threshold: allow lateral moves within this tolerance
PROMOTION_TOLERANCE = 1.0  # pts -- promote if holdout >= best - tolerance

# Holdout split: 65/35 (increased from 70/30 for more signal)
HOLDOUT_FRACTION = 0.35

# Number of holdout evaluation passes (averaged to reduce noise)
HOLDOUT_EVAL_PASSES = 2


# =============================================================================
# 1. Expanded Test Cases (30 per harness)
# =============================================================================

# --- Customer Support: expanded to 30 cases (was 23) ---
CUSTOMER_SUPPORT_CASES = _mh.CUSTOMER_SUPPORT_CASES + [
    {"id": 24, "query": "I was promised a promotional rate but I'm being charged full price",
     "expected_category": "billing",
     "quality_criteria": "Should investigate the promotional offer and resolve the billing discrepancy"},
    {"id": 25, "query": "Can I pause my subscription instead of canceling?",
     "expected_category": "billing",
     "quality_criteria": "Should explain pause/hold options if available, or alternatives to cancellation"},
    {"id": 26, "query": "My app shows a spinning wheel and won't load any content",
     "expected_category": "technical",
     "quality_criteria": "Should suggest clearing cache, checking network, reinstalling, or updating the app"},
    {"id": 27, "query": "I keep getting logged out every time I close the browser",
     "expected_category": "technical",
     "quality_criteria": "Should suggest checking cookie settings, browser extensions, and session preferences"},
    {"id": 28, "query": "Do you offer accessibility features for visually impaired users?",
     "expected_category": "general",
     "quality_criteria": "Should describe accessibility features, screen reader support, or direct to accessibility page"},
    {"id": 29, "query": "I'm moving to a different country, can I keep my account?",
     "expected_category": "general",
     "quality_criteria": "Should explain international availability, account transfer, or region-specific limitations"},
    {"id": 30, "query": "My family member passed away and I need to close their account",
     "expected_category": "general",
     "quality_criteria": "Should handle sensitively, explain the account closure process for deceased users"},
]

# --- IT Helpdesk: expanded to 30 cases (was 18) ---
IT_HELPDESK_CASES = _mh.IT_HELPDESK_CASES + [
    {"id": 19, "query": "I need to reset my MFA device, I got a new phone",
     "expected_category": "password-reset",
     "quality_criteria": "Should explain MFA reset process, mention IT security verification requirements"},
    {"id": 20, "query": "My temporary password expired before I could use it",
     "expected_category": "password-reset",
     "quality_criteria": "Should offer to issue a new temporary password and explain the expiration policy"},
    {"id": 21, "query": "I need Zoom installed for a client meeting tomorrow",
     "expected_category": "software-install",
     "quality_criteria": "Should prioritize the urgent timeline, explain fast-track install or web browser fallback"},
    {"id": 22, "query": "My antivirus keeps blocking a legitimate program I need",
     "expected_category": "software-install",
     "quality_criteria": "Should explain how to add exceptions or whitelist programs, mention security policy"},
    {"id": 23, "query": "My external monitor shows no signal when I plug it in",
     "expected_category": "hardware",
     "quality_criteria": "Should suggest checking cable, display settings, trying different ports, driver updates"},
    {"id": 24, "query": "My laptop battery dies after only 2 hours, it used to last all day",
     "expected_category": "hardware",
     "quality_criteria": "Should suggest battery diagnostics, power settings check, and potential replacement"},
    {"id": 25, "query": "The shared drive is extremely slow to load files",
     "expected_category": "network",
     "quality_criteria": "Should check network speed, suggest mapping drive, check for sync issues"},
    {"id": 26, "query": "I can't access any websites but Slack and email work fine",
     "expected_category": "network",
     "quality_criteria": "Should check proxy settings, DNS config, firewall rules, browser-specific issues"},
    {"id": 27, "query": "My webcam stopped working during a Teams meeting",
     "expected_category": "hardware",
     "quality_criteria": "Should suggest checking camera permissions, device manager, reinstalling drivers"},
    {"id": 28, "query": "I need to connect to the guest Wi-Fi for a contractor visiting",
     "expected_category": "network",
     "quality_criteria": "Should explain guest network access procedure, mention time-limited credentials"},
    {"id": 29, "query": "The Docker Desktop installer says I need admin rights",
     "expected_category": "software-install",
     "quality_criteria": "Should explain admin rights request process, mention developer tool provisioning"},
    {"id": 30, "query": "My docking station keeps disconnecting all peripherals randomly",
     "expected_category": "hardware",
     "quality_criteria": "Should suggest firmware update, trying different USB-C port, driver reinstall"},
]

# --- Sales Inquiry: expanded to 30 cases (was 17) ---
SALES_INQUIRY_CASES = _mh.SALES_INQUIRY_CASES + [
    {"id": 18, "query": "What's your uptime SLA for the enterprise plan?",
     "expected_category": "features",
     "quality_criteria": "Should mention specific SLA (e.g., 99.9%), explain what's covered and remedies"},
    {"id": 19, "query": "Do you have a startup or nonprofit discount?",
     "expected_category": "pricing",
     "quality_criteria": "Should explain startup/nonprofit programs, eligibility, and how to apply"},
    {"id": 20, "query": "Can we get a sandbox environment to test before committing?",
     "expected_category": "demo-request",
     "quality_criteria": "Should offer sandbox/trial options, explain setup process, and limitations"},
    {"id": 21, "query": "How does your data export work if we decide to leave?",
     "expected_category": "features",
     "quality_criteria": "Should explain data portability, export formats, and no lock-in policy"},
    {"id": 22, "query": "We need HIPAA compliance, do you support that?",
     "expected_category": "features",
     "quality_criteria": "Should address HIPAA compliance status, BAA availability, and security measures"},
    {"id": 23, "query": "What's the typical ROI your customers see?",
     "expected_category": "competitor-comparison",
     "quality_criteria": "Should share ROI metrics, case studies, or typical customer outcomes"},
    {"id": 24, "query": "Can you match the pricing we got from your competitor?",
     "expected_category": "pricing",
     "quality_criteria": "Should address price matching policy, focus on value differences, and offer to discuss"},
    {"id": 25, "query": "We want to pilot with 10 users before rolling out to 500",
     "expected_category": "demo-request",
     "quality_criteria": "Should explain pilot program, timeline, and how it transitions to full deployment"},
    {"id": 26, "query": "Does your platform support custom branding and white-labeling?",
     "expected_category": "features",
     "quality_criteria": "Should explain white-label options, customization level, and which plan includes it"},
    {"id": 27, "query": "What happens if we exceed our plan's user limit mid-contract?",
     "expected_category": "pricing",
     "quality_criteria": "Should explain overage policy, upgrade options, and flexibility during the contract"},
    {"id": 28, "query": "Can we see a case study from a company in our industry?",
     "expected_category": "competitor-comparison",
     "quality_criteria": "Should offer industry-specific case studies or references, ask about their industry"},
    {"id": 29, "query": "I'd like to schedule a technical deep-dive with your engineering team",
     "expected_category": "demo-request",
     "quality_criteria": "Should offer to arrange technical session, ask about specific areas of interest"},
    {"id": 30, "query": "Do you offer multi-year contract discounts?",
     "expected_category": "pricing",
     "quality_criteria": "Should explain multi-year pricing benefits, typical discounts, and commitment terms"},
]

# --- Healthcare Triage: 30 cases ---
HEALTHCARE_TRIAGE_CASES = [
    # Emergency (6 cases)
    {"id": 1, "query": "I have severe chest pain and it's radiating to my left arm",
     "expected_category": "emergency",
     "quality_criteria": "Should urge immediate 911 call, mention possible heart attack symptoms"},
    {"id": 2, "query": "My child swallowed a handful of pills from the medicine cabinet",
     "expected_category": "emergency",
     "quality_criteria": "Should urge calling Poison Control (1-800-222-1222) and 911 immediately"},
    {"id": 3, "query": "I can't breathe and my throat is swelling up after eating shellfish",
     "expected_category": "emergency",
     "quality_criteria": "Should recognize anaphylaxis, urge EpiPen use if available and call 911"},
    {"id": 4, "query": "My father suddenly can't speak and one side of his face is drooping",
     "expected_category": "emergency",
     "quality_criteria": "Should recognize stroke symptoms (FAST), urge immediate 911 call"},
    {"id": 5, "query": "I've been having thoughts of harming myself",
     "expected_category": "emergency",
     "quality_criteria": "Should take seriously, provide 988 Suicide & Crisis Lifeline, urge immediate help"},
    {"id": 6, "query": "I cut my hand deeply while cooking and it won't stop bleeding",
     "expected_category": "emergency",
     "quality_criteria": "Should advise applying pressure, elevating, and going to ER if bleeding doesn't stop"},
    # Appointment (6 cases)
    {"id": 7, "query": "How do I schedule a physical exam?",
     "expected_category": "appointment",
     "quality_criteria": "Should explain how to schedule, mention annual wellness visits and what to expect"},
    {"id": 8, "query": "I need to see a dermatologist for a skin concern",
     "expected_category": "appointment",
     "quality_criteria": "Should explain specialist referral process and how to schedule"},
    {"id": 9, "query": "Can I reschedule my appointment from Tuesday to Thursday?",
     "expected_category": "appointment",
     "quality_criteria": "Should explain rescheduling process, mention cancellation policy and portal"},
    {"id": 10, "query": "My doctor recommended I see a cardiologist, how do I get a referral?",
     "expected_category": "appointment",
     "quality_criteria": "Should explain referral process, insurance requirements, and scheduling steps"},
    {"id": 11, "query": "I need a same-day appointment, I'm feeling really sick",
     "expected_category": "appointment",
     "quality_criteria": "Should explain urgent/same-day appointment options, walk-in clinic alternatives"},
    {"id": 12, "query": "Do you offer telehealth appointments?",
     "expected_category": "appointment",
     "quality_criteria": "Should explain telehealth availability, how to schedule, and platform requirements"},
    # Prescription (6 cases)
    {"id": 13, "query": "I need to refill my blood pressure medication",
     "expected_category": "prescription",
     "quality_criteria": "Should explain refill process, mention portal/pharmacy options, verify prescription status"},
    {"id": 14, "query": "My new medication is making me dizzy and nauseous",
     "expected_category": "prescription",
     "quality_criteria": "Should acknowledge side effects, advise contacting prescribing doctor, mention severity thresholds"},
    {"id": 15, "query": "Can I get a 90-day supply instead of 30 days?",
     "expected_category": "prescription",
     "quality_criteria": "Should explain 90-day supply options, mail-order pharmacy, and insurance coverage"},
    {"id": 16, "query": "I'm switching pharmacies, how do I transfer my prescriptions?",
     "expected_category": "prescription",
     "quality_criteria": "Should explain pharmacy transfer process, mention 24-48 hour timeline"},
    {"id": 17, "query": "Is it safe to take ibuprofen with my current medications?",
     "expected_category": "prescription",
     "quality_criteria": "Should advise checking with pharmacist/doctor for interactions, not provide direct medical advice"},
    {"id": 18, "query": "My prescription expired and I need it renewed urgently",
     "expected_category": "prescription",
     "quality_criteria": "Should explain renewal process, mention contacting prescribing physician, urgent options"},
    # Insurance (6 cases)
    {"id": 19, "query": "Is this procedure covered by my insurance?",
     "expected_category": "insurance",
     "quality_criteria": "Should explain how to verify coverage, mention pre-authorization, and contact insurance"},
    {"id": 20, "query": "How much will my copay be for a specialist visit?",
     "expected_category": "insurance",
     "quality_criteria": "Should explain typical copay structures and how to find specific amount in plan"},
    {"id": 21, "query": "I received a bill I think is incorrect, the amount seems too high",
     "expected_category": "insurance",
     "quality_criteria": "Should explain billing review process, mention patient advocate, dispute steps"},
    {"id": 22, "query": "Does my plan cover mental health counseling?",
     "expected_category": "insurance",
     "quality_criteria": "Should explain mental health coverage parity, how to find in-network therapists"},
    {"id": 23, "query": "I need pre-authorization for an MRI, how does that work?",
     "expected_category": "insurance",
     "quality_criteria": "Should explain pre-auth process, typical timeline, and provider's role"},
    {"id": 24, "query": "Can you help me find an in-network provider near me?",
     "expected_category": "insurance",
     "quality_criteria": "Should explain how to use insurance portal/directory to find in-network providers"},
    # General Health (6 cases)
    {"id": 25, "query": "Is this rash something to worry about?",
     "expected_category": "general-health",
     "quality_criteria": "Should provide general rash info, mention when to see a doctor, avoid diagnosing"},
    {"id": 26, "query": "What vaccines do I need before traveling to Southeast Asia?",
     "expected_category": "general-health",
     "quality_criteria": "Should mention common travel vaccines, recommend scheduling a travel medicine appointment"},
    {"id": 27, "query": "How much water should I be drinking daily?",
     "expected_category": "general-health",
     "quality_criteria": "Should provide evidence-based hydration guidelines, mention individual variation"},
    {"id": 28, "query": "I've been having trouble sleeping for the past month",
     "expected_category": "general-health",
     "quality_criteria": "Should suggest sleep hygiene tips, mention when to consult a doctor for chronic insomnia"},
    {"id": 29, "query": "At what age should I start getting regular health screenings?",
     "expected_category": "general-health",
     "quality_criteria": "Should mention common screening recommendations by age, encourage talking to PCP"},
    {"id": 30, "query": "What's the difference between a cold and the flu?",
     "expected_category": "general-health",
     "quality_criteria": "Should explain symptom differences, mention flu testing, when to see a doctor"},
]

# --- E-commerce Support: 30 cases ---
ECOMMERCE_SUPPORT_CASES = [
    # Order Status (6 cases)
    {"id": 1, "query": "Where is my order? I placed it 5 days ago",
     "expected_category": "order-status",
     "quality_criteria": "Should ask for order number, explain tracking steps, mention typical delivery windows"},
    {"id": 2, "query": "My order shows delivered but I never received it",
     "expected_category": "order-status",
     "quality_criteria": "Should suggest checking with neighbors, provide missing package claim process"},
    {"id": 3, "query": "I placed an order but never got a confirmation email",
     "expected_category": "order-status",
     "quality_criteria": "Should suggest checking spam, verifying email address, and looking up order by account"},
    {"id": 4, "query": "Can I add another item to my existing order before it ships?",
     "expected_category": "order-status",
     "quality_criteria": "Should explain order modification window and process, or suggest placing a new order"},
    {"id": 5, "query": "My package has been stuck in transit for a week",
     "expected_category": "order-status",
     "quality_criteria": "Should explain common delays, offer to file a carrier inquiry, mention reshipment options"},
    {"id": 6, "query": "I want to cancel my order, it hasn't shipped yet",
     "expected_category": "order-status",
     "quality_criteria": "Should explain cancellation window, process, and refund timeline"},
    # Returns (6 cases)
    {"id": 7, "query": "I want to return this item, it doesn't fit properly",
     "expected_category": "returns",
     "quality_criteria": "Should explain return process, mention size exchange option, and return policy"},
    {"id": 8, "query": "The product arrived damaged, the screen is cracked",
     "expected_category": "returns",
     "quality_criteria": "Should apologize, offer replacement or refund, explain damaged goods process"},
    {"id": 9, "query": "I received the wrong color, I ordered blue but got red",
     "expected_category": "returns",
     "quality_criteria": "Should apologize for the error, offer exchange for correct color, free return shipping"},
    {"id": 10, "query": "Can I return something I bought during a sale?",
     "expected_category": "returns",
     "quality_criteria": "Should explain return policy for sale items, any exceptions or restocking fees"},
    {"id": 11, "query": "How do I print a return shipping label?",
     "expected_category": "returns",
     "quality_criteria": "Should explain label generation from returns portal, drop-off locations"},
    {"id": 12, "query": "I returned an item 2 weeks ago but still haven't gotten my refund",
     "expected_category": "returns",
     "quality_criteria": "Should explain refund processing timeline, suggest checking return tracking"},
    # Payment (6 cases)
    {"id": 13, "query": "My payment was declined but I have enough funds",
     "expected_category": "payment",
     "quality_criteria": "Should suggest verifying card details, contacting bank, trying another payment method"},
    {"id": 14, "query": "Do you accept PayPal or Apple Pay?",
     "expected_category": "payment",
     "quality_criteria": "Should list accepted payment methods clearly"},
    {"id": 15, "query": "I have a promo code but it's not working at checkout",
     "expected_category": "payment",
     "quality_criteria": "Should troubleshoot promo code issues, check expiry and terms"},
    {"id": 16, "query": "Can I pay in installments for this expensive item?",
     "expected_category": "payment",
     "quality_criteria": "Should explain installment/buy-now-pay-later options and eligibility"},
    {"id": 17, "query": "I was charged twice for the same order",
     "expected_category": "payment",
     "quality_criteria": "Should acknowledge the issue, explain investigation process, mention refund timeline"},
    {"id": 18, "query": "How do I check my gift card balance?",
     "expected_category": "payment",
     "quality_criteria": "Should explain gift card balance check process, mention where to find card number"},
    # Product Info (6 cases)
    {"id": 19, "query": "Does this come in blue?",
     "expected_category": "product-info",
     "quality_criteria": "Should explain how to check available colors, mention product page variants"},
    {"id": 20, "query": "Is this product compatible with iPhone 15?",
     "expected_category": "product-info",
     "quality_criteria": "Should suggest checking compatibility in specifications, offer to verify"},
    {"id": 21, "query": "What are the dimensions of this desk?",
     "expected_category": "product-info",
     "quality_criteria": "Should direct to specifications, mention where to find exact dimensions"},
    {"id": 22, "query": "Is this item currently in stock or backordered?",
     "expected_category": "product-info",
     "quality_criteria": "Should explain how to check real-time availability, mention backorder timeline"},
    {"id": 23, "query": "What's the difference between the standard and premium version?",
     "expected_category": "product-info",
     "quality_criteria": "Should compare versions clearly, highlight key differences and value"},
    {"id": 24, "query": "Does this product come with a warranty?",
     "expected_category": "product-info",
     "quality_criteria": "Should explain warranty terms, duration, and how to make a claim"},
    # Shipping (6 cases)
    {"id": 25, "query": "How long does shipping take?",
     "expected_category": "shipping",
     "quality_criteria": "Should explain shipping speed options and typical delivery timeframes"},
    {"id": 26, "query": "Do you ship internationally to Canada?",
     "expected_category": "shipping",
     "quality_criteria": "Should confirm international shipping availability, mention customs and additional costs"},
    {"id": 27, "query": "Can I change my shipping address after placing an order?",
     "expected_category": "shipping",
     "quality_criteria": "Should explain address change window and process"},
    {"id": 28, "query": "Is there free shipping on orders over a certain amount?",
     "expected_category": "shipping",
     "quality_criteria": "Should explain free shipping threshold and any conditions"},
    {"id": 29, "query": "I need this delivered by Friday, what are my express options?",
     "expected_category": "shipping",
     "quality_criteria": "Should explain express/next-day options, cutoff times, and costs"},
    {"id": 30, "query": "Can I pick up my order at a store instead of having it shipped?",
     "expected_category": "shipping",
     "quality_criteria": "Should explain in-store pickup options if available, or suggest alternatives"},
]

# --- Legal Intake: 30 cases ---
LEGAL_INTAKE_CASES = [
    # Personal Injury (6 cases)
    {"id": 1, "query": "I was in a car accident last week and the other driver was at fault",
     "expected_category": "personal-injury",
     "quality_criteria": "Should express empathy, ask about injuries, mention statute of limitations and contingency fee"},
    {"id": 2, "query": "I slipped and fell at a grocery store and broke my wrist",
     "expected_category": "personal-injury",
     "quality_criteria": "Should discuss premises liability, importance of incident report, medical records"},
    {"id": 3, "query": "My doctor made a mistake during surgery and now I have complications",
     "expected_category": "personal-injury",
     "quality_criteria": "Should discuss medical malpractice, mention expert review needed, timeline concerns"},
    {"id": 4, "query": "I was injured at work when a shelf collapsed on me",
     "expected_category": "personal-injury",
     "quality_criteria": "Should distinguish workers comp vs personal injury claim, mention employer liability"},
    {"id": 5, "query": "A defective product exploded and burned my hand",
     "expected_category": "personal-injury",
     "quality_criteria": "Should discuss product liability, importance of preserving the product as evidence"},
    {"id": 6, "query": "I was bitten by my neighbor's dog while jogging",
     "expected_category": "personal-injury",
     "quality_criteria": "Should mention dog bite laws, homeowner's insurance, medical documentation"},
    # Family Law (6 cases)
    {"id": 7, "query": "I need help with my divorce, we can't agree on anything",
     "expected_category": "family-law",
     "quality_criteria": "Should explain contested divorce process, mention mediation option, timeline"},
    {"id": 8, "query": "My ex won't let me see my kids on my scheduled weekends",
     "expected_category": "family-law",
     "quality_criteria": "Should discuss custody enforcement, contempt of court, documentation importance"},
    {"id": 9, "query": "I want to adopt my stepchild, what's the process?",
     "expected_category": "family-law",
     "quality_criteria": "Should explain stepparent adoption process, consent requirements, home study"},
    {"id": 10, "query": "My spouse and I want a prenuptial agreement before our wedding",
     "expected_category": "family-law",
     "quality_criteria": "Should explain prenup purpose, what it can cover, recommend both parties have counsel"},
    {"id": 11, "query": "I need to modify my child support payments, I lost my job",
     "expected_category": "family-law",
     "quality_criteria": "Should explain modification process, material change in circumstances, court filing"},
    {"id": 12, "query": "I'm worried about domestic violence, I need a protective order",
     "expected_category": "family-law",
     "quality_criteria": "Should take seriously, explain protective order process, mention immediate resources"},
    # Business (6 cases)
    {"id": 13, "query": "How do I incorporate an LLC?",
     "expected_category": "business",
     "quality_criteria": "Should explain LLC formation steps, filing requirements, operating agreement importance"},
    {"id": 14, "query": "I received a cease and desist letter about my company name",
     "expected_category": "business",
     "quality_criteria": "Should explain C&D response options, trademark implications, timeline to respond"},
    {"id": 15, "query": "My business partner is stealing from the company",
     "expected_category": "business",
     "quality_criteria": "Should discuss partnership disputes, fiduciary duties, legal remedies and evidence"},
    {"id": 16, "query": "I need someone to review a commercial lease before I sign",
     "expected_category": "business",
     "quality_criteria": "Should explain lease review importance, common issues to watch for, flat fee options"},
    {"id": 17, "query": "A customer is threatening to sue us over a contract dispute",
     "expected_category": "business",
     "quality_criteria": "Should discuss contract dispute resolution, mediation/arbitration clauses, documentation"},
    {"id": 18, "query": "I want to trademark my company logo and slogan",
     "expected_category": "business",
     "quality_criteria": "Should explain trademark registration process, search importance, costs and timeline"},
    # Criminal (6 cases)
    {"id": 19, "query": "I was arrested for DUI last night, what should I do?",
     "expected_category": "criminal",
     "quality_criteria": "Should emphasize right to counsel, not to discuss case, explain typical DUI process"},
    {"id": 20, "query": "I've been accused of shoplifting but I didn't do it",
     "expected_category": "criminal",
     "quality_criteria": "Should advise not to discuss with anyone, explain defense options, mention consequences"},
    {"id": 21, "query": "Can I get my criminal record expunged?",
     "expected_category": "criminal",
     "quality_criteria": "Should explain expungement eligibility, process, and typical timeline"},
    {"id": 22, "query": "The police want to question me about an incident, should I go?",
     "expected_category": "criminal",
     "quality_criteria": "Should strongly advise having an attorney present, explain 5th Amendment rights"},
    {"id": 23, "query": "My son was charged with drug possession, he's only 17",
     "expected_category": "criminal",
     "quality_criteria": "Should explain juvenile vs adult court, diversion programs, parental rights"},
    {"id": 24, "query": "I need help posting bail for a family member",
     "expected_category": "criminal",
     "quality_criteria": "Should explain bail process, bond options, conditions of release"},
    # Real Estate (6 cases)
    {"id": 25, "query": "I want to buy commercial property for my restaurant",
     "expected_category": "real-estate",
     "quality_criteria": "Should discuss commercial purchase process, zoning verification, due diligence"},
    {"id": 26, "query": "My landlord won't return my security deposit",
     "expected_category": "real-estate",
     "quality_criteria": "Should explain tenant rights, security deposit laws, demand letter process"},
    {"id": 27, "query": "I'm buying a house and need an attorney for the closing",
     "expected_category": "real-estate",
     "quality_criteria": "Should explain attorney role at closing, title review, typical flat fees"},
    {"id": 28, "query": "My neighbor's fence is on my property, what can I do?",
     "expected_category": "real-estate",
     "quality_criteria": "Should discuss boundary disputes, survey options, resolution approaches"},
    {"id": 29, "query": "I'm facing foreclosure, what are my options?",
     "expected_category": "real-estate",
     "quality_criteria": "Should explain foreclosure defense options, loan modification, short sale, timeline"},
    {"id": 30, "query": "My HOA is fining me and I think it's unfair",
     "expected_category": "real-estate",
     "quality_criteria": "Should explain HOA dispute process, governing documents review, appeal rights"},
]


# =============================================================================
# 2. Harness Definitions (6 harnesses)
# =============================================================================

ALL_HARNESS_DEFS = [
    {"name": "customer_support",  "yaml": "customer_support_adk.yaml", "cases": CUSTOMER_SUPPORT_CASES},
    {"name": "it_helpdesk",       "yaml": "it_helpdesk.yaml",          "cases": IT_HELPDESK_CASES},
    {"name": "sales_inquiry",     "yaml": "sales_inquiry.yaml",        "cases": SALES_INQUIRY_CASES},
    {"name": "healthcare_triage", "yaml": "healthcare_triage.yaml",    "cases": HEALTHCARE_TRIAGE_CASES},
    {"name": "ecommerce_support", "yaml": "ecommerce_support.yaml",    "cases": ECOMMERCE_SUPPORT_CASES},
    {"name": "legal_intake",      "yaml": "legal_intake.yaml",         "cases": LEGAL_INTAKE_CASES},
]


# =============================================================================
# 3. Held-Out Split (65/35)
# =============================================================================

def split_cases(cases: list[dict], holdout_frac: float = HOLDOUT_FRACTION, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split cases into (1-holdout_frac) training and holdout_frac held-out.

    Returns (train_cases, holdout_cases).
    """
    rng = random.Random(seed)
    shuffled = list(cases)
    rng.shuffle(shuffled)
    split_point = int(len(shuffled) * (1.0 - holdout_frac))
    train_cases = shuffled[:split_point]
    holdout_cases = shuffled[split_point:]
    return train_cases, holdout_cases


# Build harnesses with train/holdout splits
HARNESSES = []
for h in ALL_HARNESS_DEFS:
    train, holdout = split_cases(h["cases"])
    HARNESSES.append({
        "name": h["name"],
        "yaml": h["yaml"],
        "cases": h["cases"],
        "train_cases": train,
        "holdout_cases": holdout,
    })


# =============================================================================
# 4. Insight Memory (same as sandbox/17)
# =============================================================================

@dataclass
class Insight:
    """Record of what worked or didn't in a hypothesis test."""
    iteration: int
    harness: str
    strategy: str
    description: str
    train_delta: float
    holdout_delta: float
    accepted: bool

    def __str__(self) -> str:
        status = "ACCEPTED" if self.accepted else "REJECTED"
        return (
            f"Iter {self.iteration}: \"{self.strategy}\" on {self.harness}: "
            f"train {self.train_delta:+.1f}, holdout {self.holdout_delta:+.1f} ({status})"
        )


insight_memory: list[Insight] = []


def get_relevant_insights(harness_name: str, max_insights: int = 10) -> str:
    """Format relevant past insights for the hypothesis generation prompt."""
    relevant = [i for i in insight_memory if i.harness == harness_name]
    general = [i for i in insight_memory if i.harness != harness_name and abs(i.train_delta) > 3.0]
    all_insights = relevant + general[-3:]
    all_insights = all_insights[-max_insights:]
    if not all_insights:
        return "No past insights available yet."
    lines = ["Past insights for this harness:"]
    for insight in all_insights:
        prefix = "" if insight.harness == harness_name else f"[from {insight.harness}] "
        lines.append(f"  - {prefix}{insight}")
    return "\n".join(lines)


# =============================================================================
# 5. Hypothesis Generation (same strategies as sandbox/17)
# =============================================================================

STRATEGY_PROMPTS = {
    "templates": """You are an AI workflow optimizer specializing in ISSUE-SPECIFIC TEMPLATES.
Your strategy: Add step-by-step templates for specific sub-issues that the weak cases reveal.

For each weak case pattern, create a concrete template like:
  "For SCREEN FLICKERING: 1. Check refresh rate settings 2. Update display driver
   3. Test with external monitor 4. If persists, submit hardware ticket"

Focus on ACTIONABLE STEPS for specific issue types seen in the weak cases.
Each template should be a numbered, specific troubleshooting or response flow.""",

    "knowledge": """You are an AI workflow optimizer specializing in DOMAIN KNOWLEDGE INJECTION.
Your strategy: Add concrete, realistic details that make responses more actionable.

Inject knowledge like:
  - Server names, URLs, phone numbers, specific portals
  - Specific policy details (timelines, fees, requirements)
  - Tool names, commands, or procedures
  - Industry-specific terminology and standards

Focus on CONCRETE DETAILS that the handler currently lacks.""",

    "clarity": """You are an AI workflow optimizer specializing in INSTRUCTION CLARITY.
Your strategy: Make handler instructions clearer and more focused to improve response quality.

Improvements include:
  - Remove ambiguous or redundant phrases
  - Add explicit quality expectations: "ALWAYS mention X when the user asks about Y"
  - Add explicit structure requirements: "Structure your response as: 1. Acknowledge 2. Diagnose 3. Steps 4. Escalation path"
  - Add response format hints: "Keep responses under 200 words. Use numbered steps."

Focus on making the instruction UNAMBIGUOUS about what a good response looks like.""",
}


async def generate_hypothesis_variant(
    client: genai.Client,
    current_config: dict,
    strategy: str,
    harness_result: HarnessResult,
    harness_name: str,
    insights_text: str,
    iteration: int,
) -> dict:
    """Generate one hypothesis variant using a specific strategy."""
    strategy_prompt = STRATEGY_PROMPTS[strategy]
    valid_categories = extract_categories_from_config(current_config)
    focused_context = _build_agent_driven_context(harness_result, current_config, harness_name)
    breakdown = harness_result.category_breakdown()

    analysis = f"""Benchmark Results for {harness_name} (Iteration {iteration}, TRAINING SET ONLY):
- Aggregate Score: {harness_result.aggregate_score:.1f}/100
- Category Accuracy: {harness_result.category_accuracy:.0%}
- Quality Score: {harness_result.avg_quality:.1f}/30
- Helpfulness Score: {harness_result.avg_helpfulness:.1f}/20

Valid Categories: {', '.join(valid_categories)}

Category Breakdown:
"""
    for cat, info in breakdown.items():
        analysis += f"  {cat}: avg={info['avg_score']:.1f}, accuracy={info['accuracy']:.0%} ({info['total']} cases)\n"

    analysis += f"\n{focused_context}"

    prompt = f"""{strategy_prompt}

{analysis}

{insights_text}

IMPORTANT RULES:
1. You must NOT rewrite handler instructions from scratch
2. You must ONLY propose ADDITIONS to append to the existing handler instructions
3. Focus on the weakest category shown above
4. Use your specific strategy ({strategy}) to guide what kind of additions you propose

Propose 2-3 specific ADDITIONS to append to the weakest handler.
For each proposal, specify:
- The exact text to APPEND to the handler instruction
- Which handler node it applies to
- Why (referencing specific weak cases)

Reply with a numbered list. Be specific and concrete."""

    await asyncio.sleep(LLM_DELAY)
    resp = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
    )
    proposals_text = resp.candidates[0].content.parts[0].text.strip()

    proposals = []
    current = ""
    for line in proposals_text.split("\n"):
        if re.match(r"^\d+[\.\)]\s", line.strip()) and current:
            proposals.append(current.strip())
            current = line
        else:
            current += "\n" + line
    if current.strip():
        proposals.append(current.strip())

    new_config = await _apply_proposals_to_config(
        client, current_config, proposals, iteration, harness_name, strategy,
    )
    return new_config


async def _apply_proposals_to_config(
    client: genai.Client,
    current_config: dict,
    proposals: list[str],
    iteration: int,
    harness_name: str,
    strategy: str,
) -> dict:
    """Apply proposals to a deep copy of the config."""
    proposals_text = "\n".join(proposals)

    extract_prompt = f"""You are a precise text extractor. Given the improvement proposals below,
extract the text that should be APPENDED to handler node instructions.

PROPOSALS:
{proposals_text}

CURRENT HANDLER NODE IDs: {', '.join(n['id'] for n in current_config['nodes'] if n['id'] != 'classify')}

For each proposal, output a JSON object with:
- "node_id": which handler node to append to (must be one of the node IDs above)
- "append_text": the exact text to append to that node's instruction

Output a JSON array of these objects. No markdown, no explanation.
Example: [{{"node_id": "handle_network", "append_text": "\\nFor VPN issues:\\n1. Check credentials..."}}]"""

    await asyncio.sleep(LLM_DELAY)
    try:
        resp = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=[{"role": "user", "parts": [{"text": extract_prompt}]}],
            config={"temperature": 0.0},
        )
        extract_text = resp.candidates[0].content.parts[0].text.strip()

        if "```" in extract_text:
            match = re.search(r"```(?:json)?\s*(.*?)```", extract_text, re.DOTALL)
            if match:
                extract_text = match.group(1).strip()

        json_match = re.search(r'\[.*\]', extract_text, re.DOTALL)
        if json_match:
            extract_text = json_match.group(0)

        additions = json.loads(extract_text)
        if not isinstance(additions, list):
            additions = [additions]

    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Failed to extract additions for %s/%s: %s. Fallback.", harness_name, strategy, e)
        handler_ids = [n["id"] for n in current_config["nodes"] if n["id"] != "classify"]
        target_handler = handler_ids[0]
        for hid in handler_ids:
            if hid in proposals_text.lower() or hid.replace("handle_", "") in proposals_text.lower():
                target_handler = hid
                break
        additions = [{
            "node_id": target_handler,
            "append_text": f"\n\nADDITIONAL GUIDANCE ({strategy}, iter {iteration}):\n" + proposals_text[:1500],
        }]

    new_config = copy.deepcopy(current_config)
    new_config["version"] = f"4.0.{iteration}-{strategy}"

    nodes_by_id = {n["id"]: n for n in new_config["nodes"]}
    applied_count = 0

    for addition in additions:
        node_id = addition.get("node_id", "")
        append_text = addition.get("append_text", "")
        if not node_id or not append_text:
            continue

        target_node = nodes_by_id.get(node_id)
        if target_node is None:
            for nid, node in nodes_by_id.items():
                if node_id in nid or nid in node_id:
                    target_node = node
                    break

        if target_node is None or target_node["id"] == "classify":
            continue

        current_instruction = target_node.get("instruction", "")
        if len(current_instruction) + len(append_text) > 4000:
            append_text = append_text[:4000 - len(current_instruction)]

        target_node["instruction"] = current_instruction.rstrip() + "\n\n" + append_text.strip() + "\n"
        applied_count += 1

    if applied_count == 0:
        logger.warning("No additions applied for %s/%s. Returning copy.", harness_name, strategy)
        new_config["version"] = f"4.0.{iteration}-{strategy}"

    _validate_dag_config(new_config, harness_name)
    return new_config


async def generate_hypotheses(
    client: genai.Client,
    config: dict,
    harness_result: HarnessResult,
    harness_name: str,
    iteration: int,
) -> list[tuple[str, dict]]:
    """Generate 3 hypothesis configs, one per strategy."""
    insights_text = get_relevant_insights(harness_name)
    variants = []
    for strategy in STRATEGIES:
        try:
            variant_config = await generate_hypothesis_variant(
                client, config, strategy,
                harness_result, harness_name, insights_text, iteration,
            )
            variants.append((strategy, variant_config))
        except Exception as e:
            logger.warning("Failed %s variant for %s: %s", strategy, harness_name, e)
            print(f"    WARNING: {strategy} variant generation failed: {e}")
    return variants


# =============================================================================
# 6. Benchmark Runner
# =============================================================================

async def run_harness_benchmark(
    client: genai.Client,
    config: dict,
    config_path: str,
    cases: list[dict],
    harness_name: str,
    iteration: int,
    label: str = "",
) -> HarnessResult:
    """Run benchmark cases for one harness and score them."""
    result = HarnessResult(
        harness_name=harness_name,
        iteration=iteration,
        config_path=config_path,
    )

    prefix = f"[{label}] " if label else ""

    for i, case in enumerate(cases):
        dag_result = await run_dag_query(client, config, case["query"])
        await asyncio.sleep(LLM_DELAY)

        score = await score_response(
            client,
            query=case["query"],
            expected_category=case["expected_category"],
            actual_category=dag_result["category"],
            response=dag_result["response"],
            quality_criteria=case["quality_criteria"],
            harness_name=harness_name,
        )

        record = InteractionRecord(
            query=case["query"],
            case_id=case["id"],
            expected_category=case["expected_category"],
            actual_category=dag_result["category"],
            response=dag_result["response"],
            score=score,
            quality_criteria=case["quality_criteria"],
        )
        result.records.append(record)

        cat_ok = "correct" if dag_result["category"] == case["expected_category"] else f"WRONG:{dag_result['category']}"
        print(f"    {prefix}Case {case['id']:2d}: \"{case['query'][:40]}...\" -> {dag_result['category']} ({cat_ok}) -> {score['total_score']}")

        await asyncio.sleep(LLM_DELAY)

    return result


# =============================================================================
# 7. Score Tracking
# =============================================================================

def save_scores(
    all_iterations: list[dict],
    insights: list[Insight],
    promotions: list[dict],
) -> None:
    """Save score history, insights, and promotions to scores_broad.json."""
    data = {
        "iterations": all_iterations,
        "insights": [
            {
                "iteration": i.iteration,
                "harness": i.harness,
                "strategy": i.strategy,
                "description": i.description,
                "train_delta": round(i.train_delta, 2),
                "holdout_delta": round(i.holdout_delta, 2),
                "accepted": i.accepted,
            }
            for i in insights
        ],
        "promotions": promotions,
    }
    with open(SCORES_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nScores saved to {SCORES_FILE.name}")


# =============================================================================
# 8. Main Loop: 6-Harness Hypothesis-Tree with Tuned Threshold
# =============================================================================

async def run_broad_harness_loop(iterations: int = 10) -> None:
    """Run the 6-harness hypothesis-tree learning loop with tuned promotion.

    Key change from sandbox/17:
      - Promotion threshold: accept if holdout >= (best_holdout - PROMOTION_TOLERANCE)
      - Holdout evaluated 2x and averaged
      - 35% holdout split (was 30%)
      - 6 harnesses, 30 cases each
    """
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "alanblount-demo"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
    )

    global insight_memory
    insight_memory = []

    all_iteration_data: list[dict] = []
    promotion_log: list[dict] = []

    # Best-config tracking
    best_config_content: dict[str, str] = {}
    best_harness_train: dict[str, float] = {}
    best_harness_holdout: dict[str, float] = {}
    best_aggregate_holdout: float = 0.0

    # Baseline scores (from iteration 1)
    baseline_scores: dict[str, float] = {}

    # Current config paths
    current_configs: dict[str, Path] = {}
    for h in HARNESSES:
        base_path = SANDBOX_DIR / h["yaml"]
        current_configs[h["name"]] = base_path
        with open(base_path) as f:
            best_config_content[h["name"]] = f.read()
        best_harness_train[h["name"]] = 0.0
        best_harness_holdout[h["name"]] = 0.0

    summary_rows: list[dict] = []

    for iteration in range(1, iterations + 1):
        print(f"\n{'='*70}")
        print(f"=== ITERATION {iteration}/{iterations} ===")
        print(f"{'='*70}")

        iter_data = {
            "iteration": iteration,
            "harnesses": {},
            "hypotheses_tested": 0,
            "winner_strategy": None,
        }

        # ---- Step 1: Evaluate all harnesses on TRAINING set ----
        train_results: dict[str, HarnessResult] = {}
        for harness in HARNESSES:
            hname = harness["name"]
            config_path = current_configs[hname]
            config = load_dag_config(config_path)
            dag_version = config.get("version", "?")

            print(f"\n  --- {hname} TRAINING ({config_path.name}, v{dag_version}) ---")
            print(f"  Running {len(harness['train_cases'])} training cases...")

            hr = await run_harness_benchmark(
                client, config, str(config_path),
                harness["train_cases"], hname, iteration, label="train",
            )
            train_results[hname] = hr

        # ---- Step 2: Evaluate all harnesses on HELD-OUT set ----
        holdout_results: dict[str, HarnessResult] = {}
        for harness in HARNESSES:
            hname = harness["name"]
            config_path = current_configs[hname]
            config = load_dag_config(config_path)

            print(f"\n  --- {hname} HELD-OUT ({config_path.name}) ---")
            print(f"  Running {len(harness['holdout_cases'])} held-out cases...")

            hr = await run_harness_benchmark(
                client, config, str(config_path),
                harness["holdout_cases"], hname, iteration, label="holdout",
            )
            holdout_results[hname] = hr

        # Compute aggregates
        train_agg = sum(hr.aggregate_score for hr in train_results.values()) / len(train_results) if train_results else 0.0
        holdout_agg = sum(hr.aggregate_score for hr in holdout_results.values()) / len(holdout_results) if holdout_results else 0.0

        # Print summary
        print(f"\n=== ITERATION {iteration} SCORES ===")
        for hname in [h["name"] for h in HARNESSES]:
            ts = train_results[hname].aggregate_score
            hs = holdout_results[hname].aggregate_score
            print(f"  {hname:20s}: train={ts:.1f}, holdout={hs:.1f}")
        print(f"  {'AGGREGATE':20s}: train={train_agg:.1f}, holdout={holdout_agg:.1f}")

        # Save baseline
        if iteration == 1:
            for hname in [h["name"] for h in HARNESSES]:
                baseline_scores[hname] = holdout_results[hname].aggregate_score

        # Store per-harness data
        for harness in HARNESSES:
            hname = harness["name"]
            iter_data["harnesses"][hname] = {
                "train_score": round(train_results[hname].aggregate_score, 2),
                "holdout_score": round(holdout_results[hname].aggregate_score, 2),
                "train_cases": len(harness["train_cases"]),
                "holdout_cases": len(harness["holdout_cases"]),
                "category_accuracy": round(train_results[hname].category_accuracy, 4),
            }

        # ---- Update best scores ----
        for hname in [h["name"] for h in HARNESSES]:
            ts = train_results[hname].aggregate_score
            hs = holdout_results[hname].aggregate_score
            if ts > best_harness_train[hname]:
                best_harness_train[hname] = ts
            if hs > best_harness_holdout[hname]:
                best_harness_holdout[hname] = hs
                config_path = current_configs[hname]
                with open(config_path) as f:
                    best_config_content[hname] = f.read()

        if holdout_agg > best_aggregate_holdout:
            best_aggregate_holdout = holdout_agg

        # ---- Step 3: Rollback check ----
        rollback_triggered = False
        if iteration > 1 and holdout_agg < (best_aggregate_holdout - ROLLBACK_THRESHOLD):
            rollback_triggered = True
            print(f"\n*** ROLLBACK: holdout {holdout_agg:.1f} < best {best_aggregate_holdout:.1f} - {ROLLBACK_THRESHOLD} ***")
            for harness in HARNESSES:
                hname = harness["name"]
                base_name = Path(harness["yaml"]).stem
                rollback_path = SANDBOX_DIR / f"{base_name}_broad_v{iteration + 1}.yaml"
                with open(rollback_path, "w") as f_out:
                    f_out.write(best_config_content[hname])
                current_configs[hname] = rollback_path
                print(f"  [{hname}] Rolled back -> {rollback_path.name}")

        # ---- Step 4: Hypothesis generation ----
        winner_strategy = None
        hypotheses_tested = 0
        promotion_entry = {
            "iteration": iteration,
            "target_harness": None,
            "accepted": False,
            "strategy": None,
            "holdout_delta": 0.0,
        }

        if iteration < iterations and not rollback_triggered:
            # Find weakest harness by training score
            harness_scores = [(h["name"], train_results[h["name"]].aggregate_score) for h in HARNESSES]
            harness_scores.sort(key=lambda x: x[1])  # lowest first
            weakest_name, weakest_score = harness_scores[0]

            print(f"\nWeakest harness: {weakest_name} (train={weakest_score:.1f})")

            print(f"Generating 3 hypotheses for {weakest_name}...")

            # Load best config for this harness
            best_yaml = best_config_content[weakest_name]
            best_parsed = yaml.safe_load(best_yaml)
            best_config = best_parsed.get("dag", best_parsed)

            harness_obj = next(h for h in HARNESSES if h["name"] == weakest_name)
            train_hr = train_results[weakest_name]

            # Generate 3 hypothesis variants
            variants = await generate_hypotheses(
                client, best_config, train_hr, weakest_name, iteration,
            )
            hypotheses_tested = len(variants)

            if not variants:
                print("  No valid hypothesis variants generated. Skipping.")
            else:
                # Evaluate each variant on TRAINING cases of weakest harness
                variant_train_scores: list[tuple[str, float, dict]] = []

                for strategy, variant_config in variants:
                    print(f"\n  Evaluating Hypothesis ({strategy}) on training set...")
                    variant_hr = await run_harness_benchmark(
                        client, variant_config, f"hypothesis-{strategy}",
                        harness_obj["train_cases"], weakest_name, iteration,
                        label=f"hyp-{strategy[0].upper()}",
                    )
                    variant_train_scores.append((strategy, variant_hr.aggregate_score, variant_config))
                    print(f"  Hypothesis ({strategy}): train={variant_hr.aggregate_score:.1f}")

                # Pick winner: highest training score
                variant_train_scores.sort(key=lambda x: -x[1])
                winner_strategy, winner_train_score, winner_config = variant_train_scores[0]

                # Validate winner on HELD-OUT cases (2x averaged)
                print(f"\n  Winner: {winner_strategy} (train={winner_train_score:.1f})")
                print(f"  Validating on held-out set ({HOLDOUT_EVAL_PASSES}x averaged)...")

                holdout_scores_list = []
                for pass_num in range(1, HOLDOUT_EVAL_PASSES + 1):
                    holdout_hr = await run_harness_benchmark(
                        client, winner_config, f"winner-{winner_strategy}-pass{pass_num}",
                        harness_obj["holdout_cases"], weakest_name, iteration,
                        label=f"holdout-v{pass_num}",
                    )
                    holdout_scores_list.append(holdout_hr.aggregate_score)
                    print(f"  Holdout pass {pass_num}: {holdout_hr.aggregate_score:.1f}")

                winner_holdout_score = sum(holdout_scores_list) / len(holdout_scores_list)
                current_holdout_score = holdout_results[weakest_name].aggregate_score
                best_holdout_for_harness = best_harness_holdout[weakest_name]

                holdout_delta = winner_holdout_score - current_holdout_score
                train_delta = winner_train_score - train_results[weakest_name].aggregate_score

                print(f"\n  Hypothesis results for {weakest_name}:")
                for strategy, train_score, _ in variant_train_scores:
                    marker = " <- WINNER" if strategy == winner_strategy else ""
                    print(f"    ({strategy}): train={train_score:.1f}{marker}")
                print(f"  Winner holdout (avg): {winner_holdout_score:.1f}")
                print(f"  Current holdout:      {current_holdout_score:.1f}")
                print(f"  Best-ever holdout:    {best_holdout_for_harness:.1f}")
                print(f"  Delta vs current:     {holdout_delta:+.1f}")

                # TUNED PROMOTION: accept if within PROMOTION_TOLERANCE of best-ever
                accepted = winner_holdout_score >= (best_holdout_for_harness - PROMOTION_TOLERANCE)
                promotion_reason = ""
                if winner_holdout_score > best_holdout_for_harness:
                    promotion_reason = "NEW_BEST"
                elif winner_holdout_score >= (best_holdout_for_harness - PROMOTION_TOLERANCE):
                    promotion_reason = "LATERAL_MOVE"
                else:
                    promotion_reason = "REGRESSION"

                if accepted:
                    base_name = Path(harness_obj["yaml"]).stem
                    new_config_path = SANDBOX_DIR / f"{base_name}_broad_v{iteration + 1}.yaml"
                    with open(new_config_path, "w") as f_out:
                        yaml.dump({"dag": winner_config}, f_out, default_flow_style=False, sort_keys=False)
                    current_configs[weakest_name] = new_config_path
                    print(f"  PROMOTING ({promotion_reason}): {winner_strategy} -> {new_config_path.name}")

                    # Update best if this is actually better
                    if winner_holdout_score > best_holdout_for_harness:
                        best_harness_holdout[weakest_name] = winner_holdout_score
                        with open(new_config_path) as f:
                            best_config_content[weakest_name] = f.read()
                else:
                    print(f"  REJECTING ({promotion_reason}): holdout {winner_holdout_score:.1f} < "
                          f"best {best_holdout_for_harness:.1f} - {PROMOTION_TOLERANCE}")

                promotion_entry = {
                    "iteration": iteration,
                    "target_harness": weakest_name,
                    "accepted": accepted,
                    "strategy": winner_strategy,
                    "holdout_avg": round(winner_holdout_score, 2),
                    "holdout_delta": round(holdout_delta, 2),
                    "reason": promotion_reason,
                    "train_delta": round(train_delta, 2),
                }

                # Record insights
                for strategy, train_score, _ in variant_train_scores:
                    is_winner = strategy == winner_strategy
                    s_train_delta = train_score - train_results[weakest_name].aggregate_score
                    insight = Insight(
                        iteration=iteration,
                        harness=weakest_name,
                        strategy=strategy,
                        description=f"{strategy} on {weakest_name}",
                        train_delta=s_train_delta,
                        holdout_delta=holdout_delta if is_winner else 0.0,
                        accepted=accepted and is_winner,
                    )
                    insight_memory.append(insight)

                iter_data["winner_strategy"] = winner_strategy if accepted else f"{winner_strategy}(rej)"
                iter_data["hypotheses_tested"] = hypotheses_tested

        promotion_log.append(promotion_entry)

        # Build summary row
        summary_row = {
            "iteration": iteration,
            "train_agg": round(train_agg, 1),
            "holdout_agg": round(holdout_agg, 1),
            "hypotheses_tested": hypotheses_tested,
            "winner_strategy": iter_data.get("winner_strategy", "-"),
        }
        summary_rows.append(summary_row)

        iter_data["train_aggregate"] = round(train_agg, 2)
        iter_data["holdout_aggregate"] = round(holdout_agg, 2)
        all_iteration_data.append(iter_data)

        # Save after each iteration (crash-safe)
        save_scores(all_iteration_data, insight_memory, promotion_log)

    # =================================================================
    # Final Summary
    # =================================================================
    print(f"\n{'='*70}")
    print(f"=== 6-HARNESS BENCHMARK: {iterations}-ITERATION SUMMARY ===")
    print(f"{'='*70}\n")

    # Main results table
    hnames = [h["name"] for h in HARNESSES]
    print(f"{'Harness':20s} | {'Cases':>5s} | {'Baseline':>8s} | {'Best':>6s} | {'Delta':>6s} | Promoted?")
    print("-" * 80)

    total_promotions = 0
    for hname in hnames:
        baseline = baseline_scores.get(hname, 0.0)
        best = best_harness_holdout.get(hname, 0.0)
        delta = best - baseline
        # Check if promoted
        promoted_iters = [p for p in promotion_log if p.get("target_harness") == hname and p.get("accepted")]
        if promoted_iters:
            promo_str = f"YES (iter {promoted_iters[0]['iteration']})"
            total_promotions += len(promoted_iters)
        else:
            promo_str = "NO"

        harness_obj = next(h for h in HARNESSES if h["name"] == hname)
        n_cases = len(harness_obj["cases"])
        print(f"{hname:20s} | {n_cases:5d} | {baseline:8.1f} | {best:6.1f} | {delta:+6.1f} | {promo_str}")

    baseline_agg = sum(baseline_scores.values()) / len(baseline_scores) if baseline_scores else 0.0
    best_agg = sum(best_harness_holdout.values()) / len(best_harness_holdout) if best_harness_holdout else 0.0
    agg_delta = best_agg - baseline_agg
    total_cases = sum(len(h["cases"]) for h in HARNESSES)

    print(f"\n{'Aggregate':20s}: {baseline_agg:.1f} -> {best_agg:.1f} ({agg_delta:+.1f})")
    print(f"{'Total promotions':20s}: {total_promotions}")
    print(f"{'Total cases':20s}: {total_cases}")

    # Per-iteration table
    print(f"\nPer-iteration aggregates:")
    print(f"{'Iter':>4s} | {'Train':>7s} | {'Holdout':>8s} | {'Hypotheses':>10s} | {'Winner':>15s}")
    print("-" * 55)
    for row in summary_rows:
        winner = row['winner_strategy'] or "-"
        print(f"  {row['iteration']:2d} | {row['train_agg']:7.1f} | {row['holdout_agg']:8.1f} | "
              f"{row['hypotheses_tested']:10d} | {winner:>15s}")

    # Strategy analysis
    print(f"\nStrategy performance:")
    strategy_data: dict[str, dict] = {s: {"wins": 0, "accepts": 0} for s in STRATEGIES}
    for p in promotion_log:
        s = p.get("strategy")
        if s and s in strategy_data:
            strategy_data[s]["wins"] += 1
            if p.get("accepted"):
                strategy_data[s]["accepts"] += 1
    for s in STRATEGIES:
        d = strategy_data[s]
        print(f"  {s:12s}: {d['wins']} times winner, {d['accepts']} promoted")

    # Promotion threshold analysis
    print(f"\nPromotion threshold analysis:")
    print(f"  Tolerance: {PROMOTION_TOLERANCE} pts (promote if holdout >= best - {PROMOTION_TOLERANCE})")
    print(f"  Holdout fraction: {HOLDOUT_FRACTION*100:.0f}%")
    print(f"  Holdout eval passes: {HOLDOUT_EVAL_PASSES}")

    strict_accepts = sum(1 for p in promotion_log if p.get("reason") == "NEW_BEST")
    lateral_accepts = sum(1 for p in promotion_log if p.get("reason") == "LATERAL_MOVE")
    rejections = sum(1 for p in promotion_log if p.get("reason") == "REGRESSION")
    print(f"  New best (strict):    {strict_accepts}")
    print(f"  Lateral moves:        {lateral_accepts}")
    print(f"  Rejections:           {rejections}")

    # Domain pattern analysis
    print(f"\nDomain patterns (original 3 vs new 3):")
    original = ["customer_support", "it_helpdesk", "sales_inquiry"]
    new_domains = ["healthcare_triage", "ecommerce_support", "legal_intake"]
    orig_baselines = [baseline_scores.get(h, 0) for h in original]
    new_baselines = [baseline_scores.get(h, 0) for h in new_domains]
    orig_bests = [best_harness_holdout.get(h, 0) for h in original]
    new_bests = [best_harness_holdout.get(h, 0) for h in new_domains]

    orig_avg_base = sum(orig_baselines) / len(orig_baselines) if orig_baselines else 0
    new_avg_base = sum(new_baselines) / len(new_baselines) if new_baselines else 0
    orig_avg_best = sum(orig_bests) / len(orig_bests) if orig_bests else 0
    new_avg_best = sum(new_bests) / len(new_bests) if new_bests else 0

    print(f"  Original 3 avg: baseline={orig_avg_base:.1f}, best={orig_avg_best:.1f}, delta={orig_avg_best-orig_avg_base:+.1f}")
    print(f"  New 3 avg:      baseline={new_avg_base:.1f}, best={new_avg_best:.1f}, delta={new_avg_best-new_avg_base:+.1f}")

    # Insight memory dump
    print(f"\nInsight Memory ({len(insight_memory)} entries):")
    for insight in insight_memory:
        print(f"  {insight}")

    # Train vs holdout tracking
    if len(summary_rows) >= 2:
        train_scores = [r["train_agg"] for r in summary_rows]
        holdout_scores = [r["holdout_agg"] for r in summary_rows]
        train_range = max(train_scores) - min(train_scores)
        holdout_range = max(holdout_scores) - min(holdout_scores)
        correlation_check = sum(
            1 for i in range(1, len(summary_rows))
            if (train_scores[i] - train_scores[i-1]) * (holdout_scores[i] - holdout_scores[i-1]) >= 0
        )
        tracking_pct = correlation_check / (len(summary_rows) - 1) * 100
        print(f"\nTrain vs Held-Out Tracking:")
        print(f"  Train range:   {min(train_scores):.1f} - {max(train_scores):.1f} ({train_range:.1f} pts)")
        print(f"  Holdout range: {min(holdout_scores):.1f} - {max(holdout_scores):.1f} ({holdout_range:.1f} pts)")
        print(f"  Direction agreement: {tracking_pct:.0f}%")


# =============================================================================
# 9. Reset & CLI
# =============================================================================

def reset_all():
    """Remove all broad-harness versioned configs and scores."""
    patterns = [
        "customer_support_adk_broad_v*.yaml",
        "it_helpdesk_broad_v*.yaml",
        "sales_inquiry_broad_v*.yaml",
        "healthcare_triage_broad_v*.yaml",
        "ecommerce_support_broad_v*.yaml",
        "legal_intake_broad_v*.yaml",
    ]
    removed = []
    for pattern in patterns:
        for f in SANDBOX_DIR.glob(pattern):
            f.unlink()
            removed.append(f.name)

    if SCORES_FILE.exists():
        SCORES_FILE.unlink()
        removed.append(SCORES_FILE.name)

    if removed:
        print(f"Reset: removed {', '.join(removed)}")
    else:
        print("Reset: no broad-harness configs or scores found.")
    print("Base DAGs unchanged:")
    for h in HARNESSES:
        print(f"  {h['yaml']}")


async def main():
    parser = argparse.ArgumentParser(
        description="6-Harness Benchmark with Tuned Promotion Threshold"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations (default: 10)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove all broad-harness versioned configs and scores, then exit",
    )
    args = parser.parse_args()

    if args.reset:
        reset_all()
        return

    total_train = sum(len(h["train_cases"]) for h in HARNESSES)
    total_holdout = sum(len(h["holdout_cases"]) for h in HARNESSES)
    total_all = sum(len(h["cases"]) for h in HARNESSES)

    print("=" * 70)
    print("6-Harness Benchmark with Tuned Promotion Threshold")
    print("=" * 70)
    print(f"Harnesses: {len(HARNESSES)}")
    for h in HARNESSES:
        print(f"  {h['name']:20s}: {len(h['train_cases'])} train + {len(h['holdout_cases'])} holdout = {len(h['cases'])} total ({h['yaml']})")
    print(f"Total per iteration: {total_train} train + {total_holdout} holdout = {total_all}")
    print(f"Iterations: {args.iterations}")
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Strategies: {', '.join(STRATEGIES)}")
    print(f"Promotion tolerance: {PROMOTION_TOLERANCE} pts (lateral moves allowed)")
    print(f"Holdout fraction: {HOLDOUT_FRACTION*100:.0f}%")
    print(f"Holdout eval passes: {HOLDOUT_EVAL_PASSES}")
    print(f"Stability threshold: {STABILITY_THRESHOLD} pts")
    print(f"Rollback threshold: {ROLLBACK_THRESHOLD} pts")
    print(f"LLM delay: {LLM_DELAY}s")
    print(f"Split seed: 42")

    # Estimate LLM calls
    avg_train = total_train // len(HARNESSES)
    avg_holdout = total_holdout // len(HARNESSES)
    base_eval = args.iterations * (total_train + total_holdout) * 3
    hyp_gen = (args.iterations - 1) * 3 * 2
    hyp_eval = (args.iterations - 1) * (3 * avg_train * 3 + avg_holdout * 3 * HOLDOUT_EVAL_PASSES)
    est_total = base_eval + hyp_gen + hyp_eval
    print(f"Estimated LLM calls: ~{est_total}")

    await run_broad_harness_loop(args.iterations)


if __name__ == "__main__":
    asyncio.run(main())
