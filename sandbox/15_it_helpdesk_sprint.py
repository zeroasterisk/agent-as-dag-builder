"""Sprint D: IT Helpdesk Ceiling Breaker.

Isolated experiment to push IT helpdesk past the 76-78/100 ceiling.

Approach:
  1. Diagnose: Run all 18 cases, examine low-scorers in detail
  2. Test 4 alternative DAG structures (A/B/C/D)
  3. Compare all approaches
  4. Combine best-per-category into a final DAG

Usage:
    python sandbox/15_it_helpdesk_sprint.py

Environment:
    GOOGLE_GENAI_USE_VERTEXAI=1
    GOOGLE_CLOUD_PROJECT=alanblount-demo
    GOOGLE_CLOUD_LOCATION=global
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
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

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("it_helpdesk_sprint")

SANDBOX_DIR = Path(__file__).parent
JUDGE_MODEL = "gemini-3.5-flash"
SCORES_FILE = SANDBOX_DIR / "scores_it_helpdesk_sprint.json"
LLM_DELAY = 0.3


# =============================================================================
# 1. Test Cases (same as 11_multi_harness.py)
# =============================================================================

IT_HELPDESK_CASES = [
    # --- Password Reset (4 cases) ---
    {"id": 1, "query": "I forgot my password and can't log in", "expected_category": "password-reset",
     "quality_criteria": "Should guide through password reset process, mention self-service portal"},
    {"id": 2, "query": "My account got locked after too many login attempts", "expected_category": "password-reset",
     "quality_criteria": "Should explain account unlock process and how to prevent future lockouts"},
    {"id": 3, "query": "I need to change my password, it's been 90 days", "expected_category": "password-reset",
     "quality_criteria": "Should provide steps for password change and mention password policy requirements"},
    {"id": 4, "query": "My SSO login isn't working with the company portal", "expected_category": "password-reset",
     "quality_criteria": "Should troubleshoot SSO issues, suggest clearing cookies or checking IdP status"},
    # --- Software Install (5 cases) ---
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
    # --- Hardware (4 cases) ---
    {"id": 10, "query": "My laptop screen is flickering constantly", "expected_category": "hardware",
     "quality_criteria": "Should suggest display driver update, external monitor test, and hardware repair if needed"},
    {"id": 11, "query": "The printer on the 3rd floor isn't working", "expected_category": "hardware",
     "quality_criteria": "Should suggest basic troubleshooting (power cycle, paper jam) and offer to dispatch support"},
    {"id": 12, "query": "My laptop won't turn on at all", "expected_category": "hardware",
     "quality_criteria": "Should suggest checking power adapter, battery reset, and offer replacement if needed"},
    {"id": 13, "query": "My keyboard is typing the wrong characters", "expected_category": "hardware",
     "quality_criteria": "Should suggest checking language/layout settings, trying external keyboard, driver update"},
    # --- Network (5 cases) ---
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


# =============================================================================
# 2. DAG Configs for Each Approach
# =============================================================================

# --- CONTROL: Original (from it_helpdesk.yaml) ---
CONTROL_CONFIG = {
    "name": "it_helpdesk_control",
    "version": "1.0.0",
    "default_model": "gemini-3.5-flash",
    "nodes": [
        {
            "id": "classify",
            "type": "agent",
            "model": "gemini-3.5-flash",
            "instruction": (
                "Classify the IT helpdesk request into one of these categories.\n"
                "Reply with ONLY the category name, nothing else:\n"
                "- password-reset\n"
                "- software-install\n"
                "- hardware\n"
                "- network\n"
            ),
        },
        {
            "id": "handle_password_reset",
            "type": "agent",
            "model": "gemini-3.5-flash",
            "instruction": (
                "You are an IT helpdesk specialist for password and account access issues.\n"
                "Help the user reset their password, unlock their account, or resolve\n"
                "authentication problems. Provide clear step-by-step instructions.\n"
                "Mention self-service portal options when applicable.\n"
                "Be concise and helpful.\n"
            ),
        },
        {
            "id": "handle_software_install",
            "type": "agent",
            "model": "gemini-3.5-flash",
            "instruction": (
                "You are an IT helpdesk specialist for software installation and licensing.\n"
                "Help the user install, update, or troubleshoot software applications.\n"
                "Mention the company software catalog or approval process when relevant.\n"
                "Provide clear steps and mention any prerequisites.\n"
                "Be concise and helpful.\n"
            ),
        },
        {
            "id": "handle_hardware",
            "type": "agent",
            "model": "gemini-3.5-flash",
            "instruction": (
                "You are an IT helpdesk specialist for hardware issues.\n"
                "Help the user diagnose and resolve hardware problems with laptops,\n"
                "desktops, monitors, printers, and peripherals. Suggest basic\n"
                "troubleshooting first, then escalation to on-site support if needed.\n"
                "Be concise and helpful.\n"
            ),
        },
        {
            "id": "handle_network",
            "type": "agent",
            "model": "gemini-3.5-flash",
            "instruction": (
                "You are an IT helpdesk specialist for network and connectivity issues.\n"
                "Help the user resolve VPN, Wi-Fi, ethernet, DNS, and general\n"
                "network connectivity problems. Provide step-by-step troubleshooting.\n"
                "Mention common fixes like flushing DNS, resetting network adapter,\n"
                "or reconnecting to VPN. Be concise and helpful.\n"
            ),
        },
    ],
    "edges": [
        {"from": "START", "to": "classify"},
        {"from": "classify", "to": "handle_password_reset", "condition": "password-reset"},
        {"from": "classify", "to": "handle_software_install", "condition": "software-install"},
        {"from": "classify", "to": "handle_hardware", "condition": "hardware"},
        {"from": "classify", "to": "handle_network", "condition": "network"},
    ],
}

# --- APPROACH A: Chain-of-thought handler ---
APPROACH_A_CONFIG = copy.deepcopy(CONTROL_CONFIG)
APPROACH_A_CONFIG["name"] = "it_helpdesk_A_cot"
APPROACH_A_CONFIG["version"] = "A.1.0"

COT_PREFIX = (
    "Think step by step before responding. Structure your response as follows:\n"
    "1. UNDERSTAND: Restate the user's problem in one sentence\n"
    "2. DIAGNOSE: Identify the most likely cause(s)\n"
    "3. SOLUTION: Provide specific, numbered troubleshooting steps\n"
    "4. PREVENTION: Offer one tip to prevent this issue in the future\n"
    "5. ESCALATION: If steps don't resolve it, explain what to do next\n\n"
)

for node in APPROACH_A_CONFIG["nodes"]:
    if node["id"] != "classify":
        node["instruction"] = COT_PREFIX + node["instruction"]

# --- APPROACH B: Two-stage handler (triage + detailed) ---
APPROACH_B_CONFIG = {
    "name": "it_helpdesk_B_twostage",
    "version": "B.1.0",
    "default_model": "gemini-3.5-flash",
    "nodes": [
        {
            "id": "classify",
            "type": "agent",
            "model": "gemini-3.5-flash",
            "instruction": (
                "Classify the IT helpdesk request into one of these categories.\n"
                "Reply with ONLY the category name, nothing else:\n"
                "- password-reset\n"
                "- software-install\n"
                "- hardware\n"
                "- network\n"
            ),
        },
        # --- Password Reset triage + detail ---
        {
            "id": "triage_password_reset",
            "type": "agent",
            "model": "gemini-3.5-flash",
            "instruction": (
                "You are an IT helpdesk triage specialist for password/account issues.\n"
                "Quickly assess the user's request and provide:\n"
                "1. Issue type: (forgotten password / locked account / expired password / SSO issue / other)\n"
                "2. Severity: (low / medium / high)\n"
                "3. Is this a known common issue? (yes/no)\n"
                "4. Immediate action needed: (one sentence)\n\n"
                "Be brief and structured. This triage will be used to provide a detailed response.\n"
            ),
        },
        {
            "id": "handle_password_reset",
            "type": "agent",
            "model": "gemini-3.5-flash",
            "instruction": (
                "You are an IT helpdesk specialist for password and account access issues.\n"
                "Based on the triage assessment, provide a detailed, helpful response.\n"
                "Include specific step-by-step instructions for the user.\n"
                "Mention self-service portal options (e.g., company SSO portal, Active Directory self-service).\n"
                "Mention password policy requirements (length, complexity, expiration).\n"
                "Offer to escalate to IT admin if self-service doesn't work.\n"
            ),
        },
        # --- Software Install triage + detail ---
        {
            "id": "triage_software_install",
            "type": "agent",
            "model": "gemini-3.5-flash",
            "instruction": (
                "You are an IT helpdesk triage specialist for software issues.\n"
                "Quickly assess the user's request and provide:\n"
                "1. Issue type: (new install / update / license / configuration / other)\n"
                "2. Requires admin approval? (yes/no/unknown)\n"
                "3. Is this a standard company-approved application? (yes/no/unknown)\n"
                "4. Immediate action needed: (one sentence)\n\n"
                "Be brief and structured.\n"
            ),
        },
        {
            "id": "handle_software_install",
            "type": "agent",
            "model": "gemini-3.5-flash",
            "instruction": (
                "You are an IT helpdesk specialist for software installation and licensing.\n"
                "Based on the triage assessment, provide a detailed, helpful response.\n"
                "Include steps for the company software catalog or self-service portal.\n"
                "Mention approval workflows, license management, and MDM policies.\n"
                "Provide clear prerequisites and system requirements when relevant.\n"
            ),
        },
        # --- Hardware triage + detail ---
        {
            "id": "triage_hardware",
            "type": "agent",
            "model": "gemini-3.5-flash",
            "instruction": (
                "You are an IT helpdesk triage specialist for hardware issues.\n"
                "Quickly assess the user's request and provide:\n"
                "1. Device type: (laptop / desktop / monitor / printer / peripheral / other)\n"
                "2. Likely cause: (software/driver / physical damage / wear / configuration / unknown)\n"
                "3. Can user self-troubleshoot? (yes/no)\n"
                "4. Immediate action needed: (one sentence)\n\n"
                "Be brief and structured.\n"
            ),
        },
        {
            "id": "handle_hardware",
            "type": "agent",
            "model": "gemini-3.5-flash",
            "instruction": (
                "You are an IT helpdesk specialist for hardware issues.\n"
                "Based on the triage assessment, provide a detailed, helpful response.\n"
                "Include specific diagnostic steps (driver updates, BIOS reset, hardware tests).\n"
                "Mention warranty status checks, RMA process, and when to request replacement.\n"
                "For peripherals, suggest compatibility checks and alternative devices.\n"
                "Offer to dispatch on-site support if self-troubleshooting fails.\n"
            ),
        },
        # --- Network triage + detail ---
        {
            "id": "triage_network",
            "type": "agent",
            "model": "gemini-3.5-flash",
            "instruction": (
                "You are an IT helpdesk triage specialist for network issues.\n"
                "Quickly assess the user's request and provide:\n"
                "1. Issue type: (VPN / Wi-Fi / DNS / proxy / bandwidth / access / other)\n"
                "2. Location: (office / remote / unknown)\n"
                "3. Likely cause: (configuration / infrastructure / ISP / firewall / unknown)\n"
                "4. Immediate action needed: (one sentence)\n\n"
                "Be brief and structured.\n"
            ),
        },
        {
            "id": "handle_network",
            "type": "agent",
            "model": "gemini-3.5-flash",
            "instruction": (
                "You are an IT helpdesk specialist for network and connectivity issues.\n"
                "Based on the triage assessment, provide a detailed, helpful response.\n"
                "Include specific troubleshooting steps for the identified issue type.\n"
                "Mention VPN configs (split tunneling, certificates), DNS settings (flush cache, check servers),\n"
                "DHCP lease renewal, proxy configuration, and common ISP-related issues.\n"
                "Provide OS-specific commands where applicable (Windows, macOS, Linux).\n"
                "Suggest network diagnostic tools (ping, traceroute, nslookup).\n"
            ),
        },
    ],
    "edges": [
        {"from": "START", "to": "classify"},
        {"from": "classify", "to": "triage_password_reset", "condition": "password-reset"},
        {"from": "classify", "to": "triage_software_install", "condition": "software-install"},
        {"from": "classify", "to": "triage_hardware", "condition": "hardware"},
        {"from": "classify", "to": "triage_network", "condition": "network"},
        {"from": "triage_password_reset", "to": "handle_password_reset"},
        {"from": "triage_software_install", "to": "handle_software_install"},
        {"from": "triage_hardware", "to": "handle_hardware"},
        {"from": "triage_network", "to": "handle_network"},
    ],
}

# --- APPROACH C: Knowledge-enriched handler ---
APPROACH_C_CONFIG = copy.deepcopy(CONTROL_CONFIG)
APPROACH_C_CONFIG["name"] = "it_helpdesk_C_knowledge"
APPROACH_C_CONFIG["version"] = "C.1.0"

KNOWLEDGE_INSTRUCTIONS = {
    "handle_password_reset": (
        "You are an expert IT helpdesk specialist for password and account access issues.\n\n"
        "KNOWLEDGE BASE:\n"
        "- Self-service password reset portal: https://passwordreset.company.com\n"
        "- Active Directory password policies: minimum 12 characters, must include uppercase,\n"
        "  lowercase, number, and special character. Passwords expire every 90 days.\n"
        "- Account lockout: triggered after 5 failed attempts, auto-unlocks after 30 minutes\n"
        "  or can be manually unlocked by IT admin\n"
        "- SSO uses Okta/Azure AD federation. Common SSO issues: expired session, browser cache,\n"
        "  IdP certificate renewal, conditional access policies\n"
        "- LDAP sync runs every 15 minutes; password changes may take up to 15 min to propagate\n"
        "- MFA methods: authenticator app (preferred), SMS, hardware token\n\n"
        "RESPONSE GUIDELINES:\n"
        "1. Acknowledge the user's frustration\n"
        "2. Provide specific, numbered steps to resolve the issue\n"
        "3. Mention the self-service portal URL\n"
        "4. Include relevant policy details (lockout duration, password requirements)\n"
        "5. Offer escalation path if self-service fails\n"
        "Be concise but thorough. Use bullet points for steps.\n"
    ),
    "handle_software_install": (
        "You are an expert IT helpdesk specialist for software installation and licensing.\n\n"
        "KNOWLEDGE BASE:\n"
        "- Company Software Center: accessible via Start Menu > Company Software Center\n"
        "  or https://software.company.com\n"
        "- Software approval workflow: Standard apps (Office, Slack, Zoom) = auto-approved.\n"
        "  Specialized apps (Adobe CC, JetBrains, dev tools) = manager approval required.\n"
        "- License types: per-user (Office 365), per-device (AutoCAD), floating (MATLAB)\n"
        "- MDM platform: Microsoft Intune for Windows, Jamf for macOS\n"
        "- Developer tools (Python, VS Code, Git): available in Developer Self-Service portal\n"
        "  at https://devtools.company.com, no approval needed for standard dev stack\n"
        "- Update policies: security updates = forced within 48h, app updates = user-initiated\n"
        "  unless admin-restricted\n"
        "- Common issues: UAC prompts (need local admin), proxy blocking downloads,\n"
        "  conflicting software versions, insufficient disk space\n\n"
        "RESPONSE GUIDELINES:\n"
        "1. Identify whether this is an install, update, or licensing issue\n"
        "2. Point to the correct self-service portal or process\n"
        "3. Mention approval requirements if applicable\n"
        "4. Provide specific troubleshooting steps for the issue\n"
        "5. Include workarounds if the standard process is blocked\n"
        "Be concise but thorough.\n"
    ),
    "handle_hardware": (
        "You are an expert IT helpdesk specialist for hardware issues.\n\n"
        "KNOWLEDGE BASE:\n"
        "- Standard hardware: Dell Latitude laptops, Dell OptiPlex desktops, Dell UltraSharp monitors\n"
        "- Warranty: 3-year on-site warranty for laptops/desktops, 1-year for peripherals\n"
        "- RMA process: submit ticket at https://hardware.company.com/rma with asset tag\n"
        "- Common laptop issues:\n"
        "  * Screen flickering: usually display driver or cable issue. Try: update Intel/NVIDIA driver,\n"
        "    test with external monitor, check display cable seating\n"
        "  * Won't power on: try hard reset (hold power 15s), remove from dock, try different charger,\n"
        "    check LED indicators. If no LEDs = likely motherboard or battery failure\n"
        "  * Keyboard issues: check language/layout settings (Win+Space), try external USB keyboard,\n"
        "    update keyboard driver, check for physical debris\n"
        "- Printers: HP LaserJet fleet, managed by PaperCut. Common fixes: power cycle,\n"
        "  clear print queue, re-add printer via \\\\printserver\\printername\n"
        "- Peripheral compatibility: check approved hardware list at https://hardware.company.com/approved\n"
        "- On-site support: available M-F 8am-6pm, request via ServiceNow or email hardware@company.com\n\n"
        "RESPONSE GUIDELINES:\n"
        "1. Identify the specific hardware and likely cause\n"
        "2. Provide targeted diagnostic steps (not generic)\n"
        "3. Mention warranty/RMA if hardware replacement may be needed\n"
        "4. Offer on-site support as an escalation option\n"
        "5. Include preventive tips\n"
        "Be concise but thorough.\n"
    ),
    "handle_network": (
        "You are an expert IT helpdesk specialist for network and connectivity issues.\n\n"
        "KNOWLEDGE BASE:\n"
        "- Corporate VPN: Cisco AnyConnect, server: vpn.company.com\n"
        "  * Split tunneling enabled (only company traffic goes through VPN)\n"
        "  * Requires valid AD credentials + MFA\n"
        "  * Common issues: outdated client, certificate expired, port 443/UDP 4500 blocked by ISP,\n"
        "    conflicting VPN software, DNS not resolving internal domains\n"
        "  * Fix: reinstall client from https://vpn.company.com, check firewall ports,\n"
        "    verify credentials, clear AnyConnect profile cache\n"
        "- Office Wi-Fi: WPA2-Enterprise (802.1X), SSID: CorpWiFi (5GHz preferred), GuestWiFi\n"
        "  * Auth via machine certificate or AD credentials\n"
        "  * Issues: forget network and reconnect, verify certificate, switch 2.4GHz/5GHz,\n"
        "    check if too far from AP (use Wi-Fi analyzer app)\n"
        "- DNS: internal DNS servers 10.0.0.53, 10.0.0.54. External fallback: 8.8.8.8\n"
        "  * Flush cache: Windows: ipconfig /flushdns, macOS: sudo dscacheutil -flushcache\n"
        "  * Test: nslookup wiki.company.com 10.0.0.53\n"
        "- Proxy: proxy.company.com:8080 for web traffic, auto-configured via PAC file\n"
        "  * Some apps need manual proxy config\n"
        "- Bandwidth/QoS: video calls require minimum 2Mbps up/down\n"
        "  * Prefer wired connection for video calls\n"
        "  * Close bandwidth-heavy apps (large downloads, streaming)\n"
        "  * Check: speedtest at https://speedtest.company.com\n"
        "- Network diagnostic commands:\n"
        "  * ping gateway: ping 10.0.0.1\n"
        "  * trace route: tracert wiki.company.com (Windows) / traceroute (macOS)\n"
        "  * DNS lookup: nslookup company.com\n"
        "  * IP config: ipconfig /all (Windows) / ifconfig (macOS)\n"
        "  * Release/renew DHCP: ipconfig /release && ipconfig /renew\n\n"
        "RESPONSE GUIDELINES:\n"
        "1. Identify the specific network issue type (VPN/Wi-Fi/DNS/proxy/bandwidth)\n"
        "2. Provide OS-specific commands for troubleshooting\n"
        "3. Mention specific server addresses and configuration details\n"
        "4. Include diagnostic tool suggestions\n"
        "5. Escalation: if issue persists, contact network team at network@company.com\n"
        "   or submit ticket with traceroute/ping results attached\n"
        "Be concise but thorough.\n"
    ),
}

for node in APPROACH_C_CONFIG["nodes"]:
    if node["id"] in KNOWLEDGE_INSTRUCTIONS:
        node["instruction"] = KNOWLEDGE_INSTRUCTIONS[node["id"]]

# --- APPROACH D: Category-specific tuning (detailed instructions for weak categories) ---
APPROACH_D_CONFIG = copy.deepcopy(CONTROL_CONFIG)
APPROACH_D_CONFIG["name"] = "it_helpdesk_D_tuned"
APPROACH_D_CONFIG["version"] = "D.1.0"

# Keep password-reset and software-install simple (they score well already)
# Give network and hardware much more detailed instructions

TUNED_INSTRUCTIONS = {
    "handle_hardware": (
        "You are an expert IT helpdesk specialist for hardware issues. You must provide\n"
        "SPECIFIC, ACTIONABLE troubleshooting steps -- never generic advice.\n\n"
        "For SCREEN FLICKERING issues:\n"
        "1. Right-click desktop > Display Settings > check refresh rate (should be 60Hz+)\n"
        "2. Update display driver: Device Manager > Display Adapters > Update Driver\n"
        "3. Test with external monitor to isolate display vs GPU issue\n"
        "4. Check display cable connection (open back panel if comfortable, or request IT)\n"
        "5. If persists: likely hardware -- submit RMA request\n\n"
        "For LAPTOP WON'T TURN ON:\n"
        "1. Check power adapter LED (should be solid white/green)\n"
        "2. Try different power outlet and adapter if available\n"
        "3. Hard reset: unplug, hold power button 15 seconds, plug back in\n"
        "4. Remove from dock/peripherals, try again\n"
        "5. Check for LED indicators -- blinking patterns indicate specific failures\n"
        "6. If no signs of power: likely motherboard or battery -- request replacement\n\n"
        "For PRINTER issues:\n"
        "1. Power cycle the printer (off 30s, back on)\n"
        "2. Check paper tray and clear any paper jams\n"
        "3. On your PC: clear print queue (Services > Print Spooler > Restart)\n"
        "4. Remove and re-add printer: Settings > Printers > Add via \\\\printserver\\name\n"
        "5. If still not working: check printer display for error codes, report to IT\n\n"
        "For KEYBOARD issues:\n"
        "1. Check language/layout: press Win+Space to cycle input methods\n"
        "2. Go to Settings > Time & Language > Language -- ensure correct layout\n"
        "3. Try an external USB keyboard to isolate hardware vs software\n"
        "4. Update keyboard driver: Device Manager > Keyboards > Update\n"
        "5. Check for physical damage or debris under keys\n\n"
        "Always end with an escalation path: 'If these steps don't resolve the issue, "
        "please submit a ticket and we can arrange on-site support.'\n"
    ),
    "handle_network": (
        "You are an expert IT helpdesk specialist for network and connectivity issues.\n"
        "You must provide SPECIFIC, ACTIONABLE troubleshooting steps with actual commands.\n\n"
        "For VPN CONNECTION issues:\n"
        "1. Verify credentials: ensure you're using your current AD password\n"
        "2. Check VPN client version: Help > About -- should be 4.10+ for AnyConnect\n"
        "3. Try disconnecting and reconnecting to VPN\n"
        "4. Clear VPN profile cache: navigate to C:\\ProgramData\\Cisco\\AnyConnect and delete profiles\n"
        "5. Check if ports 443 and UDP 4500 are open (some ISPs/routers block these)\n"
        "6. Temporarily disable local firewall to test: netsh advfirewall set allprofiles state off\n"
        "7. Try connecting via a mobile hotspot to rule out ISP blocking\n"
        "8. Reinstall VPN client from https://vpn.company.com\n\n"
        "For WI-FI DISCONNECTION issues:\n"
        "1. Forget the network: Settings > Wi-Fi > Manage Known Networks > Forget\n"
        "2. Reconnect with your AD credentials\n"
        "3. Check signal strength -- move closer to access point or switch to 5GHz band\n"
        "4. Update wireless driver: Device Manager > Network Adapters > Update\n"
        "5. Reset network stack: netsh winsock reset && netsh int ip reset (run as admin, restart)\n"
        "6. Check if power saving is disabling Wi-Fi: Device Manager > Network Adapter > Properties\n"
        "   > Power Management > uncheck 'Allow computer to turn off this device'\n"
        "7. If office-wide: report to network team -- may be AP issue\n\n"
        "For DNS RESOLUTION issues:\n"
        "1. Flush DNS cache: ipconfig /flushdns (Windows) or sudo dscacheutil -flushcache (macOS)\n"
        "2. Test DNS: nslookup company.com\n"
        "3. Check DNS settings: ipconfig /all -- should show corporate DNS servers\n"
        "4. Try alternate DNS: nslookup company.com 8.8.8.8\n"
        "5. If only internal sites fail: check VPN connection (internal DNS requires VPN)\n"
        "6. Release and renew IP: ipconfig /release && ipconfig /renew\n\n"
        "For VIDEO CALL QUALITY issues:\n"
        "1. Test bandwidth: run speed test at speedtest.net -- need 2Mbps+ up/down\n"
        "2. Switch to wired ethernet connection if possible\n"
        "3. Close bandwidth-heavy applications (large downloads, streaming, cloud sync)\n"
        "4. Reduce video quality in call settings (720p instead of 1080p)\n"
        "5. Check if VPN is routing all traffic: split tunnel should be enabled\n"
        "6. Restart router/modem if on home network\n\n"
        "For INTERNAL SITE ACCESS issues:\n"
        "1. Verify you're on the corporate network (check VPN if remote)\n"
        "2. Try accessing by IP instead of hostname to isolate DNS\n"
        "3. Check proxy settings: Settings > Network > Proxy -- should be auto-configured\n"
        "4. Clear browser cache and cookies for the specific site\n"
        "5. Try a different browser (Edge, Chrome, Firefox)\n"
        "6. Test DNS resolution: nslookup wiki.company.com\n\n"
        "Always include relevant OS-specific commands and an escalation path.\n"
    ),
}

for node in APPROACH_D_CONFIG["nodes"]:
    if node["id"] in TUNED_INSTRUCTIONS:
        node["instruction"] = TUNED_INSTRUCTIONS[node["id"]]


# =============================================================================
# 3. DAG Executor (copied from 11_multi_harness.py, simplified)
# =============================================================================

def build_routing(config: dict) -> dict:
    """Parse edges into routing structure."""
    start_nodes = []
    conditional: dict[str, list] = {}
    unconditional: dict[str, list] = {}

    for edge in config["edges"]:
        src = edge["from"]
        dst = edge["to"]
        cond = edge.get("condition")

        if src == "START":
            start_nodes.append(dst)
        elif cond:
            conditional.setdefault(src, []).append({"to": dst, "condition": cond})
        else:
            unconditional.setdefault(src, []).append(dst)

    return {
        "start_nodes": start_nodes,
        "conditional": conditional,
        "unconditional": unconditional,
    }


def extract_categories_from_config(config: dict) -> list[str]:
    """Extract valid categories from DAG edges."""
    categories = []
    routing = build_routing(config)
    for node_id in routing["start_nodes"]:
        if node_id in routing["conditional"]:
            for route in routing["conditional"][node_id]:
                categories.append(route["condition"].lower())
    return categories


async def run_dag_query(
    client: genai.Client,
    config: dict,
    query: str,
) -> dict:
    """Execute a query through the DAG."""
    nodes_by_id = {n["id"]: n for n in config["nodes"]}
    routing = build_routing(config)
    default_model = config.get("default_model", "gemini-3.5-flash")
    valid_categories = extract_categories_from_config(config)

    results = {
        "category": "unknown",
        "response": "",
        "nodes_visited": [],
    }

    async def call_node(node_id: str, user_input: str) -> str:
        node = nodes_by_id.get(node_id)
        if node is None or node.get("type", "agent") != "agent":
            return ""

        model = node.get("model", default_model)
        instruction = node.get("instruction", "Help the user.")
        prompt = f"{instruction}\n\nUser request: {user_input}"

        try:
            response = client.models.generate_content(
                model=model,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
            )
            text = ""
            for p in response.candidates[0].content.parts:
                if hasattr(p, "text") and p.text:
                    text += p.text
            results["nodes_visited"].append(node_id)
            return text.strip()
        except Exception as e:
            logger.error("Node %s failed: %s", node_id, e)
            results["nodes_visited"].append(node_id)
            return f"ERROR: {e}"

    # Step 1: classify
    classify_output = ""
    for node_id in routing["start_nodes"]:
        classify_output = await call_node(node_id, query)

    # Extract category
    classify_lower = classify_output.lower().strip()
    detected_category = "unknown"
    for cat in valid_categories:
        if cat in classify_lower:
            detected_category = cat
            break
    results["category"] = detected_category

    # Step 2: route
    to_visit = []
    for node_id in routing["start_nodes"]:
        if node_id in routing["conditional"]:
            for route in routing["conditional"][node_id]:
                if route["condition"].lower() in classify_lower:
                    to_visit.append(route["to"])
                    break

    # Step 3: run handler nodes (follow unconditional edges too)
    final_output = classify_output
    visited = set(routing["start_nodes"])
    for node_id in to_visit:
        if node_id in visited:
            continue
        visited.add(node_id)
        await asyncio.sleep(LLM_DELAY)
        handler_output = await call_node(node_id, query)
        if handler_output:
            final_output = handler_output

        # Follow unconditional edges from this node
        if node_id in routing.get("unconditional", {}):
            to_visit.extend(routing["unconditional"][node_id])

    results["response"] = final_output
    return results


# =============================================================================
# 4. LLM-as-Judge Scoring (same as 11_multi_harness.py)
# =============================================================================

async def _single_judge_call(client: genai.Client, judge_prompt: str) -> dict:
    """Make a single LLM judge call and return parsed scores dict."""
    resp = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=[{"role": "user", "parts": [{"text": judge_prompt}]}],
        config={"temperature": 0.0},
    )
    judge_text = resp.candidates[0].content.parts[0].text.strip()

    if "```" in judge_text:
        match = re.search(r"```(?:json)?\s*(.*?)```", judge_text, re.DOTALL)
        if match:
            judge_text = match.group(1).strip()

    json_match = re.search(r'\{[^{}]*\}', judge_text)
    if json_match:
        judge_text = json_match.group(0)

    return json.loads(judge_text)


async def score_response(
    client: genai.Client,
    query: str,
    expected_category: str,
    actual_category: str,
    response: str,
    quality_criteria: str,
) -> dict:
    """Score a response using LLM-as-judge with 2x averaging."""
    category_score = 50 if actual_category == expected_category else 0

    judge_prompt = f"""You are a strict scoring judge for an IT helpdesk system.
Evaluate the response below and output EXACTLY one JSON object.

INPUT:
- User query: "{query}"
- Expected category: {expected_category}
- Actual category: {actual_category}
- Quality criteria: {quality_criteria}

RESPONSE:
{response[:1500]}

SCORING RUBRIC:
quality_score (integer 0-30):
  0-7   = completely misses criteria, generic or wrong advice
  8-15  = partially addresses criteria, missing key details
  16-22 = mostly meets criteria, has relevant specifics
  23-30 = fully meets criteria with specific, actionable details

helpfulness_score (integer 0-20):
  0-5   = unhelpful, confusing, or missing steps
  6-10  = somewhat helpful but vague
  11-15 = helpful, clear, with actionable steps
  16-20 = excellent -- specific, concise, clear, actionable, with escalation path

OUTPUT FORMAT (no markdown, no explanation, ONLY this JSON):
{{"quality_score": <int>, "helpfulness_score": <int>, "reasoning": "<one sentence>"}}"""

    quality_scores = []
    helpfulness_scores = []
    reasonings = []

    for attempt in range(2):
        try:
            await asyncio.sleep(LLM_DELAY)
            scores = await _single_judge_call(client, judge_prompt)
            q = max(0, min(30, int(scores.get("quality_score", 15))))
            h = max(0, min(20, int(scores.get("helpfulness_score", 10))))
            quality_scores.append(q)
            helpfulness_scores.append(h)
            reasonings.append(scores.get("reasoning", ""))
        except Exception as e:
            logger.warning("Judge attempt %d failed: %s", attempt + 1, e)
            quality_scores.append(15)
            helpfulness_scores.append(10)
            reasonings.append(f"Scoring error: {e}")

    quality_score = round(sum(quality_scores) / len(quality_scores))
    helpfulness_score = round(sum(helpfulness_scores) / len(helpfulness_scores))
    reasoning = reasonings[0] if reasonings else ""

    total = category_score + quality_score + helpfulness_score
    return {
        "total_score": total,
        "category_score": category_score,
        "quality_score": quality_score,
        "helpfulness_score": helpfulness_score,
        "reasoning": reasoning,
    }


# =============================================================================
# 5. Data Structures
# =============================================================================

@dataclass
class CaseResult:
    case_id: int
    query: str
    expected_category: str
    actual_category: str
    response: str
    score: dict
    quality_criteria: str
    nodes_visited: list[str] = field(default_factory=list)


@dataclass
class ApproachResult:
    approach_name: str
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def avg_score(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.score["total_score"] for c in self.cases) / len(self.cases)

    @property
    def avg_quality(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.score["quality_score"] for c in self.cases) / len(self.cases)

    @property
    def avg_helpfulness(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.score["helpfulness_score"] for c in self.cases) / len(self.cases)

    def category_breakdown(self) -> dict[str, dict]:
        breakdown: dict[str, dict] = {}
        for c in self.cases:
            cat = c.expected_category
            if cat not in breakdown:
                breakdown[cat] = {"scores": [], "quality": [], "helpfulness": [], "correct": 0, "total": 0}
            breakdown[cat]["scores"].append(c.score["total_score"])
            breakdown[cat]["quality"].append(c.score["quality_score"])
            breakdown[cat]["helpfulness"].append(c.score["helpfulness_score"])
            breakdown[cat]["total"] += 1
            if c.actual_category == c.expected_category:
                breakdown[cat]["correct"] += 1
        for cat in breakdown:
            b = breakdown[cat]
            b["avg_score"] = sum(b["scores"]) / len(b["scores"])
            b["avg_quality"] = sum(b["quality"]) / len(b["quality"])
            b["avg_helpfulness"] = sum(b["helpfulness"]) / len(b["helpfulness"])
            b["accuracy"] = b["correct"] / b["total"]
        return breakdown

    def low_scorers(self, threshold: int = 75) -> list[CaseResult]:
        return [c for c in self.cases if c.score["total_score"] < threshold]


# =============================================================================
# 6. Run a Single Approach
# =============================================================================

async def run_approach(
    client: genai.Client,
    config: dict,
    approach_name: str,
    cases: list[dict],
) -> ApproachResult:
    """Run all test cases through one DAG configuration and score them."""
    result = ApproachResult(approach_name=approach_name)

    for case in cases:
        # Run query through DAG
        dag_result = await run_dag_query(client, config, case["query"])
        await asyncio.sleep(LLM_DELAY)

        # Score
        score = await score_response(
            client,
            query=case["query"],
            expected_category=case["expected_category"],
            actual_category=dag_result["category"],
            response=dag_result["response"],
            quality_criteria=case["quality_criteria"],
        )

        cr = CaseResult(
            case_id=case["id"],
            query=case["query"],
            expected_category=case["expected_category"],
            actual_category=dag_result["category"],
            response=dag_result["response"],
            score=score,
            quality_criteria=case["quality_criteria"],
            nodes_visited=dag_result.get("nodes_visited", []),
        )
        result.cases.append(cr)

        cat_ok = "OK" if dag_result["category"] == case["expected_category"] else f"WRONG:{dag_result['category']}"
        print(f"    [{approach_name}] Case {case['id']:2d}: score={score['total_score']:3d} "
              f"(q={score['quality_score']:2d} h={score['helpfulness_score']:2d}) "
              f"cat={cat_ok} \"{case['query'][:40]}\"")

        await asyncio.sleep(LLM_DELAY)

    return result


# =============================================================================
# 7. Combined Best-of-Each Approach
# =============================================================================

def build_combined_config(
    approach_results: dict[str, ApproachResult],
) -> tuple[dict, dict[str, str]]:
    """Build a combined DAG using the best approach per category.

    Returns (config, mapping) where mapping is {category: approach_name}.
    """
    categories = ["password-reset", "software-install", "hardware", "network"]
    best_per_category: dict[str, str] = {}

    for cat in categories:
        best_score = -1.0
        best_approach = "Control"
        for approach_name, result in approach_results.items():
            bd = result.category_breakdown()
            if cat in bd:
                if bd[cat]["avg_score"] > best_score:
                    best_score = bd[cat]["avg_score"]
                    best_approach = approach_name
        best_per_category[cat] = best_approach

    print(f"\n  Best approach per category:")
    for cat, approach in best_per_category.items():
        score = approach_results[approach].category_breakdown()[cat]["avg_score"]
        print(f"    {cat:20s} -> {approach:25s} (avg={score:.1f})")

    # Build combined config -- use the handler instruction from the winning approach
    # Map approach names to their configs
    approach_configs = {
        "Control": CONTROL_CONFIG,
        "A (chain-of-thought)": APPROACH_A_CONFIG,
        "B (two-stage)": APPROACH_B_CONFIG,
        "C (knowledge-enriched)": APPROACH_C_CONFIG,
        "D (category-tuned)": APPROACH_D_CONFIG,
    }

    # Category -> handler node IDs
    cat_to_handler = {
        "password-reset": "handle_password_reset",
        "software-install": "handle_software_install",
        "hardware": "handle_hardware",
        "network": "handle_network",
    }

    # Start with a copy of the control config
    combined = copy.deepcopy(CONTROL_CONFIG)
    combined["name"] = "it_helpdesk_combined_best"
    combined["version"] = "COMBINED.1.0"

    for cat, approach_name in best_per_category.items():
        handler_id = cat_to_handler[cat]
        source_config = approach_configs[approach_name]
        source_nodes = {n["id"]: n for n in source_config["nodes"]}

        if handler_id in source_nodes:
            # Replace handler instruction
            for node in combined["nodes"]:
                if node["id"] == handler_id:
                    node["instruction"] = source_nodes[handler_id]["instruction"]
                    break

        # For approach B (two-stage), we also need to add the triage node and edge
        if approach_name == "B (two-stage)":
            triage_id = f"triage_{handler_id.replace('handle_', '')}"
            if triage_id in source_nodes:
                # Add triage node
                combined["nodes"].append(copy.deepcopy(source_nodes[triage_id]))
                # Replace the direct classify->handler edge with classify->triage->handler
                new_edges = []
                for edge in combined["edges"]:
                    if edge.get("to") == handler_id and edge.get("condition"):
                        # classify -> triage (conditional)
                        new_edges.append({"from": edge["from"], "to": triage_id, "condition": edge["condition"]})
                        # triage -> handler (unconditional)
                        new_edges.append({"from": triage_id, "to": handler_id})
                    else:
                        new_edges.append(edge)
                combined["edges"] = new_edges

    return combined, best_per_category


# =============================================================================
# 8. Main Runner
# =============================================================================

async def main():
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "alanblount-demo"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
    )

    all_results: dict[str, ApproachResult] = {}

    approaches = [
        ("Control", CONTROL_CONFIG),
        ("A (chain-of-thought)", APPROACH_A_CONFIG),
        ("B (two-stage)", APPROACH_B_CONFIG),
        ("C (knowledge-enriched)", APPROACH_C_CONFIG),
        ("D (category-tuned)", APPROACH_D_CONFIG),
    ]

    print("=" * 70)
    print("IT HELPDESK SPRINT: CEILING BREAKER EXPERIMENT")
    print("=" * 70)
    print(f"Test cases: {len(IT_HELPDESK_CASES)}")
    print(f"Approaches: {len(approaches)}")
    print(f"Model: gemini-3.5-flash")
    print(f"Judge: {JUDGE_MODEL}")
    print(f"LLM delay: {LLM_DELAY}s")
    est_calls = len(IT_HELPDESK_CASES) * len(approaches) * 3  # classify + handle + 2x judge ~= 3 avg
    print(f"Estimated LLM calls: ~{est_calls}")
    print()

    # ---- Run each approach ----
    for approach_name, config in approaches:
        print(f"\n{'='*60}")
        print(f"  APPROACH: {approach_name}")
        print(f"{'='*60}")
        t0 = time.time()

        result = await run_approach(client, config, approach_name, IT_HELPDESK_CASES)
        all_results[approach_name] = result

        elapsed = time.time() - t0
        print(f"\n  {approach_name}: avg={result.avg_score:.1f} "
              f"(q={result.avg_quality:.1f} h={result.avg_helpfulness:.1f}) "
              f"[{elapsed:.0f}s]")

    # ================================================================
    # EXPERIMENT 1: DIAGNOSIS -- Low-scoring cases from Control
    # ================================================================
    print(f"\n\n{'='*70}")
    print("=== EXPERIMENT 1: DIAGNOSIS ===")
    print(f"{'='*70}")

    control_result = all_results["Control"]
    bd = control_result.category_breakdown()

    print(f"\nControl baseline: {control_result.avg_score:.1f}/100")
    print(f"\nCategory breakdown:")
    for cat, info in sorted(bd.items(), key=lambda x: x[1]["avg_score"]):
        print(f"  {cat:20s}: avg={info['avg_score']:.1f} (q={info['avg_quality']:.1f} h={info['avg_helpfulness']:.1f})")

    low_scorers = control_result.low_scorers(75)
    print(f"\nLow-scoring cases (< 75):")
    if not low_scorers:
        # Expand threshold
        low_scorers = sorted(control_result.cases, key=lambda c: c.score["total_score"])[:6]
        print(f"  (none below 75; showing 6 weakest instead)")

    for c in sorted(low_scorers, key=lambda x: x.score["total_score"]):
        print(f"\n  Case {c.case_id}: \"{c.query}\"")
        print(f"    Category: expected={c.expected_category}, got={c.actual_category}")
        print(f"    Score: {c.score['total_score']} (quality={c.score['quality_score']}, helpfulness={c.score['helpfulness_score']})")
        print(f"    Criteria: {c.quality_criteria}")
        print(f"    Judge: {c.score.get('reasoning', 'N/A')}")
        # Truncate response for readability
        resp_preview = c.response[:300].replace('\n', ' ')
        print(f"    Response preview: {resp_preview}...")

    # ================================================================
    # EXPERIMENT 3: A/B/C/D COMPARISON
    # ================================================================
    print(f"\n\n{'='*70}")
    print("=== EXPERIMENT 3: A/B/C/D RESULTS ===")
    print(f"{'='*70}")

    control_avg = all_results["Control"].avg_score

    # Summary table
    print(f"\n{'Approach':<30s} {'Avg':>6s} {'Delta':>7s} {'Quality':>8s} {'Helpful':>8s}")
    print("-" * 65)
    for name in ["Control", "A (chain-of-thought)", "B (two-stage)", "C (knowledge-enriched)", "D (category-tuned)"]:
        r = all_results[name]
        delta = r.avg_score - control_avg
        delta_str = f"{'+' if delta >= 0 else ''}{delta:.1f}"
        marker = " ***" if delta > 3 else " *" if delta > 1 else ""
        print(f"  {name:<28s} {r.avg_score:5.1f} {delta_str:>7s} {r.avg_quality:7.1f} {r.avg_helpfulness:7.1f}{marker}")

    # Category breakdown per approach
    print(f"\nPer-category breakdown:")
    categories = ["password-reset", "software-install", "hardware", "network"]
    header = f"  {'Category':<20s}"
    for name in ["Control", "A (CoT)", "B (2stg)", "C (Know)", "D (Tune)"]:
        header += f" {name:>10s}"
    print(header)
    print("  " + "-" * 72)

    short_names = {
        "Control": "Control",
        "A (chain-of-thought)": "A (CoT)",
        "B (two-stage)": "B (2stg)",
        "C (knowledge-enriched)": "C (Know)",
        "D (category-tuned)": "D (Tune)",
    }
    approach_names = list(all_results.keys())

    for cat in categories:
        row = f"  {cat:<20s}"
        scores_for_cat = []
        for aname in approach_names:
            bd = all_results[aname].category_breakdown()
            if cat in bd:
                score = bd[cat]["avg_score"]
                scores_for_cat.append((aname, score))
                row += f" {score:10.1f}"
            else:
                row += f" {'N/A':>10s}"
        # Mark best
        if scores_for_cat:
            best_approach, best_score = max(scores_for_cat, key=lambda x: x[1])
            row += f"  <- {short_names.get(best_approach, best_approach)}"
        print(row)

    # ================================================================
    # EXPERIMENT 4: COMBINED BEST
    # ================================================================
    print(f"\n\n{'='*70}")
    print("=== EXPERIMENT 4: COMBINED BEST-OF-EACH ===")
    print(f"{'='*70}")

    combined_config, category_mapping = build_combined_config(all_results)

    print(f"\n  Running combined DAG...")
    t0 = time.time()
    combined_result = await run_approach(client, combined_config, "Combined", IT_HELPDESK_CASES)
    elapsed = time.time() - t0

    all_results["Combined"] = combined_result

    print(f"\n  Combined: avg={combined_result.avg_score:.1f} "
          f"(q={combined_result.avg_quality:.1f} h={combined_result.avg_helpfulness:.1f}) "
          f"[{elapsed:.0f}s]")

    combined_bd = combined_result.category_breakdown()
    print(f"\n  Combined category breakdown:")
    for cat in categories:
        if cat in combined_bd:
            info = combined_bd[cat]
            source = category_mapping.get(cat, "?")
            print(f"    {cat:20s}: avg={info['avg_score']:.1f} (from {source})")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n\n{'='*70}")
    print("=== FINAL SUMMARY ===")
    print(f"{'='*70}")

    print(f"\n{'Approach':<30s} {'Avg':>6s} {'Delta':>7s}")
    print("-" * 50)
    for name in ["Control", "A (chain-of-thought)", "B (two-stage)",
                  "C (knowledge-enriched)", "D (category-tuned)", "Combined"]:
        r = all_results[name]
        delta = r.avg_score - control_avg
        delta_str = f"{'+' if delta >= 0 else ''}{delta:.1f}"
        marker = " *** CEILING BROKEN" if r.avg_score > 82 else ""
        if name == "Combined" and r.avg_score > control_avg + 3:
            marker = " *** CEILING BROKEN"
        print(f"  {name:<28s} {r.avg_score:5.1f} {delta_str:>7s}{marker}")

    # Best and worst cases across Combined
    print(f"\nCombined: Top 3 scores:")
    for c in sorted(combined_result.cases, key=lambda x: -x.score["total_score"])[:3]:
        print(f"  Case {c.case_id}: \"{c.query[:40]}\" score={c.score['total_score']}")

    print(f"\nCombined: Bottom 3 scores:")
    for c in sorted(combined_result.cases, key=lambda x: x.score["total_score"])[:3]:
        print(f"  Case {c.case_id}: \"{c.query[:40]}\" score={c.score['total_score']} -- {c.score.get('reasoning', '')[:60]}")

    # ================================================================
    # Save Results
    # ================================================================
    save_data = {
        "experiment": "it_helpdesk_sprint_d",
        "model": "gemini-3.5-flash",
        "judge_model": JUDGE_MODEL,
        "cases": len(IT_HELPDESK_CASES),
        "approaches": {},
    }

    for name, result in all_results.items():
        approach_data = {
            "avg_score": round(result.avg_score, 2),
            "avg_quality": round(result.avg_quality, 2),
            "avg_helpfulness": round(result.avg_helpfulness, 2),
            "category_breakdown": {},
            "per_case": [],
        }
        for cat, info in result.category_breakdown().items():
            approach_data["category_breakdown"][cat] = {
                "avg_score": round(info["avg_score"], 2),
                "avg_quality": round(info["avg_quality"], 2),
                "avg_helpfulness": round(info["avg_helpfulness"], 2),
                "count": info["total"],
            }
        for c in result.cases:
            approach_data["per_case"].append({
                "case_id": c.case_id,
                "query": c.query,
                "expected_category": c.expected_category,
                "actual_category": c.actual_category,
                "score": c.score["total_score"],
                "quality": c.score["quality_score"],
                "helpfulness": c.score["helpfulness_score"],
                "reasoning": c.score.get("reasoning", ""),
                "response_length": len(c.response),
                "nodes_visited": c.nodes_visited,
            })
        save_data["approaches"][name] = approach_data

    if "Combined" in all_results:
        save_data["combined_mapping"] = category_mapping
        save_data["combined_score"] = round(all_results["Combined"].avg_score, 2)
        save_data["control_score"] = round(control_avg, 2)
        save_data["improvement"] = round(all_results["Combined"].avg_score - control_avg, 2)

    with open(SCORES_FILE, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {SCORES_FILE.name}")


if __name__ == "__main__":
    asyncio.run(main())
