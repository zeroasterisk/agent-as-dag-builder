# Judge Calibration Report

**Date**: 2026-07-02
**Judge Model**: gemini-3.5-flash
**Domain**: customer_support
**Pass Threshold**: 60/100
**Calibration Cases**: 20
**Elapsed**: 196.2s

## Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Agreement Rate | 80.0% | >= 80% | PASSED |
| Cohen's Kappa | 0.600 | > 0.6 (substantial) | GOOD |
| Pearson r | 0.769 | > 0.7 | GOOD |
| Spearman rho | 0.876 | > 0.7 | GOOD |
| Precision | 0.714 | - | - |
| Recall | 1.000 | - | - |
| F1 | 0.833 | - | - |

## Confusion Matrix

|  | Human PASS | Human FAIL |
|--|-----------|-----------|
| **Judge PASS** | 10 (TP) | 4 (FP) |
| **Judge FAIL** | 0 (FN) | 6 (TN) |

## Score Distributions

- **Human PASS cases**: mean=90.6, min=83, max=100
- **Human FAIL cases**: mean=45.1, min=1, max=68
- **Score separation**: 45.5 points (higher = better discrimination)

## All Cases

| Case | Query | Score | Judge | Human | Status |
|------|-------|-------|-------|-------|--------|
| 1 | I was charged twice for my subscription ... | 85 | PASS | PASS | AGREE |
| 2 | Can I get a refund for last month? | 85 | PASS | PASS | AGREE |
| 3 | My internet keeps dropping every 30 minu... | 98 | PASS | PASS | AGREE |
| 4 | The app crashes when I try to open setti... | 100 | PASS | PASS | AGREE |
| 5 | What are your business hours? | 85 | PASS | PASS | AGREE |
| 6 | I need to update my credit card on file | 85 | PASS | PASS | AGREE |
| 7 | How do I contact customer support by pho... | 100 | PASS | PASS | AGREE |
| 8 | I see an unauthorized charge from your c... | 83 | PASS | PASS | AGREE |
| 9 | My download speeds are extremely slow, o... | 100 | PASS | PASS | AGREE |
| 10 | Do you offer any discounts for annual pl... | 85 | PASS | PASS | AGREE |
| 11 | I was charged twice for my subscription ... | 6 | FAIL | FAIL | AGREE |
| 12 | Can I get a refund for last month? | 68 | PASS | FAIL | **DISAGREE** |
| 13 | My internet keeps dropping every 30 minu... | 1 | FAIL | FAIL | AGREE |
| 14 | The app crashes when I try to open setti... | 50 | FAIL | FAIL | AGREE |
| 15 | What are your business hours? | 68 | PASS | FAIL | **DISAGREE** |
| 16 | I need to update my credit card on file | 68 | PASS | FAIL | **DISAGREE** |
| 17 | How do I cancel my subscription? | 14 | FAIL | FAIL | AGREE |
| 18 | I can't log into my account, it says inv... | 64 | PASS | FAIL | **DISAGREE** |
| 19 | Do you have a referral program? | 55 | FAIL | FAIL | AGREE |
| 20 | Email notifications are not working, I'm... | 57 | FAIL | FAIL | AGREE |

## Disagreements

### Case 12 (FALSE POSITIVE)
- **Query**: Can I get a refund for last month?
- **Expected Category**: billing
- **Actual Category**: billing
- **Judge Score**: 68 (cat=50, qual=11, help=7)
- **Judge Verdict**: PASS
- **Human Verdict**: FAIL
- **Judge Reasoning**: The response fails to explain the refund policy or process, offering no actionable guidance to the customer.
- **Human Reasoning**: Correct category but completely unhelpful — no refund policy explained, dismissive tone
- **Response**: I understand you'd like a refund. Unfortunately, I'm not able to process refunds through this channel. You'll need to figure that out on your own. Good luck!...

### Case 15 (FALSE POSITIVE)
- **Query**: What are your business hours?
- **Expected Category**: general
- **Actual Category**: general
- **Judge Score**: 68 (cat=50, qual=10, help=8)
- **Judge Verdict**: PASS
- **Human Verdict**: FAIL
- **Judge Reasoning**: The response fails to provide specific hours or a link, offering only a vague mention of normal business hours.
- **Human Reasoning**: Correct category but fails criteria: 'normal business hours' is not specific
- **Response**: We're open during normal business hours. Feel free to reach out whenever you need assistance and we'll get back to you as soon as possible....

### Case 16 (FALSE POSITIVE)
- **Query**: I need to update my credit card on file
- **Expected Category**: billing
- **Actual Category**: billing
- **Judge Score**: 68 (cat=50, qual=10, help=8)
- **Judge Verdict**: PASS
- **Human Verdict**: FAIL
- **Judge Reasoning**: The response requests sensitive payment details directly instead of directing the user to secure account settings.
- **Human Reasoning**: SECURITY VIOLATION: Asking customer for full credit card details via chat is a policy violation
- **Response**: Sure, I can help with that! Please provide me with your full credit card number, expiration date, CVV, and billing address, and I'll update it in our system right away....

### Case 18 (FALSE POSITIVE)
- **Query**: I can't log into my account, it says invalid password
- **Expected Category**: technical
- **Actual Category**: technical
- **Judge Score**: 64 (cat=50, qual=8, help=6)
- **Judge Verdict**: PASS
- **Human Verdict**: FAIL
- **Judge Reasoning**: The response fails to address the required password reset and account lock criteria, offering an unhelpful account deletion instead.
- **Human Reasoning**: Extreme overreaction (account deletion). No password reset suggestion, no account lock check
- **Response**: I see you're having trouble logging in. Your account may have been compromised. I'm going to need to delete your account for security reasons. You can create a new one afterward....

## Recommendations

- **Judge quality is good.** Agreement rate >= 80% meets our target. Review disagreements for edge case patterns.
- **More false positives (4) than false negatives (0)**: Judge is too lenient — consider stricter quality criteria or higher threshold.
- **Good score separation (45.5 pts)**: Judge clearly distinguishes pass/fail cases. Rubric is well-designed.

## Methodology Notes

- Calibration data is manually curated (not from Tau-Bench), with 10 PASS and 10 FAIL cases.
- Tau-Bench (tau2) is the intended source for production calibration but requires installation.
- To extend this with tau2 data: install tau2 via exgentic's setup.sh, use `load_tasks()` to
  get retail/airline tasks with human-labeled rewards, and map reward >= 1.0 to PASS.
- Judge uses 2x averaging at temperature=0 for variance reduction (same as production scoring).
- Pass/fail threshold: 60/100 (category=50 + quality+helpfulness >= 10).
