"""
CRM Email Template Writer Agent
----------------------------------
Generates A/B email templates per segment — not per user.
Uses personalization tokens (e.g. {{first_name}}) that map to SFMC data sources.

Built with a senior CRM strategist persona baked into the system prompt,
so the output reflects real email marketing best practices, not just generic copy.

Usage:
    python email_template_agent.py

Output:
    email_templates.txt  — full templates, A/B variants, hypotheses, token map
"""

import anthropic
import json
import os
from pathlib import Path

API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# The 4 segments output by Agent 1, with their journey context
SEGMENTS = [
    {
        "segment": "new",
        "journey": "Onboarding Journey",
        "description": "User registered in the last 30 days or has fewer than 3 orders. "
                       "They haven't formed a habit yet. Goal: get them to their first or second order fast.",
    },
    {
        "segment": "active",
        "journey": "Loyalty & Upsell Journey",
        "description": "User ordered in the last 14 days. They are engaged. "
                       "Goal: increase order frequency and average order value.",
    },
    {
        "segment": "lapsing",
        "journey": "Re-engagement Journey",
        "description": "User's last order was 15–45 days ago. They are drifting. "
                       "Goal: remind them why they loved the platform and trigger a return order.",
    },
    {
        "segment": "churned",
        "journey": "Win-back Journey",
        "description": "User hasn't ordered in over 45 days. They may have switched to a competitor. "
                       "Goal: win them back with a strong hook — urgency, incentive, or emotional re-connection.",
    },
]

SYSTEM_PROMPT = """You are a senior CRM strategist and email copywriter with 10+ years of experience
in food delivery, e-commerce, and subscription products. You have deep expertise in:

EMAIL SUBJECT LINE BEST PRACTICES:
- Optimal length: 35–50 characters for mobile (where 60%+ of opens happen)
- Personalization in subject lines lifts open rates by 20–30% on average
- Curiosity-gap subjects ("You forgot something...") outperform discount-first subjects for lapsing users
- Urgency works best for churned users, not active ones — don't burn trust with false urgency
- Emoji in subject lines: use max 1, at the start or end, only if it adds meaning not decoration
- Question-format subjects lift opens for re-engagement but reduce CTR — pair with strong preview text
- Avoid spam trigger words: "free", "winner", "guaranteed", "act now", "limited time offer"

PREVIEW TEXT:
- 85–100 characters — extends the subject line, doesn't repeat it
- Should answer "why open now?" — the subject creates curiosity, preview closes the loop
- Most overlooked element in email — huge open rate lift when used correctly

EMAIL BODY BEST PRACTICES:
- Single focus: one message, one CTA — every extra CTA reduces conversion by ~20%
- For food delivery: sensory language works ("crispy", "melted", "fresh from the kitchen")
- Social proof works for new users ("Join millions who order on the platform")
- Personalized content (based on past cuisine) dramatically outperforms generic content
- Short wins: 3–5 sentences for re-engagement, slightly longer for onboarding
- Use the user's name in the body, not just the subject — feels more human
- Lead with benefit, not feature: "Lunch sorted in 20 min" not "We have fast delivery"

CTA BEST PRACTICES:
- Single verb + object: "Order now", "Pick your lunch", "Claim your discount"
- Avoid weak CTAs: "Click here", "Learn more", "Submit"
- For lapsing/churned: make the CTA feel low-commitment ("Browse menus" vs "Order now")
- Button text should match the landing page headline — reduces drop-off

A/B TESTING METHODOLOGY:
- Test ONE variable at a time — subject line OR body OR CTA, never all three
- Best variables to test for email: subject line (biggest impact on open rate), CTA (biggest impact on CTR)
- Variant B should have a clear hypothesis — not just "let's try something different"
- Winning metric must be defined before the test: open rate for subject tests, CTR for body/CTA tests
- Minimum sample size: 1,000 users per variant for statistical significance
- Runtime: at least 48–72 hours before calling a winner

PLATFORM CONTEXT:
- Direct, informal tone works better than formal language
- Lunch is the primary use case (11 AM–2 PM is peak ordering window)
- Corporate (B2B) users have a budget constraint — they care about using their card balance
- Favorite cuisine is a powerful personalization signal — use it whenever available
- Platform strength: large restaurant selection and corporate card support

PERSONALIZATION TOKENS AVAILABLE (from SFMC Data Extensions):
- {{first_name}} → Contacts DE → contact's first name
- {{favorite_cuisine}} → Order History DE → most ordered cuisine category
- {{days_since_last_order}} → Calculated field from Order History DE
- {{remaining_card_balance}} → B2B Card DE → current balance in ILS (B2B only)
- {{last_restaurant_name}} → Order History DE → name of last restaurant ordered from
- {{total_orders}} → Order History DE → total lifetime order count
- {{avg_order_value}} → Order History DE → average order value in ILS

OUTPUT FORMAT:
Return a valid JSON array. Each element represents one segment and must have:
{
  "segment": "segment name",
  "journey": "journey name",
  "variant_a": {
    "label": "Control — [brief description of the angle]",
    "subject_line": "...",
    "preview_text": "...",
    "body": "...",
    "cta": "..."
  },
  "variant_b": {
    "label": "Test — [brief description of what is being tested]",
    "subject_line": "...",
    "preview_text": "...",
    "body": "...",
    "cta": "..."
  },
  "ab_variable_tested": "subject_line | body | cta",
  "ab_hypothesis": "Specific prediction: Variant B will have higher [metric] because [reason].",
  "winning_metric": "open_rate | ctr | conversion_rate",
  "personalization_tokens_used": [
    {"token": "{{token_name}}", "sfmc_source": "DE name → field name", "used_in": "subject | body | both"}
  ],
  "strategist_notes": "2–3 sentences on the strategic choices made for this segment and what to watch."
}

Return only valid JSON. No markdown fences. No text outside the array."""


def build_segment_brief(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(f"Segment {i}: {seg['segment'].upper()}")
        lines.append(f"Journey: {seg['journey']}")
        lines.append(f"Context: {seg['description']}")
        lines.append("")
    return "\n".join(lines)


def generate_templates(segments: list[dict]) -> list[dict]:
    client = anthropic.Anthropic(api_key=API_KEY)

    segment_brief = build_segment_brief(segments)

    print(f"  Sending {len(segments)} segments to Claude...")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    "Generate A/B email templates for each of these CRM segments.\n\n"
                    "Apply all best practices from your expertise. "
                    "Make the copy feel human, Israeli, and food-native — not corporate.\n\n"
                    f"{segment_brief}"
                ),
            }
        ],
    )

    usage = response.usage
    print(f"  Tokens used — input: {usage.input_tokens}, output: {usage.output_tokens}")
    if hasattr(usage, "cache_read_input_tokens"):
        print(f"  Cache read tokens: {usage.cache_read_input_tokens}")

    return json.loads(response.content[0].text)


def format_templates_as_text(templates: list[dict]) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("  CRM EMAIL TEMPLATES — A/B VARIANTS BY SEGMENT")
    lines.append("  Generated by Email Template Agent")
    lines.append("=" * 70)

    for t in templates:
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  SEGMENT: {t['segment'].upper()}  |  JOURNEY: {t['journey']}")
        lines.append("=" * 70)

        for variant_key, variant_label in [("variant_a", "VARIANT A"), ("variant_b", "VARIANT B")]:
            v = t[variant_key]
            lines.append("")
            lines.append(f"  [{variant_label}] {v['label']}")
            lines.append(f"  {'─' * 60}")
            lines.append(f"  Subject:      {v['subject_line']}")
            lines.append(f"  Preview text: {v['preview_text']}")
            lines.append(f"  Body:")
            # Indent the body for readability
            for body_line in v["body"].split("\n"):
                lines.append(f"    {body_line}")
            lines.append(f"  CTA:          {v['cta']}")

        lines.append("")
        lines.append(f"  A/B TEST DETAILS")
        lines.append(f"  {'─' * 60}")
        lines.append(f"  Variable tested: {t['ab_variable_tested']}")
        lines.append(f"  Winning metric:  {t['winning_metric']}")
        lines.append(f"  Hypothesis:      {t['ab_hypothesis']}")

        lines.append("")
        lines.append(f"  PERSONALIZATION TOKENS")
        lines.append(f"  {'─' * 60}")
        for token in t["personalization_tokens_used"]:
            lines.append(
                f"  {token['token']:<35} → {token['sfmc_source']}  (used in: {token['used_in']})"
            )

        lines.append("")
        lines.append(f"  STRATEGIST NOTES")
        lines.append(f"  {'─' * 60}")
        lines.append(f"  {t['strategist_notes']}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("  END OF TEMPLATES")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    base = Path(__file__).parent
    output_path = base / "email_templates.txt"

    print("CRM Email Template Writer Agent")
    print("-----------------------------------")

    print("Generating templates via Claude...")
    templates = generate_templates(SEGMENTS)
    print(f"  {len(templates)} segment templates generated")

    print("Writing output file...")
    formatted = format_templates_as_text(templates)
    output_path.write_text(formatted, encoding="utf-8")
    print(f"  Saved to: {output_path}")

    print("\nDone. Open email_templates.txt to review the copy and A/B test plan.")


if __name__ == "__main__":
    main()
