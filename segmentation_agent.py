"""
CRM Segmentation Agent
------------------------
Reads a CSV of user order history, sends it to Claude for behavioral
segmentation, and outputs a new CSV with segment labels and recommended
journey enrollments.

Usage:
    python segmentation_agent.py

Requires:
    ANTHROPIC_API_KEY environment variable to be set.
"""

import anthropic
import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path

import os

API_KEY = os.environ.get("ANTHROPIC_API_KEY")


# Reference date — in production this would be date.today()
TODAY = "2026-05-10"

SYSTEM_PROMPT = f"""You are a CRM segmentation engine for a food delivery platform.
Today's date is {TODAY}.

For each user provided, classify them into exactly one of these segments based on their order behavior:

- new: registered within the last 30 days OR has fewer than 3 total orders ever
- active: last order was within the last 14 days (and is not "new")
- lapsing: last order was 15 to 45 days ago
- churned: no order in more than 45 days

Then assign the recommended journey:
- new      → "Onboarding Journey"
- active   → "Loyalty & Upsell Journey"
- lapsing  → "Re-engagement Journey"
- churned  → "Win-back Journey"

Special rule for B2B users:
If a B2B user has been registered for more than 60 days but has fewer than 5 orders
in the last 90 days, override their journey to "B2B Activation Journey" regardless
of their segment — these are underperforming corporate accounts that need direct attention.

Return a JSON array. Each element must have exactly these keys:
{{
  "user_id": "string",
  "segment": "new | active | lapsing | churned",
  "recommended_journey": "string",
  "reason": "one sentence explaining the classification"
}}

Return only valid JSON. No markdown fences, no explanation text outside the JSON array."""


def load_users(filepath: Path) -> list[dict]:
    with open(filepath, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def format_user_block(user: dict) -> str:
    return (
        f"User ID: {user['user_id']}\n"
        f"Type: {user['user_type']}\n"
        f"Registration date: {user['registration_date']}\n"
        f"Last order date: {user['last_order_date']}\n"
        f"Orders last 30 days: {user['orders_last_30d']}\n"
        f"Orders last 90 days: {user['orders_last_90d']}\n"
        f"Avg order value (ILS): {user['avg_order_value_ils']}\n"
        f"Favorite cuisine: {user['favorite_cuisine']}"
    )


def segment_users(users: list[dict]) -> list[dict]:
    """Send all users to Claude and get back segmentation results."""
    client = anthropic.Anthropic(api_key=API_KEY)

    # Build one block of text containing all users, separated by dividers
    user_blocks = "\n\n---\n\n".join(format_user_block(u) for u in users)

    print(f"  Sending {len(users)} users to Claude...")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                # Cache the system prompt — it's the same for every run,
                # so this saves tokens and latency on repeated calls
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"Segment these users:\n\n{user_blocks}",
            }
        ],
    )

    # Log token usage so you can track costs
    usage = response.usage
    print(f"  Tokens used — input: {usage.input_tokens}, output: {usage.output_tokens}")
    if hasattr(usage, "cache_read_input_tokens"):
        print(f"  Cache read tokens: {usage.cache_read_input_tokens}")

    raw = response.content[0].text
    return json.loads(raw)


def write_output(users: list[dict], segments: list[dict], output_path: Path):
    """Merge original user data with segmentation results and write to CSV."""
    segment_map = {s["user_id"]: s for s in segments}

    original_fields = list(users[0].keys())
    added_fields = ["segment", "recommended_journey", "reason"]
    fieldnames = original_fields + added_fields

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for user in users:
            row = dict(user)
            seg = segment_map.get(user["user_id"], {})
            row["segment"] = seg.get("segment", "unknown")
            row["recommended_journey"] = seg.get("recommended_journey", "unknown")
            row["reason"] = seg.get("reason", "")
            writer.writerow(row)


def print_summary(segments: list[dict]):
    print("\n=============================")
    print("  SEGMENTATION SUMMARY")
    print("=============================")

    seg_counts = Counter(s["segment"] for s in segments)
    print("\nSegments:")
    for seg, count in seg_counts.most_common():
        print(f"  {seg:<10} {count} users")

    journey_counts = Counter(s["recommended_journey"] for s in segments)
    print("\nJourney Enrollment:")
    for journey, count in journey_counts.most_common():
        print(f"  {journey:<30} {count} users")

    print("\nUser Detail:")
    for s in segments:
        print(f"  {s['user_id']} | {s['segment']:<10} | {s['recommended_journey']:<30} | {s['reason']}")


def main():
    base = Path(__file__).parent
    input_path = base / "sample_data.csv"
    output_path = base / "segmented_output.csv"

    print("CRM Segmentation Agent")
    print("------------------------")

    print("Loading users from CSV...")
    users = load_users(input_path)
    print(f"  {len(users)} users loaded")

    print("Running segmentation via Claude...")
    segments = segment_users(users)
    print(f"  {len(segments)} users classified")

    print("Writing output CSV...")
    write_output(users, segments, output_path)
    print(f"  Saved to: {output_path}")

    print_summary(segments)


if __name__ == "__main__":
    main()
