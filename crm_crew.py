"""
CRM Multi-Agent Crew (Level 3)
---------------------------------
A crew of 4 specialized agents that collaborate to produce
a full CRM campaign plan from raw user data.

Agent flow:
  1. Data Analyst     → segments users from CSV
  2. CRM Strategist   → defines campaign priorities per segment
  3. Email Copywriter → writes A/B email templates
  4. QA Reviewer      → critiques and approves the output

Usage:
    python crm_crew.py

Output:
    crew_output.txt — full campaign plan approved by QA
"""

import os

from crewai import Agent, Crew, LLM, Process, Task
from pathlib import Path

API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# ── LLM Configuration ────────────────────────────────────────────────────────
llm = LLM(
    model="anthropic/claude-sonnet-4-6",
    api_key=API_KEY,
    max_tokens=4096,
)

# ── Load user data ────────────────────────────────────────────────────────────
def load_csv_as_text(filepath: Path) -> str:
    return filepath.read_text(encoding="utf-8")


# ── Define Agents ─────────────────────────────────────────────────────────────

data_analyst = Agent(
    role="CRM Data Analyst",
    goal="Analyze user order history and classify every user into a behavioral segment with clear reasoning.",
    backstory=(
        "You are a data analyst specializing in food delivery platforms. "
        "You have 7 years of experience at leading food-delivery platforms. "
        "You understand that behind every row of data is a real person with ordering habits, "
        "and your job is to translate raw numbers into actionable behavioral segments. "
        "You are precise, data-driven, and always explain your reasoning."
    ),
    llm=llm,
    verbose=True,
)

crm_strategist = Agent(
    role="Senior CRM Strategist",
    goal="Define the campaign strategy for each user segment — what to say, when, and why.",
    backstory=(
        "You are a senior CRM strategist with deep knowledge of the food delivery market. "
        "You have run retention and reactivation campaigns for food delivery apps with millions of users. "
        "You know that lunch is the primary use case, "
        "and that B2B users respond differently to messaging than B2C users. "
        "You think in journeys, not emails — each touchpoint should build on the last."
    ),
    llm=llm,
    verbose=True,
)

email_copywriter = Agent(
    role="Email Copywriter",
    goal="Write creative, personalized A/B email templates that feel human and drive action.",
    backstory=(
        "You are a copywriter who specializes in food, lifestyle, and consumer apps. "
        "You have written campaigns for food delivery brands across Europe and the Middle East. "
        "You write short, punchy copy that respects the reader's time. "
        "You always use sensory language for food ('melted cheese', 'crispy falafel'), "
        "you never use spam words, and you believe a great subject line is worth more than a great email body. "
        "You use personalization tokens like {{first_name}} and {{favorite_cuisine}} naturally, not mechanically."
    ),
    llm=llm,
    verbose=True,
)

qa_reviewer = Agent(
    role="CRM QA & Deliverability Specialist",
    goal="Review all campaign materials for quality, tone, spam risk, token errors, and strategic alignment.",
    backstory=(
        "You are a QA specialist who has reviewed thousands of CRM campaigns before they hit send. "
        "You check for: spam trigger words, broken personalization tokens, CTA clarity, "
        "tone consistency, subject line length, and whether the copy actually matches the segment strategy. "
        "You are constructive but direct — you flag problems and suggest specific fixes. "
        "You never approve something just to be nice. If something is off, you say so and explain why."
    ),
    llm=llm,
    verbose=True,
)


# ── Define Tasks ──────────────────────────────────────────────────────────────

def build_tasks(user_data_text: str) -> list[Task]:

    task_segment = Task(
        description=(
            f"Today's date is 2026-05-10.\n\n"
            f"Here is the raw user data from the platform:\n\n"
            f"{user_data_text}\n\n"
            f"Analyze each user and classify them into one of these segments:\n"
            f"- new: registered in last 30 days OR fewer than 3 total orders\n"
            f"- active: last order within 14 days\n"
            f"- lapsing: last order 15–45 days ago\n"
            f"- churned: no order in over 45 days\n\n"
            f"Special B2B rule: if a B2B user has been registered over 60 days "
            f"but has fewer than 5 orders in the last 90 days, flag them as 'B2B underperforming'.\n\n"
            f"For each user provide: user_id, segment, and one sentence of reasoning."
        ),
        expected_output=(
            "A clean segmentation report listing every user with their segment and reasoning. "
            "End with a summary count per segment."
        ),
        agent=data_analyst,
    )

    task_strategy = Task(
        description=(
            "Based on the segmentation report above, define the campaign strategy for each segment.\n\n"
            "For each segment provide:\n"
            "1. Core message angle (what emotional or rational hook to use)\n"
            "2. Recommended send time (day of week, time of day — think about lunch ordering behavior)\n"
            "3. Personalization priority (which tokens matter most for this segment)\n"
            "4. What to avoid (tone mistakes, wrong CTAs, etc.)\n"
            "5. Success metric to optimize for\n\n"
            "Remember: lunch is peak (11AM–2PM), "
            "B2B users care about card balance, B2C users care about discovery and habit."
        ),
        expected_output=(
            "A strategy brief per segment — clear enough for a copywriter to write from without asking questions."
        ),
        agent=crm_strategist,
        context=[task_segment],
    )

    task_copy = Task(
        description=(
            "Using the strategy brief above, write A/B email templates for each segment.\n\n"
            "For each segment produce:\n"
            "- Variant A (control): subject line, preview text, body (3–5 sentences), CTA\n"
            "- Variant B (test): one variable changed from Variant A — state clearly what you are testing\n"
            "- A/B hypothesis: which variant you predict will win and why\n"
            "- Personalization tokens used and where they come from\n\n"
            "Available tokens: {{first_name}}, {{favorite_cuisine}}, {{days_since_last_order}}, "
            "{{last_restaurant_name}}, {{remaining_card_balance}} (B2B only), {{total_orders}}\n\n"
            "Rules: max 1 emoji per subject line, no spam words, single CTA per email, "
            "sensory food language in the body, informal Israeli tone."
        ),
        expected_output=(
            "Complete A/B email templates for all segments, clearly labeled, "
            "with personalization token map and A/B hypothesis for each."
        ),
        agent=email_copywriter,
        context=[task_strategy],
    )

    task_qa = Task(
        description=(
            "Review all email templates produced by the copywriter against the strategy defined by the strategist.\n\n"
            "Check each template for:\n"
            "1. Spam trigger words (flag any found)\n"
            "2. Subject line length (should be 35–50 characters)\n"
            "3. Personalization token syntax — are they correctly formatted as {{token}}?\n"
            "4. CTA quality — is it a strong action verb? Is there only one?\n"
            "5. Tone — does it match the segment? (e.g., urgency for churned, not for active)\n"
            "6. Strategy alignment — does the copy match what the strategist recommended?\n"
            "7. A/B test validity — is exactly ONE variable being tested per pair?\n\n"
            "For each issue found: state what it is, which template it's in, and your suggested fix.\n"
            "At the end: give an overall APPROVED or NEEDS REVISION verdict per segment."
        ),
        expected_output=(
            "A QA report with issues flagged per template, suggested fixes, "
            "and a final verdict (APPROVED / NEEDS REVISION) per segment."
        ),
        agent=qa_reviewer,
        context=[task_strategy, task_copy],
    )

    return [task_segment, task_strategy, task_copy, task_qa]


# ── Assemble and Run the Crew ─────────────────────────────────────────────────

def main():
    base = Path(__file__).parent
    user_data_path = base / "sample_data.csv"
    output_path = base / "crew_output.txt"

    print("CRM Multi-Agent Crew")
    print("----------------------")
    print("Agents: Data Analyst → CRM Strategist → Email Copywriter → QA Reviewer")
    print("")

    user_data_text = load_csv_as_text(user_data_path)
    tasks = build_tasks(user_data_text)

    crew = Crew(
        agents=[data_analyst, crm_strategist, email_copywriter, qa_reviewer],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    print("Kicking off the crew...\n")
    result = crew.kickoff()

    # Save the final output
    output_path.write_text(str(result), encoding="utf-8")
    print(f"\nFinal output saved to: {output_path}")
    print("\n--- FINAL OUTPUT PREVIEW ---")
    print(str(result)[:1000])


if __name__ == "__main__":
    main()
