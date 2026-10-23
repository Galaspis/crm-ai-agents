# CRM AI Agents

A learning project exploring AI agent architectures for CRM use cases, built around a simulated food-delivery platform dataset (`sample_data.csv`, 15 invented users with order history).

## Agents

### Agent 1 — Segmentation (`segmentation_agent.py`)
Reads `sample_data.csv` and asks Claude to classify each user into one behavioral segment:
- `new` — registered in the last 30 days or fewer than 3 orders ever
- `active` — last order within the last 14 days
- `lapsing` — last order 15–45 days ago
- `churned` — no order in more than 45 days

Each segment maps to a journey (Onboarding, Loyalty & Upsell, Re-engagement, Win-back). **B2B override:** a B2B user registered for more than 60 days with fewer than 5 orders in the last 90 days is sent to the "B2B Activation Journey" regardless of segment.

The reference date is fixed in the code (`TODAY = "2026-05-10"`) so the sample data always segments the same way; in production it would be `date.today()`. Output: `segmented_output.csv`.

### Agent 2 — Email Copywriter (`email_template_agent.py`)
Generates A/B email templates per segment (subject line, preview text, body, CTA, hypothesis, winning metric) with a senior-CRM-strategist system prompt. Personalization tokens available to the copy:
`{{first_name}}`, `{{favorite_cuisine}}`, `{{days_since_last_order}}`, `{{remaining_card_balance}}` (B2B only), `{{last_restaurant_name}}`, `{{total_orders}}`, `{{avg_order_value}}`. Output: `email_templates.txt`.

### Agent 3 — Multi-Agent Crew (`crm_crew.py`)
A 4-agent CrewAI pipeline that runs end-to-end on the same CSV:
1. **Data Analyst** → segments users
2. **CRM Strategist** → defines campaign strategy per segment
3. **Email Copywriter** → writes A/B templates
4. **QA Reviewer** → checks quality, spam risk and strategy alignment

Output: `crew_output.txt`.

## Tech stack

- Python 3.10+
- [Anthropic Claude](https://docs.anthropic.com), model `claude-sonnet-4-6` (Agents 1 and 2 call the SDK directly with prompt caching; CrewAI uses `anthropic/claude-sonnet-4-6`)
- [CrewAI](https://docs.crewai.com), multi-agent orchestration (Agent 3)

## Setup

```bash
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY=your_key_here
# PowerShell:
$env:ANTHROPIC_API_KEY="your_key_here"

# Run individual agents
python segmentation_agent.py
python email_template_agent.py

# Run the full crew
python crm_crew.py
```

The three output files are gitignored.

## Purpose

Built to learn progressive agent patterns: from a single-call agent, to a multi-step agent with structured output, to a sequential crew of role-based agents.
