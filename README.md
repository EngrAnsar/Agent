# Government Engineering AI Copilot

A live demo agent for the **41st In-Service Training Course for Junior Engineers
(BS-17–18)**, Punjab Engineering Academy, Lahore.

Built to accompany the workshop *"AI Is Already Replacing Tasks, Not Humans"* —
it shows an SDO the difference between a chatbot and an agent by actually
running six engineering-workflow tasks against one fictional case file:

> **Rehabilitation of Metalled Road, Chowk Azam – Kot Sultan (12.4 km), District
> Layyah — Punjab C&W Department**

Every tab produces a **draft**. Nothing in this app approves, certifies, or
signs an engineering document — that stays with the engineer, on purpose. Say
this out loud at least once during the demo.

## What's in it

| Tab | What it does |
|---|---|
| 📄 Document Analyzer | Extracts project identity, contract terms, scope, milestones from the project brief |
| 📊 BOQ Assistant | Deterministic rate-range check on the approved BOQ, then asks the agent to explain flagged items |
| 📈 Progress Monitor | Planned vs. actual progress, variance, and an AI risk analysis grounded in the contractor's own report |
| 🔍 Inspection Assistant | Turns rough, handwritten-style field notes into a structured record with action items |
| 📝 Report Generator | Combines whatever you've already generated into a one-page draft report with a sign-off block |
| 🤖 Delay Risk (ML) | A `scikit-learn` linear regression trained live on 14 past projects, scoring the current one — the agent explains the score, it doesn't decide anything |

## Setup (once, tonight)

```bash
cd gov-engineering-copilot
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Your Groq key is already in `.env` in this folder (`GROQ_API_KEY=...`) — the
app loads it automatically. If you ever need to swap it, edit `.env` or paste
a different key straight into the sidebar at runtime (session-only, not saved).

## Run it

```bash
source .venv/bin/activate
streamlit run app.py
```

It opens at `http://localhost:8501`. Test every tab once **before** the
session — Groq is fast, but venue wifi is the real variable. If wifi is weak
tomorrow, switch the sidebar model to **GPT-OSS 20B (fastest)** — it trades a
little depth for noticeably lower latency in front of a room.

## Suggested live-demo order (matches the slide deck)

1. **Document Analyzer** — click *Extract project parameters* on the pre-loaded
   brief. Point out: nothing was retyped, and it says "not stated" rather than
   guessing at anything missing.
2. **BOQ Assistant** — the shoulder-earthwork line is pre-loaded with a rate
   10x too high. Show the table (it's already highlighted), then click *Ask
   the agent to review flagged items* — the flag was math, the explanation is
   the agent, the correction is still the engineer's to make.
3. **Progress Monitor** — planned 72% / actual 58% is pre-loaded from the
   contractor's own report. Click *Generate AI risk analysis* and read the
   three follow-up questions aloud — that's the SDO's next site meeting,
   drafted in seconds.
4. **Inspection Assistant** — the rough field notes are real-looking and
   messy on purpose. Click *Structure inspection notes* and compare the
   output side-by-side with the raw text on screen.
5. **Report Generator** — click *Generate one-page progress report*. Because
   the earlier three tabs already ran, this one stitches their outputs
   together automatically — that's the "one architecture, one pipeline" idea
   from the slides, made concrete.
6. **Delay Risk (ML)** — show the historical table, then click *Explain this
   prediction*. Emphasize the caveat the agent is instructed to state: 14
   rows is a demo, not a statistically robust model — the same honesty you'd
   want from any tool an engineer signs off on.

## If something breaks mid-session

- **No internet / API down:** every tab still shows the raw sample data and
  computed numbers (BOQ table, progress chart, ML prediction) without calling
  the model — only the "AI explains this" step needs connectivity. Keep
  talking through the deterministic parts.
- **A generate button errors:** the app catches it and prints a plain-English
  reason instead of crashing (auth, rate limit, network, or "unexpected") —
  read it aloud, it's part of the "know the failure modes" section of the
  talk anyway.
- **Need a clean slate:** refresh the browser tab — Streamlit resets session
  state.

## Adapting it live for another department

Swap the files in `sample_data/` for a real (or another fictional) PHED or
GDA case file with the same shape — a brief, a quantities/BOQ-like table, a
progress or maintenance report, and field notes — and the same six tabs work
unchanged. That's the "one architecture, every department" slide, and it's
worth saying so if a participant asks whether this only works for roads.

## Files

```
gov-engineering-copilot/
├── app.py                       # the Streamlit app
├── requirements.txt
├── .env                         # GROQ_API_KEY (already set, keep this out of git)
├── .env.example
└── sample_data/
    ├── project_brief.txt
    ├── boq.csv
    ├── progress_report.txt
    ├── inspection_notes.txt
    └── historical_projects.csv
```
