"""
Government Engineering AI Copilot
----------------------------------
A demo agent built for the 41st In-Service Training Course for Junior
Engineers (BS-17-18), Punjab Engineering Academy, Lahore.

Shows how AI can take over the repetitive information-processing tasks
around an engineering project - document extraction, BOQ checks, progress
variance, inspection notes, report drafting, and delay-risk scoring -
while every output stays a draft for the engineer to review and sign.

Run:
    streamlit run app.py
"""

import os
import io
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import streamlit as st
import groq
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

SAMPLE_DIR = Path(__file__).parent / "sample_data"


def _load_dotenv():
    """Tiny .env loader so the demo works with no extra dependency."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

MODEL_OPTIONS = {
    "GPT-OSS 120B (most capable, default)": "openai/gpt-oss-120b",
    "Llama 3.3 70B (balanced)": "llama-3.3-70b-versatile",
    "GPT-OSS 20B (fastest)": "openai/gpt-oss-20b",
}

st.set_page_config(
    page_title="Government Engineering AI Copilot",
    page_icon="\U0001F4D0",
    layout="wide",
)


@st.cache_data
def load_text(name: str) -> str:
    return (SAMPLE_DIR / name).read_text()


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / name)


def get_client() -> Optional["groq.Groq"]:
    api_key = st.session_state.get("api_key") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    return groq.Groq(api_key=api_key)


def ask_agent(system: str, user_prompt: str, max_tokens: int = 3000) -> str:
    """Single-call helper with friendly, demo-safe error handling."""
    client = get_client()
    if client is None:
        return (
            "**No API key configured.** Enter a Groq API key in the sidebar "
            "to run this step."
        )
    model = st.session_state.get("model", "openai/gpt-oss-120b")
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""
    except groq.AuthenticationError:
        return "**Authentication failed.** Check that the API key in the sidebar is correct."
    except groq.RateLimitError:
        return "**Rate limited.** Wait a moment and try again."
    except groq.APIStatusError as e:
        return f"**API error ({e.status_code}).** {e.message}"
    except groq.APIConnectionError:
        return "**Network error.** Check the internet connection and try again."
    except Exception as e:  # last resort, keeps the live demo from crashing
        return f"**Unexpected error.** {e}"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### \U0001F4D0 Engineering AI Copilot")
    st.caption("41st In-Service Training Course · Punjab Engineering Academy")

    default_key = os.environ.get("GROQ_API_KEY", "")
    st.session_state["api_key"] = st.text_input(
        "Groq API key",
        value=st.session_state.get("api_key", default_key),
        type="password",
        help="Reads GROQ_API_KEY from the environment (.env file) if set. "
             "Pasting a key here overrides it for this session only.",
    )
    model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=0)
    st.session_state["model"] = MODEL_OPTIONS[model_label]

    st.divider()
    st.markdown(
        "**Human-in-the-loop, by design.** \n"
        "Every output below is a draft. Nothing here approves, certifies, "
        "or signs an engineering document - that stays with you."
    )
    st.divider()
    st.caption("Case file: Rehabilitation of Link Road, Chowk Azam - Kot Sultan "
               "(12.4 km), District Layyah, Punjab C&W Department.")

st.title("Government Engineering AI Copilot")
st.caption(
    "One architecture, six modules - document intelligence, BOQ checks, progress "
    "monitoring, inspection support, report drafting, and ML delay-risk scoring."
)

tabs = st.tabs([
    "\U0001F4C4 Document Analyzer",
    "\U0001F4CA BOQ Assistant",
    "\U0001F4C8 Progress Monitor",
    "\U0001F50D Inspection Assistant",
    "\U0001F4DD Report Generator",
    "\U0001F916 Delay Risk (ML)",
])

# --------------------------------------------------------------------------
# Tab 1 - Document Analyzer
# --------------------------------------------------------------------------

with tabs[0]:
    st.subheader("Document Analyzer")
    st.write("Paste (or use the sample) project document. The agent extracts the "
             "parameters an engineer would otherwise copy out by hand.")

    doc_text = st.text_area(
        "Project document",
        value=load_text("project_brief.txt"),
        height=280,
        key="doc_text",
    )

    if st.button("Extract project parameters", type="primary", key="btn_doc"):
        with st.spinner("Reading document..."):
            result = ask_agent(
                system=(
                    "You are an assistant to a government civil engineer in Pakistan. "
                    "Extract project parameters from documents precisely and concisely. "
                    "Never invent a figure that is not in the source text - write "
                    "'not stated' instead. Output clean Markdown with short sections."
                ),
                user_prompt=(
                    "Extract the following from the project document below, as a "
                    "Markdown summary with these headings: Project Identity (name, "
                    "location, executing department), Contract (contractor, approved "
                    "cost, time allowed, commencement and completion dates), Scope "
                    "Highlights (bullet list, max 6 items), Key Milestones (bullet "
                    "list), and Reporting Requirement.\n\n---\n\n" + doc_text
                ),
                max_tokens=1600,
            )
        st.session_state["doc_analysis"] = result

    if "doc_analysis" in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state["doc_analysis"])

# --------------------------------------------------------------------------
# Tab 2 - BOQ Assistant
# --------------------------------------------------------------------------

# Engineering-judgment reference ranges for this specific BOQ, PKR per unit.
# Used for deterministic outlier detection - the AI explains, it doesn't guess.
BOQ_TYPICAL_RANGE = {
    1: (300, 500), 2: (2800, 3800), 3: (3800, 4800), 4: (80, 120),
    5: (45, 70), 6: (19000, 23000), 7: (500, 900), 8: (26000, 31000),
    9: (26000, 31000), 10: (35000, 48000), 11: (10000, 14000), 12: (900, 1400),
}

with tabs[1]:
    st.subheader("BOQ Assistant")
    st.write("The approved Bill of Quantities for the case file. Rate checks below "
             "are computed directly from the numbers - the agent only explains what "
             "it finds, it never overrides a figure against the approved BOQ.")

    boq = load_csv("boq.csv").copy()
    boq["typical_low"] = boq["item_no"].map(lambda i: BOQ_TYPICAL_RANGE[i][0])
    boq["typical_high"] = boq["item_no"].map(lambda i: BOQ_TYPICAL_RANGE[i][1])
    boq["flag"] = (boq["rate_pkr"] < boq["typical_low"]) | (boq["rate_pkr"] > boq["typical_high"])

    total = boq["amount_pkr"].sum()
    flagged = boq[boq["flag"]]

    c1, c2 = st.columns(2)
    c1.metric("BOQ items", len(boq))
    c2.metric("Items flagged for rate review", len(flagged))

    def highlight_flag(row):
        color = "background-color: #f3e2cf" if row["flag"] else ""
        return [color] * len(row)

    st.dataframe(
        boq[["item_no", "description", "unit", "quantity", "rate_pkr", "amount_pkr", "flag"]]
        .style.apply(highlight_flag, axis=1)
        .format({"quantity": "{:,.0f}", "rate_pkr": "{:,.0f}", "amount_pkr": "{:,.0f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Total BOQ value shown: PKR {total:,.0f}")

    if len(flagged):
        if st.button("Ask the agent to review flagged items", type="primary", key="btn_boq"):
            with st.spinner("Reviewing..."):
                items_text = flagged[["item_no", "description", "unit", "quantity", "rate_pkr", "amount_pkr"]].to_csv(index=False)
                result = ask_agent(
                    system=(
                        "You are assisting a government engineer reviewing a Bill of "
                        "Quantities. You are given items whose rate falls outside the "
                        "engineer-set typical range for that work item. Explain plainly "
                        "what to verify - do not state a corrected rate as fact, since "
                        "only the engineer can confirm against the approved BOQ and "
                        "current market rate schedule."
                    ),
                    user_prompt=(
                        "These BOQ items were flagged as outside the typical rate range "
                        "for this class of work:\n\n" + items_text +
                        "\n\nFor each, give: a one-line plain-English flag, the most "
                        "likely explanation (e.g. data entry error, decimal shift, "
                        "unit mismatch, or a genuinely justified higher rate), and one "
                        "concrete action to verify it. Keep it under 120 words per item."
                    ),
                    max_tokens=800,
                )
            st.session_state["boq_review"] = result

    if "boq_review" in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state["boq_review"])

# --------------------------------------------------------------------------
# Tab 3 - Progress Monitor
# --------------------------------------------------------------------------

ACTIVITY_PROGRESS = pd.DataFrame([
    {"activity": "Mobilization & site clearance", "planned_pct": 100, "actual_pct": 100},
    {"activity": "Earthwork & sub-grade", "planned_pct": 100, "actual_pct": 100},
    {"activity": "Sub-base (crushed stone)", "planned_pct": 100, "actual_pct": 95},
    {"activity": "WBM base course", "planned_pct": 90, "actual_pct": 70},
    {"activity": "Culvert reconstruction", "planned_pct": 80, "actual_pct": 33},
    {"activity": "Prime / tack coat", "planned_pct": 40, "actual_pct": 28},
    {"activity": "Wearing course", "planned_pct": 20, "actual_pct": 0},
    {"activity": "Shoulders / signage", "planned_pct": 10, "actual_pct": 0},
])

with tabs[2]:
    st.subheader("Progress Monitor")
    st.write("Planned vs. actual progress from the Month 7 contractor report.")

    c1, c2, c3 = st.columns(3)
    planned = c1.number_input("Planned progress %", value=72, min_value=0, max_value=100)
    actual = c2.number_input("Actual progress %", value=58, min_value=0, max_value=100)
    variance = actual - planned
    c3.metric("Variance", f"{variance:+d}%", delta=f"{variance:+d}%")

    st.bar_chart(ACTIVITY_PROGRESS.set_index("activity")[["planned_pct", "actual_pct"]])

    if variance <= -10:
        st.error("Status: AT RISK - progress variance exceeds -10%.")
    elif variance < 0:
        st.warning("Status: WATCH - behind schedule, within tolerance.")
    else:
        st.success("Status: ON TRACK.")

    if st.button("Generate AI risk analysis", type="primary", key="btn_progress"):
        with st.spinner("Analyzing..."):
            result = ask_agent(
                system=(
                    "You are assisting a supervising engineer reading a contractor's "
                    "monthly progress report on a road project. Be specific and "
                    "practical - reference the actual activities and remarks given, "
                    "don't write generic project-management filler."
                ),
                user_prompt=(
                    f"Overall planned progress: {planned}%. Actual progress: {actual}%. "
                    f"Variance: {variance}%.\n\nActivity-wise status:\n"
                    + ACTIVITY_PROGRESS.to_csv(index=False)
                    + "\n\nContractor's monthly report (full text):\n\n"
                    + load_text("progress_report.txt")
                    + "\n\nWrite, as Markdown: (1) the most likely causes of the "
                      "variance based on this specific report, (2) which activity is "
                      "the critical bottleneck for finishing on time, (3) three "
                      "pointed follow-up questions the SDO should put to the "
                      "contractor at the next site meeting."
                ),
                max_tokens=1200,
            )
        st.session_state["progress_analysis"] = result

    if "progress_analysis" in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state["progress_analysis"])

# --------------------------------------------------------------------------
# Tab 4 - Inspection Assistant
# --------------------------------------------------------------------------

with tabs[3]:
    st.subheader("Inspection Assistant")
    st.write("Rough, unedited field notes from today's site visit. The agent turns "
             "them into a structured record the engineer can act on.")

    notes = st.text_area(
        "Field notes (as taken on site)",
        value=load_text("inspection_notes.txt"),
        height=280,
        key="inspection_notes",
    )

    if st.button("Structure inspection notes", type="primary", key="btn_inspect"):
        with st.spinner("Structuring..."):
            result = ask_agent(
                system=(
                    "You turn rough, handwritten-style site inspection notes from a "
                    "government engineer into a structured record. Preserve every "
                    "specific fact (chainage/RD, quantities, names) exactly as given. "
                    "Do not add observations that are not in the notes."
                ),
                user_prompt=(
                    "Structure these field notes as Markdown with headings: "
                    "Observations (by location/RD), Non-Compliances & Concerns, "
                    "Documentation Requested From Contractor, Action Items (with a "
                    "suggested owner and timeframe for each), and Follow-Up Required "
                    "Before Next Payment Certificate.\n\n---\n\n" + notes
                ),
                max_tokens=1400,
            )
        st.session_state["inspection_report"] = result

    if "inspection_report" in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state["inspection_report"])

# --------------------------------------------------------------------------
# Tab 5 - Report Generator
# --------------------------------------------------------------------------

with tabs[4]:
    st.subheader("Report Generator")
    st.write("Combines whatever has already been generated in the other tabs - or "
             "falls back to the raw case file - into a one-page draft for the "
             "supervising engineer to review, edit, and sign.")

    have_doc = "doc_analysis" in st.session_state
    have_progress = "progress_analysis" in st.session_state
    have_inspection = "inspection_report" in st.session_state
    st.caption(
        f"Using: document analysis {'✓' if have_doc else '(raw brief)'} · "
        f"progress analysis {'✓' if have_progress else '(raw report)'} · "
        f"inspection findings {'✓' if have_inspection else '(raw notes)'}"
    )

    if st.button("Generate one-page progress report", type="primary", key="btn_report"):
        with st.spinner("Drafting..."):
            doc_part = st.session_state.get("doc_analysis", load_text("project_brief.txt"))
            progress_part = st.session_state.get("progress_analysis",
                                                   f"Planned 72%, Actual 58%.\n" + load_text("progress_report.txt"))
            inspection_part = st.session_state.get("inspection_report", load_text("inspection_notes.txt"))

            result = ask_agent(
                system=(
                    "You draft one-page engineering progress reports for a supervising "
                    "engineer (SDO) in a Pakistani government engineering department. "
                    "Tone: formal, factual, no marketing language. The report is a "
                    "draft for review - end it with a sign-off block, not a conclusion "
                    "that claims authority the writer doesn't have."
                ),
                user_prompt=(
                    "Draft a one-page engineering progress report from the material "
                    "below. Sections: Project Summary, Progress Status (with the "
                    "planned/actual/variance figures), Key Risks, Recommended Actions, "
                    "and a closing sign-off block reading "
                    "'Reviewed by: ______________  SDO, [Sub-Division]   Date: ______'.\n\n"
                    "PROJECT INFORMATION:\n" + doc_part +
                    "\n\nPROGRESS FINDINGS:\n" + progress_part +
                    "\n\nINSPECTION FINDINGS:\n" + inspection_part
                ),
                max_tokens=1800,
            )
        st.session_state["final_report"] = result

    if "final_report" in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state["final_report"])
        st.download_button(
            "Download report (Markdown)",
            data=st.session_state["final_report"],
            file_name="progress_report_draft.md",
            mime="text/markdown",
        )

# --------------------------------------------------------------------------
# Tab 6 - Delay Risk (ML)
# --------------------------------------------------------------------------

with tabs[5]:
    st.subheader("Delay Risk Predictor")
    st.write("A small regression model, trained live on past department projects, "
             "scores schedule risk on the current one. The agent then explains the "
             "score in engineering terms - it does not decide anything on its own.")

    hist = load_csv("historical_projects.csv").copy()
    hist["variance_pct"] = hist["actual_progress_pct"] - hist["planned_progress_pct"]

    st.dataframe(hist, use_container_width=True, hide_index=True)

    features = ["duration_planned_months", "cost_million_pkr", "variance_pct"]
    X = hist[features].values
    y = hist["delay_months"].values
    model = LinearRegression().fit(X, y)

    st.markdown("##### Current project")
    c1, c2, c3 = st.columns(3)
    cur_duration = c1.number_input("Time allowed (months)", value=10)
    cur_cost = c2.number_input("Approved cost (PKR million)", value=486.5)
    cur_variance = c3.number_input("Progress variance (actual - planned, %)", value=-14)

    x_cur = np.array([[cur_duration, cur_cost, cur_variance]])
    predicted_delay = max(0.0, float(model.predict(x_cur)[0]))

    if predicted_delay >= 5:
        risk = "HIGH"
    elif predicted_delay >= 2:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    c1, c2 = st.columns(2)
    c1.metric("Predicted delay", f"{predicted_delay:.1f} months")
    c2.metric("Risk category", risk)

    st.bar_chart(hist.set_index("project_name")["delay_months"])

    if st.button("Explain this prediction", type="primary", key="btn_ml"):
        with st.spinner("Explaining..."):
            result = ask_agent(
                system=(
                    "You explain a linear regression delay-risk prediction to a "
                    "government engineer who is not a data scientist. Be concrete: "
                    "connect the prediction to the specific evidence given, not "
                    "generic project-management advice. Never claim the model is "
                    "more certain than a small linear regression trained on 14 rows "
                    "actually is - state that limitation plainly."
                ),
                user_prompt=(
                    f"A linear regression trained on {len(hist)} past projects "
                    f"predicts {predicted_delay:.1f} months of delay "
                    f"(risk category: {risk}) for the current project, given: "
                    f"time allowed {cur_duration} months, approved cost "
                    f"PKR {cur_cost} million, progress variance {cur_variance}%.\n\n"
                    "Supporting evidence from the field - contractor's progress "
                    "report and today's site inspection notes:\n\n"
                    + load_text("progress_report.txt") + "\n\n"
                    + load_text("inspection_notes.txt") +
                    "\n\nWrite, as Markdown: a short explanation of why this project "
                    "scores where it does, which 2-3 factors in the evidence above "
                    "matter most, one caveat about the model's limits, and what the "
                    "SDO should check or escalate this week."
                ),
                max_tokens=1200,
            )
        st.session_state["ml_explanation"] = result

    if "ml_explanation" in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state["ml_explanation"])
