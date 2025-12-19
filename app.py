# =========================================
# app.py — JD → KPI Agentic System (GPT-5.2)
# =========================================

import streamlit as st
import re
import json
from typing import List
from pydantic import BaseModel, ValidationError

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# =========================================
# MODEL CONFIG (EXPLICIT)
# =========================================

JD_MODEL = "gpt-4o-mini"
KPI_MODEL = "gpt-4o-mini"

# Sub-Agent 1 — JD Normalizer
jd_llm = ChatOpenAI(
    model=JD_MODEL,
    temperature=0
)

# Sub-Agent 2 — KPI Engineer
kpi_llm = ChatOpenAI(
    model=KPI_MODEL,
    temperature=0
)

# =========================================
# SCHEMAS (HARD CONSTRAINTS)
# =========================================

class StructuredResponsibility(BaseModel):
    action: str
    object: str
    outcome: str
    control_scope: str

class KPI(BaseModel):
    name: str
    formula: str
    target: str
    unit: str
    data_source: str
    frequency: str
    indicator_type: str  # leading / lagging

# =========================================
# JSON SAFETY
# =========================================

def extract_json(text: str):
    if not text or not text.strip():
        raise ValueError("Model returned empty output")

    text = text.strip()

    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()

    return json.loads(text)

# =========================================
# PROMPTS
# =========================================

JD_REWRITE_PROMPT = ChatPromptTemplate.from_template("""
You are Sub-Agent 1: JD Normalizer.

Convert the job description into structured responsibilities.

MANDATORY:
- NEVER return empty output
- Rewrite each responsibility even if already in bullet form
- One structured object per responsibility
- Infer outcomes conservatively if implicit

OUTPUT RULES:
- JSON ARRAY ONLY
- Each object MUST include:
  action, object, outcome, control_scope
- No commentary, no markdown

Job Description:
{jd_text}
""")

KPI_PROMPT = ChatPromptTemplate.from_template("""
You are Sub-Agent 2: KPI Engineer.

Generate measurable KPIs ONLY for the responsibility below.

MANDATORY:
- NEVER return empty output
- Max 3 KPIs
- Quantifiable only
- MUST include formulas (math only)
- Allowed units: %, count, hours, days
- No subjective language
- KPI must be controllable by the role
- Output JSON ARRAY ONLY
- No commentary

Structured Responsibility:
{responsibility}
""")

# =========================================
# SUB-AGENT 1 — JD NORMALIZER (RETRY)
# =========================================

def jd_normalizer(jd_text: str, max_retries: int = 2) -> List[StructuredResponsibility]:
    last_error = None

    for _ in range(max_retries + 1):
        response = jd_llm.invoke(
            JD_REWRITE_PROMPT.format(jd_text=jd_text)
        )

        try:
            raw = extract_json(response.content)

            if not isinstance(raw, list) or not raw:
                raise ValueError("JD Normalizer returned invalid JSON")

            return [
                StructuredResponsibility(**item)
                for item in raw
            ]

        except Exception as e:
            last_error = e

    raise RuntimeError(f"JD Normalizer failed: {last_error}")

# =========================================
# KPI FALLBACK (DETERMINISTIC)
# =========================================

def fallback_kpis(resp: StructuredResponsibility) -> List[KPI]:
    outcome = resp.outcome.lower()

    if "record" in outcome or "data" in outcome:
        return [KPI(
            name="Record Accuracy Rate",
            formula="(Correct records / Total records audited) * 100",
            target=">= 99%",
            unit="%",
            data_source="TMIS / Audit logs",
            frequency="Quarterly",
            indicator_type="Lagging"
        )]

    if "test" in outcome or "assessment" in outcome:
        return [KPI(
            name="On-Time Test Delivery Rate",
            formula="(Tests delivered on time / Total scheduled tests) * 100",
            target=">= 98%",
            unit="%",
            data_source="TMIS / Test schedule",
            frequency="Monthly",
            indicator_type="Lagging"
        )]

    if "teach" in outcome or "instruction" in outcome:
        return [KPI(
            name="Lesson Delivery Completion Rate",
            formula="(Lessons delivered / Lessons scheduled) * 100",
            target=">= 95%",
            unit="%",
            data_source="Timetable / LMS",
            frequency="Monthly",
            indicator_type="Lagging"
        )]

    if "compliance" in outcome or "procedure" in outcome:
        return [KPI(
            name="Compliance Adherence Rate",
            formula="(Compliant activities / Activities audited) * 100",
            target="100%",
            unit="%",
            data_source="Compliance reports",
            frequency="Quarterly",
            indicator_type="Lagging"
        )]

    return [KPI(
        name="Task Completion Rate",
        formula="(Completed tasks / Assigned tasks) * 100",
        target=">= 95%",
        unit="%",
        data_source="Department records",
        frequency="Monthly",
        indicator_type="Lagging"
    )]

# =========================================
# SUB-AGENT 2 — KPI ENGINEER (RETRY + FALLBACK)
# =========================================

def kpi_engineer(
    responsibility: StructuredResponsibility,
    max_retries: int = 2
) -> List[KPI]:

    last_error = None

    for _ in range(max_retries + 1):
        response = kpi_llm.invoke(
            KPI_PROMPT.format(responsibility=responsibility.dict())
        )

        try:
            raw = extract_json(response.content)

            if not isinstance(raw, list) or not raw:
                raise ValueError("KPI Engineer returned invalid JSON")

            return [KPI(**item) for item in raw]

        except Exception as e:
            last_error = e

    # Guaranteed safe fallback
    return fallback_kpis(responsibility)

# =========================================
# VALIDATION (ANTI-SUBJECTIVITY)
# =========================================

def validate_kpis(kpis: List[KPI]):
    forbidden = ["quality", "effective", "initiative", "excellent", "good", "poor"]

    for kpi in kpis:
        if not re.search(r"[*/+\-%]", kpi.formula):
            raise ValueError(f"Invalid formula: {kpi.formula}")

        if any(word in kpi.name.lower() for word in forbidden):
            raise ValueError(f"Subjective KPI detected: {kpi.name}")

# =========================================
# MAIN AGENT (NO AI)
# =========================================

def main_agent(jd_text: str):
    if len(jd_text.strip()) < 50:
        raise ValueError("Job description too short")

    responsibilities = jd_normalizer(jd_text)
    output = []

    for resp in responsibilities:
        kpis = kpi_engineer(resp)
        validate_kpis(kpis)

        output.append({
            "responsibility": resp.dict(),
            "kpis": [k.dict() for k in kpis]
        })

    return responsibilities, output

# =========================================
# STREAMLIT UI
# =========================================

st.set_page_config(page_title="JD → KPI Generator (GPT-5.2)", layout="wide")

st.title("JD → KPI Generator")
st.caption("Main Orchestrator + 2 Explicit Sub-Agents (GPT-5.2, Hardened)")

st.info("Paste JOB PURPOSE + RESPONSIBILITIES. Bullet points are fine.")

jd_text = st.text_area(
    "Paste Job Description",
    height=360,
    placeholder="Paste the full job description here..."
)

if st.button("Run Agent System"):
    if not jd_text.strip():
        st.error("Please paste a job description.")
        st.stop()

    try:
        responsibilities, kpi_output = main_agent(jd_text)

        st.subheader("1️⃣ Structured Responsibilities")
        st.json([r.dict() for r in responsibilities])

        st.subheader("2️⃣ KPIs")
        st.json(kpi_output)

    except Exception as e:
        st.error("Processing failed")
        st.code(str(e))
