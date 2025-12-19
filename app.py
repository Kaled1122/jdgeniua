# =========================================
# app.py — JD → KPI Agentic System (HARDENED)
# =========================================

import streamlit as st
import re
import json
from typing import List
from pydantic import BaseModel, ValidationError

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# =========================================
# SUB-AGENT MODELS (EXPLICIT SEPARATION)
# =========================================

# Sub-Agent 1 — JD Normalizer
jd_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Sub-Agent 2 — KPI Engineer
kpi_llm = ChatOpenAI(
    model="gpt-4o-mini",
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
# JSON SAFETY (CRITICAL)
# =========================================

def extract_json(text: str):
    if not text or not text.strip():
        raise ValueError("Model returned empty output")

    text = text.strip()

    # Remove markdown fences if present
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()

    return json.loads(text)

# =========================================
# PROMPTS (ROLE-BOUND)
# =========================================

JD_REWRITE_PROMPT = ChatPromptTemplate.from_template("""
You are Sub-Agent 1: JD Normalizer.

Convert the following job description into structured responsibilities.

MANDATORY RULES:
- NEVER return empty output
- Even if input is already a bullet list, rewrite each item
- Output one structured object per responsibility
- Infer outcomes conservatively if not explicit

STRICT OUTPUT FORMAT:
- JSON ARRAY ONLY
- Each object MUST contain:
  action, object, outcome, control_scope
- No commentary, no markdown

Job Description:
{jd_text}
""")

KPI_PROMPT = ChatPromptTemplate.from_template("""
You are Sub-Agent 2: KPI Engineer.

Generate measurable KPIs ONLY for the given responsibility.

MANDATORY RULES:
- NEVER return empty output
- Max 3 KPIs
- KPIs MUST be quantifiable
- KPIs MUST include formulas (math only)
- Allowed units: %, count, hours, days
- No subjective words (quality, effectiveness, initiative, etc.)
- KPIs must be controllable by the role
- Output JSON ARRAY ONLY
- No commentary

Structured Responsibility:
{responsibility}
""")

# =========================================
# SUB-AGENT 1 — JD NORMALIZER (WITH RETRY)
# =========================================

def jd_normalizer(jd_text: str, max_retries: int = 2) -> List[StructuredResponsibility]:
    last_error = None

    for attempt in range(max_retries + 1):
        response = jd_llm.invoke(
            JD_REWRITE_PROMPT.format(jd_text=jd_text)
        )

        try:
            raw = extract_json(response.content)

            if not isinstance(raw, list) or len(raw) == 0:
                raise ValueError("JD Normalizer returned empty or invalid JSON")

            return [
                StructuredResponsibility(**item)
                for item in raw
            ]

        except Exception as e:
            last_error = e

    raise RuntimeError(
        f"JD Normalizer failed after {max_retries + 1} attempts: {last_error}"
    )

# =========================================
# SUB-AGENT 2 — KPI ENGINEER (WITH RETRY)
# =========================================

def kpi_engineer(
    responsibility: StructuredResponsibility,
    max_retries: int = 2
) -> List[KPI]:

    last_error = None

    for attempt in range(max_retries + 1):
        response = kpi_llm.invoke(
            KPI_PROMPT.format(
                responsibility=responsibility.dict()
            )
        )

        try:
            raw = extract_json(response.content)

            if not isinstance(raw, list) or len(raw) == 0:
                raise ValueError("KPI Engineer returned empty or invalid JSON")

            return [
                KPI(**item)
                for item in raw
            ]

        except Exception as e:
            last_error = e

    raise RuntimeError(
        f"KPI Engineer failed after {max_retries + 1} attempts: {last_error}"
    )

# =========================================
# VALIDATION LAYER (NO AI)
# =========================================

def validate_kpis(kpis: List[KPI]):
    forbidden = [
        "quality", "effective",
        "initiative", "excellent",
        "good", "poor"
    ]

    for kpi in kpis:
        if not re.search(r"[*/+\-÷%]", kpi.formula):
            raise ValueError(f"Invalid formula: {kpi.formula}")

        if any(w in kpi.name.lower() for w in forbidden):
            raise ValueError(f"Subjective KPI detected: {kpi.name}")

# =========================================
# MAIN AGENT (ORCHESTRATOR — NO AI)
# =========================================

def main_agent(jd_text: str):
    if len(jd_text.strip()) < 50:
        raise ValueError("Job description too short to extract responsibilities")

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

st.set_page_config(
    page_title="JD → KPI Generator",
    layout="wide"
)

st.title("JD → KPI Generator")
st.caption("Main Orchestrator + 2 Explicit Sub-Agents (Hardened & Deterministic)")

st.info(
    "Paste JOB PURPOSE + RESPONSIBILITIES. "
    "Bullet points are fine. Do NOT paste qualifications only."
)

jd_text = st.text_area(
    "Paste Job Description",
    height=340,
    placeholder="Paste the full job description here..."
)

if st.button("Run Agent System"):
    if not jd_text.strip():
        st.error("Please paste a job description.")
        st.stop()

    try:
        responsibilities, kpi_output = main_agent(jd_text)

        st.subheader("1️⃣ Structured Responsibilities (JD Normalizer)")
        st.json([r.dict() for r in responsibilities])

        st.subheader("2️⃣ KPIs (KPI Engineer)")
        st.json(kpi_output)

    except ValidationError as e:
        st.error("Schema validation failed")
        st.code(str(e))

    except Exception as e:
        st.error("Processing failed")
        st.code(str(e))
