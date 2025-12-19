# =========================================
# app.py — Clearly Separated Agentic System
# =========================================

import streamlit as st
import re
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
# PROMPTS (ROLE-BOUND)
# =========================================

JD_REWRITE_PROMPT = ChatPromptTemplate.from_template("""
You are Sub-Agent 1: JD Normalizer.

Your ONLY task is to rewrite job responsibilities into
structured, outcome-focused responsibilities.

STRICT RULES:
- Do NOT generate KPIs
- Do NOT evaluate performance
- Do NOT add commentary
- Each item MUST include:
  action, object, outcome, control_scope
- Output JSON ARRAY ONLY

Job Description:
{jd_text}
""")

KPI_PROMPT = ChatPromptTemplate.from_template("""
You are Sub-Agent 2: KPI Engineer.

Your ONLY task is to generate measurable KPIs.

STRICT RULES:
- Max 3 KPIs
- KPIs MUST be quantifiable
- KPIs MUST include formulas (math only)
- Allowed units: %, count, hours, days
- No subjective words
- KPIs must be controllable by the role
- Output JSON ARRAY ONLY

Structured Responsibility:
{responsibility}
""")

# =========================================
# SUB-AGENT 1 — JD NORMALIZER
# =========================================

def jd_normalizer(jd_text: str) -> List[StructuredResponsibility]:
    response = jd_llm.invoke(
        JD_REWRITE_PROMPT.format(jd_text=jd_text)
    )

    raw = eval(response.content)

    return [
        StructuredResponsibility(**item)
        for item in raw
    ]

# =========================================
# SUB-AGENT 2 — KPI ENGINEER
# =========================================

def kpi_engineer(
    responsibility: StructuredResponsibility
) -> List[KPI]:

    response = kpi_llm.invoke(
        KPI_PROMPT.format(
            responsibility=responsibility.dict()
        )
    )

    raw = eval(response.content)

    return [
        KPI(**item)
        for item in raw
    ]

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
            raise ValueError(
                f"Invalid formula: {kpi.formula}"
            )

        if any(w in kpi.name.lower() for w in forbidden):
            raise ValueError(
                f"Subjective KPI detected: {kpi.name}"
            )

# =========================================
# MAIN AGENT (ORCHESTRATOR — NO LLM)
# =========================================

def main_agent(jd_text: str):
    # Step 1 — Call Sub-Agent 1
    responsibilities = jd_normalizer(jd_text)

    # Step 2 — Call Sub-Agent 2 per responsibility
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
    page_title="JD → KPI (Clearly Separated Agents)",
    layout="wide"
)

st.title("JD → KPI Generator")
st.caption("Main Orchestrator + 2 Explicit Sub-Agents")

jd_text = st.text_area(
    "Paste Job Description",
    height=320
)

if st.button("Run Agent System"):
    if not jd_text.strip():
        st.error("Please paste a job description.")
        st.stop()

    try:
        responsibilities, kpi_output = main_agent(jd_text)

        st.subheader("1️⃣ Structured Responsibilities (Sub-Agent 1)")
        st.json([r.dict() for r in responsibilities])

        st.subheader("2️⃣ KPIs (Sub-Agent 2)")
        st.json(kpi_output)

    except ValidationError as e:
        st.error("Schema validation failed")
        st.code(str(e))

    except Exception as e:
        st.error("Processing failed")
        st.code(str(e))
