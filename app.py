import streamlit as st
import re
import json
from typing import List
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# =========================================
# MODEL CONFIG (VALID + FAST)
# =========================================

JD_MODEL = "gpt-4o-mini"
KPI_MODEL = "gpt-4o-mini"

jd_llm = ChatOpenAI(model=JD_MODEL, temperature=0)
kpi_llm = ChatOpenAI(model=KPI_MODEL, temperature=0)

# =========================================
# SCHEMAS
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
    indicator_type: str

# =========================================
# JSON SAFETY
# =========================================

def extract_json(text: str):
    if not text or not text.strip():
        raise ValueError("Empty model output")

    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].strip()

    return json.loads(text)

# =========================================
# PROMPTS
# =========================================

JD_REWRITE_PROMPT = ChatPromptTemplate.from_template("""
You are Sub-Agent 1: JD Normalizer.

Convert the job description into structured responsibilities.

RULES:
- Rewrite each responsibility
- NEVER return empty output
- JSON ARRAY ONLY
- Each item MUST include:
  action, object, outcome, control_scope

Job Description:
{jd_text}
""")

KPI_BATCH_PROMPT = ChatPromptTemplate.from_template("""
You are Sub-Agent 2: KPI Engineer.

Generate measurable KPIs for EACH responsibility below.

RULES:
- Max 3 KPIs per responsibility
- Quantifiable only
- MUST include formulas
- Allowed units: %, count, hours, days
- No subjective language
- JSON ARRAY ONLY
- Each item must be:
  {{
    "responsibility_index": number,
    "kpis": [ ... ]
  }}

Structured Responsibilities:
{responsibilities}
""")

# =========================================
# JD NORMALIZER
# =========================================

def jd_normalizer(jd_text: str) -> List[StructuredResponsibility]:
    response = jd_llm.invoke(
        JD_REWRITE_PROMPT.format(jd_text=jd_text)
    )
    raw = extract_json(response.content)

    if not isinstance(raw, list) or not raw:
        raise ValueError("JD Normalizer failed")

    return [StructuredResponsibility(**r) for r in raw]

# =========================================
# KPI FALLBACK (DETERMINISTIC)
# =========================================

def fallback_kpis(resp: StructuredResponsibility) -> List[KPI]:
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
# BATCHED KPI ENGINE (FAST FIX)
# =========================================

def batched_kpi_engine(responsibilities: List[StructuredResponsibility]):

    response = kpi_llm.invoke(
        KPI_BATCH_PROMPT.format(
            responsibilities=[r.dict() for r in responsibilities]
        )
    )

    try:
        raw = extract_json(response.content)
    except Exception:
        raw = []

    results = {}

    # Build map from batch output
    if isinstance(raw, list):
        for item in raw:
            idx = item.get("responsibility_index")
            try:
                results[idx] = [KPI(**k) for k in item["kpis"]]
            except Exception:
                pass

    # Ensure every responsibility has KPIs
    final = []
    for i, resp in enumerate(responsibilities):
        kpis = results.get(i) or fallback_kpis(resp)
        final.append({
            "responsibility": resp.dict(),
            "kpis": [k.dict() for k in kpis]
        })

    return final

# =========================================
# MAIN AGENT (NO AI)
# =========================================

def main_agent(jd_text: str):
    if len(jd_text.strip()) < 50:
        raise ValueError("Job description too short")

    responsibilities = jd_normalizer(jd_text)

    # Limit to avoid junk bullets
    responsibilities = responsibilities[:8]

    kpi_output = batched_kpi_engine(responsibilities)

    return responsibilities, kpi_output

# =========================================
# STREAMLIT UI
# =========================================

st.set_page_config(page_title="JD → KPI Generator (FAST)", layout="wide")

st.title("JD → KPI Generator")
st.caption("Batched KPI Engine • Fast • Stable • Railway-Ready")

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
        with st.spinner("Processing job description..."):
            responsibilities, kpi_output = main_agent(jd_text)

        st.subheader("1️⃣ Structured Responsibilities")
        st.json([r.dict() for r in responsibilities])

        st.subheader("2️⃣ KPIs (Batched Generation)")
        st.json(kpi_output)

    except Exception as e:
        st.error("Processing failed")
        st.code(str(e))
