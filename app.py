import streamlit as st
import re
import json
import os
from typing import List
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# =========================================
# MODEL CONFIG (VALID + FAST)
# =========================================

JD_MODEL = "gpt-4o-mini"
KPI_MODEL = "gpt-4o-mini"

# Initialize LLMs (will be checked when used)
def get_jd_llm():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it to use this application.")
    return ChatOpenAI(model=JD_MODEL, temperature=0)

def get_kpi_llm():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it to use this application.")
    return ChatOpenAI(model=KPI_MODEL, temperature=0)

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
    
    # Remove markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        # Try to extract JSON from any code block
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part and (part.startswith("{") or part.startswith("[")):
                text = part
                break
    
    # Remove any leading/trailing non-JSON text
    text = text.strip()
    
    # Try to find JSON object/array in the text
    if not (text.startswith("{") or text.startswith("[")):
        # Try to extract JSON from text using regex
        json_match = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', text)
        if json_match:
            text = json_match.group(1)
        else:
            raise ValueError(f"No JSON found in text. First 200 chars: {text[:200]}")
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {str(e)}\nFirst 500 chars: {text[:500]}")

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
- Return ONLY valid JSON, no markdown, no explanations
- Each KPI must have: name, formula, target, unit, data_source, frequency, indicator_type

CRITICAL: Return a JSON array. Each item must be:
{{
  "responsibility_index": 0,
  "kpis": [
    {{
      "name": "Course Completion Rate",
      "formula": "(Completed courses / Total courses) * 100",
      "target": ">= 95%",
      "unit": "%",
      "data_source": "Course records",
      "frequency": "Monthly",
      "indicator_type": "Lagging"
    }}
  ]
}}

Structured Responsibilities:
{responsibilities}

Return ONLY the JSON array, nothing else:
""")

# =========================================
# JD NORMALIZER
# =========================================

def jd_normalizer(jd_text: str) -> List[StructuredResponsibility]:
    jd_llm = get_jd_llm()
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
    kpi_llm = get_kpi_llm()
    
    try:
        response = kpi_llm.invoke(
            KPI_BATCH_PROMPT.format(
                responsibilities=json.dumps([r.model_dump() for r in responsibilities], indent=2)
            )
        )
        
        # Debug: Show raw response in expander
        with st.expander("🔍 Debug: LLM Response (click to view)", expanded=False):
            st.code(response.content, language="json")
        
        if not response.content or not response.content.strip():
            st.warning("⚠️ LLM returned empty response. Using fallback KPIs.")
            raw = []
        else:
            try:
                raw = extract_json(response.content)
                if not isinstance(raw, list):
                    st.warning(f"⚠️ LLM response is not a list. Got: {type(raw)}")
                    raw = []
            except ValueError as e:
                st.warning(f"⚠️ Failed to parse JSON: {str(e)}")
                raw = []
            except Exception as e:
                st.warning(f"⚠️ Unexpected error parsing response: {str(e)}")
                raw = []

        results = {}

        # Build map from batch output
        if isinstance(raw, list):
            for item in raw:
                idx = item.get("responsibility_index")
                if idx is None:
                    continue
                try:
                    kpis = item.get("kpis", [])
                    if kpis:
                        results[idx] = [KPI(**k) for k in kpis]
                except Exception as e:
                    st.warning(f"⚠️ Failed to parse KPIs for index {idx}: {str(e)}")
                    continue

        # Ensure every responsibility gets KPIs
        final = []
        fallback_count = 0
        for i, resp in enumerate(responsibilities):
            kpis = results.get(i)
            if not kpis:
                kpis = fallback_kpis(resp)
                fallback_count += 1

            final.append({
                "responsibility": resp.model_dump(),
                "kpis": [k.model_dump() for k in kpis]
            })
        
        if fallback_count > 0:
            st.info(f"ℹ️ Used fallback KPIs for {fallback_count} out of {len(responsibilities)} responsibilities. Check debug output above.")

        return final
        
    except Exception as e:
        st.error(f"❌ KPI generation failed: {str(e)}")
        # Return fallback for all
        return [{
            "responsibility": resp.model_dump(),
            "kpis": [k.model_dump() for k in fallback_kpis(resp)]
        } for resp in responsibilities]

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

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ **OPENAI_API_KEY environment variable is not set.**")
    st.info("Please set your OpenAI API key as an environment variable to use this application.")
    st.code("export OPENAI_API_KEY='your-api-key-here'", language="bash")
    st.stop()

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
        st.json([r.model_dump() for r in responsibilities])

        st.subheader("2️⃣ KPIs (Batched Generation)")
        st.json(kpi_output)

    except Exception as e:
        st.error("Processing failed")
        st.code(str(e))
