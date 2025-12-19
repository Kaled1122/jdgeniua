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

class ResponsibilityKPIs(BaseModel):
    responsibility_index: int
    kpis: List[KPI]

class KPIBatchResponse(BaseModel):
    results: List[ResponsibilityKPIs]

class ResponsibilitiesResponse(BaseModel):
    responsibilities: List[StructuredResponsibility]

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
- Extract ALL responsibilities from the job description
- Rewrite each responsibility in structured format
- NEVER return empty output
- Each item MUST include:
  - action: The verb/action being performed
  - object: What the action is performed on
  - outcome: The expected result or goal
  - control_scope: The area or domain of control

Job Description:
{jd_text}

Extract and structure all responsibilities from the above job description.
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
- Each KPI must be specific to the responsibility (not generic)
- Make KPIs relevant to the action, object, and outcome

Each KPI must have:
- name: Clear, descriptive name
- formula: Mathematical formula for calculation
- target: Specific target value (e.g., ">= 95%", "<= 2 hours")
- unit: Measurement unit (%, count, hours, days)
- data_source: Where the data comes from
- frequency: How often measured (Daily, Weekly, Monthly, Quarterly)
- indicator_type: Leading or Lagging

Structured Responsibilities:
{responsibilities}

Generate specific, measurable KPIs for each responsibility. Make them relevant to the actual work being done.
""")

# =========================================
# JD NORMALIZER
# =========================================

def jd_normalizer(jd_text: str) -> List[StructuredResponsibility]:
    """
    Stage 1: Extract structured responsibilities from job description.
    Uses Model 1 (jd_llm) to parse and structure the responsibilities.
    """
    jd_llm = get_jd_llm()
    
    try:
        # Use structured output for reliable parsing
        structured_llm = jd_llm.with_structured_output(ResponsibilitiesResponse)
        
        response = structured_llm.invoke(
            JD_REWRITE_PROMPT.format(jd_text=jd_text)
        )

        if not response.responsibilities or len(response.responsibilities) == 0:
            raise ValueError("JD Normalizer returned empty results")

        return response.responsibilities
        
    except Exception as e:
        # Fallback to JSON parsing if structured output fails
        try:
            response = jd_llm.invoke(
                JD_REWRITE_PROMPT.format(jd_text=jd_text)
            )
            raw = extract_json(response.content)
            
            if not isinstance(raw, list) or not raw:
                raise ValueError("JD Normalizer failed")
            
            return [StructuredResponsibility(**r) for r in raw]
        except Exception as fallback_error:
            raise ValueError(f"JD Normalizer failed: {str(e)} (fallback also failed: {str(fallback_error)})")

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
    """
    Stage 2: Generate KPIs for each responsibility.
    Uses Model 2 (kpi_llm) to create measurable KPIs from structured responsibilities.
    """
    kpi_llm = get_kpi_llm()
    
    try:
        # Use structured output with Pydantic - much more reliable!
        structured_llm = kpi_llm.with_structured_output(KPIBatchResponse)
        
        # Format responsibilities for the prompt
        responsibilities_json = json.dumps([r.model_dump() for r in responsibilities], indent=2)
        
        response = structured_llm.invoke(
            KPI_BATCH_PROMPT.format(
                responsibilities=responsibilities_json
            )
        )
        
        results = {}
        for item in response.results:
            idx = item.responsibility_index
            if idx is not None and item.kpis:
                results[idx] = item.kpis
        
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
            st.warning(f"⚠️ Used fallback KPIs for {fallback_count} out of {len(responsibilities)} responsibilities.")
        else:
            total_kpis = sum(len(f['kpis']) for f in final)
            st.success(f"✅ Generated {total_kpis} KPIs across {len(final)} responsibilities")

        return final
        
    except Exception as e:
        st.error(f"❌ KPI generation failed: {str(e)}")
        st.exception(e)
        # Return fallback for all
        return [{
            "responsibility": resp.model_dump(),
            "kpis": [k.model_dump() for k in fallback_kpis(resp)]
        } for resp in responsibilities]

# =========================================
# MAIN AGENT (NO AI)
# =========================================

def main_agent(jd_text: str):
    """
    Main pipeline: Job Description → Responsibilities → KPIs
    
    Stage 1: Extract structured responsibilities using Model 1 (jd_llm)
    Stage 2: Generate KPIs using Model 2 (kpi_llm)
    """
    if len(jd_text.strip()) < 50:
        raise ValueError("Job description too short")

    # Stage 1: Extract responsibilities (Model 1)
    with st.spinner("🔍 Stage 1: Extracting responsibilities..."):
        responsibilities = jd_normalizer(jd_text)

    # Limit to avoid too many responsibilities
    responsibilities = responsibilities[:8]
    
    if not responsibilities:
        raise ValueError("No responsibilities extracted from job description")

    # Stage 2: Generate KPIs (Model 2)
    with st.spinner("📊 Stage 2: Generating KPIs..."):
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
        responsibilities, kpi_output = main_agent(jd_text)

        st.subheader("1️⃣ Structured Responsibilities")
        st.json([r.model_dump() for r in responsibilities])

        st.subheader("2️⃣ KPIs (Batched Generation)")
        st.json(kpi_output)
        
        # Export functionality
        st.subheader("3️⃣ Export Results")
        
        # CSV Export
        try:
            import pandas as pd
            import io
            
            # Flatten the data for CSV
            rows = []
            for item in kpi_output:
                resp = item['responsibility']
                resp_text = f"{resp['action']} {resp['object']}"
                for kpi in item['kpis']:
                    rows.append({
                        'Responsibility': resp_text,
                        'Action': resp['action'],
                        'Object': resp['object'],
                        'Outcome': resp['outcome'],
                        'Control Scope': resp['control_scope'],
                        'KPI Name': kpi['name'],
                        'Formula': kpi['formula'],
                        'Target': kpi['target'],
                        'Unit': kpi['unit'],
                        'Data Source': kpi['data_source'],
                        'Frequency': kpi['frequency'],
                        'Indicator Type': kpi['indicator_type']
                    })
            
            df = pd.DataFrame(rows)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                "📥 Download as CSV",
                csv_data,
                "kpis.csv",
                "text/csv",
                key="download-csv"
            )
            
            # Show preview
            with st.expander("📊 Preview CSV Data"):
                st.dataframe(df, use_container_width=True)
                
        except ImportError:
            st.info("💡 Install pandas for CSV export: `pip install pandas`")
        except Exception as e:
            st.warning(f"Export failed: {str(e)}")

    except Exception as e:
        st.error("Processing failed")
        st.code(str(e))
        st.exception(e)
