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
- MUST include detailed, understandable formulas with explanations
- Allowed units: %, count, hours, days
- No subjective language
- Each KPI must be specific to the responsibility (not generic)
- Make KPIs relevant to the action, object, and outcome

FORMULA REQUIREMENTS:
- Formulas must be clear and easy to understand
- Include variable definitions (what each part means)
- Use simple mathematical operations (+, -, *, /)
- Format: "Formula: (X / Y) * 100 | X = description of X, Y = description of Y"
- Make it clear what each variable represents

Each KPI must have:
- name: Clear, descriptive name
- formula: Detailed mathematical formula with variable explanations. Format: "Formula: (X / Y) * 100 | X = what X represents, Y = what Y represents"
- target: Specific target value (e.g., ">= 95%", "<= 2 hours")
- unit: Measurement unit (%, count, hours, days)
- data_source: Where the data comes from (be specific)
- frequency: How often measured (Daily, Weekly, Monthly, Quarterly)
- indicator_type: Leading or Lagging

FORMULA EXAMPLES:
Good: "Formula: (Courses Completed On Time / Total Courses Scheduled) * 100 | Courses Completed On Time = number of courses finished by deadline, Total Courses Scheduled = all courses planned for the period"
Good: "Formula: (Number of Observations Completed / Total Team Members) | Number of Observations Completed = performance reviews conducted, Total Team Members = size of team being managed"
Bad: "(A/B)*100" (too vague, no explanation)

Structured Responsibilities:
{responsibilities}

Generate specific, measurable KPIs for each responsibility. Make formulas detailed, clear, and easy to understand with full variable explanations.
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

st.set_page_config(
    page_title="JD → KPI Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 {
        color: white;
        margin: 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .kpi-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .responsibility-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1565a0 0%, #e66f00 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 JD → KPI Generator</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">Transform Job Descriptions into Measurable KPIs</p>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Batched KPI Engine • Fast • Stable • Railway-Ready</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.markdown("### 📋 About")
    st.info("""
    This tool uses AI to:
    1. Extract structured responsibilities from job descriptions
    2. Generate measurable KPIs for each responsibility
    
    **Two-Stage Pipeline:**
    - Stage 1: Responsibility Extraction (Model 1)
    - Stage 2: KPI Generation (Model 2)
    """)
    
    st.markdown("### 🔑 API Status")
    if os.getenv("OPENAI_API_KEY"):
        st.success("✅ API Key Configured")
    else:
        st.error("❌ API Key Missing")
        st.code("export OPENAI_API_KEY='your-key'", language="bash")
    
    st.markdown("### 📊 Model Info")
    st.caption(f"**JD Model:** {JD_MODEL}")
    st.caption(f"**KPI Model:** {KPI_MODEL}")
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Paste complete job descriptions
    - Include all responsibilities
    - Bullet points work fine
    - Minimum 50 characters required
    """)

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ **OPENAI_API_KEY environment variable is not set.**")
    st.info("Please set your OpenAI API key as an environment variable to use this application.")
    st.code("export OPENAI_API_KEY='your-api-key-here'", language="bash")
    st.stop()

# Main input area
st.header("📝 Input Job Description")
st.info("💡 **Tip:** Paste JOB PURPOSE + RESPONSIBILITIES. Bullet points are fine.")

jd_text = st.text_area(
    "Paste Job Description",
    height=360,
    placeholder="Paste the full job description here...\n\nExample:\n• Manage team of 10 developers\n• Deliver projects on time\n• Ensure code quality standards",
    help="Enter a complete job description with responsibilities"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run_button = st.button("🚀 Generate KPIs", type="primary", use_container_width=True)

if run_button:
    if not jd_text.strip():
        st.error("❌ Please paste a job description.")
        st.stop()
    
    if len(jd_text.strip()) < 50:
        st.warning("⚠️ Job description seems too short. Please provide more details.")
        st.stop()

    try:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        responsibilities, kpi_output = main_agent(jd_text)
        
        progress_bar.progress(100)
        status_text.empty()
        
        # Success message
        total_kpis = sum(len(item['kpis']) for item in kpi_output)
        st.success(f"✅ Successfully generated {total_kpis} KPIs across {len(responsibilities)} responsibilities!")
        
        # Metrics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📋 Responsibilities", len(responsibilities))
        with col2:
            st.metric("📊 Total KPIs", total_kpis)
        with col3:
            avg_kpis = round(total_kpis / len(responsibilities), 1) if responsibilities else 0
            st.metric("📈 Avg KPIs/Responsibility", avg_kpis)
        with col4:
            st.metric("✅ Status", "Complete")
        
        st.markdown("---")
        
        # Section 1: Structured Responsibilities
        st.header("1️⃣ Structured Responsibilities")
        st.caption("Responsibilities extracted and structured from the job description")
        
        # Display responsibilities in cards
        for idx, resp in enumerate(responsibilities):
            with st.container():
                st.markdown(f"""
                <div class="responsibility-card">
                    <h3 style="margin: 0; color: white;">Responsibility {idx + 1}</h3>
                    <p style="margin: 0.5rem 0; font-size: 1.1rem; font-weight: bold;">{resp.action} {resp.object}</p>
                    <p style="margin: 0.3rem 0; opacity: 0.9;"><strong>Outcome:</strong> {resp.outcome}</p>
                    <p style="margin: 0.3rem 0; opacity: 0.9;"><strong>Scope:</strong> {resp.control_scope}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Collapsible JSON view
        with st.expander("🔍 View Raw JSON"):
            st.json([r.model_dump() for r in responsibilities])
        
        st.markdown("---")
        
        # Section 2: KPIs
        st.header("2️⃣ Generated KPIs")
        st.caption("Measurable KPIs for each responsibility")
        
        # Display KPIs in a more readable format
        for idx, item in enumerate(kpi_output):
            resp = item['responsibility']
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;">
                <h3 style="margin: 0; color: white;">📋 Responsibility {idx + 1}: {resp['action']} {resp['object']}</h3>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;"><strong>Outcome:</strong> {resp['outcome']}</p>
                <p style="margin: 0.3rem 0 0 0; opacity: 0.9;"><strong>Scope:</strong> {resp['control_scope']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # KPI cards
            for kpi_idx, kpi in enumerate(item['kpis']):
                with st.container():
                    st.markdown(f"""
                    <div class="kpi-card">
                        <h4 style="margin: 0 0 1rem 0; color: #1f77b4;">📊 KPI {kpi_idx + 1}: {kpi['name']}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**📐 Formula**")
                        # Split formula if it has explanations (| separator)
                        if " | " in kpi['formula']:
                            formula_parts = kpi['formula'].split(" | ", 1)
                            st.code(formula_parts[0], language="text")
                            st.markdown("**Variable Definitions:**")
                            st.info(formula_parts[1])
                        else:
                            st.code(kpi['formula'], language="text")
                        
                        st.markdown("**🎯 Target**")
                        st.success(kpi['target'])
                        
                        st.markdown("**📏 Unit**")
                        st.info(kpi['unit'])
                    with col2:
                        st.markdown("**📂 Data Source**")
                        st.info(kpi['data_source'])
                        
                        st.markdown("**🔄 Frequency**")
                        st.info(kpi['frequency'])
                        
                        st.markdown("**📈 Indicator Type**")
                        if kpi['indicator_type'].lower() == 'leading':
                            st.success(f"🟢 {kpi['indicator_type']}")
                        else:
                            st.info(f"🔵 {kpi['indicator_type']}")
            
            st.divider()
        
        # Collapsible JSON view
        with st.expander("🔍 View Raw JSON Output"):
            st.json(kpi_output)
        
        st.markdown("---")
        
        # Section 3: Export
        st.header("3️⃣ Export Results")
        st.caption("Download your KPIs in various formats")
        
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
            
            # Export buttons in columns
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download as CSV",
                    csv_data,
                    "kpis.csv",
                    "text/csv",
                    key="download-csv",
                    use_container_width=True,
                    type="primary"
                )
            with col2:
                # JSON export
                json_data = json.dumps(kpi_output, indent=2)
                st.download_button(
                    "📄 Download as JSON",
                    json_data,
                    "kpis.json",
                    "application/json",
                    key="download-json",
                    use_container_width=True
                )
            
            # Show preview
            with st.expander("📊 Preview Data Table"):
                st.dataframe(df, use_container_width=True, height=400)
                
        except ImportError:
            st.warning("💡 Install pandas for CSV export: `pip install pandas`")
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")

    except Exception as e:
        st.error("❌ Processing failed")
        st.error(f"**Error:** {str(e)}")
        
        with st.expander("🔍 Error Details"):
            st.exception(e)
        
        st.info("💡 **Tips to fix:**\n- Check your API key is set correctly\n- Ensure job description is complete\n- Try with a shorter description first")
