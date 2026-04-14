import streamlit as st
import os, json, requests,re

from pypdf import PdfReader

# --- 1. UI  ---
st.set_page_config(page_title="KACST Satellite ", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #E3F2FD !important; }
    .stApp { background-color: #ffffff; color: #002E5D; }
    .header-text { font-family: 'Open Sans', sans-serif; color: #002E5D; text-align: center; }
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p { color: #002E5D !important; font-weight: 600; }
    [data-testid="stSidebar"] .stRadio > label { color: #002E5D !important; font-weight: bold; }
    .stRadio > div { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #002E5D; }
    .st-success { background-color: #E8F5E9 !important; border: 2px solid #43A047 !important; color: #1B5E20 !important; border-radius: 10px !important; }
    .st-info { background-color: #F3E5F5 !important; border: 2px solid #8E24AA !important; color: #4A148C !important; border-radius: 10px !important; }
    header[data-testid="stHeader"] {
        visibility: hidden;
        height: 0%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Header ---
with st.container():
    col_logo, col_titles = st.columns([1, 4])
    with col_logo:
        if os.path.exists("Kacst_logo.png"): st.image("Kacst_logo.png", width=180)
        else: st.write("🚀 [LOGO NOT FOUND]")
    with col_titles:
        st.markdown("<div class='header-text'><h1>Satellite Safe Mode: QA & Decision Support System</h1></div>", unsafe_allow_html=True)

# --- 3. PDF Extraction ---
@st.cache_resource
def get_operational_data():
    relevant_chunks = []
    pdf_files = [f for f in os.listdir() if f.endswith(".pdf")]
    
    pdf_files.sort(key=lambda x: "NOS3" in x.upper(), reverse=True)
    
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                text = page.extract_text()
                if not text: continue 
                
                lines = text.split('\n')
                for line in lines:
                    if any(cmd in line.lower() for cmd in ['switch', 'disable', 'transition',
                   'voltage', 'connection', 'failure', 'error', 'high temp', 'low temp', 'tumbling', 'reboot', 'safe mode', 'status']):
                        relevant_chunks.append(line.strip())
        except: 
            continue
            
    return "\n".join(relevant_chunks[:100]) 

context = get_operational_data()

# --- 4. Domain Validtion ---
def is_in_domain(query):
    check_prompt = f"Task: Categorize input. Respond with 'YES' if the input is about satellites, satellite safe mode, or technical space anomalies. Respond with 'NO' for anything else.\nInput: {query}\nResponse:"
    try:
        r = requests.post("http://localhost:11434/api/generate", 
                         json={"model": "qwen2.5:1.5b", "prompt": check_prompt, "stream": False, "options": {"temperature": 0.0}})
        return "YES" in r.json().get("response", "").upper()
    except:
        return True # Default to True to avoid blocking during connection issues

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown("---")
    mode = st.radio("SELECT SERVICE:", ["🚨 Immediate Action", "🔍 Q&A"])
    st.markdown("---")

# --- 6. User Input ---
st.markdown("### <span style='color: #002E5D;'>⌨️ Enter a Satellite Scenario or Question: </span>", unsafe_allow_html=True)
status = st.text_input("")

if status:
    # --- Input Validtion ---
    is_gibberish = not re.search(r'\b[a-zA-Z]{3,}\b', status)
    has_symbols_only = not re.search(r'[a-zA-Z0-9]', status)

    if is_gibberish or has_symbols_only:
        st.error("⚠️ Invalid Input: Please enter a valid technical question or scenario.")
    # --- Domain Validtion ---
    elif not is_in_domain(status):
        st.error("❌ Out of Domain: This query does not relate to Satellite Safe Mode or technical operations.")
    else:
        # --- Prompt Generation ---
        if "Immediate Action" in mode:
            prompt = f"""
            SYSTEM: You are a NASA Ground Operator. Use the PROVIDED DATA ONLY.
            DATA: {context}
            USER REPORT: {status}
            REQUIRED RESPONSE FORMAT:
            1. IMMEDIATE ACTION: (What to turn off/on)
            2. MODE CHANGE: (Target Mode)
            3. SAFETY CHECK: (Telemetry to verify)
            STRICT RULE: NO EXPLANATIONS. NO 'DESIGN' TALK. JUST THE CHECKLIST.
            """
            style = "success"
            res_header = " MISSION RECOVERY PROTOCOL"
        else:
            prompt = f"""
            SYSTEM: Technical Expert. Explain the following concept using the provided data.
            DATA: {context}
            QUESTION: {status}
            TASK: 2-3 technical bullet points only.
            RESPONSE:
            """
            style = "info"
            res_header = " TECHNICAL ANALYSIS"
        # --- AI Generation ---
        placeholder = st.empty()
        full_res = ""
        try:
            with requests.post("http://localhost:11434/api/generate", 
                json={"model": "qwen2.5:1.5b", "prompt": prompt, "stream": True, "options": {"temperature": 0.0}}, 
                stream=True) as r:
                for line in r.iter_lines():
                    if line:
                        chunk = json.loads(line.decode('utf-8'))
                        full_res += chunk.get("response", "")
                        
                        if style == "success": 
                         placeholder.success(f"### 🚨 {res_header}\n{full_res}")
                        else: 
                         placeholder.info(f"### 📚 {res_header}\n{full_res}")
        except: 
            st.error("Ollama connection lost. Ensure server is running.")
