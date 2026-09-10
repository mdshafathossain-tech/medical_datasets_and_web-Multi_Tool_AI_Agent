"""
app.py
------
Streamlit UI for the Multi-Tool Medical AI Agent.

Provides a modern, chat-based interface over the routing agent defined
in agent.py, with:
    - Custom "medical" themed CSS (soft cyan/blue gradients, rounded cards)
    - A sidebar with project info, live status indicators, and clickable
      sample questions
    - Persistent chat history via st.session_state
    - An expandable "Thinking & Tool Details" panel per response, showing
      exactly which tool(s) the agent called and what they returned
    - Graceful error handling and loading spinners

Run with:
    streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Env loading -- must happen before importing agent/tools/db_setup, since
# they read GROQ_API_KEY etc. from the environment at import time.
# --------------------------------------------------------------------------
load_dotenv()

from db_setup import build_all_databases, DATASET_CONFIG

# --------------------------------------------------------------------------
# Build the SQLite databases BEFORE importing agent/tools.
#
# tools.py constructs its 3 SQL tools at *import time* (module-level code),
# and each one requires its .db file to already exist on disk. On a fresh
# deploy (e.g. Streamlit Cloud), the databases/ folder doesn't exist yet
# (it's git-ignored, since it's meant to be generated, not committed) --
# so this build step must run before `from agent import ...` below, or
# that import raises FileNotFoundError before the app ever gets a chance
# to build the databases itself.
#
# This intentionally runs before st.set_page_config(), since it's plain
# Python with no Streamlit calls -- st.set_page_config() must still be
# the *first* Streamlit command, which it is, just below.
# --------------------------------------------------------------------------
db_init_error = None
try:
    build_all_databases(force_rebuild=False)
    databases_ready = True
except Exception as exc:  # noqa: BLE001
    databases_ready = False
    db_init_error = str(exc)

# Only import agent/tools (which build the SQL tools) if the databases
# were built successfully -- otherwise skip straight to showing the error
# banner further down, instead of crashing with a second, confusing error.
if databases_ready:
    from agent import query_agent_with_trace
else:
    query_agent_with_trace = None


# ==========================================================================
# Page config (must be the first Streamlit command)
# ==========================================================================
st.set_page_config(
    page_title="Medical Multi-Tool AI Agent",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================================
# Custom CSS -- soft cyan/blue medical theme, rounded cards, chat bubbles
# ==========================================================================
CUSTOM_CSS = """
<style>
    /* ---- Global background & typography ---- */
    .stApp {
        background: linear-gradient(160deg, #f0fbfc 0%, #e8f4fb 45%, #eef7f5 100%);
        font-family: 'Segoe UI', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide default Streamlit chrome for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ---- Main title / hero card ---- */
    .hero-card {
        background: linear-gradient(120deg, #0891b2 0%, #06b6d4 45%, #22d3ee 100%);
        padding: 1.75rem 2rem;
        border-radius: 20px;
        color: white;
        box-shadow: 0 8px 24px rgba(6, 182, 212, 0.25);
        margin-bottom: 1.5rem;
    }
    .hero-card h1 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .hero-card p {
        margin: 0.35rem 0 0 0;
        font-size: 0.95rem;
        opacity: 0.92;
    }

    /* ---- Sidebar styling ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f0fbfc 100%);
        border-right: 1px solid #d5eef2;
    }
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.5rem 0 1rem 0;
    }
    .sidebar-logo .icon {
        font-size: 2.1rem;
    }
    .sidebar-logo .title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0e7490;
        line-height: 1.2;
    }
    .sidebar-logo .subtitle {
        font-size: 0.75rem;
        color: #64748b;
    }

    /* Status pill cards in the sidebar */
    .status-card {
        background: white;
        border: 1px solid #d5eef2;
        border-radius: 12px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.55rem;
        font-size: 0.85rem;
        font-weight: 500;
        color: #0f172a;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    /* Section labels in the sidebar */
    .sidebar-section-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #0e7490;
        margin: 1.1rem 0 0.5rem 0;
    }

    /* ---- Sample question buttons ---- */
    div[data-testid="stSidebar"] .stButton button {
        width: 100%;
        text-align: left;
        background: white;
        border: 1px solid #d5eef2;
        border-radius: 10px;
        color: #0f172a;
        font-size: 0.82rem;
        padding: 0.55rem 0.75rem;
        margin-bottom: 0.4rem;
        transition: all 0.15s ease;
    }
    div[data-testid="stSidebar"] .stButton button:hover {
        background: #ecfeff;
        border-color: #22d3ee;
        color: #0e7490;
        transform: translateX(2px);
    }

    /* "Clear Chat" button -- make it stand out slightly differently */
    div[data-testid="stSidebar"] .clear-chat-btn button {
        background: #fff1f2;
        border-color: #fecdd3;
        color: #be123c;
        font-weight: 600;
        text-align: center;
    }
    div[data-testid="stSidebar"] .clear-chat-btn button:hover {
        background: #ffe4e6;
        border-color: #fda4af;
    }

    /* ---- Chat bubbles ---- */
    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 0.25rem 0.4rem;
        margin-bottom: 0.6rem;
    }

    /* ---- Expander (Thinking & Tool Details) ---- */
    div[data-testid="stExpander"] {
        border: 1px solid #d5eef2;
        border-radius: 12px;
        background: #f8fefe;
    }

    /* ---- Chat input box ---- */
    div[data-testid="stChatInput"] {
        border-radius: 14px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================================================
# Dynamic DB initialization (runs once per session, cached across reruns)
# ==========================================================================
# ==========================================================================
# Session state setup
# ==========================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ..., "trace": [...]}

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ==========================================================================
# Sidebar
# ==========================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
            <div class="icon">🩺</div>
            <div>
                <div class="title">Medical AI Agent</div>
                <div class="subtitle">Multi-Tool Dataset &amp; Web Assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Status indicators ----
    st.markdown('<div class="sidebar-section-label">System Status</div>', unsafe_allow_html=True)

    if databases_ready:
        st.markdown('<div class="status-card">🟢 Databases Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-card">🔴 Database Setup Failed</div>', unsafe_allow_html=True)

    groq_key_present = bool(os.getenv("GROQ_API_KEY"))
    if groq_key_present:
        st.markdown('<div class="status-card">⚡ LLM Engine: Groq API</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-card">⚠️ GROQ_API_KEY not set</div>', unsafe_allow_html=True)

    st.markdown('<div class="status-card">🔎 Web Search: DuckDuckGo (free)</div>', unsafe_allow_html=True)

    dataset_names = ", ".join(name.capitalize() for name in DATASET_CONFIG.keys())
    st.markdown(f'<div class="status-card">📊 Datasets: {dataset_names}</div>', unsafe_allow_html=True)

    # ---- Quick usage guide / sample questions ----
    st.markdown('<div class="sidebar-section-label">Try Asking</div>', unsafe_allow_html=True)

    sample_questions = [
        "What is the average cholesterol level in the heart disease dataset?",
        "How many malignant cases are in the cancer dataset?",
        "What is the average BMI for diabetic patients?",
        "What are the common symptoms of type 2 diabetes?",
        "What causes high blood pressure?",
    ]

    for i, question in enumerate(sample_questions):
        if st.button(question, key=f"sample_q_{i}", use_container_width=True):
            st.session_state.pending_prompt = question
            st.rerun()

    # ---- Clear chat history ----
    st.markdown('<div class="sidebar-section-label">Session</div>', unsafe_allow_html=True)
    st.markdown('<div class="clear-chat-btn">', unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat History", key="clear_chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="sidebar-section-label">About</div>
        <div class="status-card" style="font-weight:400; line-height:1.4;">
            This assistant routes your question to a dataset-specific SQL
            tool (Heart Disease, Cancer, Diabetes) or to a web search tool
            for general medical knowledge -- automatically.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================================
# Main area -- hero header
# ==========================================================================
st.markdown(
    """
    <div class="hero-card">
        <h1>🩺 Medical Multi-Tool AI Agent</h1>
        <p>Ask about Heart Disease, Cancer, or Diabetes datasets -- or any general medical question.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not databases_ready:
    st.error(
        "⚠️ Could not initialize the medical databases. "
        f"Details: {db_init_error}\n\n"
        "Make sure your CSV files exist in the `data/` folder and that "
        "`db_setup.py` runs without errors, then restart the app."
    )
    st.stop()

if not groq_key_present:
    st.error(
        "⚠️ GROQ_API_KEY is not set. Add it to your `.env` file "
        "(GROQ_API_KEY=your_key_here) and restart the app to enable the AI agent."
    )
    st.stop()


# ==========================================================================
# Render existing chat history
# ==========================================================================
for message in st.session_state.messages:
    avatar = "🧑‍⚕️" if message["role"] == "user" else "🩺"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

        # Show tool trace (if any) for past assistant messages too, so the
        # transparency panel persists across reruns, not just for the
        # most recent answer.
        trace = message.get("trace")
        if trace:
            with st.expander("🔍 Thinking & Tool Details"):
                for step in trace:
                    if step["type"] == "tool_call":
                        st.markdown(f"**🔧 Called tool:** `{step['tool']}`")
                        st.code(str(step["input"]), language="text")
                    elif step["type"] == "tool_result":
                        st.markdown(f"**📄 Result from `{step['tool']}`:**")
                        st.code(step["output"], language="text")


# ==========================================================================
# Handle a new question -- either typed, or from a sidebar sample button
# ==========================================================================
typed_prompt = st.chat_input("Ask about heart disease, cancer, diabetes, or general medical topics...")
prompt = st.session_state.pending_prompt or typed_prompt
st.session_state.pending_prompt = None  # consume it so it doesn't repeat on next rerun

if prompt:
    # Show and store the user's message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(prompt)

    # Run the agent and show the response
    with st.chat_message("assistant", avatar="🩺"):
        try:
            with st.spinner("🔬 Analyzing dataset & query..."):
                answer, trace = query_agent_with_trace(prompt)

            st.markdown(answer)

            if trace:
                with st.expander("🔍 Thinking & Tool Details"):
                    for step in trace:
                        if step["type"] == "tool_call":
                            st.markdown(f"**🔧 Called tool:** `{step['tool']}`")
                            st.code(str(step["input"]), language="text")
                        elif step["type"] == "tool_result":
                            st.markdown(f"**📄 Result from `{step['tool']}`:**")
                            st.code(step["output"], language="text")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "trace": trace}
            )

        except Exception as exc:  # noqa: BLE001
            error_message = (
                "⚠️ Something went wrong while processing your question. "
                f"Details: {exc}"
            )
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message, "trace": []}
            )