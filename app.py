"""
RAG AI Assistant — Main Streamlit Application.

A production-ready RAG chatbot that lets users upload PDFs,
ask questions, and get AI-powered answers with source citations.

Run with:  streamlit run app.py
"""

import streamlit as st
from rag.ingest import ingest_pdfs
from rag.chat import ask_question
from rag.memory import create_memory, save_to_memory, clear_memory
from utils.helpers import load_environment, validate_environment, format_source_documents


# =========================================================================
# Page Config & Environment
# =========================================================================
st.set_page_config(
    page_title="DocuRAG AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_environment()

# =========================================================================
# Custom CSS — Ultra Premium Dark Glassmorphism Theme
# =========================================================================
st.markdown("""
<style>
/* ---- Import Google Fonts ---- */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* ---- Global ---- */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}
.block-container { 
    padding-top: 3rem; 
    padding-bottom: 5rem;
    max-width: 900px;
}

/* Background gradient */
.stApp {
    background: radial-gradient(circle at 15% 50%, rgba(124, 58, 237, 0.08), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(6, 182, 212, 0.08), transparent 25%);
    background-color: #09090b; /* Deep zinc-950 */
}

/* ---- Main Header ---- */
.main-header {
    text-align: center;
    padding: 1rem 0 2.5rem 0;
}
.main-header h1 {
    background: linear-gradient(135deg, #A78BFA 0%, #38BDF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    letter-spacing: -0.03em;
}
.main-header p {
    color: #94A3B8;
    font-size: 1.2rem;
    font-weight: 300;
    margin-top: 0;
}

/* ---- Center Upload Zone (Glassmorphism) ---- */
.upload-container {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.4);
    border-radius: 1.5rem;
    padding: 2.5rem;
    margin: 1rem auto 3rem auto;
    transition: all 0.3s ease;
}
.upload-container:hover {
    border: 1px solid rgba(167, 139, 250, 0.2);
    box-shadow: 0 10px 40px 0 rgba(124, 58, 237, 0.1);
}
.upload-header {
    text-align: center;
    color: #E2E8F0;
    font-size: 1.4rem;
    font-weight: 500;
    margin-bottom: 1.5rem;
}

/* ---- Source Citation Card (Glassmorphism) ---- */
.source-card {
    background: rgba(30, 27, 75, 0.3);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(124, 58, 237, 0.15);
    border-left: 4px solid #8B5CF6;
    border-radius: 0.75rem;
    padding: 1.25rem;
    margin: 0.75rem 0;
    transition: all 0.2s ease;
}
.source-card:hover { 
    border-color: rgba(167, 139, 250, 0.5); 
    transform: translateY(-2px);
    background: rgba(30, 27, 75, 0.5);
}
.source-card .src-title {
    color: #DDD6FE;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
    margin-bottom: 0.5rem;
}
.source-card .src-preview {
    color: #94A3B8;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ---- Stats Pill ---- */
.stat-pill {
    background: rgba(30, 41, 59, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 1rem;
    padding: 1.25rem;
    text-align: center;
    backdrop-filter: blur(12px);
}
.stat-pill .stat-value {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #A78BFA, #38BDF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-pill .stat-label {
    color: #94A3B8;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
    margin-top: 0.25rem;
}

/* ---- Sidebar Tweaks ---- */
section[data-testid="stSidebar"] {
    background: #09090b !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}
.stChatMessage {
    background: transparent !important;
}

/* ---- Buttons Glow ---- */
button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #7C3AED 0%, #0284C7 100%) !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.25) !important;
    transition: all 0.3s ease !important;
}
button[data-testid="baseButton-primary"]:hover {
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4) !important;
    transform: translateY(-2px) !important;
}

/* ---- File Uploader Customization ---- */
[data-testid="stFileUploadDropzone"] {
    border: 2px dashed rgba(167, 139, 250, 0.4) !important;
    border-radius: 1rem !important;
    background: rgba(124, 58, 237, 0.03) !important;
    padding: 2rem !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #8B5CF6 !important;
    background: rgba(124, 58, 237, 0.06) !important;
}

/* ---- Success / Error Banners ---- */
.success-banner {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.5) 0%, rgba(6, 95, 70, 0.5) 100%);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 1rem;
    padding: 1.25rem;
    color: #A7F3D0;
    text-align: center;
    margin-top: 1.5rem;
}
.error-banner {
    background: linear-gradient(135deg, rgba(69, 10, 10, 0.5) 0%, rgba(127, 29, 29, 0.5) 100%);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 1rem;
    padding: 1.25rem;
    color: #FCA5A5;
    text-align: center;
    margin-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# =========================================================================
# Session State Initialization
# =========================================================================
def init_session_state():
    """Initialize all Streamlit session-state keys."""
    defaults = {
        "memory": create_memory(),
        "messages": [],
        "documents_ingested": False,
        "ingestion_stats": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# =========================================================================
# Sidebar — Setup & Controls
# =========================================================================
def render_sidebar():
    """Render the sidebar with settings and controls."""
    with st.sidebar:
        st.markdown("## 🧠 Configuration")
        st.markdown("---")

        # ---- Environment Check ----
        env_status = validate_environment()
        if not env_status["all_required_set"]:
            st.error("⚠️ Missing API keys! Check your `.env` file.")
            missing = [k for k in ("OPENAI_API_KEY", "PINECONE_API_KEY") if not env_status[k]]
            for m in missing:
                st.code(m)
            st.stop()
        else:
            st.success("✅ Systems Online", icon="⚡")

        st.markdown("---")

        # ---- Ingestion Stats ----
        if st.session_state.ingestion_stats:
            s = st.session_state.ingestion_stats
            st.markdown("### 📊 Database Stats")
            c1, c2 = st.columns(2)
            c1.metric("Files Processed", s["num_files"])
            c2.metric("Pages Read", s["num_pages"])
            st.metric("Vector Chunks", s["num_chunks"])
            st.markdown("---")

        # ---- Controls ----
        st.markdown("### ⚙️ Chat Controls")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            clear_memory(st.session_state.memory)
            st.session_state.messages = []
            st.rerun()

        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align: center; color: #475569; font-size: 0.8rem;">'
            'Built by Anish with 🚀 using LangChain & Pinecone</div>',
            unsafe_allow_html=True,
        )


# =========================================================================
# Center Upload & Chat Interface
# =========================================================================
def render_main():
    """Render the main central interface."""

    # ---- Main Header ----
    st.markdown(
        '<div class="main-header">'
        '<h1>🧠 DocuRAG AI</h1>'
        '<p>Upload your knowledge base. Ask anything. Get cited answers instantly.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- Center Upload Section ----
    # Render if no documents are ingested yet, OR inside an expander if they want to add more
    if not st.session_state.documents_ingested:
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        st.markdown('<div class="upload-header">📄 Initialize Knowledge Base</div>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Select PDFs to construct your RAG database",
            type=["pdf"],
            accept_multiple_files=True,
            key="main_uploader",
            label_visibility="collapsed",
        )
        
        if uploaded_files:
            if st.button("✨ Initialize Vector Space", use_container_width=True, type="primary"):
                with st.spinner("⏳ Extracting text, generating embeddings, and upserting to Pinecone..."):
                    stats = ingest_pdfs(uploaded_files)

                if stats["status"] == "success":
                    st.session_state.documents_ingested = True
                    st.session_state.ingestion_stats = stats
                    st.rerun() # Refresh to hide the upload box and show chat
                else:
                    st.markdown(
                        f'<div class="error-banner">❌ Failed: {stats.get("error", "Unknown error")}</div>',
                        unsafe_allow_html=True,
                    )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # If documents are already ingested, show a sleek minimized version in an expander
        with st.expander("➕ Expand Knowledge Base (Upload more PDFs)"):
            uploaded_files = st.file_uploader(
                "Upload additional PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                key="extra_uploader",
                label_visibility="collapsed"
            )
            if uploaded_files:
                if st.button("✨ Add to Vector Space", use_container_width=True, type="primary"):
                    with st.spinner("⏳ Upserting new documents..."):
                        stats = ingest_pdfs(uploaded_files)
                    
                    if stats["status"] == "success":
                        # Update stats cumulatively
                        s = st.session_state.ingestion_stats
                        s["num_files"] += stats["num_files"]
                        s["num_pages"] += stats["num_pages"]
                        s["num_chunks"] += stats["num_chunks"]
                        st.markdown(
                            f'<div class="success-banner">✅ Successfully appended <b>{stats["num_files"]}</b> new files!</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.error(f"❌ Error: {stats.get('error')}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- Chat History ----
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="✨" if msg["role"] == "user" else "🧠"):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    render_sources(msg["sources"])

        # ---- Chat Input ----
        if prompt := st.chat_input("Ask a question about your knowledge base...", key="chat_input"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="✨"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar="🧠"):
                with st.spinner("🔮 Querying vector space..."):
                    try:
                        response_stream, source_docs = ask_question(
                            question=prompt,
                            memory=st.session_state.memory,
                        )
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        return

                full_response = st.write_stream(response_stream)
                sources = format_source_documents(source_docs) if source_docs else []
                if sources:
                    render_sources(sources)

            save_to_memory(st.session_state.memory, prompt, full_response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
            })


# =========================================================================
# Source Citations Renderer
# =========================================================================
def render_sources(sources: list):
    """Display formatted source citations with glassmorphism styling."""
    if not sources:
        return

    with st.expander("📚 View Extracted Evidence", expanded=False):
        for src in sources:
            st.markdown(
                f'<div class="source-card">'
                f'  <div class="src-title">📄 {src["source"]} — Page {src["page"]}</div>'
                f'  <div class="src-preview">{src["content"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# =========================================================================
# App Entry Point
# =========================================================================
def main():
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
