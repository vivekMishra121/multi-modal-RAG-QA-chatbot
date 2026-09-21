"""Enterprise-grade Streamlit UI for the RAG QA Chatbot API."""

from __future__ import annotations

import os
import uuid
from typing import Any, Optional

import httpx
import streamlit as st

API_BASE_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Vega — Intelligent Document Search",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; }

/* ── App background: clean white ── */
.stApp { background: #f8fafc; color: #0f172a; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.2rem 1rem !important; }

.sidebar-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.5rem;
    margin-top: 0.2rem;
}

/* ── Sidebar buttons ── */
[data-testid="stSidebar"] .stButton > button {
    background: #f1f5f9 !important;
    color: #475569 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #e2e8f0 !important;
    color: #0f172a !important;
    border-color: #3b82f6 !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}
.stButton > button[kind="primary"]:hover { opacity: 0.88 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #f8fafc !important;
    border: 1.5px dashed #cbd5e1 !important;
    border-radius: 10px !important;
}

/* ── Text inputs ── */
[data-testid="stTextInput"] input {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #0f172a !important;
    font-size: 0.82rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
}

/* ── Top nav ── */
.top-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0 1rem 0;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 1.5rem;
    background: #ffffff;
}
.brand-name { font-size: 1.2rem; font-weight: 700; color: #0f172a; letter-spacing: -0.3px; }
.brand-tag  { font-size: 0.6rem; color: #94a3b8; letter-spacing: 1.3px; text-transform: uppercase; }

.status-pill {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.28rem 0.75rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600;
}
.status-pill.online  { background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }
.status-pill.offline { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.15rem 0 !important;
}
[data-testid="stChatMessage"] > div { background: transparent !important; }

[data-testid="stChatMessageContent"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 0.8rem 1rem !important;
    color: #0f172a !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}
[data-testid="stChatMessageContent"] *,
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] span,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] code {
    color: #0f172a !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #0f172a !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
}

/* ── Source cards ── */
.source-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 0.65rem 0.9rem;
    margin-bottom: 0.5rem;
    font-size: 0.8rem;
}
.source-card .src-title  { font-weight: 600; color: #1d4ed8; margin-bottom: 0.2rem; }
.source-card .src-meta   { color: #94a3b8; font-size: 0.7rem; margin-bottom: 0.25rem; }
.source-card .src-preview{ color: #475569; line-height: 1.55; }

/* ── Doc list items ── */
.doc-item {
    display: flex; align-items: center; justify-content: space-between;
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 0.45rem 0.7rem;
    margin-bottom: 0.35rem; font-size: 0.78rem;
}
.doc-item .doc-name   { color: #334155; font-weight: 500; }
.doc-item .doc-chunks {
    background: #eff6ff; color: #2563eb;
    border-radius: 999px; padding: 0.1rem 0.5rem;
    font-size: 0.68rem; font-weight: 700;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #64748b !important; font-size: 0.82rem !important; font-weight: 500 !important; }

/* ── Divider ── */
hr { border-color: #e2e8f0 !important; }

/* ── Alerts ── */
.stSuccess { background: #f0fdf4 !important; border-color: #bbf7d0 !important; color: #15803d !important; border-radius: 8px !important; }
.stError   { background: #fef2f2 !important; border-color: #fecaca !important; color: #dc2626 !important; border-radius: 8px !important; }
.stInfo    { background: #eff6ff !important; border-color: #bfdbfe !important; color: #1d4ed8 !important; border-radius: 8px !important; }

/* ── Welcome ── */
.welcome-wrap {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 4rem 2rem; text-align: center;
}
.welcome-title { font-size: 1.6rem; font-weight: 700; color: #0f172a; margin-bottom: 0.5rem; margin-top: 1rem; }
.welcome-sub   { font-size: 0.95rem; color: #64748b; max-width: 480px; line-height: 1.65; }
.welcome-chips { display: flex; gap: 0.5rem; flex-wrap: wrap; justify-content: center; margin-top: 1.4rem; }
.chip {
    background: #f1f5f9; border: 1px solid #e2e8f0;
    border-radius: 999px; padding: 0.32rem 0.85rem;
    font-size: 0.78rem; color: #475569;
}

/* ── Meta pills ── */
.meta-pill {
    display: inline-flex; align-items: center;
    background: #f1f5f9; border: 1px solid #e2e8f0;
    border-radius: 999px; padding: 0.15rem 0.6rem;
    font-size: 0.68rem; color: #64748b; margin-right: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────── http helpers

def _get(url: str, timeout: float = 10.0) -> Optional[dict]:
    with httpx.Client(timeout=timeout) as c:
        r = c.get(url); r.raise_for_status(); return r.json()

def _post(url: str, **kwargs) -> Optional[dict]:
    with httpx.Client(timeout=kwargs.pop("timeout", 120.0)) as c:
        r = c.post(url, **kwargs); r.raise_for_status(); return r.json()

def api_health() -> Optional[dict]:
    try: return _get(f"{API_BASE_URL}/health", timeout=5.0)
    except Exception: return None

def ask_backend(question: str, conversation_id: str, project_id: Optional[str] = None) -> Optional[dict]:
    try:
        payload: dict[str, Any] = {"question": question, "conversation_id": conversation_id}
        if project_id: payload["project_id"] = project_id
        return _post(f"{API_BASE_URL}/api/v1/chat", json=payload, timeout=120.0)
    except httpx.HTTPStatusError as e:
        st.error(f"API {e.response.status_code}: {e.response.text}"); return None
    except Exception as e:
        st.error(f"Request failed: {e}"); return None

def upload_documents(files, project_id: Optional[str] = None) -> Optional[dict]:
    try:
        multipart = [("files", (f.name, f.getvalue(), f.type)) for f in files]
        data = {"project_id": project_id} if project_id else {}
        return _post(f"{API_BASE_URL}/api/v1/documents/upload", files=multipart, data=data, timeout=300.0)
    except httpx.HTTPStatusError as e:
        st.error(f"Upload failed {e.response.status_code}: {e.response.text}"); return None
    except Exception as e:
        st.error(f"Upload error: {e}"); return None

def list_documents() -> Optional[dict]:
    try: return _get(f"{API_BASE_URL}/api/v1/documents")
    except Exception as e:
        st.error(f"Failed to list documents: {e}"); return None


# ─────────────────────────────────────────────────────────── session init

for key, val in [("messages", []), ("conversation_id", "ui-" + uuid.uuid4().hex),
                 ("api_status", None), ("docs_cache", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

LOGO = """<svg width="{s}" height="{s}" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
  <defs><linearGradient id="vlg{uid}" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="#3b82f6"/><stop offset="100%" stop-color="#6366f1"/>
  </linearGradient></defs>
  <polygon points="16,2 28,9 28,23 16,30 4,23 4,9" fill="url(#vlg{uid})"/>
  <polygon points="16,8 20,16 16,14 16,24 12,16 16,18" fill="white"/>
</svg>"""


# ─────────────────────────────────────────────────────────── sidebar

with st.sidebar:
    st.markdown(f"""
    <div style="padding:0.4rem 0 1rem 0;border-bottom:1px solid #e2e8f0;margin-bottom:1rem;">
      <div style="display:flex;align-items:center;gap:0.55rem;">
        {LOGO.format(s=34, uid="sb")}
        <div>
          <div style="font-size:1rem;font-weight:700;color:#0f172a;letter-spacing:-0.2px;">Vega</div>
          <div style="font-size:0.58rem;color:#94a3b8;letter-spacing:1.2px;text-transform:uppercase;">Intelligent Document Search</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Status
    st.markdown('<div class="sidebar-label">System Status</div>', unsafe_allow_html=True)
    health = api_health()
    st.session_state.api_status = health
    if health:
        backend = health.get("vector_store_backend", "—")
        chunks  = health.get("index_size", 0)
        st.markdown(f"""
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:0.55rem 0.75rem;margin-bottom:0.8rem;">
          <div style="font-size:0.75rem;color:#16a34a;font-weight:600;">🟢 API Online</div>
          <div style="font-size:0.68rem;color:#64748b;margin-top:0.15rem;">{backend} &nbsp;·&nbsp; {chunks} chunks</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:0.55rem 0.75rem;margin-bottom:0.8rem;">
          <div style="font-size:0.75rem;color:#dc2626;font-weight:600;">🔴 API Offline</div>
          <div style="font-size:0.68rem;color:#64748b;margin-top:0.15rem;">Check that the API server is running</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Conversation
    st.markdown('<div class="sidebar-label">Conversation</div>', unsafe_allow_html=True)
    project_id = st.text_input("Project ID", value=st.session_state.get("project_id", ""),
                                placeholder="optional — scope retrieval",
                                label_visibility="collapsed") or None
    st.session_state.project_id = project_id

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = "ui-" + uuid.uuid4().hex
            st.rerun()
    with c2:
        if st.button("🔄 New", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = "ui-" + uuid.uuid4().hex
            st.rerun()
    st.caption(f"Session `{st.session_state.conversation_id[-8:]}`")

    st.divider()

    # Upload
    st.markdown('<div class="sidebar-label">Upload Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("files", type=["pdf", "docx", "txt"],
                                       accept_multiple_files=True, label_visibility="collapsed")
    upload_project = st.text_input("proj", placeholder="project ID (optional)",
                                    key="upload_project_id", label_visibility="collapsed") or None
    if st.button("⬆ Upload & Index", use_container_width=True,
                 disabled=not uploaded_files, type="primary"):
        with st.spinner("Uploading and indexing…"):
            result = upload_documents(uploaded_files, upload_project)
        if result:
            st.success(
                f"✅ {result['files_uploaded']} file(s) · {result['total_chunks']} chunks · "
                f"{result['tables_extracted']} tables · {result['images_extracted']} images"
            )
            st.session_state.docs_cache = None

    st.divider()

    # Knowledge base
    st.markdown('<div class="sidebar-label">Knowledge Base</div>', unsafe_allow_html=True)
    if st.button("↻ Refresh", use_container_width=True):
        st.session_state.docs_cache = list_documents()

    docs_data = st.session_state.docs_cache
    if docs_data:
        docs = docs_data.get("documents", [])
        total_chunks = docs_data.get("total_chunks", 0)
        st.markdown(f'<div style="font-size:0.68rem;color:#94a3b8;margin-bottom:0.4rem;">'
                    f'{len(docs)} doc(s) · {total_chunks} chunks</div>', unsafe_allow_html=True)
        for doc in docs:
            name = doc.get("file_name", "unknown")
            chunks = doc.get("chunks", 0)
            ext  = name.rsplit(".", 1)[-1].upper() if "." in name else "FILE"
            icon = {"PDF": "📕", "DOCX": "📘", "TXT": "📄"}.get(ext, "📎")
            st.markdown(f"""
            <div class="doc-item">
              <span class="doc-name">{icon} {name}</span>
              <span class="doc-chunks">{chunks}</span>
            </div>""", unsafe_allow_html=True)
    elif docs_data is not None:
        st.caption("No documents indexed yet.")


# ─────────────────────────────────────────────────────────── main area

num_msgs = len(st.session_state.messages)
status_html = ('<span class="status-pill online">● Online</span>'
               if st.session_state.api_status
               else '<span class="status-pill offline">● Offline</span>')

st.markdown(f"""
<div class="top-nav">
  <div style="display:flex;align-items:center;gap:0.6rem;">
    {LOGO.format(s=30, uid="nav")}
    <div>
      <div class="brand-name">Vega</div>
      <div class="brand-tag">Intelligent Document Search</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:1rem;">
    <span style="font-size:0.72rem;color:#94a3b8;">{num_msgs} messages</span>
    {status_html}
  </div>
</div>
""", unsafe_allow_html=True)

# Welcome screen
if not st.session_state.messages:
    st.markdown(f"""
    <div class="welcome-wrap">
      {LOGO.format(s=60, uid="wlc")}
      <div class="welcome-title">Ask anything about your documents</div>
      <div class="welcome-sub">
        Upload PDFs, Word documents, or text files using the sidebar,
        then ask questions in natural language. Vega retrieves the most
        relevant passages and generates a cited answer.
      </div>
      <div class="welcome-chips">
        <span class="chip">📊 Tables &amp; charts</span>
        <span class="chip">🖼 Images &amp; diagrams</span>
        <span class="chip">📝 Multi-document Q&amp;A</span>
        <span class="chip">💬 Conversation memory</span>
        <span class="chip">🔍 Hybrid retrieval</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── helper (must be defined before the loops below)
def _render_source(i: int, src: dict) -> None:
    fname   = src.get("file_name", "unknown")
    page    = src.get("page", "?")
    ctype   = src.get("chunk_type", "text")
    preview = (src.get("content_preview") or "")[:350]
    tc = {"table": "#d97706", "image": "#7c3aed", "text": "#2563eb"}.get(ctype, "#2563eb")
    st.markdown(f"""
    <div class="source-card">
      <div class="src-title">📄 {fname}</div>
      <div class="src-meta">Source {i} &nbsp;·&nbsp; Page {page} &nbsp;·&nbsp;
        <span style="color:{tc};font-weight:600;">{ctype.upper()}</span>
      </div>
      <div class="src-preview">{preview}</div>
    </div>""", unsafe_allow_html=True)


# Chat history
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role, avatar="🔷" if role == "assistant" else "🧑"):
        st.markdown(msg["content"])
        if role == "assistant" and msg.get("sources"):
            sources = msg["sources"]
            nc = msg.get("num_chunks", len(sources))
            with st.expander(f"📎 {len(sources)} source(s) · {nc} chunks retrieved"):
                for i, src in enumerate(sources, 1):
                    _render_source(i, src)

# Chat input
if prompt := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🔷"):
        with st.spinner("Retrieving and generating answer…"):
            result = ask_backend(prompt, st.session_state.conversation_id,
                                 st.session_state.get("project_id"))

        if result and result.get("success"):
            answer   = result.get("answer") or "_No answer produced._"
            strategy = result.get("strategy", "auto")
            nc       = result.get("num_chunks", 0)
            sources  = result.get("sources", [])

            st.markdown(answer)
            st.markdown(f"""
            <div style="margin-top:0.5rem;">
              <span class="meta-pill">🔍 {strategy}</span>
              <span class="meta-pill">📦 {nc} chunks</span>
              <span class="meta-pill">📎 {len(sources)} sources</span>
            </div>""", unsafe_allow_html=True)

            if sources:
                with st.expander(f"📎 {len(sources)} source(s) · {nc} chunks retrieved"):
                    for i, src in enumerate(sources, 1):
                        _render_source(i, src)

            st.session_state.messages.append({
                "role": "assistant", "content": answer,
                "sources": sources, "num_chunks": nc,
            })
        else:
            error = (result or {}).get("error") or "Backend did not return a result."
            st.error(f"⚠ {error}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {error}"})
