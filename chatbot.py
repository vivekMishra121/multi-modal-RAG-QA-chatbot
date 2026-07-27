"""Modern RAG Chatbot UI - ChatGPT Style"""

import streamlit as st
from main import get_chatbot

st.set_page_config(
    page_title="Document AI Assistant",
    page_icon="💬",
    layout="centered"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem; max-width: 800px;}
    .stChatMessage {border-radius: 12px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_chatbot():
    try:
        return get_chatbot()
    except:
        return None


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main():
    init_session()
    
    st.title("💬 Document AI Assistant")
    st.caption("Ask questions about your documents")
    
    chatbot = load_chatbot()
    
    if not chatbot:
        st.error("⚠️ Run: `python main.py build <document_path>` first")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("💡 Sample Questions")
        samples = [
            "What is Qatar's GDP growth?",
            "What are the main economic drivers?",
            "What is the inflation rate?"
        ]
        
        for q in samples:
            if st.button(q, key=q, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                result = chatbot.chat(q)
                if result['success']:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result['answer'],
                        "sources": result['sources']
                    })
                else:
                    st.session_state.messages.append({"role": "assistant", "content": f"❌ {result['error']}"})
                st.rerun()
        
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander(f"📚 {len(msg['sources'])} sources"):
                    for src in msg["sources"]:
                        st.markdown(f"**{src['file_name']}** - Page {src['page']}")
                        st.caption(src['content_preview'][:150])
                        st.divider()
    
    # Chat input - always visible
    prompt = st.chat_input("Ask a question...")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = chatbot.chat(prompt)
                
                if result['success']:
                    st.markdown(result['answer'])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result['answer'],
                        "sources": result['sources']
                    })
                    
                    if result['sources']:
                        with st.expander(f"📚 {len(result['sources'])} sources"):
                            for src in result['sources']:
                                st.markdown(f"**{src['file_name']}** - Page {src['page']}")
                                st.caption(src['content_preview'][:150])
                                st.divider()
                else:
                    error = f"❌ {result['error']}"
                    st.error(error)
                    st.session_state.messages.append({"role": "assistant", "content": error})
        
        st.rerun()


if __name__ == "__main__":
    main()
