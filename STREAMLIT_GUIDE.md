# 🚀 Quick Start Guide - Streamlit Chatbot

## Prerequisites

1. **Install Dependencies**
```bash
pip install streamlit
```

2. **Set API Key**
Create `.env` file:
```
OPENAI_API_KEY=xai-your_grok_api_key_here
```

3. **Build Index (First Time Only)**
```bash
python main.py build "C:\Users\HP\Downloads\Qatar Test Document.pdf"
```

## Run Chatbot

```bash
streamlit run app.py
```

The app will open at: http://localhost:8501

## Features

✅ **Modern UI**
- Clean, professional design
- Gradient color scheme
- Responsive layout

✅ **Chat Interface**
- Message history
- User/Assistant distinction
- Expandable source citations

✅ **Sidebar Controls**
- Model selection (Grok, GPT-4, GPT-3.5)
- Retrieval settings (top_k, max_tokens)
- Statistics dashboard
- Clear chat button
- Sample questions

✅ **Source Citations**
- View retrieved chunks
- File name, page number, chunk type
- Content preview

## Usage Tips

1. **Sample Questions**: Click buttons in sidebar for quick queries
2. **View Sources**: Expand "View Sources" to see retrieved chunks
3. **Adjust Settings**: Change model and retrieval parameters in sidebar
4. **Clear History**: Use "Clear Chat" button to start fresh

## Troubleshooting

**Error: "Chatbot not initialized"**
- Run: `python main.py build <document_path>` first

**Error: "API key not found"**
- Check `.env` file exists with `OPENAI_API_KEY`

**Slow responses**
- Reduce `top_k` in sidebar
- Use smaller `max_tokens`

## Architecture

```
User Input → Streamlit UI → RAGChatbot → Retriever → LLM → Response
                                ↓
                          Vector Store (FAISS)
```

Enjoy your RAG chatbot! 🤖
