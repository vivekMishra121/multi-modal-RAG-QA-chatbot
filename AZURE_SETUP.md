# Azure OpenAI Setup Guide

## 1. Set Environment Variables

Add to `.env` file:

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-api-key

# Deployment names (must match your Azure deployments)
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large
AZURE_CHAT_DEPLOYMENT=gpt-5.1
```

## 2. Azure Deployments Required

Create these deployments in Azure Portal:

1. **Embeddings**: `text-embedding-3-large`
2. **Chat**: `gpt-5.1` (or `gpt-4o`)

## 3. Rebuild Index

```bash
python rebuild_with_openai.py "path/to/document.pdf"
```

The system will auto-detect Azure and use it.

## 4. Test

```bash
python run_evaluation.py
```

## Cost Comparison

| Provider | Embeddings | Chat (GPT-5.1) |
|----------|-----------|----------------|
| OpenAI | $0.13/1M tokens | $X/1M tokens |
| Azure | Same pricing | Same pricing |

**Benefits of Azure:**
- Enterprise security
- Private endpoints
- SLA guarantees
- Regional deployment
