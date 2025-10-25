# How to Change Model Settings

## Latest OpenAI Models (2025)

### Most Advanced Models Available

**GPT-4.1 Series** (Latest & Best):
- `gpt-4.1` - Most advanced, 1M context, best at coding & reasoning
- `gpt-4.1-mini` - 83% cheaper, beats GPT-4o, 50% faster
- `gpt-4.1-nano` - Fastest & cheapest, good for classification

**GPT-4o Series** (Still Available):
- `gpt-4o` - Multimodal (text + images)
- `gpt-4o-mini` - Cheap alternative to GPT-3.5-turbo
- `gpt-4o-audio` - Supports audio inputs/outputs

**Legacy Models**:
- `gpt-4-turbo` - Previous generation
- `gpt-4` - Original GPT-4
- `gpt-3.5-turbo` - Cheapest text model

**Embeddings**:
- `text-embedding-3-large` - Best quality (3072 dimensions)
- `text-embedding-3-small` - Best value (1536 dimensions) ✅ Current
- `text-embedding-ada-002` - Legacy

## Where to Change Settings

### Option 1: Edit .env File (Recommended)

**Location:** `/home/xinra/NotreDame/Calcium_pipeline/.env`

```bash
# Open in editor
nano .env

# Add these lines (or modify existing):
OPENAI_MODEL=gpt-4.1              # Main model
ROUTER_MODEL=gpt-4.1-mini         # Routing/planning/verification
EMBEDDING_MODEL=text-embedding-3-small  # RAG embeddings
```

**Recommended Configuration (Best Performance):**
```bash
# .env
OPENAI_MODEL=gpt-4.1              # Best reasoning & code generation
ROUTER_MODEL=gpt-4.1-mini         # 83% cheaper, still excellent
EMBEDDING_MODEL=text-embedding-3-small  # Good enough, cheap
```

**Budget Configuration:**
```bash
# .env
OPENAI_MODEL=gpt-4o-mini          # Much cheaper than GPT-4
ROUTER_MODEL=gpt-4o-mini          # Same model for consistency
EMBEDDING_MODEL=text-embedding-3-small  # Already cheap
```

**Premium Configuration (Maximum Quality):**
```bash
# .env
OPENAI_MODEL=gpt-4.1              # Best available
ROUTER_MODEL=gpt-4.1              # Use best for everything
EMBEDDING_MODEL=text-embedding-3-large  # Better embeddings
```

### Option 2: Edit config.py Directly

**Location:** `/home/xinra/NotreDame/Calcium_pipeline/config.py`

**Lines 22-24:**
```python
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")  # Change default
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
```

## Model Comparison

### Performance

| Model | Reasoning | Coding | Speed | Cost |
|-------|-----------|--------|-------|------|
| gpt-4.1 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | $$$$ |
| gpt-4.1-mini | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $$ |
| gpt-4o | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | $$$ |
| gpt-4o-mini | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $ |
| gpt-3.5-turbo | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | $ |

### Cost Estimates

**Per 1M input tokens:**
- `gpt-4.1`: ~$2.50
- `gpt-4.1-mini`: ~$0.40 (83% cheaper than gpt-4o!)
- `gpt-4o`: ~$2.50
- `gpt-4o-mini`: ~$0.15
- `gpt-3.5-turbo`: ~$0.50
- `text-embedding-3-small`: ~$0.02

## What Each Setting Controls

### OPENAI_MODEL
**Used for:**
- Code generation (capability_manager.py)
- Informational responses (graph/nodes.py)

**Recommendation:** `gpt-4.1` (best) or `gpt-4o` (good)

### ROUTER_MODEL
**Used for:**
- Query routing (router.py)
- Clarification (clarifier.py)
- Planning (planner.py)
- Verification (verifier.py)

**Recommendation:** `gpt-4.1-mini` (excellent value) or `gpt-4o-mini` (cheap)

### EMBEDDING_MODEL
**Used for:**
- RAG paper search (rag_system.py)
- Capability store similarity (capability_store.py)

**Recommendation:** `text-embedding-3-small` (best value)

## How to Apply Changes

### After Editing .env:

```bash
# No restart needed - loaded on next run
streamlit run app.py
```

### After Editing config.py:

```bash
# Restart required
# Press Ctrl+C if running
streamlit run app.py
```

## Testing Your Changes

```bash
# Check what models are loaded
python -c "import config; print(f'Main: {config.OPENAI_MODEL}'); print(f'Router: {config.ROUTER_MODEL}'); print(f'Embeddings: {config.EMBEDDING_MODEL}')"
```

## My Recommendation

For your calcium imaging RAG system:

```bash
# .env file
OPENAI_MODEL=gpt-4.1              # Best for scientific explanations
ROUTER_MODEL=gpt-4.1-mini         # Excellent quality, much cheaper
EMBEDDING_MODEL=text-embedding-3-small  # Perfect for your use case
```

This gives you:
- ✅ Best quality responses from RAG
- ✅ 50% faster than old GPT-4o
- ✅ Lower costs overall
- ✅ 1M context window (can handle huge papers!)

## Special Features

### GPT-4.1 Advantages
- 1 million token context (can read entire papers!)
- Better at following instructions
- Improved coding ability
- Knowledge cutoff: June 2024

### GPT-4.1-mini Advantages
- 83% cheaper than GPT-4o
- 50% faster (lower latency)
- Beats GPT-4o in many benchmarks
- Perfect for routing/planning tasks

## Notes

- GPT-5 exists but is NOT available via API yet (ChatGPT only)
- GPT-4.1 series released recently (2025)
- All models support function calling
- Embeddings are backward compatible

Update your models to GPT-4.1 series for best results! 🚀
