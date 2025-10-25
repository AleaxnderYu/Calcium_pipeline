# Model Usage Across Workflow Nodes

Here's what model is used in each part of the calcium imaging analysis system.

## Configuration (config.py)

```python
OPENAI_MODEL = "gpt-4"                          # Main model for heavy tasks
ROUTER_MODEL = "gpt-3.5-turbo"                  # Lightweight model for routing
EMBEDDING_MODEL = "text-embedding-3-small"      # Embedding model for RAG
```

## Model Usage by Node

### 1. Router Node (core/router.py)
**Model:** `gpt-3.5-turbo` (ROUTER_MODEL)
- **Purpose:** Classify query as "analysis" or "informational"
- **Why lightweight:** Simple classification task
- **Current status:** ⚠️ BYPASSED (test mode - always returns "informational")

### 2. Clarifier Node (core/clarifier.py)
**Model:** `gpt-3.5-turbo` (ROUTER_MODEL)
- **Purpose:** Extract assumptions and clarify ambiguous requests
- **Why lightweight:** Straightforward text analysis
- **Example:** "Detect cells" → Assumes user wants ROI detection

### 3. RAG System (layers/rag_system.py)
**Model:** `text-embedding-3-small` (EMBEDDING_MODEL)
- **Purpose:** Convert paper chunks and queries to embeddings
- **Why this model:** Cost-effective, fast, good quality embeddings
- **Database:** ChromaDB vector store

### 4. Planner Node (core/planner.py)
**Model:** `gpt-3.5-turbo` (ROUTER_MODEL)
- **Purpose:** Create multi-step execution plan from request
- **Why lightweight:** Plan generation is structured task
- **Output:** Sequential/parallel/DAG execution plan

### 5. Capability Manager (layers/capability_manager.py)
**Model:** `gpt-4` (OPENAI_MODEL)
- **Purpose:** Generate Python code for analysis steps
- **Why GPT-4:** Complex code generation requires stronger model
- **Also uses:** `text-embedding-3-small` for capability store search

### 6. Informational Response Node (graph/nodes.py)
**Model:** `gpt-4` (OPENAI_MODEL)
- **Purpose:** Generate comprehensive answers using RAG context
- **Why GPT-4:** Synthesizing scientific literature requires reasoning
- **Input:** Retrieved paper chunks from RAG
- **Output:** Detailed, scientifically accurate response

### 7. Verifier Node (core/verifier.py)
**Model:** `gpt-3.5-turbo` (ROUTER_MODEL)
- **Purpose:** Verify execution results meet requirements
- **Why lightweight:** Result validation is straightforward
- **Output:** Success/failure with suggested fixes

### 8. Capability Store (layers/capability_store.py)
**Model:** `text-embedding-3-small` (EMBEDDING_MODEL)
- **Purpose:** Semantic search for similar past capabilities
- **Why embeddings:** Efficient similarity matching
- **Benefit:** Reuse code from similar queries

## Summary Table

| Component | Model | Cost | Speed | Purpose |
|-----------|-------|------|-------|---------|
| **Router** | gpt-3.5-turbo | $ | Fast | Classification |
| **Clarifier** | gpt-3.5-turbo | $ | Fast | Extract assumptions |
| **RAG Embeddings** | text-embedding-3-small | $ | Very Fast | Vector search |
| **Planner** | gpt-3.5-turbo | $ | Fast | Create plan |
| **Code Generator** | gpt-4 | $$$ | Slower | Generate code |
| **Informational** | gpt-4 | $$$ | Slower | Answer questions |
| **Verifier** | gpt-3.5-turbo | $ | Fast | Validate results |
| **Capability Store** | text-embedding-3-small | $ | Very Fast | Reuse past code |

## Cost Optimization Strategy

The system uses **GPT-4 only where necessary**:
- ✅ Code generation (requires precision)
- ✅ Scientific explanations (requires reasoning)

Everything else uses **GPT-3.5-turbo**:
- ✅ Routing, clarification, planning, verification
- ✅ 10x cheaper than GPT-4
- ✅ Fast enough for these tasks

## Changing Models

Edit `.env` or `config.py`:

```bash
# .env file
OPENAI_MODEL=gpt-4o              # Use GPT-4o for main tasks
ROUTER_MODEL=gpt-4o-mini         # Use GPT-4o-mini for routing
EMBEDDING_MODEL=text-embedding-3-large  # Better embeddings (more expensive)
```

## Model Costs (Approximate)

Based on OpenAI pricing:

| Model | Input | Output |
|-------|-------|--------|
| gpt-4 | $0.03/1K tokens | $0.06/1K tokens |
| gpt-3.5-turbo | $0.0005/1K tokens | $0.0015/1K tokens |
| text-embedding-3-small | $0.00002/1K tokens | N/A |

**Typical query cost:**
- Informational (RAG-only): ~$0.01-0.02
- Code execution: ~$0.05-0.10
- With capability reuse: ~$0.001-0.005 (90% savings!)

## Current Test Mode

Since router is forced to "informational", you're currently using:
1. **RAG embeddings**: text-embedding-3-small
2. **Response generation**: gpt-4

No router, clarifier, planner, code generator, or verifier models are being called.
