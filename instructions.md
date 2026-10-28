# First match ≠ right match

> **How you'll submit this lab**
>
> This repo is your lab. Fork it, do the work described below in your fork, then open a pull
> request back into this repository. An AI reviewer will check your PR against `rubric.md` and
> leave feedback directly on the PR. See `README.md` for the full workflow.

## Learning Goal

You will build or evaluate one Relevance Scoring and Rerankers for Trustworthy AI & EU AI Act path and prove the answer is grounded in inspectable evidence.

## Working Context

This lab focuses on the chain between source material, tool behavior, and final answer.
You keep the system small enough to trace one query, one retrieval or tool step, and one final output.
The proof should show grounding, not just a successful response.

## Proof of Learning

Submit one `lab_proof.md` with the code or workflow, one query, retrieved or tool evidence, final output, and one failure or limitation.

## Pseudocode

```text
INPUT: source documents, tool result, or user query
DECIDE: retrieval, tool, agent, or guardrail path
BUILD: prepare source -> run path -> capture output
VERIFY: inspect evidence that supports the answer
EXPLAIN: identify one grounding risk or failure mode
```

## Prompt Discipline

Use AI for review, debugging, or critique. Do not ask it to produce the final artifact or final answer.

```text
Role: Retrieval or agent reviewer
Task: Review my plan for grounding and verification.
Inputs: source material, query, tool path, and proof evidence.
Constraints: Do not produce the final answer. Identify weak grounding and missing traces.
Output format: three grounding risks, two trace checks, and one simplification.
Verification: Name the evidence that ties the answer to the source.
```

## Build

## Step-by-Step Instructions

### Step 1: Setup and Data Preparation

Do: Set up your environment and prepare documents with metadata.

Do:
1. Create a new notebook: `relevance_scoring_rerankers.ipynb`
2. Load Trustworthy AI podcast transcripts and EU AI Act PDF
3. Chunk documents with metadata (source type, section, etc.)
4. Generate embeddings for all chunks

---

### Step 2: Generate Embeddings and Initial Retrieval

Do: Set up basic vector search as baseline. Refer to yesterday's lab and the lecture for examples if you are stuck.

---

### Step 3: Implement Relevance Scoring via an LLM (Optional - Advanced)

Do: Add relevance scoring using LLM to evaluate chunk relevance.

Do:
1. Use LLM (i.e. chatgpt with a prompt) to score each retrieved chunk's relevance to the query
2. Combine similarity score with relevance score
3. Reorder results based on combined score

---

### Step 4: Implement Reranker (Cohere or Cross-Encoder) (Optional - Advanced)

Do: Use a dedicated reranker model for better performance.

Do:
1. Use Cohere Rerank API or sentence-transformers cross-encoder
2. Rerank initial retrieval results
3. Compare with relevance scoring approach

---

### Step 5: Add Metadata Filtering

Do: Filter results by metadata (source type, section, etc.). If you don't have enough metadata, don't panic. Re-create your vectorstore with good metadata schema! This is an important lesson to learn.

---

### Step 6: Complete RAG Pipeline with Reranking

Do: Build complete RAG system with reranking. Integrate your new vectorstore into a rag pipeline.

Tip: this should be a 1-2 line change.

---

### Step 7: Evaluate Performance

Do: Compare retrieval quality with and without reranking. For this part, you will manually evaluate whether the answers are correct.

---

## Submission Guidelines


## Extension Activities (Optional)

1. **Hybrid search:** Combine keyword search (BM25) with semantic search + reranking
2. **Custom reranker:** Fine-tune a reranker on your specific domain
3. **A/B testing:** Compare different reranking models (spoiler for next weeks!)
4. **Cost analysis:** Compare costs of different reranking approaches

---

## Resources

- [Cohere Rerank API](https://cohere.com/rerank)
- [Sentence Transformers Cross-Encoders](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [LangChain Rerankers](https://docs.langchain.com/oss/python/integrations/rerankers/)
- [Reranking Explained](https://www.pinecone.io/learn/reranking/)

---

## Troubleshooting

**Issue:** Reranking is too slow
- **Solution:** Reduce initial retrieval size, use faster models, or cache results

**Issue:** Reranker doesn't improve results
- **Solution:** Your initial retrieval might already be good, or try different reranker models

**Issue:** Cohere API rate limits
- **Solution:** Use cross-encoder for local reranking, or implement batching and rate limiting

## Verify

The reviewer can trace one answer back to retrieved content, tool output, or guardrail evidence.

## Explain

Where could the system produce a plausible but unsupported answer?

## Submission Sanity Check

Before you submit:
- Keep only files for this lab.
- Name files clearly.
- Remove secrets, API keys, tokens, and unrelated data.
- Make `lab_proof.md` the entry point for the reviewer.

## Optional Stretch

- Add one targeted improvement: better chunking, reranking, a guardrail, a trace view, or a failure-case test.
- Create a small demo UI with vibecoding only if it helps inspect sources and answers. Deploy only if it can run safely without exposed secrets.
