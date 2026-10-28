# First Match ≠ Right Match

Ironhack AI Bootcamp lab comparing semantic retrieval, relevance scoring, and
cross-encoder reranking on the supplied EU AI Act PDF and podcast transcript.
The notebook preserves source and page metadata so rankings can be inspected
against the documents rather than treated as relevance ground truth.

## Project files

- `relevance_scoring_rerankers.ipynb` — the complete, executable Steps 1–7
  workflow, including metadata filtering, the RAG answer, and qualitative review.
- `lab_proof.md` — actual run evidence, cited answer, checks, and limitations.
- `retrieval_results.json` — baseline retrieval and metadata-filter examples.
- `relevance_scoring_results.json` — top-20 relevance scores and comparison.
- `reranking_results.json` — three rankings, filtered legal evidence, cited
  answer, and evidence-quality review.
- `data/eu_ai_act.pdf` and `data/podcast.txt` — the source documents.

## Environment and run

Use Python 3.11. From the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name first-match-lab --display-name "First Match Lab (.venv)"
jupyter notebook relevance_scoring_rerankers.ipynb
```

In Jupyter, select the `first-match-lab` kernel and use **Restart Kernel and Run
All Cells**. Run from the project root because the notebook reads the relative
`data/` paths and writes the JSON results there.

The notebook pins both Hugging Face model revisions and runs them on CPU. The
first run may download their files into the normal Hugging Face cache. The
cross-encoder uses the PyTorch checkpoint because the cached safetensors
checkpoint produced non-finite CPU scores in the tested environment; the
notebook rejects non-finite output rather than saving it.

The relevance scorer defaults to a deterministic keyword rubric. An optional
OpenAI-compatible scorer is available only after explicitly setting
`USE_LLM_RELEVANCE_SCORER = True` in the notebook and providing
`OPENAI_API_KEY` through the environment. Enabling it sends the query and
candidate text to the configured OpenAI API; do not enable it for confidential
documents. The verified run did not use an API key, and no key or secret belongs
in this repository.

## Tests and verification

Run the artifact and evidence tests from the project root:

```bash
python -m unittest discover -s tests -v
```

The notebook also contains assertions for chunk metadata, filtering, retrieval
rankings, finite model scores, and source citations. `lab_proof.md` records the
actual run outputs and calls out the cross-encoder failure case. The cross-
encoder score is a raw ranking score, not a calibrated probability; do not
compare its numeric scale with cosine or combined relevance scores.

## Interpretation notes

The EU AI Act PDF's recital 27 calls the AI HLEG items “seven non-binding
ethical principles” and distinguishes them from the Regulation's legally
binding requirements. The final answer uses the Act itself—not the podcast
transcript—as the authority for legal status. Its cited Articles 9, 10, 14, and
16 are examples, not an exhaustive summary of every high-risk requirement.
The retrieval and scoring outputs are evidence for this lab, not independently
human-judged relevance metrics.
