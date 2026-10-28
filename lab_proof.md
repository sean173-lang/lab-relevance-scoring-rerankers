# First Match ≠ Right Match — Lab Proof

This proof records outputs from running every code cell in
[`relevance_scoring_rerankers.ipynb`](./relevance_scoring_rerankers.ipynb) from
the project root with the supplied documents. Scores and excerpts below come
from the generated JSON evidence files; they are not hand-invented evaluation
metrics.

## Test query

> What is the difference between the seven ethical requirements for
> trustworthy AI and the legally binding requirements for high-risk AI systems
> under the EU AI Act?

## Inputs and document preparation

- [`data/eu_ai_act.pdf`](./data/eu_ai_act.pdf): extracted 144 PDF pages.
- [`data/podcast.txt`](./data/podcast.txt): extracted 2,815 whitespace-separated
  words; the source is a single transcript without page numbers.
- The loaders produced 145 text records: one per non-empty PDF page and one
  transcript record.
- Word-window chunking uses 250 words with 40 words of overlap. Chunks are
  created separately for each PDF page, so page numbers remain attributable.
- The run produced 516 chunks: 502 PDF chunks and 14 transcript chunks.
  Metadata fields include `source`, `document_type`, `page`, and
  `chunk_index`; transcript `page` values are `null`.

The source checksums used for this run:

| Source | Bytes | SHA-256 |
|---|---:|---|
| `data/eu_ai_act.pdf` | 2,583,319 | `bba630444b3278e881066774002a1d7824308934f49ccfa203e65be43692f55e` |
| `data/podcast.txt` | 16,414 | `4769962966b3eba0ddc8ac509a9efc5f42bde2694bc60c83d909c1da67c50927` |

## Embeddings and retrieval

The notebook embeds all 516 chunks with
`sentence-transformers/all-MiniLM-L6-v2`, producing 384-dimensional vectors.
It computes cosine similarity against the query, retains the original text
and metadata, and saves the baseline top five in
[`retrieval_results.json`](./retrieval_results.json).

### Baseline cosine-similarity top five

| Rank | Source | PDF page | Cosine similarity | Retrieved excerpt |
|---:|---|---:|---:|---|
| 1 | `data/eu_ai_act.pdf` | 41 | 0.7654 | “The development of AI systems other than high-risk AI systems ... may lead to a larger uptake of ethical and trustworthy AI...” |
| 2 | `data/eu_ai_act.pdf` | 8 | 0.7621 | “...2019 Ethics guidelines for trustworthy AI... seven non-binding ethical principles...” |
| 3 | `data/eu_ai_act.pdf` | 7 | 0.7112 | “...requirements for high-risk AI systems and obligations for the relevant operators...” |
| 4 | `data/eu_ai_act.pdf` | 86 | 0.7011 | “...general-purpose AI models with systemic risk... keep track of, document, and report...” |
| 5 | `data/eu_ai_act.pdf` | 77 | 0.6940 | “...obligations set out in Sections 2 and 3 of Chapter V...” |

Cosine similarity is a ranking signal, not a relevance label. The PDF p. 41
result is related but discusses voluntary codes and uptake of trustworthy AI;
it is less direct than the p. 8 passage for distinguishing legal status.

## Relevance scoring

The 20 cosine candidates were rescored with the notebook's
`deterministic_keyword_baseline`. `USE_LLM_RELEVANCE_SCORER` was `False`; no API
key was used. This is a fixed keyword rubric, not LLM output or human-labeled
ground truth.

Combined score:

```text
0.5 * min-max-normalized cosine similarity within the top 20
+ 0.5 * (relevance score / 5)
```

### Combined relevance-scoring top five

| Rank | Original rank | Source | PDF page | Relevance / 5 | Combined score | Retrieved excerpt |
|---:|---:|---|---:|---:|---:|---|
| 1 | 2 | `data/eu_ai_act.pdf` | 8 | 5 | 0.9812 | “...seven non-binding ethical principles... legally binding requirements of this Regulation...” |
| 2 | 1 | `data/eu_ai_act.pdf` | 41 | 2 | 0.7000 | “...AI systems other than high-risk AI systems... ethical and trustworthy AI...” |
| 3 | 3 | `data/eu_ai_act.pdf` | 7 | 3 | 0.4948 | “...requirements for high-risk AI systems and obligations for the relevant operators...” |
| 4 | 13 | `data/eu_ai_act.pdf` | 62 | 4 | 0.4390 | “Article 16 Obligations of providers of high-risk AI systems. Providers ... shall ensure ... compliance...” |
| 5 | 5 | `data/eu_ai_act.pdf` | 77 | 3 | 0.3976 | “...obligations set out in Sections 2 and 3 of Chapter V...” |

The relevance score and combined score are different fields. The combined
score reorders only the 20 cosine candidates.

## Cross-encoder reranking

The notebook scores the same 20 baseline candidates with
`cross-encoder/ms-marco-MiniLM-L-6-v2` and sorts by its raw pair score.
Cross-encoder scores are **not calibrated probabilities** and their numeric
scale is not compared with cosine similarity or the combined relevance score.

### Cross-encoder top five

| Rank | Original rank | Source | PDF page | Raw cross-encoder score | Retrieved excerpt |
|---:|---:|---|---:|---:|---|
| 1 | 2 | `data/eu_ai_act.pdf` | 8 | 5.7758 | “...seven non-binding ethical principles... legally binding requirements of this Regulation...” |
| 2 | 1 | `data/eu_ai_act.pdf` | 41 | 4.6744 | “...AI systems other than high-risk AI systems... ethical and trustworthy AI...” |
| 3 | 8 | `data/eu_ai_act.pdf` | 55 | 3.9481 | “...Article 97 to amend the list in Annex III by removing high-risk AI systems...” |
| 4 | 12 | `data/eu_ai_act.pdf` | 2 | 3.7982 | “...common rules for high-risk AI systems... take into account ... Ethics guidelines...” |
| 5 | 11 | `data/eu_ai_act.pdf` | 43 | 3.1601 | “...assessment of the need to amend the list of high-risk AI systems...” |

### Observed evidence movement

| Evidence passage | Baseline cosine rank | Relevance-scored rank | Cross-encoder rank |
|---|---:|---:|---:|
| Recital 27: seven non-binding principles vs. legally binding requirements, p. 8 | 2 | 1 | 1 |
| Article 16 provider duties for high-risk AI, p. 62 | 13 | 4 | 12 |

The cross-encoder moved the direct comparison passage from rank 2 to rank 1.
It did **not** put the concrete Article 16 provider duties in its top five;
those duties were rank 4 after the deterministic relevance score and rank 12
after cross-encoder reranking. This is a documented failure case, not a claim
that the cross-encoder is generally ineffective.

## Metadata filtering and focused legal retrieval

`search_chunks` accepts exact `source`, `document_type`, and inclusive,
one-based `page_range` filters. Filtering occurs before similarity scoring;
page filters exclude transcript chunks because their page is `null`.

Verified examples saved in `retrieval_results.json`:

| Filter | Observed result |
|---|---|
| PDF only | 502 eligible chunks; top-five results retain PDF source and page |
| PDF pages 56–62 | 21 eligible chunks, all returned in the saved focused retrieval |
| Transcript only | 14 eligible chunks; top-five results retain transcript source and `page: null` |
| PDF page 8 only | Five matching chunks returned, each with `page: 8` |
| Unknown source | Empty result list |
| Inverted page range | Raises `ValueError` |

For the answer, the notebook performs a second retrieval limited to PDF pages
56–62 and reranks the 21 resulting chunks. This targets the section containing
Articles 9–16; it supplements rather than retroactively changes the original
top-20 comparison.

## Source-grounded final answer

> The question calls them “seven ethical requirements”, but recital 27 of the
> EU AI Act identifies the 2019 AI HLEG guidance as seven **non-binding ethical
> principles**: human agency and oversight; technical robustness and safety;
> privacy and data governance; transparency; diversity, non-discrimination and
> fairness; societal and environmental well-being; and accountability. The
> recital expressly distinguishes those guidelines from the Regulation’s
> legally binding requirements [EU AI Act PDF, p. 8, recital 27].
>
> By contrast, the Act imposes legal duties for high-risk AI systems. Examples
> in the supplied text include establishing and maintaining a risk-management
> system (Article 9, p. 56), applying data-governance practices to relevant
> training, validation and testing data (Article 10, p. 57), designing systems
> for effective human oversight (Article 14, p. 60), and provider duties such
> as ensuring compliance with Section 2 (Article 16, p. 62). These are
> examples, not an exhaustive list [EU AI Act PDF, pp. 56–57, 60, 62].
>
> In short: the seven principles are broad, non-binding ethical guidance; the
> Act’s high-risk requirements are specific legal obligations. They can address
> overlapping values, but they are not interchangeable checklists.

### Citation evidence inspected from the actual PDF

| Citation | Retrieved source excerpt |
|---|---|
| `data/eu_ai_act.pdf`, p. 8, recital 27 | “...the AI HLEG developed seven non-binding ethical principles... Without prejudice to the legally binding requirements of this Regulation...” |
| `data/eu_ai_act.pdf`, p. 56, Article 9 | “A risk management system shall be established, implemented, documented and maintained in relation to high-risk AI systems.” |
| `data/eu_ai_act.pdf`, p. 57, Article 10 | “Training, validation and testing data sets shall be subject to data governance and management practices...” |
| `data/eu_ai_act.pdf`, p. 60, Article 14 | “High-risk AI systems shall be designed and developed in such a way ... that they can be effectively overseen by natural persons...” |
| `data/eu_ai_act.pdf`, p. 62, Article 16 | “Providers of high-risk AI systems shall: (a) ensure that their high-risk AI systems are compliant with the requirements set out in Section 2...” |

The notebook checks that each saved citation excerpt is present on the cited
extracted PDF page. The answer uses the EU AI Act as the authority for legal
status and describes Articles 9, 10, 14, and 16 as examples, not an exhaustive
statement of all high-risk obligations.

## Manual answer and evidence review

This is a qualitative source-to-claim review, not a numeric metric or
independent human benchmark:

- **Correctness:** supported. Recital 27 explicitly calls the AI HLEG items
  “non-binding ethical principles” and distinguishes them from the Act’s
  “legally binding requirements”.
- **Legal examples:** supported. The cited Articles contain mandatory “shall”
  language for risk management, data governance, human oversight, and provider
  duties.
- **Citation quality:** all five saved excerpts were checked against their
  corresponding extracted PDF pages; the notebook assertions and artifact tests
  repeat this check.
- **Scope:** appropriately qualified as examples, not a complete list of all
  statutory requirements.
- **Known retrieval weakness:** the cross-encoder rank for Article 16 was 12,
  outside its top five. A filtered page-range retrieval surfaces the statutory
  section, but this does not prove that the original top-20 candidate set is
  complete.

The podcast is a secondary source. One passage says “chapter two of the source
lists seven of them” and calls them “seven concrete requirements”; that wording
is not used to determine legal status. The distinction in the answer is
grounded in the Act’s own recital and Articles.

## Reproducibility and verification record

- Python: 3.11.
- Direct dependency versions are pinned in [`requirements.txt`](./requirements.txt).
- The direct pins and model revisions are recorded; transitive dependencies
  are not captured in a complete lockfile.
- Embedding model revision:
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Cross-encoder revision:
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`.
- Both models ran on CPU. The tested cross-encoder used
  `use_safetensors=False` after the safetensors CPU load produced NaNs; finite
  score checks run before sorting and saving.
- The deterministic relevance baseline was used; no API key was used or saved.
- All 25 notebook cells were saved after execution; every code cell executed
  without errors.
- `python -m unittest discover -s tests -v`: 9 tests passed.
- All direct pins in `requirements.txt` match the installed, tested environment.
- Generated evidence is stored in
  [`retrieval_results.json`](./retrieval_results.json),
  [`relevance_scoring_results.json`](./relevance_scoring_results.json), and
  [`reranking_results.json`](./reranking_results.json).

## Audit against the original lab requirements

| # | Requirement | Status | Evidence |
|---:|---|---|---|
| 1 | Load the PDF and podcast transcript | PASS | 144 PDF pages and 2,815 transcript words were loaded from the supplied files. |
| 2 | Chunk documents with page and source metadata | PASS | 516 chunks; source/type/page/chunk index retained; transcript page is `null`. |
| 3 | Generate sentence-transformer embeddings | PASS | 516 vectors with 384 dimensions; model revision is recorded. |
| 4 | Implement baseline semantic retrieval | PASS | Query cosine top five and scores are saved in `retrieval_results.json`. |
| 5 | Implement cross-encoder reranking | PASS | The pinned local model scored 20 candidates; all saved scores were checked finite. |
| 6 | Compare rankings before and after reranking | PASS | Baseline, Step 3 relevance, and cross-encoder rankings are saved and displayed. |
| 7 | Add metadata filtering | PASS | Source, type, page-range, no-match, and invalid-range checks pass. |
| 8 | Generate a source-grounded answer with citations | PASS | Five quotations are checked against the cited PDF pages; article references appear in the answer. |
| 9 | Save retrieval evidence and scores to JSON | PASS | Baseline, relevance, reranking, filtered evidence, and answer citations are persisted in the three JSON files. |
| 10 | Create `relevance_scoring_rerankers.ipynb` and `lab_proof.md` | PASS | Both files exist; the full notebook was executed and the proof records its actual evidence. |

## Additional submission audit

| Requirement | Status | Evidence / qualification |
|---|---|---|
| Working metadata filtering | PASS | Source, type, inclusive page ranges, transcript `null` pages, empty matches, and invalid ranges are tested. |
| Complete RAG pipeline with cross-encoder | PASS | Baseline top-20 retrieval, cross-encoder ranking, focused page retrieval, source excerpts, and cited answer are saved. |
| Manual correctness and evidence review | PASS (qualitative) | Claims and quotations were checked against the actual PDF; this is not an independent human benchmark. |
| Documented failure case / limitation | PASS | Article 16 ranks 13 → 4 → 12 across baseline, Step 3, and cross-encoder; top-20 candidate recall is also noted as a limit. |
| Complete, reproducible notebook | PASS | All code cells executed; dependency versions and both model revisions are pinned. Model downloads require the pinned checkpoint to be available locally or from the model cache/Hub. |
| Clear README and requirements | PASS | Setup, kernel selection, execution, output files, optional scorer, and tested direct dependency pins are documented. |
| No saved API keys or secrets | PASS | Submission-file secret-pattern test passed; LLM scoring is opt-in and disabled in the verified run. |
| Optional OpenAI relevance branch | NOT VERIFIED | No API key was available or used; the deterministic fallback was exercised and tested. |
| No unrelated files added | PASS | Added files are the README, dependency manifest, artifact tests, proof, and requested evidence outputs; `.gitignore` excludes the pre-existing virtual environment and `.DS_Store`. |
| Automated GitHub review result | NOT VERIFIED | This workspace is not a Git repository and no GitHub remote or CI review is available here. Passing local checks does not guarantee an external reviewer result. |
