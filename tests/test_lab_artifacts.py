import json
import math
import re
import unittest
from importlib.metadata import version
from pathlib import Path

import nbformat
import pymupdf


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_QUERY = (
    "What is the difference between the seven ethical requirements for "
    "trustworthy AI and the legally binding requirements for high-risk AI "
    "systems under the EU AI Act?"
)


def read_json(filename: str) -> dict:
    return json.loads((ROOT / filename).read_text(encoding="utf-8"))


def evidence_key(result: dict) -> tuple:
    metadata = result["metadata"]
    return (
        metadata["source"],
        metadata["page"],
        metadata["chunk_index"],
    )


class LabArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.retrieval = read_json("retrieval_results.json")
        cls.relevance = read_json("relevance_scoring_results.json")
        cls.reranking = read_json("reranking_results.json")

    def test_notebook_is_valid_and_all_steps_were_executed(self) -> None:
        notebook = nbformat.read(
            ROOT / "relevance_scoring_rerankers.ipynb", as_version=4
        )
        nbformat.validate(notebook)

        headings = [
            line
            for cell in notebook.cells
            if cell.cell_type == "markdown"
            for line in cell.source.splitlines()
            if line.startswith("## Step")
        ]
        self.assertEqual(len(headings), 7)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        self.assertTrue(code_cells)
        self.assertTrue(all(cell.execution_count is not None for cell in code_cells))
        self.assertFalse(
            any(
                output.output_type == "error"
                for cell in code_cells
                for output in cell.outputs
            )
        )

    def test_readme_and_requirements_match_the_tested_environment(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("Restart Kernel and Run", readme)
        self.assertIn("retrieval_results.json", readme)
        self.assertIn("OPENAI_API_KEY", readme)

        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        pins = [
            line.strip()
            for line in requirements.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        self.assertTrue(pins)
        for pin in pins:
            package, expected_version = pin.split("==", maxsplit=1)
            self.assertEqual(version(package), expected_version, package)

    def test_baseline_retrieval_uses_the_test_query_and_real_metadata(self) -> None:
        self.assertEqual(self.retrieval["query"], EXPECTED_QUERY)
        self.assertEqual(self.retrieval["chunk_count"], 516)
        self.assertEqual(self.retrieval["embedding_dimension"], 384)
        self.assertEqual(len(self.retrieval["results"]), 5)
        scores = [
            result["similarity_score"] for result in self.retrieval["results"]
        ]
        self.assertTrue(all(math.isfinite(score) for score in scores))
        self.assertTrue(all(a >= b for a, b in zip(scores, scores[1:])))
        self.assertTrue(
            all(
                result["metadata"]["source"] == "data/eu_ai_act.pdf"
                and result["metadata"]["page"] is not None
                for result in self.retrieval["results"]
            )
        )

    def test_metadata_filters_preserve_source_and_page_semantics(self) -> None:
        examples = self.retrieval["metadata_filter_examples"]
        pdf_results = examples["pdf_only_top_5"]["results"]
        transcript_results = examples["podcast_only_top_5"]["results"]
        legal_example = examples["high_risk_requirements_pages_56_to_62"]
        page_eight_results = examples["pdf_page_8_only"]["results"]

        self.assertTrue(
            all(result["metadata"]["document_type"] == "pdf" for result in pdf_results)
        )
        self.assertEqual(len(transcript_results), 5)
        self.assertTrue(
            all(
                result["metadata"]["source"] == "data/podcast.txt"
                and result["metadata"]["page"] is None
                for result in transcript_results
            )
        )
        self.assertEqual(legal_example["matching_chunk_count"], 21)
        self.assertTrue(
            all(
                56 <= result["metadata"]["page"] <= 62
                for result in legal_example["results"]
            )
        )
        self.assertTrue(
            all(result["metadata"]["page"] == 8 for result in page_eight_results)
        )

    def test_relevance_scores_and_original_top_five_are_preserved(self) -> None:
        self.assertEqual(self.relevance["candidate_count"], 20)
        self.assertEqual(
            self.relevance["relevance_scoring_method"],
            "deterministic_keyword_baseline",
        )
        original_top_five = self.relevance["comparison"]["original_cosine_top_5"]
        self.assertEqual(len(original_top_five), 5)
        self.assertEqual(
            [evidence_key(result) for result in original_top_five],
            [evidence_key(result) for result in self.retrieval["results"]],
        )
        for result in self.relevance["scored_candidates"]:
            self.assertGreaterEqual(result["relevance_score"], 0)
            self.assertLessEqual(result["relevance_score"], 5)
            expected_combined = (
                0.5 * result["normalized_similarity"]
                + 0.5 * result["relevance_score"] / 5
            )
            self.assertAlmostEqual(result["combined_score"], expected_combined)

    def test_all_three_rankings_have_twenty_finite_ordered_candidates(self) -> None:
        rankings = self.reranking["rankings"]
        self.assertEqual(
            set(rankings),
            {"baseline_cosine", "step_3_relevance_scoring", "cross_encoder"},
        )
        self.assertTrue(all(len(ranking) == 20 for ranking in rankings.values()))

        cross_encoder = rankings["cross_encoder"]
        self.assertTrue(
            all(math.isfinite(result["cross_encoder_score"]) for result in cross_encoder)
        )
        self.assertTrue(
            all(
                first["cross_encoder_score"] >= second["cross_encoder_score"]
                for first, second in zip(cross_encoder, cross_encoder[1:])
            )
        )
        self.assertIn("not calibrated probabilities", self.reranking["cross_encoder_score_note"])
        self.assertEqual(
            self.reranking["cross_encoder_revision"],
            "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        )
        self.assertTrue(
            all(
                result["metadata"]["source"] == "data/eu_ai_act.pdf"
                and result["text"]
                and result["excerpt"]
                for result in cross_encoder
            )
        )

    def test_citation_excerpts_match_the_cited_pdf_pages(self) -> None:
        answer = self.reranking["rag_answer"]
        self.assertIn("[EU AI Act PDF, p. 8, recital 27]", answer["text"])
        self.assertIn("[EU AI Act PDF, pp. 56–57, 60, 62]", answer["text"])
        self.assertEqual(len(answer["citations"]), 5)

        with pymupdf.open(ROOT / "data/eu_ai_act.pdf") as pdf:
            for citation in answer["citations"]:
                self.assertEqual(citation["source"], "data/eu_ai_act.pdf")
                source_page = " ".join(
                    pdf[citation["page"] - 1].get_text("text").split()
                )
                self.assertIn(citation["excerpt"], source_page)

    def test_failure_case_is_documented_with_observed_ranks(self) -> None:
        failure = self.reranking["manual_evaluation"]["failure_case"]
        self.assertEqual(failure["ranks"]["baseline_cosine"], 13)
        self.assertEqual(failure["ranks"]["step_3_relevance_scoring"], 4)
        self.assertEqual(failure["ranks"]["cross_encoder"], 12)
        self.assertFalse(failure["cross_encoder_in_top_5"])
        self.assertIn(
            "not a numeric metric",
            self.reranking["manual_evaluation"]["method"],
        )

    def test_no_literal_api_secret_is_saved_in_submission_files(self) -> None:
        secret_pattern = re.compile(
            r"(?:sk-[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9._~+/-]{30,})"
        )
        files = [
            ROOT / "README.md",
            ROOT / "requirements.txt",
            ROOT / "relevance_scoring_rerankers.ipynb",
            ROOT / "retrieval_results.json",
            ROOT / "relevance_scoring_results.json",
            ROOT / "reranking_results.json",
            ROOT / "lab_proof.md",
        ]
        for path in files:
            if path.exists():
                self.assertIsNone(secret_pattern.search(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
