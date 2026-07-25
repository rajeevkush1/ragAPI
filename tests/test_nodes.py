"""
tests/test_nodes.py
━━━━━━━━━━━━━━━━━━━
Isolated unit tests for each RAG pipeline node.
All tests use mocks/fixtures — no live Qdrant or Ollama required.

Run:
    python tests/test_nodes.py              # all tests
    python tests/test_nodes.py QueryTest    # specific class
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from state import RAGState, RetrievedDoc, GradedDoc


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_retrieved_doc(**kwargs) -> RetrievedDoc:
    defaults = dict(
        text="Flash Attention uses tiling to achieve O(N) memory complexity.",
        source="flash_attention.pdf",
        score=0.90,
        dense_rank=0,
        bm25_rank=0,
        h1="Introduction",
        h2="Method",
        chunk_id="abc123",
    )
    defaults.update(kwargs)
    return RetrievedDoc(**defaults)


def make_graded_doc(**kwargs) -> GradedDoc:
    defaults = dict(
        text="Flash Attention uses tiling to achieve O(N) memory complexity.",
        source="flash_attention.pdf",
        score=0.90,
        relevance_score=0.92,
        relevance_reason="Directly describes the memory complexity of Flash Attention",
        h1="Introduction",
        h2="Method",
        chunk_id="abc123",
    )
    defaults.update(kwargs)
    return GradedDoc(**defaults)


def base_state(**kwargs) -> RAGState:
    defaults = RAGState(
        original_query="What is Flash Attention?",
        rewritten_query="Flash Attention memory efficient algorithm",
        query_type="analytical",
        key_terms=["flash attention", "memory", "tiling"],
        needs_context=True,
        retrieved_docs=[],
        graded_docs=[],
        answer="",
        is_grounded=False,
        grounding_score=0.0,
        unsupported_claims=[],
        final_answer="",
        conversation_history=[],
        # Loop A
        retrieval_retry_count=0,
        max_retrieval_retries=3,
        failed_queries=[],
        # Loop B
        generation_retry_count=0,
        max_generation_retries=3,
        generation_hint="",
    )
    defaults.update(kwargs)
    return defaults


# ── Test: query_analyzer ──────────────────────────────────────────────────────

class QueryAnalyzerTest(unittest.TestCase):

    def _mock_chain(self, return_value: str) -> MagicMock:
        """Build a mock chain whose .invoke() returns the given string."""
        chain = MagicMock()
        chain.invoke.return_value = return_value
        return chain

    @patch("nodes.query_analyzer.ChatOllama")
    @patch("nodes.query_analyzer._PROMPT")
    def test_analyze_factual_query(self, mock_prompt, mock_llm_cls):
        """analyze_query returns correct structure for a factual query."""
        expected = {
            "rewritten_query": "Flash Attention IO-aware algorithm memory reduction HBM SRAM",
            "query_type": "factual",
            "key_terms": ["flash attention", "memory", "HBM", "SRAM", "tiling"],
            "needs_context": True,
        }
        mock_chain = self._mock_chain(json.dumps(expected))
        # _PROMPT | llm | parser returns mock_chain
        mock_prompt.__or__ = MagicMock(return_value=MagicMock(
            __or__=MagicMock(return_value=mock_chain)
        ))
        mock_llm_cls.return_value = MagicMock()

        from nodes.query_analyzer import analyze_query
        result = analyze_query("What is Flash Attention?")

        self.assertIn("rewritten_query", result)
        self.assertIn("query_type", result)
        self.assertIn("key_terms", result)
        self.assertIn("needs_context", result)
        self.assertIsInstance(result["key_terms"], list)
        self.assertIsInstance(result["needs_context"], bool)

    def test_graceful_fallback_on_bad_json(self):
        """analyze_query falls back gracefully when LLM outputs invalid JSON."""
        with patch("nodes.query_analyzer.ChatOllama"):
            with patch("nodes.query_analyzer._PROMPT") as mock_prompt:
                mock_chain = MagicMock()
                mock_chain.invoke.return_value = "not valid json at all !!!"
                mock_prompt.__or__ = MagicMock(return_value=MagicMock(
                    __or__=MagicMock(return_value=mock_chain)
                ))

                from nodes.query_analyzer import analyze_query
                result = analyze_query("hello world")

                self.assertIsInstance(result, dict)
                self.assertIn("rewritten_query", result)
                self.assertIn("needs_context", result)

    def test_langgraph_node_returns_correct_keys(self):
        """query_analyzer graph node updates the right state keys."""
        with patch("nodes.query_analyzer.analyze_query") as mock_analyze:
            mock_analyze.return_value = {
                "rewritten_query": "Flash Attention tiling algorithm",
                "query_type": "analytical",
                "key_terms": ["flash attention", "tiling"],
                "needs_context": True,
            }
            from nodes.query_analyzer import query_analyzer
            result = query_analyzer(base_state())

        self.assertIn("rewritten_query", result)
        self.assertIn("query_type", result)
        self.assertIn("key_terms", result)
        self.assertIn("needs_context", result)


# ── Test: vector_retriever ────────────────────────────────────────────────────

class VectorRetrieverTest(unittest.TestCase):

    def test_rrf_score_both_present(self):
        """RRF score is sum of reciprocal ranks."""
        from nodes.vector_retriever import _rrf_score
        score = _rrf_score(0, 0, k=60)
        self.assertAlmostEqual(score, 1/60 + 1/60, places=6)

    def test_rrf_score_bm25_missing(self):
        """When bm25_rank=-1, only dense contributes."""
        from nodes.vector_retriever import _rrf_score
        score = _rrf_score(0, -1, k=60)
        self.assertAlmostEqual(score, 1/60, places=6)

    def test_tokenizer_lower_alphanumeric(self):
        """_tokenize lowercases and strips punctuation."""
        from nodes.vector_retriever import _tokenize
        tokens = _tokenize("Flash-Attention: O(N²) Memory!")
        self.assertIn("flash", tokens)
        self.assertIn("attention", tokens)
        self.assertIn("memory", tokens)
        self.assertNotIn("Flash-Attention", tokens)

    @patch("nodes.vector_retriever._get_qdrant")
    @patch("nodes.vector_retriever._get_embed")
    def test_hybrid_retrieve_empty_results(self, mock_embed, mock_qdrant):
        """hybrid_retrieve returns [] when Qdrant returns nothing."""
        # Return numpy array so .tolist() works
        mock_embed.return_value.embed.return_value = iter([np.array([0.1] * 1024)])
        mock_qdrant.return_value.query_points.return_value.points = []

        from nodes.vector_retriever import hybrid_retrieve
        result = hybrid_retrieve("test query")
        self.assertEqual(result, [])

    @patch("nodes.vector_retriever._get_qdrant")
    @patch("nodes.vector_retriever._get_embed")
    def test_hybrid_retrieve_returns_rrf_sorted(self, mock_embed, mock_qdrant):
        """hybrid_retrieve returns docs sorted by RRF score descending."""
        mock_embed.return_value.embed.return_value = iter([np.array([0.1] * 1024)])

        # Simulate 3 Qdrant results
        def make_result(score, text, chunk_id):
            r = MagicMock()
            r.score = score
            r.payload = {"text": text, "source": "test.pdf", "h1": "S1", "h2": "S2",
                         "chunk_id": chunk_id}
            return r

        mock_qdrant.return_value.query_points.return_value.points = [
            make_result(0.9, "Flash Attention tiling SRAM HBM memory", "c1"),
            make_result(0.7, "Adam optimizer gradient descent", "c2"),
            make_result(0.8, "Multi-head attention transformer architecture", "c3"),
        ]

        from nodes.vector_retriever import hybrid_retrieve
        docs = hybrid_retrieve("Flash Attention memory", top_k=3)

        self.assertGreater(len(docs), 0)
        # Verify sorted by score desc (RRF)
        scores = [d["score"] for d in docs]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # Verify RetrievedDoc structure
        self.assertIn("text", docs[0])
        self.assertIn("dense_rank", docs[0])
        self.assertIn("bm25_rank", docs[0])


# ── Test: relevance_grader ────────────────────────────────────────────────────

class RelevanceGraderTest(unittest.TestCase):

    @patch("nodes.relevance_grader.ChatOllama")
    @patch("nodes.relevance_grader._PROMPT")
    def test_grade_doc_high_score(self, mock_prompt, mock_llm_cls):
        """grade_doc returns a high score for a clearly relevant doc."""
        response_str = json.dumps({"score": 0.95, "reason": "Directly relevant"})
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = response_str
        mock_prompt.__or__ = MagicMock(return_value=MagicMock(
            __or__=MagicMock(return_value=mock_chain)
        ))
        mock_llm_cls.return_value = MagicMock()

        from nodes.relevance_grader import grade_doc
        doc    = make_retrieved_doc()
        graded = grade_doc("Flash Attention memory", doc)

        self.assertIn("relevance_score", graded)
        self.assertIn("relevance_reason", graded)
        self.assertGreaterEqual(graded["relevance_score"], 0.0)
        self.assertLessEqual(graded["relevance_score"], 1.0)

    @patch("nodes.relevance_grader.ChatOllama")
    @patch("nodes.relevance_grader._PROMPT")
    def test_grade_doc_bad_json_fallback(self, mock_prompt, mock_llm_cls):
        """grade_doc gives score=0.0 when LLM returns bad JSON."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "I cannot determine the relevance."
        mock_prompt.__or__ = MagicMock(return_value=MagicMock(
            __or__=MagicMock(return_value=mock_chain)
        ))
        mock_llm_cls.return_value = MagicMock()

        from nodes.relevance_grader import grade_doc
        graded = grade_doc("test query", make_retrieved_doc())
        self.assertEqual(graded["relevance_score"], 0.0)
        self.assertEqual(graded["relevance_reason"], "parse error")

    def test_grade_all_docs_threshold_filter(self):
        """grade_all_docs filters out docs below threshold."""
        with patch("nodes.relevance_grader.grade_doc") as mock_grade:
            def fake_grade(query, doc, llm=None):
                return GradedDoc(
                    text=doc["text"], source=doc["source"],
                    score=doc["score"], relevance_score=doc["dense_rank"] * 0.3,
                    relevance_reason="test", h1="", h2="", chunk_id=doc["chunk_id"],
                )
            mock_grade.side_effect = fake_grade

            from nodes.relevance_grader import grade_all_docs
            docs = [
                make_retrieved_doc(dense_rank=0, chunk_id="c0"),  # score=0.0 → filtered
                make_retrieved_doc(dense_rank=1, chunk_id="c1"),  # score=0.3 → filtered
                make_retrieved_doc(dense_rank=2, chunk_id="c2"),  # score=0.6 → kept
                make_retrieved_doc(dense_rank=3, chunk_id="c3"),  # score=0.9 → kept
            ]
            result = grade_all_docs("test", docs, threshold=0.5, parallel=False)

        self.assertEqual(len(result), 2)
        # Should be sorted descending by relevance
        self.assertGreater(result[0]["relevance_score"], result[1]["relevance_score"])


# ── Test: generator ───────────────────────────────────────────────────────────

class GeneratorTest(unittest.TestCase):

    def test_no_docs_returns_fallback(self):
        """generate_answer with empty docs returns a no-context message."""
        from nodes.generator import generate_answer
        answer = generate_answer("What is Flash Attention?", graded_docs=[])
        self.assertIn("couldn't find", answer.lower())

    @patch("nodes.generator.ChatOllama")
    @patch("nodes.generator._PROMPTS")
    def test_query_type_selects_correct_prompt(self, mock_prompts, mock_llm_cls):
        """generate_answer selects the prompt matching query_type."""
        answer_str = "Flash Attention uses tiling [flash_attention.pdf]."
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = answer_str
        mock_template = MagicMock()
        # ChatPromptTemplate | llm | parser chain must return a string
        mock_template.__or__ = MagicMock(return_value=MagicMock(
            __or__=MagicMock(return_value=mock_chain)
        ))
        mock_prompts.__getitem__ = MagicMock(return_value=mock_template)
        mock_prompts.get        = MagicMock(return_value=mock_template)
        mock_llm_cls.return_value = MagicMock()

        from nodes.generator import generate_answer
        answer = generate_answer(
            "How does Flash Attention work?",
            graded_docs=[make_graded_doc()],
            query_type="factual",
        )
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)

    def test_build_context_respects_budget(self):
        """_build_context truncates when max_chars is exceeded."""
        from nodes.generator import _build_context
        # Create docs whose combined text exceeds 200 chars
        long_docs = [make_graded_doc(text="x" * 150, chunk_id=f"c{i}") for i in range(5)]
        context   = _build_context(long_docs, max_chars=200)
        self.assertLessEqual(len(context), 500)   # some overhead for headers

    def test_langgraph_node_increments_retry_strict(self):
        """generator node uses strict mode when generation_retry_count > 0."""
        with patch("nodes.generator.generate_answer") as mock_gen:
            mock_gen.return_value = "test answer"
            from nodes.generator import generator
            state = base_state(
                graded_docs=[make_graded_doc()],
                generation_retry_count=1,  # strict kicks in at >= 1
            )
            generator(state)
            _, kwargs = mock_gen.call_args
            self.assertTrue(kwargs.get("strict", False))


# ── Test: hallucination_checker ───────────────────────────────────────────────

class HallucinationCheckerTest(unittest.TestCase):

    def test_empty_inputs_return_ungrounded(self):
        """check_hallucination with no docs returns is_grounded=False."""
        from nodes.hallucination_checker import check_hallucination
        result = check_hallucination("some answer", graded_docs=[])
        self.assertFalse(result["is_grounded"])
        self.assertEqual(result["grounding_score"], 0.0)

    @patch("nodes.hallucination_checker.ChatOllama")
    @patch("nodes.hallucination_checker._PROMPT")
    def test_grounded_answer(self, mock_prompt, mock_llm_cls):
        """check_hallucination returns is_grounded=True for a well-sourced answer."""
        response_str = json.dumps({
            "is_grounded": True,
            "grounding_score": 0.95,
            "unsupported_claims": [],
        })
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = response_str
        mock_prompt.__or__ = MagicMock(return_value=MagicMock(
            __or__=MagicMock(return_value=mock_chain)
        ))
        mock_llm_cls.return_value = MagicMock()

        from nodes.hallucination_checker import check_hallucination
        result = check_hallucination(
            "Flash Attention uses tiling [flash_attention.pdf].",
            graded_docs=[make_graded_doc()],
        )
        self.assertTrue(result["is_grounded"])
        self.assertGreater(result["grounding_score"], 0.5)
        self.assertEqual(result["unsupported_claims"], [])

    @patch("nodes.hallucination_checker.ChatOllama")
    @patch("nodes.hallucination_checker._PROMPT")
    def test_hallucinated_answer(self, mock_prompt, mock_llm_cls):
        """check_hallucination detects fabricated claims."""
        response_str = json.dumps({
            "is_grounded": False,
            "grounding_score": 0.2,
            "unsupported_claims": [
                "Flash Attention was invented at Stanford in 2019.",
                "It achieves 10x speedup on V100 GPUs.",
            ],
        })
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = response_str
        mock_prompt.__or__ = MagicMock(return_value=MagicMock(
            __or__=MagicMock(return_value=mock_chain)
        ))
        mock_llm_cls.return_value = MagicMock()

        from nodes.hallucination_checker import check_hallucination
        result = check_hallucination(
            "Flash Attention was invented at Stanford in 2019 and achieves 10x speedup.",
            graded_docs=[make_graded_doc()],
        )
        self.assertFalse(result["is_grounded"])
        self.assertEqual(len(result["unsupported_claims"]), 2)

    def test_node_appends_caveat_when_hallucinated(self):
        """hallucination_checker node appends ⚠ caveat to final_answer."""
        with patch("nodes.hallucination_checker.check_hallucination") as mock_check:
            mock_check.return_value = {
                "is_grounded": False,
                "grounding_score": 0.2,
                "unsupported_claims": ["Fabricated claim about Flash Attention."],
            }
            from nodes.hallucination_checker import hallucination_checker
            result = hallucination_checker(base_state(
                answer="Test answer with fabrication.",
                graded_docs=[make_graded_doc()],
            ))
        self.assertIn("⚠", result["final_answer"])
        self.assertIn("Fabricated claim", result["final_answer"])

    def test_node_clean_answer_when_grounded(self):
        """hallucination_checker node passes answer through unchanged when grounded."""
        with patch("nodes.hallucination_checker.check_hallucination") as mock_check:
            mock_check.return_value = {
                "is_grounded": True,
                "grounding_score": 0.95,
                "unsupported_claims": [],
            }
            from nodes.hallucination_checker import hallucination_checker
            answer = "Flash Attention uses tiling [flash_attention.pdf]."
            result = hallucination_checker(base_state(
                answer=answer,
                graded_docs=[make_graded_doc()],
            ))
        self.assertEqual(result["final_answer"], answer)


# ── Test: graph routing ───────────────────────────────────────────────────────

class GraphRoutingTest(unittest.TestCase):

    def test_route_after_analyzer_no_context(self):
        from graph import route_after_analyzer
        self.assertEqual(route_after_analyzer(base_state(needs_context=False)), "direct_answer")

    def test_route_after_analyzer_with_context(self):
        from graph import route_after_analyzer
        self.assertEqual(route_after_analyzer(base_state(needs_context=True)), "vector_retriever")

    # ── EDGE A routing ────────────────────────────────────────────────────────

    def test_edge_a_docs_found_goes_to_generator(self):
        """Edge A: when graded docs exist, proceed to generator."""
        from graph import route_after_grader
        state = base_state(graded_docs=[make_graded_doc()])
        self.assertEqual(route_after_grader(state), "generator")

    def test_edge_a_no_docs_first_attempt_loops_back(self):
        """Edge A: no docs + first attempt → record_failed_query (retrieval retry)."""
        from graph import route_after_grader
        state = base_state(graded_docs=[], retrieval_retry_count=0)
        self.assertEqual(route_after_grader(state), "record_failed_query")

    def test_edge_a_no_docs_retries_left(self):
        """Edge A: no docs + retries remaining → record_failed_query."""
        from graph import route_after_grader
        state = base_state(graded_docs=[], retrieval_retry_count=2)
        self.assertEqual(route_after_grader(state), "record_failed_query")

    def test_edge_a_no_docs_retries_exhausted(self):
        """Edge A: no docs + MAX_RETRIEVAL_RETRIES reached → no_context."""
        from graph import route_after_grader, MAX_RETRIEVAL_RETRIES
        state = base_state(graded_docs=[], retrieval_retry_count=MAX_RETRIEVAL_RETRIES)
        self.assertEqual(route_after_grader(state), "no_context")

    # ── EDGE B routing ────────────────────────────────────────────────────────

    def test_edge_b_grounded_goes_to_end(self):
        """Edge B: grounded answer → END."""
        from graph import route_after_checker
        from langgraph.graph import END
        state = base_state(is_grounded=True, generation_retry_count=0)
        self.assertEqual(route_after_checker(state), END)

    def test_edge_b_not_grounded_retries_left(self):
        """Edge B: not grounded + retries remaining → prepare_strict_gen."""
        from graph import route_after_checker
        state = base_state(is_grounded=False, generation_retry_count=0)
        self.assertEqual(route_after_checker(state), "prepare_strict_gen")

    def test_edge_b_not_grounded_retries_exhausted(self):
        """Edge B: not grounded + MAX_GENERATION_RETRIES reached → END."""
        from graph import route_after_checker, MAX_GENERATION_RETRIES
        from langgraph.graph import END
        state = base_state(is_grounded=False, generation_retry_count=MAX_GENERATION_RETRIES)
        self.assertEqual(route_after_checker(state), END)


# ── Test: Edge A transition node ──────────────────────────────────────────────

class EdgeALoopTest(unittest.TestCase):

    def test_record_failed_query_appends_query(self):
        """record_failed_query adds rewritten_query to failed_queries."""
        from graph import record_failed_query
        state = base_state(
            rewritten_query="Flash Attention tiling algorithm",
            failed_queries=[],
            retrieval_retry_count=0,
        )
        result = record_failed_query(state)
        self.assertIn("Flash Attention tiling algorithm", result["failed_queries"])

    def test_record_failed_query_increments_counter(self):
        """record_failed_query increments retrieval_retry_count."""
        from graph import record_failed_query
        state = base_state(retrieval_retry_count=1, failed_queries=[])
        result = record_failed_query(state)
        self.assertEqual(result["retrieval_retry_count"], 2)

    def test_record_failed_query_no_duplicates(self):
        """record_failed_query does not add a query that is already in failed_queries."""
        from graph import record_failed_query
        q     = "Flash Attention memory"
        state = base_state(
            rewritten_query=q,
            failed_queries=[q],
            retrieval_retry_count=1,
        )
        result = record_failed_query(state)
        self.assertEqual(result["failed_queries"].count(q), 1)

    def test_record_failed_query_clears_docs(self):
        """record_failed_query resets retrieved_docs and graded_docs."""
        from graph import record_failed_query
        state = base_state(
            retrieved_docs=[make_retrieved_doc()],
            graded_docs=[make_graded_doc()],
        )
        result = record_failed_query(state)
        self.assertEqual(result["retrieved_docs"], [])
        self.assertEqual(result["graded_docs"], [])

    def test_query_analyzer_uses_retry_prompt_on_loop(self):
        """query_analyzer node passes failed_queries to analyze_query on retry."""
        with patch("nodes.query_analyzer.analyze_query") as mock_analyze:
            mock_analyze.return_value = {
                "rewritten_query": "attention mechanism memory efficiency neural networks",
                "query_type": "analytical",
                "key_terms": ["attention", "memory", "neural"],
                "needs_context": True,
            }
            from nodes.query_analyzer import query_analyzer
            state = base_state(
                failed_queries=["Flash Attention tiling"],
                retrieval_retry_count=1,
            )
            query_analyzer(state)

            _, kwargs = mock_analyze.call_args
            self.assertEqual(kwargs.get("attempt", 0), 1)
            self.assertIn("Flash Attention tiling", kwargs.get("failed_queries", []))


# ── Test: Edge B transition node ──────────────────────────────────────────────

class EdgeBLoopTest(unittest.TestCase):

    def test_prepare_strict_gen_builds_hint_from_claims(self):
        """prepare_strict_gen encodes unsupported claims into generation_hint."""
        from graph import prepare_strict_gen
        state = base_state(
            unsupported_claims=["Flash Attention was invented in 2019."],
            grounding_score=0.3,
            generation_retry_count=0,
        )
        result = prepare_strict_gen(state)
        self.assertIn("generation_hint", result)
        self.assertIn("2019", result["generation_hint"])
        self.assertGreater(len(result["generation_hint"]), 20)

    def test_prepare_strict_gen_increments_counter(self):
        """prepare_strict_gen increments generation_retry_count."""
        from graph import prepare_strict_gen
        state = base_state(generation_retry_count=1, unsupported_claims=[])
        result = prepare_strict_gen(state)
        self.assertEqual(result["generation_retry_count"], 2)

    def test_prepare_strict_gen_clears_answer(self):
        """prepare_strict_gen clears the old answer so generator starts fresh."""
        from graph import prepare_strict_gen
        state = base_state(answer="old hallucinated answer", unsupported_claims=[])
        result = prepare_strict_gen(state)
        self.assertEqual(result["answer"], "")

    def test_prepare_strict_gen_no_claims_generic_hint(self):
        """prepare_strict_gen produces a generic hint when unsupported_claims is empty."""
        from graph import prepare_strict_gen
        state = base_state(
            unsupported_claims=[],
            grounding_score=0.2,
            generation_retry_count=0,
        )
        result = prepare_strict_gen(state)
        self.assertIn("conservative", result["generation_hint"].lower())

    def test_generator_reads_generation_hint(self):
        """generator node passes generation_hint to generate_answer."""
        with patch("nodes.generator.generate_answer") as mock_gen:
            mock_gen.return_value = "strict answer"
            from nodes.generator import generator
            state = base_state(
                graded_docs=[make_graded_doc()],
                generation_retry_count=1,
                generation_hint="Do not infer beyond the source text.",
            )
            generator(state)
            _, kwargs = mock_gen.call_args
            self.assertEqual(kwargs.get("generation_hint"), "Do not infer beyond the source text.")
            self.assertTrue(kwargs.get("strict", False))


# ── Test: hallucination_checker parse-failure is fail-closed (WR-01) ──────────

class HallucinationCheckerParseFailTest(unittest.TestCase):

    @patch("nodes.hallucination_checker.ChatOllama")
    @patch("nodes.hallucination_checker._PROMPT")
    def test_parse_failure_returns_ungrounded(self, mock_prompt, mock_llm_cls):
        """WR-01: JSON parse failure must return is_grounded=False (fail-closed)."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "this is not json"  # deliberately broken
        mock_prompt.__or__ = MagicMock(return_value=MagicMock(
            __or__=MagicMock(return_value=mock_chain)
        ))
        mock_llm_cls.return_value = MagicMock()

        from nodes.hallucination_checker import check_hallucination
        result = check_hallucination("some answer", graded_docs=[make_graded_doc()])

        self.assertFalse(
            result["is_grounded"],
            "Parse failure must be fail-closed (is_grounded=False), not fail-open",
        )
        self.assertEqual(result["grounding_score"], 0.0)
        self.assertGreater(
            len(result["unsupported_claims"]), 0,
            "Parse failure should populate unsupported_claims with an error marker",
        )

    @patch("nodes.hallucination_checker.ChatOllama")
    @patch("nodes.hallucination_checker._PROMPT")
    def test_markdown_wrapped_json_is_parsed(self, mock_prompt, mock_llm_cls):
        """Markdown-fenced JSON (```json ... ```) is stripped and parsed correctly."""
        payload = {"is_grounded": True, "grounding_score": 0.9, "unsupported_claims": []}
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = f"```json\n{json.dumps(payload)}\n```"
        mock_prompt.__or__ = MagicMock(return_value=MagicMock(
            __or__=MagicMock(return_value=mock_chain)
        ))
        mock_llm_cls.return_value = MagicMock()

        from nodes.hallucination_checker import check_hallucination
        result = check_hallucination("some answer", graded_docs=[make_graded_doc()])

        self.assertTrue(result["is_grounded"])
        self.assertAlmostEqual(result["grounding_score"], 0.9)


# ── Test: model resolution per node ──────────────────────────────────────────

class ModelResolutionTest(unittest.TestCase):
    """
    Verifies which Ollama model each node selects under different conditions.

    Expected model assignments (from node source files):
        query_analyzer         local fallback : llama3.2:1b
                               preferred cloud: gemma4:31b-cloud  (first in list)

        relevance_grader       local fallback : qwen3.5:0.8b
                               preferred cloud: nemotron-3-super:cloud  (first in list)

        generator              local fallback : qwen3.5:0.8b
                               preferred cloud: nemotron-3-super:cloud  (first in list)

        hallucination_checker  local fallback : llama3.2:1b
                               preferred cloud: gemma4:31b-cloud  (first in list)
    """

    # ── Helper ────────────────────────────────────────────────────────────────

    def _resolve(self, preferred: list, fallback: str, available: set) -> str:
        """Run config.resolve_model with a controlled _AVAILABLE_MODELS."""
        import config as cfg
        original = cfg._AVAILABLE_MODELS
        try:
            cfg._AVAILABLE_MODELS = available
            return cfg.resolve_model(preferred, fallback)
        finally:
            cfg._AVAILABLE_MODELS = original

    # ── Scenario A: no cloud models → local fallbacks ─────────────────────────

    def test_query_analyzer_fallback_when_no_cloud(self):
        """query_analyzer uses llama3.2:1b when no cloud model is available."""
        result = self._resolve(
            preferred=["gemma4:31b-cloud", "nemotron-3-super:cloud"],
            fallback="llama3.2:1b",
            available=set(),
        )
        self.assertEqual(result, "llama3.2:1b")

    def test_relevance_grader_fallback_when_no_cloud(self):
        """relevance_grader uses qwen3.5:0.8b when no cloud model is available."""
        result = self._resolve(
            preferred=["nemotron-3-super:cloud", "gemma4:31b-cloud"],
            fallback="qwen3.5:0.8b",
            available=set(),
        )
        self.assertEqual(result, "qwen3.5:0.8b")

    def test_generator_fallback_when_no_cloud(self):
        """generator uses qwen3.5:0.8b when no cloud model is available."""
        result = self._resolve(
            preferred=["nemotron-3-super:cloud", "gemma4:31b-cloud"],
            fallback="qwen3.5:0.8b",
            available=set(),
        )
        self.assertEqual(result, "qwen3.5:0.8b")

    def test_hallucination_checker_fallback_when_no_cloud(self):
        """hallucination_checker uses llama3.2:1b when no cloud model is available."""
        result = self._resolve(
            preferred=["gemma4:31b-cloud", "nemotron-3-super:cloud"],
            fallback="llama3.2:1b",
            available=set(),
        )
        self.assertEqual(result, "llama3.2:1b")

    # ── Scenario B: cloud model present → preferred model wins ────────────────

    def test_query_analyzer_prefers_gemma_when_available(self):
        """query_analyzer picks gemma4:31b-cloud when it is in _AVAILABLE_MODELS."""
        result = self._resolve(
            preferred=["gemma4:31b-cloud", "nemotron-3-super:cloud"],
            fallback="llama3.2:1b",
            available={"gemma4:31b-cloud", "llama3.2:1b"},
        )
        self.assertEqual(result, "gemma4:31b-cloud")

    def test_query_analyzer_second_preference_when_first_missing(self):
        """query_analyzer falls through to nemotron when gemma is absent."""
        result = self._resolve(
            preferred=["gemma4:31b-cloud", "nemotron-3-super:cloud"],
            fallback="llama3.2:1b",
            available={"nemotron-3-super:cloud"},
        )
        self.assertEqual(result, "nemotron-3-super:cloud")

    def test_relevance_grader_prefers_nemotron_when_available(self):
        """relevance_grader picks nemotron-3-super:cloud when it is in _AVAILABLE_MODELS."""
        result = self._resolve(
            preferred=["nemotron-3-super:cloud", "gemma4:31b-cloud"],
            fallback="qwen3.5:0.8b",
            available={"nemotron-3-super:cloud", "qwen3.5:0.8b"},
        )
        self.assertEqual(result, "nemotron-3-super:cloud")

    def test_generator_prefers_nemotron_when_available(self):
        """generator picks nemotron-3-super:cloud when it is in _AVAILABLE_MODELS."""
        result = self._resolve(
            preferred=["nemotron-3-super:cloud", "gemma4:31b-cloud"],
            fallback="qwen3.5:0.8b",
            available={"nemotron-3-super:cloud"},
        )
        self.assertEqual(result, "nemotron-3-super:cloud")

    def test_hallucination_checker_prefers_gemma_when_available(self):
        """hallucination_checker picks gemma4:31b-cloud when it is in _AVAILABLE_MODELS."""
        result = self._resolve(
            preferred=["gemma4:31b-cloud", "nemotron-3-super:cloud"],
            fallback="llama3.2:1b",
            available={"gemma4:31b-cloud"},
        )
        self.assertEqual(result, "gemma4:31b-cloud")

    # ── Scenario C: live module constants match expectations ──────────────────

    def test_query_analyzer_module_constant_is_string(self):
        """_ANALYZER_MODEL is a non-empty string (resolved at import time)."""
        from nodes.query_analyzer import _ANALYZER_MODEL
        self.assertIsInstance(_ANALYZER_MODEL, str)
        self.assertGreater(len(_ANALYZER_MODEL), 0)

    def test_relevance_grader_module_constant_is_string(self):
        """_GRADER_MODEL is a non-empty string."""
        from nodes.relevance_grader import _GRADER_MODEL
        self.assertIsInstance(_GRADER_MODEL, str)
        self.assertGreater(len(_GRADER_MODEL), 0)

    def test_generator_module_constant_is_string(self):
        """_GENERATOR_MODEL is a non-empty string."""
        from nodes.generator import _GENERATOR_MODEL
        self.assertIsInstance(_GENERATOR_MODEL, str)
        self.assertGreater(len(_GENERATOR_MODEL), 0)

    def test_hallucination_checker_module_constant_is_string(self):
        """_CHECKER_MODEL is a non-empty string."""
        from nodes.hallucination_checker import _CHECKER_MODEL
        self.assertIsInstance(_CHECKER_MODEL, str)
        self.assertGreater(len(_CHECKER_MODEL), 0)

    # ── Scenario D: resolve_model edge cases ──────────────────────────────────

    def test_resolve_model_returns_fallback_when_none_match(self):
        """resolve_model always returns fallback when no preferred model is available."""
        result = self._resolve(
            preferred=["cloud-x", "cloud-y"],
            fallback="local:1b",
            available={"some-other-model"},
        )
        self.assertEqual(result, "local:1b")

    def test_resolve_model_priority_order_respected(self):
        """resolve_model returns the FIRST match in the preferred list."""
        result = self._resolve(
            preferred=["first-choice:cloud", "second-choice:cloud"],
            fallback="local:1b",
            available={"first-choice:cloud", "second-choice:cloud"},
        )
        self.assertEqual(result, "first-choice:cloud",
                         "Must return first match, not second")

    def test_resolve_model_empty_preferred_returns_fallback(self):
        """resolve_model returns fallback when preferred list is empty."""
        result = self._resolve(
            preferred=[],
            fallback="local:1b",
            available={"cloud-model:x"},
        )
        self.assertEqual(result, "local:1b")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()

    # Run specific class if given as argument
    if len(sys.argv) > 1 and sys.argv[1] in globals():
        suite.addTests(loader.loadTestsFromTestCase(globals()[sys.argv[1]]))
        sys.argv.pop(1)
    else:
        for cls in [
            QueryAnalyzerTest,
            VectorRetrieverTest,
            RelevanceGraderTest,
            GeneratorTest,
            HallucinationCheckerTest,
            HallucinationCheckerParseFailTest,
            GraphRoutingTest,
            EdgeALoopTest,
            EdgeBLoopTest,
            ModelResolutionTest,
        ]:
            suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
