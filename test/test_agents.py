"""
Tests for agents.py: helpers, LearningAgents, and LearningOrchestrator.answer_question.

Run from project root:
  python -m pytest test/test_agents.py -v
  or
  python -m unittest discover -s test -p "test_*.py" -v
"""
import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Project root on path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Stub 'retrieval' so importing agents (and tools.retrieval_tool) does not load Chroma/sentence_transformers
_fake_retrieval = MagicMock()
_fake_retrieval.RetrievalService = MagicMock()
sys.modules["retrieval"] = _fake_retrieval

import agents


# ----- Unit tests: _format_retrieved_context -----


class TestFormatRetrievedContext(unittest.TestCase):
    def test_empty_list(self) -> None:
        self.assertEqual(
            agents._format_retrieved_context([]),
            "(No segments retrieved)",
        )

    def test_single_chunk(self) -> None:
        chunks = [
            {"document": "Hello world", "metadata": {"course": "CS101", "start": 0}},
        ]
        out = agents._format_retrieved_context(chunks)
        self.assertIn("[Segment 1]", out)
        self.assertIn("Content: Hello world", out)
        self.assertIn("Metadata:", out)
        self.assertIn("CS101", out)

    def test_multiple_chunks(self) -> None:
        chunks = [
            {"document": "First", "metadata": {"a": 1}},
            {"document": "Second", "metadata": {}},
        ]
        out = agents._format_retrieved_context(chunks)
        self.assertIn("[Segment 1]", out)
        self.assertIn("[Segment 2]", out)
        self.assertIn("First", out)
        self.assertIn("Second", out)

    def test_missing_document_or_metadata(self) -> None:
        chunks = [{}]
        out = agents._format_retrieved_context(chunks)
        self.assertIn("[Segment 1]", out)
        self.assertIn("Content: ", out)
        self.assertIn("Metadata: {}", out)


# ----- Unit tests: _parse_groundedness_result -----


class TestParseGroundednessResult(unittest.TestCase):
    def test_valid_json_supported(self) -> None:
        raw = '{"status": "SUPPORTED", "reason": "All facts supported"}'
        out = agents._parse_groundedness_result(raw)
        self.assertEqual(out["status"], "SUPPORTED")
        self.assertIn("reason", out)

    def test_valid_json_unsupported(self) -> None:
        raw = '{"status": "UNSUPPORTED", "reason": "Fake citation"}'
        out = agents._parse_groundedness_result(raw)
        self.assertEqual(out["status"], "UNSUPPORTED")

    def test_json_with_extra_text_before_after(self) -> None:
        raw = 'Here is the result: {"status": "SUPPORTED", "reason": "ok"} Thanks.'
        out = agents._parse_groundedness_result(raw)
        self.assertEqual(out["status"], "SUPPORTED")

    def test_json_reason_with_braces(self) -> None:
        raw = '{"status": "UNSUPPORTED", "reason": "Reason with {nested} braces"}'
        out = agents._parse_groundedness_result(raw)
        self.assertEqual(out["status"], "UNSUPPORTED")

    def test_unparseable_defaults_to_unsupported(self) -> None:
        out = agents._parse_groundedness_result("not json at all")
        self.assertEqual(out["status"], "UNSUPPORTED")
        self.assertIn("reason", out)

    def test_fallback_supported_in_text(self) -> None:
        out = agents._parse_groundedness_result(
            "The answer is fully supported. status: SUPPORTED."
        )
        self.assertEqual(out["status"], "SUPPORTED")


# ----- Unit tests: LearningAgents -----


class TestLearningAgents(unittest.TestCase):
    @patch("agents.RetrievalTool")
    def test_create_retrieval_agent(self, mock_retrieval_tool: MagicMock) -> None:
        la = agents.LearningAgents()
        agent = la.create_retrieval_agent()
        self.assertIsNotNone(agent)
        self.assertEqual(agent.role, "Lecture Retrieval Agent")
        self.assertIn("retrieve", (agent.goal or "").lower())

    @patch("agents.RetrievalTool")
    def test_create_check_groundedness_agent(self, mock_retrieval_tool: MagicMock) -> None:
        la = agents.LearningAgents()
        agent = la.create_check_groundedness_agent()
        self.assertIsNotNone(agent)
        self.assertIn("Groundedness", agent.role)
        self.assertIn("JSON", (agent.goal or ""))

    @patch("agents.RetrievalTool")
    def test_create_search_agent(self, mock_retrieval_tool: MagicMock) -> None:
        la = agents.LearningAgents()
        agent = la.create_search_agent()
        self.assertIsNotNone(agent)
        self.assertIn("Search", agent.role)


# ----- Integration-style tests: answer_question (mocked Crew & RetrievalService) -----


def _run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestAnswerQuestion(unittest.TestCase):
    def _mock_crew_output(self, raw: str) -> MagicMock:
        m = MagicMock()
        m.raw = raw
        return m

    @patch("agents.observe", lambda *a, **kw: (lambda f: f))
    @patch("agents.RetrievalService")
    @patch("agents.Crew")
    def test_answer_question_returns_dict_with_response(
        self, mock_crew_cls: MagicMock, mock_retrieval_service_cls: MagicMock
    ) -> None:
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve_vector.return_value = [
            {"document": "Some content", "metadata": {"course": "X"}},
        ]
        mock_retrieval_service_cls.return_value = mock_retrieval

        retrieval_out = self._mock_crew_output("Retrieval answer here")
        check_out = self._mock_crew_output(
            '{"status": "SUPPORTED", "reason": "grounded"}'
        )
        call_results = [retrieval_out, check_out]

        async def fake_kickoff_async(*args, **kwargs):
            return call_results.pop(0) if call_results else self._mock_crew_output("")

        mock_crew_instance = MagicMock()
        mock_crew_instance.kickoff_async = AsyncMock(side_effect=fake_kickoff_async)
        mock_crew_cls.return_value = mock_crew_instance

        orchestrator = agents.LearningOrchestrator()
        orchestrator._retrieval_service = mock_retrieval

        async def run() -> dict:
            return await orchestrator.answer_question(
                "What is X?", user_id="u1", session_id="s1"
            )

        result = asyncio.run(run())
        self.assertIsInstance(result, dict)
        self.assertIn("response", result)
        self.assertEqual(result["response"], "Retrieval answer here")

    @patch("agents.observe", lambda *a, **kw: (lambda f: f))
    @patch("agents.RetrievalService")
    @patch("agents.Crew")
    def test_answer_question_when_unsupported_returns_search_answer(
        self, mock_crew_cls: MagicMock, mock_retrieval_service_cls: MagicMock
    ) -> None:
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve_vector.return_value = [
            {"document": "Content", "metadata": {}},
        ]
        mock_retrieval_service_cls.return_value = mock_retrieval

        retrieval_out = self._mock_crew_output("Retrieval answer")
        check_out = self._mock_crew_output(
            '{"status": "UNSUPPORTED", "reason": "not grounded"}'
        )
        search_out = self._mock_crew_output("Search-based answer with sources")
        call_results = [retrieval_out, check_out, search_out]

        async def fake_kickoff_async(*args, **kwargs):
            return call_results.pop(0) if call_results else search_out

        mock_crew_instance = MagicMock()
        mock_crew_instance.kickoff_async = AsyncMock(side_effect=fake_kickoff_async)
        mock_crew_cls.return_value = mock_crew_instance

        orchestrator = agents.LearningOrchestrator()
        orchestrator._retrieval_service = mock_retrieval

        async def run() -> dict:
            return await orchestrator.answer_question(
                "What is Y?", user_id="u2", session_id="s2"
            )

        result = asyncio.run(run())
        self.assertIsInstance(result, dict)
        self.assertIn("response", result)
        self.assertEqual(result["response"], "Search-based answer with sources")

    @patch("agents.observe", lambda *a, **kw: (lambda f: f))
    @patch("agents.RetrievalService")
    @patch("agents.Crew")
    def test_answer_question_handles_retrieval_context_unavailable(
        self, mock_crew_cls: MagicMock, mock_retrieval_service_cls: MagicMock
    ) -> None:
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve_vector.side_effect = Exception("ChromaDB down")
        mock_retrieval_service_cls.return_value = mock_retrieval

        retrieval_out = self._mock_crew_output("I don't know")
        check_out = self._mock_crew_output(
            '{"status": "SUPPORTED", "reason": "minimal answer"}'
        )
        call_results = [retrieval_out, check_out]

        async def fake_kickoff_async(*args, **kwargs):
            return call_results.pop(0) if call_results else retrieval_out

        mock_crew_instance = MagicMock()
        mock_crew_instance.kickoff_async = AsyncMock(side_effect=fake_kickoff_async)
        mock_crew_cls.return_value = mock_crew_instance

        orchestrator = agents.LearningOrchestrator()
        orchestrator._retrieval_service = mock_retrieval

        async def run() -> dict:
            return await orchestrator.answer_question(
                "Question?", user_id="u3", session_id="s3"
            )

        result = asyncio.run(run())
        self.assertIn("response", result)
        self.assertEqual(result["response"], "I don't know")


if __name__ == "__main__":
    unittest.main()
