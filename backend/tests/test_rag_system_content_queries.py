"""
Tests for RAGSystem content-query handling.

Verifies that:
- Content queries end up calling the search tool and returning a synthesized answer.
- Sources from the search tool are surfaced in the return value.
- General (non-course) queries bypass the search tool entirely.
- Conversation history is saved after each query.
- Sources are reset between queries so stale results don't bleed through.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from unittest.mock import MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# We want to avoid importing heavyweight dependencies (chromadb,
# sentence-transformers, anthropic) at module level, so we patch them
# during the import of rag_system.
# ---------------------------------------------------------------------------

def _make_rag_system():
    """
    Instantiate RAGSystem with all external I/O dependencies mocked out.
    Returns (rag_system, mocks_dict).
    """
    from rag_system import RAGSystem

    mock_config = MagicMock()
    mock_config.ANTHROPIC_API_KEY = "test-key"
    mock_config.ANTHROPIC_MODEL = "claude-test"
    mock_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    mock_config.CHROMA_PATH = "/tmp/test-chroma"
    mock_config.MAX_RESULTS = 5
    mock_config.MAX_HISTORY = 2
    mock_config.CHUNK_SIZE = 800
    mock_config.CHUNK_OVERLAP = 100

    with (
        patch("rag_system.VectorStore") as MockVectorStore,
        patch("rag_system.AIGenerator") as MockAIGenerator,
        patch("rag_system.SessionManager") as MockSessionManager,
        patch("rag_system.DocumentProcessor"),
    ):
        mock_vector_store = MagicMock()
        mock_ai_generator = MagicMock()
        mock_session_manager = MagicMock()

        MockVectorStore.return_value = mock_vector_store
        MockAIGenerator.return_value = mock_ai_generator
        MockSessionManager.return_value = mock_session_manager

        # Also patch the tools so they don't touch ChromaDB
        with (
            patch("rag_system.CourseSearchTool") as MockSearchTool,
            patch("rag_system.CourseOutlineTool") as MockOutlineTool,
        ):
            mock_search_tool = MagicMock()
            mock_search_tool.get_tool_definition.return_value = {
                "name": "search_course_content",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            }
            mock_search_tool.last_sources = []

            mock_outline_tool = MagicMock()
            mock_outline_tool.get_tool_definition.return_value = {
                "name": "get_course_outline",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            }

            MockSearchTool.return_value = mock_search_tool
            MockOutlineTool.return_value = mock_outline_tool

            rag = RAGSystem(mock_config)

    return rag, {
        "ai_generator": mock_ai_generator,
        "session_manager": mock_session_manager,
        "vector_store": mock_vector_store,
        "search_tool": mock_search_tool,
        "outline_tool": mock_outline_tool,
    }


class TestRAGSystemContentQueries(unittest.TestCase):

    # ------------------------------------------------------------------
    # Content queries: search tool is used
    # ------------------------------------------------------------------

    def test_content_query_returns_ai_response(self):
        rag, mocks = _make_rag_system()
        mocks["ai_generator"].generate_response.return_value = "Agents use tool calls."
        mocks["search_tool"].last_sources = []

        response, sources = rag.query("What are agents?")

        self.assertEqual(response, "Agents use tool calls.")

    def test_content_query_passes_tools_to_ai_generator(self):
        rag, mocks = _make_rag_system()
        mocks["ai_generator"].generate_response.return_value = "answer"
        mocks["search_tool"].last_sources = []

        rag.query("Explain RAG")

        call_kwargs = mocks["ai_generator"].generate_response.call_args[1]
        tools = call_kwargs.get("tools") or mocks["ai_generator"].generate_response.call_args[0]
        # tools are passed as keyword arg
        self.assertIsNotNone(call_kwargs.get("tools"))

    def test_content_query_passes_tool_manager_to_ai_generator(self):
        rag, mocks = _make_rag_system()
        mocks["ai_generator"].generate_response.return_value = "answer"
        mocks["search_tool"].last_sources = []

        rag.query("Explain embeddings")

        call_kwargs = mocks["ai_generator"].generate_response.call_args[1]
        self.assertIsNotNone(call_kwargs.get("tool_manager"))

    # ------------------------------------------------------------------
    # Sources are surfaced
    # ------------------------------------------------------------------

    def test_sources_returned_from_search_tool(self):
        rag, mocks = _make_rag_system()
        mocks["ai_generator"].generate_response.return_value = "RAG combines retrieval and generation."
        mocks["search_tool"].last_sources = [
            {"name": "RAG Course - Lesson 1", "url": "https://example.com/1"}
        ]

        response, sources = rag.query("Explain RAG")

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["name"], "RAG Course - Lesson 1")

    def test_sources_are_reset_after_query(self):
        rag, mocks = _make_rag_system()
        mocks["ai_generator"].generate_response.return_value = "answer"
        mocks["search_tool"].last_sources = [{"name": "Stale Source", "url": None}]

        rag.query("first query")

        # After query, reset_sources should have cleared last_sources
        self.assertEqual(mocks["search_tool"].last_sources, [])

    def test_no_sources_when_tool_not_used(self):
        rag, mocks = _make_rag_system()
        mocks["ai_generator"].generate_response.return_value = "Paris is the capital of France."
        mocks["search_tool"].last_sources = []   # no search happened
        mocks["outline_tool"].last_sources = []  # outline tool also idle

        response, sources = rag.query("What is the capital of France?")

        self.assertEqual(sources, [])

    # ------------------------------------------------------------------
    # Session / conversation history
    # ------------------------------------------------------------------

    def test_conversation_history_fetched_when_session_provided(self):
        rag, mocks = _make_rag_system()
        mocks["ai_generator"].generate_response.return_value = "answer"
        mocks["session_manager"].get_conversation_history.return_value = "prior context"
        mocks["search_tool"].last_sources = []

        rag.query("follow-up question", session_id="session-42")

        mocks["session_manager"].get_conversation_history.assert_called_once_with("session-42")

    def test_history_passed_to_ai_generator(self):
        rag, mocks = _make_rag_system()
        mocks["ai_generator"].generate_response.return_value = "answer"
        mocks["session_manager"].get_conversation_history.return_value = "User: hello\nAssistant: hi"
        mocks["search_tool"].last_sources = []

        rag.query("follow-up", session_id="s1")

        call_kwargs = mocks["ai_generator"].generate_response.call_args[1]
        self.assertIn("hello", call_kwargs.get("conversation_history", ""))

    def test_exchange_saved_to_session_after_query(self):
        rag, mocks = _make_rag_system()
        mocks["ai_generator"].generate_response.return_value = "Deep learning uses neural nets."
        mocks["search_tool"].last_sources = []

        rag.query("What is deep learning?", session_id="sess-7")

        mocks["session_manager"].add_exchange.assert_called_once_with(
            "sess-7",
            "What is deep learning?",
            "Deep learning uses neural nets.",
        )

    def test_no_session_lookup_when_session_id_not_provided(self):
        rag, mocks = _make_rag_system()
        mocks["ai_generator"].generate_response.return_value = "answer"
        mocks["search_tool"].last_sources = []

        rag.query("standalone question")

        mocks["session_manager"].get_conversation_history.assert_not_called()
        mocks["session_manager"].add_exchange.assert_not_called()

    # ------------------------------------------------------------------
    # Query prompt construction
    # ------------------------------------------------------------------

    def test_query_text_included_in_prompt_sent_to_generator(self):
        rag, mocks = _make_rag_system()
        mocks["ai_generator"].generate_response.return_value = "answer"
        mocks["search_tool"].last_sources = []

        rag.query("explain transformer architecture")

        call_kwargs = mocks["ai_generator"].generate_response.call_args[1]
        query_arg = call_kwargs.get("query") or mocks["ai_generator"].generate_response.call_args[0][0]
        self.assertIn("transformer architecture", query_arg)

    # ------------------------------------------------------------------
    # Multiple sequential queries: sources don't bleed across
    # ------------------------------------------------------------------

    def test_sources_isolated_across_two_sequential_queries(self):
        rag, mocks = _make_rag_system()
        mocks["ai_generator"].generate_response.return_value = "some answer"

        # First query has sources
        mocks["search_tool"].last_sources = [{"name": "Course A - Lesson 1", "url": None}]
        _, sources_first = rag.query("query with results")

        # Second query has no sources
        mocks["search_tool"].last_sources = []
        _, sources_second = rag.query("query without results")

        self.assertEqual(len(sources_first), 1)
        self.assertEqual(len(sources_second), 0)


if __name__ == "__main__":
    unittest.main()
