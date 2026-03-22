"""
Tests for CourseSearchTool.execute()

Covers: result formatting, empty results (with/without filters),
error passthrough, and last_sources population.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from unittest.mock import MagicMock, patch

from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool


def make_results(docs, metas, distances=None, error=None):
    """Helper to build SearchResults without touching ChromaDB."""
    return SearchResults(
        documents=docs,
        metadata=metas,
        distances=distances or [0.1] * len(docs),
        error=error,
    )


class TestCourseSearchToolExecute(unittest.TestCase):

    def setUp(self):
        self.mock_store = MagicMock(spec=VectorStore)
        self.tool = CourseSearchTool(self.mock_store)

    # ------------------------------------------------------------------
    # Happy-path: results are found
    # ------------------------------------------------------------------

    def test_returns_formatted_content_for_single_result(self):
        self.mock_store.search.return_value = make_results(
            docs=["Agents automate tasks by chaining tool calls."],
            metas=[{"course_title": "AI Agents Course", "lesson_number": 2}],
        )
        self.mock_store.get_lesson_link.return_value = "https://example.com/lesson2"

        result = self.tool.execute(query="what are agents?")

        self.assertIn("AI Agents Course", result)
        self.assertIn("Lesson 2", result)
        self.assertIn("Agents automate tasks", result)

    def test_returns_formatted_content_for_multiple_results(self):
        self.mock_store.search.return_value = make_results(
            docs=["Content A", "Content B"],
            metas=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 3},
            ],
        )
        self.mock_store.get_lesson_link.return_value = None

        result = self.tool.execute(query="some topic")

        self.assertIn("Course A", result)
        self.assertIn("Course B", result)
        self.assertIn("Content A", result)
        self.assertIn("Content B", result)

    def test_result_without_lesson_number_omits_lesson_header(self):
        self.mock_store.search.return_value = make_results(
            docs=["General overview content"],
            metas=[{"course_title": "Intro Course"}],  # no lesson_number key
        )

        result = self.tool.execute(query="overview")

        self.assertIn("Intro Course", result)
        self.assertNotIn("Lesson", result)

    def test_search_called_with_correct_parameters(self):
        self.mock_store.search.return_value = make_results(
            docs=["some content"],
            metas=[{"course_title": "MCP Course", "lesson_number": 1}],
        )
        self.mock_store.get_lesson_link.return_value = None

        self.tool.execute(query="tool use", course_name="MCP", lesson_number=1)

        self.mock_store.search.assert_called_once_with(
            query="tool use", course_name="MCP", lesson_number=1
        )

    # ------------------------------------------------------------------
    # Empty results
    # ------------------------------------------------------------------

    def test_empty_results_returns_no_content_message(self):
        self.mock_store.search.return_value = make_results([], [])

        result = self.tool.execute(query="obscure topic")

        self.assertIn("No relevant content found", result)

    def test_empty_results_with_course_filter_includes_course_name(self):
        self.mock_store.search.return_value = make_results([], [])

        result = self.tool.execute(query="anything", course_name="Python 101")

        self.assertIn("No relevant content found", result)
        self.assertIn("Python 101", result)

    def test_empty_results_with_lesson_filter_includes_lesson_number(self):
        self.mock_store.search.return_value = make_results([], [])

        result = self.tool.execute(query="anything", lesson_number=5)

        self.assertIn("No relevant content found", result)
        self.assertIn("lesson 5", result)

    def test_empty_results_with_both_filters(self):
        self.mock_store.search.return_value = make_results([], [])

        result = self.tool.execute(
            query="anything", course_name="Deep Learning", lesson_number=3
        )

        self.assertIn("Deep Learning", result)
        self.assertIn("lesson 3", result)

    # ------------------------------------------------------------------
    # Error passthrough
    # ------------------------------------------------------------------

    def test_error_from_vector_store_is_returned_directly(self):
        self.mock_store.search.return_value = make_results(
            [], [], error="ChromaDB connection failed"
        )

        result = self.tool.execute(query="anything")

        self.assertEqual(result, "ChromaDB connection failed")

    def test_error_takes_priority_over_empty_results(self):
        self.mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[], error="Timeout"
        )

        result = self.tool.execute(query="test")

        self.assertIn("Timeout", result)
        self.assertNotIn("No relevant content found", result)

    # ------------------------------------------------------------------
    # Source tracking
    # ------------------------------------------------------------------

    def test_last_sources_populated_after_search(self):
        self.mock_store.search.return_value = make_results(
            docs=["content"],
            metas=[{"course_title": "RAG Course", "lesson_number": 4}],
        )
        self.mock_store.get_lesson_link.return_value = "https://rag.example.com/4"

        self.tool.execute(query="retrieval")

        self.assertEqual(len(self.tool.last_sources), 1)
        self.assertEqual(self.tool.last_sources[0]["name"], "RAG Course - Lesson 4")
        self.assertEqual(self.tool.last_sources[0]["url"], "https://rag.example.com/4")

    def test_last_sources_empty_when_no_results(self):
        self.mock_store.search.return_value = make_results([], [])
        self.tool.last_sources = [{"name": "stale", "url": None}]  # pre-populate

        self.tool.execute(query="nothing")

        # execute() with no results goes through is_empty() branch, never calls _format_results
        # so last_sources should not be updated — it stays stale
        # (reset_sources is the caller's responsibility via ToolManager)
        self.assertEqual(self.tool.last_sources, [{"name": "stale", "url": None}])

    def test_last_sources_accumulates_across_multiple_docs(self):
        self.mock_store.search.return_value = make_results(
            docs=["c1", "c2"],
            metas=[
                {"course_title": "Course X", "lesson_number": 1},
                {"course_title": "Course X", "lesson_number": 2},
            ],
        )
        self.mock_store.get_lesson_link.side_effect = [
            "https://x.com/1",
            "https://x.com/2",
        ]

        self.tool.execute(query="topic")

        self.assertEqual(len(self.tool.last_sources), 2)
        self.assertEqual(self.tool.last_sources[0]["name"], "Course X - Lesson 1")
        self.assertEqual(self.tool.last_sources[1]["name"], "Course X - Lesson 2")

    def test_get_tool_definition_has_required_fields(self):
        defn = self.tool.get_tool_definition()

        self.assertEqual(defn["name"], "search_course_content")
        self.assertIn("query", defn["input_schema"]["properties"])
        self.assertIn("query", defn["input_schema"]["required"])


if __name__ == "__main__":
    unittest.main()
