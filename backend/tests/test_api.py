"""
API endpoint tests for the FastAPI RAG application.

Covers:
- POST /api/query  – query processing with and without an existing session
- GET  /api/courses – course statistics
"""

import pytest


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------


class TestQueryEndpoint:
    def test_returns_200_with_valid_payload(self, client, mock_rag, sample_query_payload):
        response = client.post("/api/query", json=sample_query_payload)
        assert response.status_code == 200

    def test_response_contains_required_fields(self, client, mock_rag, sample_query_payload):
        response = client.post("/api/query", json=sample_query_payload)
        body = response.json()
        assert "answer" in body
        assert "sources" in body
        assert "session_id" in body

    def test_answer_matches_rag_output(self, client, mock_rag, sample_query_payload):
        mock_rag.query.return_value = ("Mocked answer text.", ["file_a.txt"])
        response = client.post("/api/query", json=sample_query_payload)
        assert response.json()["answer"] == "Mocked answer text."

    def test_sources_is_a_list(self, client, mock_rag, sample_query_payload):
        response = client.post("/api/query", json=sample_query_payload)
        assert isinstance(response.json()["sources"], list)

    def test_auto_creates_session_when_none_provided(self, client, mock_rag, sample_query_payload):
        """Without a session_id in the request, the app should create one."""
        mock_rag.session_manager.create_session.return_value = "new-session-999"
        response = client.post("/api/query", json=sample_query_payload)
        body = response.json()
        assert body["session_id"] == "new-session-999"
        mock_rag.session_manager.create_session.assert_called_once()

    def test_uses_provided_session_id(
        self, client, mock_rag, sample_query_payload_with_session
    ):
        """When session_id is present in the request it must be forwarded to RAG and echoed back."""
        response = client.post("/api/query", json=sample_query_payload_with_session)
        body = response.json()
        assert body["session_id"] == "existing-session-xyz"
        # create_session should NOT have been called
        mock_rag.session_manager.create_session.assert_not_called()

    def test_rag_query_called_with_correct_args(
        self, client, mock_rag, sample_query_payload_with_session
    ):
        client.post("/api/query", json=sample_query_payload_with_session)
        mock_rag.query.assert_called_once_with(
            sample_query_payload_with_session["query"],
            "existing-session-xyz",
        )

    def test_returns_500_on_rag_exception(self, client, mock_rag, sample_query_payload):
        mock_rag.query.side_effect = RuntimeError("RAG failure")
        response = client.post("/api/query", json=sample_query_payload)
        assert response.status_code == 500

    def test_returns_422_for_missing_query_field(self, client):
        """Pydantic validation: 'query' is required."""
        response = client.post("/api/query", json={})
        assert response.status_code == 422

    def test_returns_422_for_wrong_content_type(self, client):
        response = client.post(
            "/api/query",
            data="not json",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code in (415, 422)


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------


class TestCoursesEndpoint:
    def test_returns_200(self, client, mock_rag):
        response = client.get("/api/courses")
        assert response.status_code == 200

    def test_response_contains_required_fields(self, client, mock_rag):
        body = client.get("/api/courses").json()
        assert "total_courses" in body
        assert "course_titles" in body

    def test_total_courses_is_integer(self, client, mock_rag):
        body = client.get("/api/courses").json()
        assert isinstance(body["total_courses"], int)

    def test_course_titles_is_list(self, client, mock_rag):
        body = client.get("/api/courses").json()
        assert isinstance(body["course_titles"], list)

    def test_values_match_rag_analytics(self, client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["Course X", "Course Y", "Course Z"],
        }
        body = client.get("/api/courses").json()
        assert body["total_courses"] == 3
        assert body["course_titles"] == ["Course X", "Course Y", "Course Z"]

    def test_returns_500_on_rag_exception(self, client, mock_rag):
        mock_rag.get_course_analytics.side_effect = RuntimeError("DB error")
        response = client.get("/api/courses")
        assert response.status_code == 500
