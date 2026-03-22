"""
Shared fixtures for API and unit tests.

Module-level patching runs before app.py is imported to prevent:
- RAGSystem from initializing ChromaDB / sentence-transformers
- StaticFiles from requiring the ../frontend directory to exist
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Build the mock RAGSystem instance that every test can share
# ---------------------------------------------------------------------------

mock_rag_instance = MagicMock()
mock_rag_instance.query.return_value = (
    "This is a test answer.",
    ["course_intro.txt", "module_1.txt"],
)
mock_rag_instance.get_course_analytics.return_value = {
    "total_courses": 2,
    "course_titles": ["Intro to Python", "Advanced FastAPI"],
}
mock_rag_instance.session_manager.create_session.return_value = "test-session-abc123"

# ---------------------------------------------------------------------------
# Inject mock modules before app.py is imported so the global
# `rag_system = RAGSystem(config)` line hits the mock, not real services.
# ---------------------------------------------------------------------------

sys.modules.setdefault(
    "rag_system",
    MagicMock(RAGSystem=MagicMock(return_value=mock_rag_instance)),
)
sys.modules.setdefault(
    "config",
    MagicMock(config=MagicMock()),
)

# Patch StaticFiles so app.mount("/", StaticFiles(...)) succeeds even though
# ../frontend doesn't exist in the test environment.
with patch("fastapi.staticfiles.StaticFiles", MagicMock()):
    import app as _app_module  # noqa: E402  (import not at top)

_fastapi_app = _app_module.app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def client():
    """HTTP test client for the FastAPI app (shared across the session)."""
    with TestClient(_fastapi_app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def mock_rag():
    """
    Return the shared RAGSystem mock and reset call history after each test
    so tests don't interfere with each other's assertions.
    """
    yield mock_rag_instance
    mock_rag_instance.reset_mock(return_value=True, side_effect=True)
    # Restore defaults that reset_mock wipes out
    mock_rag_instance.query.return_value = (
        "This is a test answer.",
        ["course_intro.txt", "module_1.txt"],
    )
    mock_rag_instance.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Intro to Python", "Advanced FastAPI"],
    }
    mock_rag_instance.session_manager.create_session.return_value = (
        "test-session-abc123"
    )


@pytest.fixture
def sample_query_payload():
    return {"query": "What topics are covered in the Python course?"}


@pytest.fixture
def sample_query_payload_with_session():
    return {
        "query": "What topics are covered in the Python course?",
        "session_id": "existing-session-xyz",
    }
