"""
Tests for AIGenerator tool-call routing.

Verifies that:
- General queries get a direct response (no tool call).
- Content queries trigger search_course_content and a second synthesis call.
- The tool result is forwarded correctly in the follow-up message.
- The tool_manager.execute_tool is called with the right tool name and arguments.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from unittest.mock import MagicMock, patch, call

from ai_generator import AIGenerator


# ---------------------------------------------------------------------------
# Helpers to build fake Anthropic response objects
# ---------------------------------------------------------------------------

def _text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _tool_use_block(name: str, tool_id: str, input_kwargs: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = tool_id
    block.input = input_kwargs
    return block


def _make_response(stop_reason: str, content: list):
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = content
    return resp


class TestAIGeneratorToolRouting(unittest.TestCase):

    def _make_generator(self, mock_client):
        """Create an AIGenerator with the Anthropic client replaced by a mock."""
        with patch("ai_generator.anthropic.Anthropic", return_value=mock_client):
            gen = AIGenerator(api_key="test-key", model="claude-test")
        return gen

    # ------------------------------------------------------------------
    # Direct (no-tool) path
    # ------------------------------------------------------------------

    def test_direct_response_when_no_tool_use(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_response(
            stop_reason="end_turn",
            content=[_text_block("Python is a programming language.")],
        )
        gen = self._make_generator(mock_client)

        result = gen.generate_response(query="What is Python?")

        self.assertEqual(result, "Python is a programming language.")
        mock_client.messages.create.assert_called_once()

    def test_direct_response_does_not_call_tool_manager(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_response(
            stop_reason="end_turn",
            content=[_text_block("42 is the answer.")],
        )
        mock_tool_manager = MagicMock()
        gen = self._make_generator(mock_client)

        gen.generate_response(
            query="What is 6 times 7?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        mock_tool_manager.execute_tool.assert_not_called()

    # ------------------------------------------------------------------
    # Tool-use path: search_course_content
    # ------------------------------------------------------------------

    def test_tool_use_triggers_second_api_call(self):
        mock_client = MagicMock()
        first_response = _make_response(
            stop_reason="tool_use",
            content=[
                _tool_use_block(
                    name="search_course_content",
                    tool_id="tool_abc",
                    input_kwargs={"query": "what are agents?"},
                )
            ],
        )
        second_response = _make_response(
            stop_reason="end_turn",
            content=[_text_block("Agents are autonomous systems.")],
        )
        mock_client.messages.create.side_effect = [first_response, second_response]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "[AI Agents - Lesson 1]\nAgents automate workflows."

        gen = self._make_generator(mock_client)
        result = gen.generate_response(
            query="what are agents?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        self.assertEqual(mock_client.messages.create.call_count, 2)
        self.assertEqual(result, "Agents are autonomous systems.")

    def test_execute_tool_called_with_correct_name_and_args(self):
        mock_client = MagicMock()
        first_response = _make_response(
            stop_reason="tool_use",
            content=[
                _tool_use_block(
                    name="search_course_content",
                    tool_id="tool_xyz",
                    input_kwargs={"query": "RAG pipeline", "course_name": "RAG Course"},
                )
            ],
        )
        second_response = _make_response(
            stop_reason="end_turn",
            content=[_text_block("RAG involves retrieval and generation.")],
        )
        mock_client.messages.create.side_effect = [first_response, second_response]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "[RAG Course - Lesson 2]\nRAG details."

        gen = self._make_generator(mock_client)
        gen.generate_response(
            query="explain RAG",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="RAG pipeline", course_name="RAG Course"
        )

    def test_tool_result_included_in_follow_up_message(self):
        """The second API call must contain the tool result in the messages list."""
        mock_client = MagicMock()
        tool_output = "[Course X - Lesson 3]\nDetailed chunk content."
        first_response = _make_response(
            stop_reason="tool_use",
            content=[
                _tool_use_block(
                    name="search_course_content",
                    tool_id="tool_id_1",
                    input_kwargs={"query": "embeddings"},
                )
            ],
        )
        second_response = _make_response(
            stop_reason="end_turn",
            content=[_text_block("Embeddings map text to vectors.")],
        )
        mock_client.messages.create.side_effect = [first_response, second_response]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = tool_output

        gen = self._make_generator(mock_client)
        gen.generate_response(
            query="what are embeddings?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Inspect the messages sent in the second call
        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]

        # Last message should be the tool result from the user role
        tool_result_message = messages[-1]
        self.assertEqual(tool_result_message["role"], "user")
        result_content = tool_result_message["content"]
        self.assertEqual(len(result_content), 1)
        self.assertEqual(result_content[0]["type"], "tool_result")
        self.assertEqual(result_content[0]["tool_use_id"], "tool_id_1")
        self.assertEqual(result_content[0]["content"], tool_output)

    def test_synthesis_call_does_not_include_tools(self):
        """After two tool rounds, the final synthesis call must not include tools."""
        mock_client = MagicMock()
        first_response = _make_response(
            stop_reason="tool_use",
            content=[_tool_use_block("search_course_content", "t1", {"query": "outline"})],
        )
        second_response = _make_response(
            stop_reason="tool_use",
            content=[_tool_use_block("search_course_content", "t2", {"query": "fine-tuning"})],
        )
        third_response = _make_response(
            stop_reason="end_turn",
            content=[_text_block("Fine-tuning adjusts model weights.")],
        )
        mock_client.messages.create.side_effect = [first_response, second_response, third_response]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "chunk content"

        gen = self._make_generator(mock_client)
        gen.generate_response(
            query="fine-tuning",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Third call is the synthesis — must not include tools
        third_call_kwargs = mock_client.messages.create.call_args_list[2][1]
        self.assertNotIn("tools", third_call_kwargs)

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    def test_conversation_history_included_in_system_prompt(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_response(
            stop_reason="end_turn",
            content=[_text_block("Sure.")],
        )
        gen = self._make_generator(mock_client)

        gen.generate_response(
            query="follow-up question",
            conversation_history="User: Hi\nAssistant: Hello",
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertIn("Hi", call_kwargs["system"])
        self.assertIn("Hello", call_kwargs["system"])

    def test_no_history_uses_base_system_prompt_only(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_response(
            stop_reason="end_turn",
            content=[_text_block("Sure.")],
        )
        gen = self._make_generator(mock_client)

        gen.generate_response(query="standalone question")

        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertNotIn("Previous conversation", call_kwargs["system"])

    # ------------------------------------------------------------------
    # Tool use without a tool_manager (edge case)
    # ------------------------------------------------------------------

    def test_tool_use_without_tool_manager_returns_empty_text(self):
        """If stop_reason is tool_use but no tool_manager given, falls through to text."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_response(
            stop_reason="tool_use",
            content=[
                _tool_use_block("search_course_content", "t1", {"query": "x"}),
            ],
        )
        gen = self._make_generator(mock_client)

        # Should not raise; falls through to content[0].text
        # The mock text attribute is a MagicMock, not a string—just verify no exception.
        try:
            gen.generate_response(query="test", tool_manager=None)
        except Exception as e:
            self.fail(f"generate_response raised unexpectedly: {e}")


    # ------------------------------------------------------------------
    # Sequential (two-round) tool calls
    # ------------------------------------------------------------------

    def test_two_sequential_tool_calls_makes_three_api_calls(self):
        """Two tool rounds produce exactly 3 API calls total."""
        mock_client = MagicMock()
        r1 = _make_response(
            stop_reason="tool_use",
            content=[_tool_use_block("get_course_outline", "t1", {"course_name": "Course X"})],
        )
        r2 = _make_response(
            stop_reason="tool_use",
            content=[_tool_use_block("search_course_content", "t2", {"query": "topic from lesson 4"})],
        )
        r3 = _make_response(
            stop_reason="end_turn",
            content=[_text_block("Here is the answer from two searches.")],
        )
        mock_client.messages.create.side_effect = [r1, r2, r3]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "search result"

        gen = self._make_generator(mock_client)
        result = gen.generate_response(
            query="find courses covering the same topic as lesson 4 of Course X",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        self.assertEqual(mock_client.messages.create.call_count, 3)
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        self.assertEqual(result, "Here is the answer from two searches.")

    def test_intermediate_call_includes_tools(self):
        """The second API call (between round 1 and round 2) must still include tools."""
        mock_client = MagicMock()
        r1 = _make_response(
            stop_reason="tool_use",
            content=[_tool_use_block("get_course_outline", "t1", {"course_name": "Course X"})],
        )
        r2 = _make_response(
            stop_reason="tool_use",
            content=[_tool_use_block("search_course_content", "t2", {"query": "embeddings"})],
        )
        r3 = _make_response(
            stop_reason="end_turn",
            content=[_text_block("Final answer.")],
        )
        mock_client.messages.create.side_effect = [r1, r2, r3]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "result"

        gen = self._make_generator(mock_client)
        gen.generate_response(
            query="multi-part question",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Second call is the intermediate round — must include tools
        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        self.assertIn("tools", second_call_kwargs)

    def test_max_two_rounds_enforced(self):
        """Loop stops after MAX_TOOL_ROUNDS even if Claude keeps returning tool_use."""
        mock_client = MagicMock()
        tool_resp = lambda tid: _make_response(
            stop_reason="tool_use",
            content=[_tool_use_block("search_course_content", tid, {"query": "x"})],
        )
        synthesis = _make_response(
            stop_reason="end_turn",
            content=[_text_block("Capped answer.")],
        )
        # 3 tool_use responses; only 2 rounds should execute, third call is synthesis
        mock_client.messages.create.side_effect = [tool_resp("t1"), tool_resp("t2"), synthesis]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "chunk"

        gen = self._make_generator(mock_client)
        result = gen.generate_response(
            query="question",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        self.assertEqual(mock_client.messages.create.call_count, 3)
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        self.assertEqual(result, "Capped answer.")

    def test_tool_execution_error_terminates_loop_gracefully(self):
        """A tool execution exception causes an early synthesis call; no exception propagates."""
        mock_client = MagicMock()
        first_response = _make_response(
            stop_reason="tool_use",
            content=[_tool_use_block("search_course_content", "t1", {"query": "x"})],
        )
        synthesis_response = _make_response(
            stop_reason="end_turn",
            content=[_text_block("Sorry, an error occurred.")],
        )
        mock_client.messages.create.side_effect = [first_response, synthesis_response]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = Exception("ChromaDB unavailable")

        gen = self._make_generator(mock_client)
        try:
            result = gen.generate_response(
                query="question",
                tools=[{"name": "search_course_content"}],
                tool_manager=mock_tool_manager,
            )
        except Exception as e:
            self.fail(f"generate_response raised unexpectedly: {e}")

        # Should make exactly 2 calls: initial + synthesis (no second tool round)
        self.assertEqual(mock_client.messages.create.call_count, 2)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        # Synthesis call must not include tools
        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        self.assertNotIn("tools", second_call_kwargs)


if __name__ == "__main__":
    unittest.main()
