import anthropic
from typing import List, Optional

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Tool Usage:
- Use `search_course_content` **only** for questions about specific course content or detailed educational materials
- Use `get_course_outline` for questions asking about the structure, outline, or list of lessons of a course; when used, present the course title, course link, and the number and title of each lesson
- You may make up to 2 sequential tool calls if the first result is insufficient or if the query requires information from multiple sources. Proceed to a final answer once you have sufficient information.
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using a tool
- **Course content questions**: Use `search_course_content` first, then answer
- **Course outline / structure questions**: Use `get_course_outline` and present the full lesson list
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Supports up to MAX_TOOL_ROUNDS sequential tool calls. Each round gets its
        own API call so Claude can reason about previous results before deciding
        whether to search again.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": query}]

        # Prepare API call parameters
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Initial API call
        response = self.client.messages.create(**api_params)

        rounds_completed = 0

        while response.stop_reason == "tool_use" and tool_manager and rounds_completed < self.MAX_TOOL_ROUNDS:
            # Execute all tool calls in this response
            tool_results = []
            error = False
            for block in response.content:
                if block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(block.name, **block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })
                    except Exception as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Tool execution failed: {e}"
                        })
                        error = True

            # Append this round's turns to the conversation
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            rounds_completed += 1

            # Determine whether the next call is an intermediate round (with tools)
            # or a final synthesis call (without tools)
            need_synthesis = error or rounds_completed >= self.MAX_TOOL_ROUNDS

            next_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }
            if not need_synthesis and tools:
                next_params["tools"] = tools
                next_params["tool_choice"] = {"type": "auto"}

            response = self.client.messages.create(**next_params)

            if need_synthesis:
                break  # This was the synthesis call; we're done

        return response.content[0].text
