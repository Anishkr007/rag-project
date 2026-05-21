"""
Conversation Memory Module for the RAG AI Assistant.

Manages chat history so the AI can understand follow-up questions
and maintain context across a conversation.

HOW MEMORY WORKS (LangChain 1.2+):
====================================
We use ChatMessageHistory to store every message (human + AI) in a list.
When the user asks a follow-up like "Tell me more about that", the history
provides previous messages so the system understands what "that" refers to.

Memory is stored in Streamlit's session_state so it persists across
reruns but resets when the user closes the tab.
"""

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Tuple


def create_memory() -> ChatMessageHistory:
    """
    Create a new ChatMessageHistory instance.

    Returns:
        A fresh ChatMessageHistory instance.
    """
    return ChatMessageHistory()


def get_chat_history_as_text(memory: ChatMessageHistory) -> str:
    """
    Convert the memory buffer into a formatted text string.

    Useful for injecting chat history into prompt templates.

    Args:
        memory: The chat message history instance.

    Returns:
        Formatted string of the conversation history.
    """
    messages = memory.messages
    if not messages:
        return "No previous conversation."

    history_lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history_lines.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            history_lines.append(f"AI: {msg.content}")

    return "\n".join(history_lines)


def get_chat_history_tuples(
    memory: ChatMessageHistory,
) -> List[Tuple[str, str]]:
    """
    Convert memory into a list of (human, ai) tuples.

    Useful for Streamlit's chat display.

    Returns:
        List of (user_message, ai_response) tuples.
    """
    messages = memory.messages
    tuples = []
    i = 0

    while i < len(messages) - 1:
        if isinstance(messages[i], HumanMessage) and isinstance(
            messages[i + 1], AIMessage
        ):
            tuples.append((messages[i].content, messages[i + 1].content))
            i += 2
        else:
            i += 1

    return tuples


def save_to_memory(
    memory: ChatMessageHistory, question: str, answer: str
) -> None:
    """
    Save a Q&A exchange to memory.

    Args:
        memory: The chat message history instance.
        question: The user's question.
        answer: The AI's response.
    """
    memory.add_user_message(question)
    memory.add_ai_message(answer)


def clear_memory(memory: ChatMessageHistory) -> None:
    """Clear all messages from the conversation memory."""
    memory.clear()
