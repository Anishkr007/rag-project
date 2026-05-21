"""
Chat Module for the RAG AI Assistant.

This is the heart of the RAG pipeline — it ties together:
  Retriever (Pinecone) + LLM (GPT-4.1-mini) + Memory + Prompts

HOW THE RAG CHAT PIPELINE WORKS:
=================================
1. User sends a question
2. If there's chat history, the question is "condensed" into a standalone form
   Example: "What about it?" → "What about machine learning?"
3. The standalone question is sent to Pinecone to retrieve relevant chunks
4. The retrieved chunks (context) + question are formatted into a prompt
5. GPT-4.1-mini generates a streaming response based on the context
6. The response and source documents are returned to the UI

HOW LANGCHAIN CONNECTS EVERYTHING:
====================================
LangChain is the orchestration layer that:
- Provides a unified interface to OpenAI models (ChatOpenAI, OpenAIEmbeddings)
- Wraps Pinecone in a Retriever interface
- Manages conversation history (ChatMessageHistory)
- Defines prompt templates that format context + history + question
- Chains these components together into a single callable pipeline
"""

from typing import Generator, List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document

from rag.prompts import SYSTEM_PROMPT, CONDENSE_QUESTION_PROMPT
from rag.retriever import similarity_search
from rag.memory import get_chat_history_as_text
from utils.helpers import get_env_var


# =========================================================================
# LLM Initialization
# =========================================================================

def get_chat_model(
    temperature: float = 0.3,
    streaming: bool = True,
) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance configured for RAG.

    Args:
        temperature: Controls randomness (0 = deterministic, 1 = creative).
                     0.3 is ideal for factual Q&A from documents.
        streaming: Enable token-by-token streaming for real-time display.

    Returns:
        Configured ChatOpenAI instance using gpt-4.1-mini.
    """
    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=temperature,
        streaming=streaming,
        openai_api_key=get_env_var("OPENAI_API_KEY"),
    )


# =========================================================================
# Question Condensing (for follow-up questions)
# =========================================================================

def condense_question(
    question: str,
    memory: ChatMessageHistory,
) -> str:
    """
    Rephrase a follow-up question into a standalone question using chat history.

    This step is critical for conversational RAG. Without it, a question like
    "Tell me more about that" wouldn't return useful search results because
    "that" has no meaning without context.

    Args:
        question: The user's raw question.
        memory: Chat message history.

    Returns:
        A standalone question suitable for vector search.
    """
    chat_history = get_chat_history_as_text(memory)

    # If no history, the question is already standalone
    if chat_history == "No previous conversation.":
        return question

    # Use a lightweight LLM call to rephrase
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        openai_api_key=get_env_var("OPENAI_API_KEY"),
    )

    prompt = CONDENSE_QUESTION_PROMPT.format(
        chat_history=chat_history,
        question=question,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


# =========================================================================
# RAG Response Generation (Streaming)
# =========================================================================

def stream_rag_response(
    question: str,
    retrieved_docs: List[Document],
    memory: ChatMessageHistory,
) -> Generator[str, None, None]:
    """
    Generate a streaming RAG response token by token.

    This is the function that Streamlit's st.write_stream() calls.
    It yields individual tokens as they arrive from the LLM.

    Args:
        question: The user's (possibly condensed) question.
        retrieved_docs: Documents retrieved from Pinecone.
        memory: Chat message history for context.

    Yields:
        String tokens as they stream from GPT-4.1-mini.
    """
    llm = get_chat_model(streaming=True)

    # Format the retrieved documents into context
    context = "\n\n---\n\n".join(
        [
            f"[Source: {doc.metadata.get('source', 'Unknown')} | "
            f"Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
            for doc in retrieved_docs
        ]
    )

    # Get chat history for context
    chat_history = get_chat_history_as_text(memory)

    # Build the system prompt with context
    system_content = SYSTEM_PROMPT.format(
        context=context,
        chat_history=chat_history,
    )

    # Create the message sequence
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=question),
    ]

    # Stream the response token by token
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


# =========================================================================
# Full RAG Pipeline (orchestrates everything)
# =========================================================================

def ask_question(
    question: str,
    memory: ChatMessageHistory,
    top_k: int = 5,
    namespace: str = "",
) -> Tuple[Generator[str, None, None], List[Document]]:
    """
    Execute the full RAG pipeline and return a streaming response + sources.

    Pipeline:
    1. Condense question (if chat history exists)
    2. Retrieve relevant documents from Pinecone
    3. Generate streaming response using context + question

    Args:
        question: The user's question.
        memory: Chat message history.
        top_k: Number of documents to retrieve.
        namespace: Pinecone namespace.

    Returns:
        Tuple of (streaming_generator, source_documents).
        - streaming_generator: yields tokens for st.write_stream()
        - source_documents: list of retrieved Documents for citations
    """
    # Step 1: Condense the question if there's chat history
    standalone_question = condense_question(question, memory)

    # Step 2: Retrieve relevant documents
    docs = similarity_search(
        query=standalone_question,
        top_k=top_k,
        namespace=namespace,
    )

    # Step 3: Create the streaming generator
    response_stream = stream_rag_response(standalone_question, docs, memory)

    return response_stream, docs
