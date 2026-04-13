"""
RAG-Based Context-Aware Chatbot
Main application with Streamlit UI and RAGEngine
"""

import os
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class RAGEngine:
    """Core RAG engine for retrieval and generation"""

    def __init__(self, vector_db_dir: str = "vector_db"):
        """Initialize RAG engine with FAISS index and Gemini API"""
        self.vector_db_dir = vector_db_dir
        self.similarity_threshold = 0.5
        self.top_k = 3

        # Load FAISS index and metadata
        self._load_vector_store()

        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Initialize Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel('models/gemini-2.5-flash')

        print("✅ RAG Engine initialized successfully")

    def _load_vector_store(self):
        """Load FAISS index and chunk metadata"""
        index_path = Path(self.vector_db_dir) / "faiss_index.bin"
        metadata_path = Path(self.vector_db_dir) / "chunks_metadata.pkl"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Please run 'python data_prep.py' first to create the index."
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at {metadata_path}. "
                "Please run 'python data_prep.py' first."
            )

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load metadata
        with open(metadata_path, 'rb') as f:
            self.chunks = pickle.load(f)

        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        print(f"Loaded metadata for {len(self.chunks)} chunks")

    def _l2_to_cosine_similarity(self, l2_distance: float, norm: float = 1.0) -> float:
        """Convert L2 distance to cosine similarity (approximate)"""
        # For normalized vectors: cosine_sim ≈ 1 - (l2_distance^2 / 2)
        return 1.0 - (l2_distance ** 2) / 2.0

    def retrieve_context(self, query: str) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Retrieve relevant document chunks for the query
        Returns: (list of retrieved chunks, is_above_threshold)
        """
        # Embed the query
        query_embedding = self.embedding_model.encode([query])[0].astype('float32')
        query_embedding = np.expand_dims(query_embedding, axis=0)

        # Search FAISS index
        distances, indices = self.index.search(query_embedding, self.top_k)

        # Convert to results with similarity scores
        retrieved_chunks = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):  # Valid index
                similarity = self._l2_to_cosine_similarity(dist)
                chunk_data = self.chunks[idx].copy()
                chunk_data['similarity_score'] = float(similarity)
                retrieved_chunks.append(chunk_data)

        # Check if any chunk meets the threshold
        is_above_threshold = any(
            chunk['similarity_score'] >= self.similarity_threshold
            for chunk in retrieved_chunks
        )

        # Log retrieval scores (for debugging)
        print(f"Query: {query[:50]}...")
        for i, chunk in enumerate(retrieved_chunks):
            print(f"  Chunk {i+1}: score={chunk['similarity_score']:.3f}, source={chunk['metadata']['source']}")

        return retrieved_chunks, is_above_threshold

    def _format_conversation_history(self, messages: List[Dict[str, str]], max_tokens: int = 1500) -> str:
        """Format conversation history with token limit"""
        if not messages:
            return ""

        # Simple token estimation: ~4 chars per token
        history_text = ""
        total_chars = 0
        max_chars = max_tokens * 4

        # Start from most recent messages
        for msg in reversed(messages):
            msg_text = f"{msg['role'].capitalize()}: {msg['content']}\n"
            msg_chars = len(msg_text)

            if total_chars + msg_chars > max_chars:
                break

            history_text = msg_text + history_text
            total_chars += msg_chars

        return history_text.strip()

    def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]]
    ) -> Tuple[str, List[str]]:
        """
        Generate response using retrieved context and conversation history
        Returns: (response_text, source_files)
        """
        # Format context from retrieved chunks
        context_parts = []
        source_files = set()

        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"[Document {i}]\n{chunk['text']}")
            source_files.add(chunk['metadata']['source'])

        context_text = "\n\n".join(context_parts)

        # Format conversation history
        history_text = self._format_conversation_history(conversation_history)

        # Build prompt
        prompt = f"""You are a helpful AI assistant that answers questions based on provided documents.

Context from documents:
{context_text}

{"Conversation history:" if history_text else ""}
{history_text}

User question: {query}

Instructions:
- Answer the question based primarily on the provided context from documents
- Be clear, concise, and conversational
- If the context doesn't fully answer the question, say so
- Maintain continuity with the conversation history if relevant

Answer:"""

        # Call Gemini API with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=500
                    )
                )
                return response.text, list(source_files)

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed to generate response after {max_retries} attempts: {e}")

    def fallback_response(
        self,
        query: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Generate fallback response using LLM without document context
        Used when no relevant documents are found
        """
        # Format conversation history
        history_text = self._format_conversation_history(conversation_history)

        # Build prompt for fallback
        prompt = f"""You are a helpful AI assistant. The user has asked a question that is outside the scope of the available documents.

{"Conversation history:" if history_text else ""}
{history_text}

User question: {query}

Instructions:
- Provide a helpful general knowledge response
- Be clear that this answer is not from the document collection
- Keep the response concise and accurate
- Maintain continuity with the conversation history if relevant

Answer:"""

        # Call Gemini API with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=500
                    )
                )
                return response.text

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed to generate fallback response after {max_retries} attempts: {e}")


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'rag_engine' not in st.session_state:
        try:
            st.session_state.rag_engine = RAGEngine()
            st.session_state.engine_loaded = True
            st.session_state.error_message = None
        except Exception as e:
            st.session_state.engine_loaded = False
            st.session_state.error_message = str(e)


def manage_conversation_memory():
    """Manage conversation memory - keep last 7 turns"""
    max_turns = 7
    max_messages = max_turns * 2  # user + assistant per turn

    if len(st.session_state.chat_history) > max_messages:
        # Remove oldest turn (2 messages: user + assistant)
        st.session_state.chat_history = st.session_state.chat_history[-max_messages:]


def main():
    """Main Streamlit application with premium UI"""
    st.set_page_config(
        page_title="AI Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    # Custom CSS for premium UI
    st.markdown("""
    <style>
    body {
        background-color: #f5f7fb;
    }

    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    .message {
        display: flex;
        align-items: flex-end;
        gap: 10px;
    }

    .user {
        justify-content: flex-end;
    }

    .bot {
        justify-content: flex-start;
    }

    .bubble {
        padding: 12px 16px;
        border-radius: 18px;
        max-width: 70%;
        font-size: 14px;
        line-height: 1.4;
    }

    .user-bubble {
        background-color: #4CAF50;
        color: white;
        border-bottom-right-radius: 5px;
    }

    .bot-bubble {
        background-color: #EAEAEA;
        color: black;
        border-bottom-left-radius: 5px;
    }

    .bot-bubble-doc {
        background-color: #E8F5E9;
        color: black;
        border-bottom-left-radius: 5px;
        border-left: 3px solid #4CAF50;
    }

    .bot-bubble-fallback {
        background-color: #FFF9C4;
        color: black;
        border-bottom-left-radius: 5px;
        border-left: 3px solid #FFC107;
    }

    .avatar {
        width: 35px;
        height: 35px;
        border-radius: 50%;
    }

    .timestamp {
        font-size: 10px;
        color: gray;
        margin-top: 3px;
    }

    .header {
        font-size: 22px;
        font-weight: bold;
        padding-bottom: 10px;
    }

    .sources {
        font-size: 11px;
        color: #666;
        margin-top: 5px;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")

        # Show stats if engine loaded
        if st.session_state.engine_loaded:
            num_docs = st.session_state.rag_engine.index.ntotal
            st.metric("📚 Documents Indexed", num_docs)

        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.write("RAG-based chatbot with Gemini")
        st.markdown("""
        - 📄 Green bubble = From documents
        - 🌐 Yellow bubble = General knowledge
        """)

    # Header
    st.markdown('<div class="header">🤖 AI Assistant</div>', unsafe_allow_html=True)

    # Check if engine loaded successfully
    if not st.session_state.engine_loaded:
        st.error("⚠️ Failed to initialize RAG engine")
        st.error(st.session_state.error_message)
        st.info("""
        **Setup Instructions:**
        1. Add documents to the `data/` directory
        2. Run: `python data_prep.py`
        3. Set `GEMINI_API_KEY` in `.env` file
        4. Restart the app
        """)
        return

    # Chat input
    user_input = st.chat_input("Type your message...")

    # Handle input
    if user_input:
        time_now = datetime.now().strftime("%H:%M")

        # Save user message
        st.session_state.chat_history.append({
            "role": "user",
            "message": user_input,
            "time": time_now
        })

        # Generate response
        with st.spinner("🤖 Thinking..."):
            try:
                # Retrieve context
                retrieved_chunks, is_above_threshold = st.session_state.rag_engine.retrieve_context(user_input)

                # Get conversation history for context (convert format)
                history = [
                    {"role": chat["role"], "content": chat["message"]}
                    for chat in st.session_state.chat_history[:-1]
                ]

                if is_above_threshold:
                    # Document-based response
                    response_text, sources = st.session_state.rag_engine.generate_response(
                        user_input, retrieved_chunks, history
                    )
                    response = f"📄 {response_text}"
                    is_doc_based = True
                    source_list = sources
                else:
                    # Fallback response
                    response_text = st.session_state.rag_engine.fallback_response(user_input, history)
                    response = f"🌐 {response_text}"
                    is_doc_based = False
                    source_list = []

                # Save bot message
                st.session_state.chat_history.append({
                    "role": "bot",
                    "message": response,
                    "time": time_now,
                    "is_document_based": is_doc_based,
                    "sources": source_list
                })

                # Manage conversation memory
                manage_conversation_memory()

            except Exception as e:
                error_msg = "⚠️ Something went wrong while processing your request. Please try again or rephrase your question."
                st.session_state.chat_history.append({
                    "role": "bot",
                    "message": error_msg,
                    "time": time_now,
                    "is_document_based": False,
                    "sources": []
                })
                print(f"Error: {str(e)}")

    # Display chat
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f"""
            <div class="message user">
                <div>
                    <div class="bubble user-bubble">{chat["message"]}</div>
                    <div class="timestamp">{chat["time"]}</div>
                </div>
                <img class="avatar" src="https://cdn-icons-png.flaticon.com/512/847/847969.png">
            </div>
            """, unsafe_allow_html=True)

        else:
            # Determine bubble style based on response type
            is_doc = chat.get("is_document_based", False)
            bubble_class = "bot-bubble-doc" if is_doc else "bot-bubble-fallback"
            sources = chat.get("sources", [])

            # Format sources if available
            sources_html = ""
            if sources:
                sources_html = f'<div class="sources">Sources: {", ".join(sources)}</div>'

            st.markdown(f"""
            <div class="message bot">
                <img class="avatar" src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png">
                <div>
                    <div class="bubble {bubble_class}">{chat["message"]}{sources_html}</div>
                    <div class="timestamp">{chat["time"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
