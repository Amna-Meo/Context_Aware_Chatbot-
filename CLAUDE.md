# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Context-Aware Chatbot Using RAG (Retrieval-Augmented Generation)** designed as a document-driven conversational assistant for students and beginners learning AI and technical subjects.

### Core Design Principles

1. **Document-First Retrieval**: All responses must prioritize retrieved document context to minimize hallucinations
2. **Controlled Fallback**: When no relevant documents exist, the system may use LLM generation but must clearly distinguish this from document-based answers
3. **Context Awareness**: Maintain full conversation memory across multi-turn conversations
4. **Traceability**: Answers must be traceable to source content when documents exist

### Knowledge Base Sources
- PDF files (lecture notes, research articles)
- Plain text documents
- Curated Wikipedia content

## Architecture

### RAG Pipeline
The system implements a standard RAG architecture:
1. **Document Ingestion**: Load and process PDFs, text files, and Wikipedia content
2. **Vector Storage**: Embed documents and store in vector database for similarity search
3. **Retrieval**: Query vector store to find relevant document chunks
4. **Generation**: Use retrieved context with LLM (Google Gemini free-tier) to generate responses
5. **Fallback**: When retrieval fails, use LLM with clear indication of non-document origin

### Key Components
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (free, offline, CPU-friendly)
- **Vector Store**: FAISS (local, fast, no server needed)
- **LLM**: Google Gemini API (free-tier with rate limits)
- **Conversation Memory**: Session-based context tracking (last 7 turns, ~1000-1500 tokens max)
- **UI**: Streamlit-based interface

### Technical Specifications

#### Document Chunking
- **Chunk Size**: 500 characters
- **Overlap**: 50 characters
- **Strategy**: 
  - PDFs: Page-based + text splitter (clean headers/footers, fix broken sentences)
  - Text files: Direct chunking with splitter

#### Retrieval Configuration
- **Top-K**: 3 documents per query
- **Similarity Threshold**: 0.5-0.7 (cosine similarity)
- **Fallback Logic**: If no chunk exceeds threshold → use LLM fallback

#### Response Indicators (UI)
- **Document-based**: `📄 Answer (From Documents):`
- **LLM Fallback**: `🌐 General Knowledge (LLM Response):`
- Optional: Color coding (green for documents, yellow for fallback)

#### Directory Structure
```
project/
├── data/              # Raw documents (PDFs, text files, Wikipedia content)
├── vector_db/         # FAISS index storage
├── app.py             # Main Streamlit application
├── data_prep.py       # Document processing and indexing
└── requirements.txt   # Dependencies
```

#### Session Management
- **Type**: Session-only (no persistence across restarts)
- **Memory**: Clears when app restarts
- **Rationale**: Simpler implementation, fits 2-day development timeline

## Development Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Required packages:
# - langchain
# - sentence-transformers
# - faiss-cpu
# - google-generativeai
# - streamlit
# - pypdf (for PDF processing)

# Prepare knowledge base and build FAISS index
python data_prep.py
```

### Running the Application
```bash
# Run Streamlit app
streamlit run app.py

# Run CLI version (if available)
python cli_chat.py
```

### Testing
```bash
# Run tests
python test_setup.py

# Quick validation
python quick_test.py
```

## Performance Requirements

- **Response Time**: < 3 seconds average
- **Session Capacity**: Handle 10+ queries without degradation
- **Consistency**: Maintain quality across rephrased queries and multi-turn conversations

## Constraints

### API Constraints
- Uses Google Gemini free-tier API
- Subject to rate limits and token limitations
- Implement appropriate error handling for API failures

### Hardware Constraints
- CPU-only execution (no GPU)
- Limited RAM for vector storage
- Optimize for local machine performance

### Development Constraints
- Simple, efficient architecture
- Minimal overhead
- No custom model training or fine-tuning

## Non-Goals

The following are explicitly out of scope:
- Voice-based interaction
- Mobile or cross-platform deployment
- Real-time web browsing
- Multilingual support
- Complex or highly interactive UI
- Custom model training or fine-tuning

## Code Guidelines

### When Implementing Retrieval
- Always check if relevant documents were retrieved before generating responses
- Log retrieval confidence scores for debugging
- Use similarity threshold (0.5-0.7) to determine when fallback is needed
- Retrieve top-3 most relevant chunks per query
- Clean PDF text (remove headers/footers, fix broken sentences)

### When Implementing Generation
- Clearly distinguish document-based vs. LLM-generated responses using UI indicators:
  - `📄 Answer (From Documents):` for document-based
  - `🌐 General Knowledge (LLM Response):` for fallback
- Include source attribution when using retrieved documents
- Implement graceful degradation when API limits are hit
- Retrieved content always takes precedence over generated knowledge

### When Handling Conversation Context
- Maintain last 7 conversation turns in memory
- Limit conversation history to ~1000-1500 tokens
- Handle follow-up questions that reference previous context
- Session memory clears on app restart (no persistence)

### Error Handling
- **User-facing**: Simple, beginner-friendly messages
  - Example: `⚠️ Something went wrong while processing your request. Please try again or rephrase your question.`
- **Backend**: Detailed logs for debugging
- Handle API rate limit errors gracefully
- Implement retry logic with exponential backoff for transient failures

### Wikipedia Integration
- Pre-fetch relevant Wikipedia content during setup
- Store locally in `/data` directory
- Avoid real-time API calls (adds complexity and latency)

## Testing Strategy

### Manual Testing Focus
- Test diverse query types (factual, conceptual, follow-up)
- Verify document retrieval accuracy
- Validate fallback behavior for out-of-scope queries
- Test multi-turn conversation coherence

### Validation Criteria
A response is correct if it:
- Semantically aligns with retrieved document content
- Is relevant to the query
- Is factually accurate
- Is free from hallucination when documents exist
