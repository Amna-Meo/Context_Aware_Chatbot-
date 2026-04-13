"""
Document Processing Pipeline - Streaming Version
Writes chunks to disk incrementally to handle very large documents
"""

import gc
import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


class SimpleTextSplitter:
    """Simple text splitter with overlap"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end < len(text):
                best_break = end
                for separator in self.separators:
                    if separator:
                        last_sep = text.rfind(separator, start, end)
                        if last_sep > start:
                            best_break = last_sep + len(separator)
                            break
                end = best_break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap
            if start <= 0:
                start = end

        return chunks


def clean_pdf_text(text: str) -> str:
    """Clean PDF text"""
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"\n\s*Page \d+\s*\n", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    text = text.replace("\x00", "")
    return text.strip()


def process_documents_streaming(data_dir: str, temp_dir: str) -> int:
    """Process documents and write chunks to temporary files"""
    data_path = Path(data_dir)
    temp_path = Path(temp_dir)

    pdf_files = [
        f for f in data_path.rglob("*.pdf") if not f.name.endswith(".Identifier")
    ]
    txt_files = [
        f for f in data_path.rglob("*.txt") if not f.name.endswith(".Identifier")
    ]
    md_files = [f for f in data_path.rglob("*.md") if f.name != "README.md"]

    print(
        f"Found {len(pdf_files)} PDF, {len(txt_files)} text, {len(md_files)} markdown files"
    )

    text_splitter = SimpleTextSplitter(chunk_size=500, chunk_overlap=50)

    chunk_count = 0
    file_index = 0

    # Process PDFs
    for pdf_file in pdf_files:
        print(f"\nProcessing PDF: {pdf_file.name}")
        reader = PdfReader(str(pdf_file))
        total_pages = len(reader.pages)
        print(f"Total pages: {total_pages}")

        page_batch = []

        for page_num in range(total_pages):
            if page_num % 100 == 0:
                print(f"  Page {page_num}/{total_pages}...")

            try:
                page = reader.pages[page_num]
                text = page.extract_text()

                if text and text.strip():
                    cleaned_text = clean_pdf_text(text)

                    if len(cleaned_text) < 500:
                        page_batch.append(
                            {
                                "text": cleaned_text,
                                "metadata": {
                                    "source": str(pdf_file.name),
                                    "page_number": page_num + 1,
                                    "doc_type": "pdf",
                                },
                            }
                        )
                    else:
                        text_chunks = text_splitter.split_text(cleaned_text)
                        for i, chunk_text in enumerate(text_chunks):
                            page_batch.append(
                                {
                                    "text": chunk_text,
                                    "metadata": {
                                        "source": str(pdf_file.name),
                                        "page_number": page_num + 1,
                                        "doc_type": "pdf",
                                        "chunk_index": i,
                                    },
                                }
                            )

                # Write batch to disk every 100 pages
                if len(page_batch) >= 200:
                    batch_file = temp_path / f"batch_{file_index}.pkl"
                    with open(batch_file, "wb") as f:
                        pickle.dump(page_batch, f)
                    chunk_count += len(page_batch)
                    print(f"    Saved batch {file_index} ({len(page_batch)} chunks)")
                    page_batch = []
                    file_index += 1
                    gc.collect()

            except Exception as e:
                print(f"    Error on page {page_num}: {e}")

        # Write remaining chunks
        if page_batch:
            batch_file = temp_path / f"batch_{file_index}.pkl"
            with open(batch_file, "wb") as f:
                pickle.dump(page_batch, f)
            chunk_count += len(page_batch)
            print(f"    Saved final batch {file_index} ({len(page_batch)} chunks)")
            file_index += 1
            gc.collect()

    print(f"\n✅ Total chunks: {chunk_count} in {file_index} batch files")
    return chunk_count


def create_embeddings_streaming(temp_dir: str, vector_db_dir: str):
    """Create embeddings by loading batches from disk"""
    temp_path = Path(temp_dir)
    batch_files = sorted(temp_path.glob("batch_*.pkl"))

    print(f"\nInitializing embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    all_embeddings = []
    all_chunks = []

    print(f"Processing {len(batch_files)} batch files...")

    for batch_idx, batch_file in enumerate(batch_files):
        print(f"  Loading batch {batch_idx + 1}/{len(batch_files)}...")

        with open(batch_file, "rb") as f:
            chunks = pickle.load(f)

        texts = [chunk["text"] for chunk in chunks]

        print(f"    Generating embeddings for {len(texts)} chunks...")
        embeddings = model.encode(texts, batch_size=8, show_progress_bar=False)

        all_embeddings.extend(embeddings)
        all_chunks.extend(chunks)

        gc.collect()

    print(f"\n✅ Generated {len(all_embeddings)} embeddings")

    # Create FAISS index
    print("Creating FAISS index...")
    embeddings_array = np.array(all_embeddings).astype("float32")
    dimension = embeddings_array.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    # Save
    vector_db_path = Path(vector_db_dir)
    vector_db_path.mkdir(parents=True, exist_ok=True)

    index_file = vector_db_path / "faiss_index.bin"
    metadata_file = vector_db_path / "chunks_metadata.pkl"

    print("Saving index and metadata...")
    faiss.write_index(index, str(index_file))

    with open(metadata_file, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\n✅ Vector database created")
    print(f"   - {len(all_chunks)} chunks")
    print(f"   - Index: {index_file}")
    print(f"   - Metadata: {metadata_file}")


def main():
    """Main function"""
    print("=" * 60)
    print("RAG Chatbot - Streaming Document Processor")
    print("=" * 60)

    data_dir = "data"
    vector_db_dir = "vector_db"

    # Create temporary directory for batches
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nUsing temporary directory: {temp_dir}")

        try:
            print("\n[Step 1/2] Processing documents...")
            chunk_count = process_documents_streaming(data_dir, temp_dir)

            if chunk_count == 0:
                print("\n❌ No chunks created")
                return

            print("\n[Step 2/2] Creating embeddings and index...")
            create_embeddings_streaming(temp_dir, vector_db_dir)

            print("\n" + "=" * 60)
            print("✅ Processing complete!")
            print("=" * 60)
            print("\nNext steps:")
            print("1. Create .env with: GEMINI_API_KEY=your_key")
            print("2. Run: streamlit run app.py")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
