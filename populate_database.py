import argparse
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from config import DATA_PATH, get_collection_name
from database import open_database


def main(): # Human Generated, Machine Edited
    parser = argparse.ArgumentParser(description="Load board-game PDFs into PostgreSQL.")
    parser.add_argument("--reset", action="store_true", help="Clear this provider/model's collection before loading.")
    args = parser.parse_args()

    # Get documents
    documents = load_documents()
    # Get chunks
    chunks = calculate_chunk_ids(split_documents(documents))
    if not chunks:
        raise ValueError("No text found in data/*.pdf; scanned PDFs need OCR")
    print(f"Loaded {len(documents)} pages; split into {len(chunks)} chunks.", flush=True)

    # Open database, if reset argument is passed, rebuild database
    with open_database() as db:
        if args.reset:
            db.delete_collection()
            db.create_collection()
            print("Cleared the selected provider/model collection.")
        add_to_postgres(db, chunks)

# Load documents
def load_documents(): # Human Generated
    return PyPDFDirectoryLoader(str(DATA_PATH), glob="*.pdf").load()

# Split documents into chunks
def split_documents(documents: list[Document]): # Human Generated
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

# Calculates and sets the ID for each chunk. Format: data/{filename}:{page 3}:{chunk #}
def calculate_chunk_ids(chunks): # Machine Generated
    last_page_id = None
    current_chunk_index = 0
    collection = get_collection_name()
    for chunk in chunks:
        source = chunk.metadata["source"]
        filename = str(Path(source).relative_to(DATA_PATH))
        page = chunk.metadata.get("page", 0)
        current_page_id = f"data/{filename}:{page}"
        current_chunk_index = current_chunk_index + 1 if current_page_id == last_page_id else 0
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
        chunk.id = str(uuid5(NAMESPACE_URL, f"{collection}:{chunk_id}"))
        last_page_id = current_page_id
    return chunks

# Checks ids, if it already exists it does not add them again
def add_to_postgres(db, chunks): # Machine Generated
    added = 0
    for start in range(0, len(chunks), 8):
        batch = chunks[start:start + 8]
        existing_ids = {doc.id for doc in db.get_by_ids([chunk.id for chunk in batch])}
        new_chunks = [chunk for chunk in batch if chunk.id not in existing_ids]
        if new_chunks:
            print(f"Embedding {len(new_chunks)} new chunks...", flush=True)
            db.add_documents(new_chunks, ids=[chunk.id for chunk in new_chunks])
            added += len(new_chunks)
            print(f"Added {added} chunks...", flush=True)
    print(f"Done. Added {added} of {len(chunks)} chunks; existing chunks were skipped.")


if __name__ == "__main__":
    main()
