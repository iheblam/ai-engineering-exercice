from pathlib import Path
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from metadata_extractor import extract_metadata

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "invoices"



all_documents = list()


folder_path = (
    Path(__file__).resolve().parent
    / "data"
    / "invoices"
    / "PDF_Invoice_Folder"
)

if not folder_path.is_dir():
    raise FileNotFoundError(f"PDF folder not found: {folder_path}")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client = QdrantClient(url=QDRANT_URL)



if client.collection_exists(collection_name=COLLECTION_NAME):
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )
else:
    vector_store = None


def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()

def hash_exists_in_qdrant(file_hash):

    if vector_store is None:
        return False
    results = vector_store.client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter={
            "must": [
                {
                    "key": "metadata.file_hash",
                    "match": {
                        "value": file_hash
                    }
                }
            ]
        },
        limit=1,
    )

    points, _ = results

    return len(points) > 0

for root, _, files in os.walk(folder_path):
    print('Processing folder:', root)
    for file in files:
        
        if file.lower().endswith(".pdf"):
            full_path = os.path.join(root, file)
            file_hash = calculate_file_hash(full_path)
            metadata = extract_metadata(full_path)
            if hash_exists_in_qdrant(file_hash):
                client.set_payload(
                    collection_name=COLLECTION_NAME,
                    payload=metadata,
                    key="metadata",
                    points=Filter(must=[FieldCondition(
                        key="metadata.file_hash",
                        match=MatchValue(value=file_hash),
                    )]),
                    wait=True,
                )
                print(f"Skipping already indexed: {file}")
                continue

            loader = PyPDFLoader(str(full_path))
            documents = loader.load()

            for document in documents:
                document.metadata.update(metadata)
                document.metadata.update({
                    "file_hash": file_hash,
                    "source_file": str(full_path),
                })
            all_documents.extend(documents)

print(f"Loaded {len(all_documents)} documents.")




splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

chunks = splitter.split_documents(all_documents)

print(f"Created {len(chunks)} chunks.")

if not chunks:
    print("No new content to index.")
    raise SystemExit(0)




if vector_store:
    vector_store.add_documents(chunks)
else:
    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )

print("PDF successfully stored in Qdrant.")
