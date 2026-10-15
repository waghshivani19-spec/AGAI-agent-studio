
"""Pinecone + OpenAI embeddings, with a JSON registry of uploaded documents."""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from .config import get_settings

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


class PineconeStore:
    """Small LangChain-compatible adapter using the official Pinecone SDK."""

    def __init__(self, index: Any, embeddings: OpenAIEmbeddings, namespace: str):
        self.index = index
        self.embeddings = embeddings
        self.namespace = namespace

    async def aadd_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        if not documents:
            return []

        if ids is None:
            ids = [
                hashlib.sha256(
                    f"{doc.metadata.get('source', '')}:{i}:"
                    f"{doc.page_content}".encode("utf-8")
                ).hexdigest()
                for i, doc in enumerate(documents)
            ]

        if len(ids) != len(documents):
            raise ValueError("The number of IDs must match the number of documents.")

        vectors = await self.embeddings.aembed_documents(
            [doc.page_content for doc in documents]
        )

        records = []
        for doc_id, vector, doc in zip(ids, vectors, documents):
            metadata = dict(doc.metadata)
            metadata["text"] = doc.page_content

            records.append(
                {
                    "id": doc_id,
                    "values": vector,
                    "metadata": metadata,
                }
            )

        # Upsert in batches to avoid oversized requests.
        for start in range(0, len(records), 100):
            batch = records[start : start + 100]
            await asyncio.to_thread(
                self.index.upsert,
                vectors=batch,
                namespace=self.namespace,
            )

        return ids

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.index.delete(ids=ids, namespace=self.namespace)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
    ) -> list[tuple[Document, float]]:
        query_vector = self.embeddings.embed_query(query)

        response = self.index.query(
            vector=query_vector,
            top_k=k,
            namespace=self.namespace,
            include_metadata=True,
            filter=filter,
        )

        results = []
        for match in response.matches:
            metadata = dict(match.metadata or {})
            content = metadata.pop("text", "")
            results.append(
                (
                    Document(page_content=content, metadata=metadata),
                    float(match.score or 0.0),
                )
            )

        return results

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
    ) -> list[Document]:
        return [
            doc
            for doc, _score in self.similarity_search_with_score(
                query, k=k, filter=filter
            )
        ]

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
    ) -> list[Document]:
        query_vector = await self.embeddings.aembed_query(query)

        response = await asyncio.to_thread(
            self.index.query,
            vector=query_vector,
            top_k=k,
            namespace=self.namespace,
            include_metadata=True,
            filter=filter,
        )

        results = []
        for match in response.matches:
            metadata = dict(match.metadata or {})
            content = metadata.pop("text", "")
            results.append(Document(page_content=content, metadata=metadata))

        return results


@lru_cache(maxsize=1)
def get_vectorstore() -> PineconeStore:
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)

    if not pc.has_index(settings.pinecone_index):
        pc.create_index(
            name=settings.pinecone_index,
            dimension=settings.embedding_dimensions,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )

        while not pc.describe_index(settings.pinecone_index).status["ready"]:
            time.sleep(1)

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )

    return PineconeStore(
        index=pc.Index(settings.pinecone_index),
        embeddings=embeddings,
        namespace=settings.pinecone_namespace,
    )


# ---------- File parsing ----------

def parse_file(filename: str, data: bytes) -> list[Document]:
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Use: {', '.join(sorted(SUPPORTED))}"
        )

    if ext == ".pdf":
        reader = PdfReader(BytesIO(data))
        documents = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": page_number,
                        },
                    )
                )

        return documents

    if ext == ".docx":
        doc = DocxDocument(BytesIO(data))
        parts = [
            paragraph.text
            for paragraph in doc.paragraphs
            if paragraph.text.strip()
        ]

        for table in doc.tables:
            for row in table.rows:
                parts.append(
                    " | ".join(cell.text.strip() for cell in row.cells)
                )

        text = "\n".join(parts)
    else:
        text = data.decode("utf-8", errors="ignore")

    if not text.strip():
        return []

    return [Document(page_content=text, metadata={"source": filename})]


# ---------- JSON document registry ----------

def _registry_path() -> Path:
    path = Path(get_settings().data_dir) / "documents.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_registry() -> dict:
    path = _registry_path()

    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


def _save_registry(registry: dict) -> None:
    _registry_path().write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------- Ingestion ----------

async def ingest_file(filename: str, data: bytes) -> dict:
    settings = get_settings()
    documents = parse_file(filename, data)

    if not documents:
        raise ValueError("No extractable text found in this file.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No document chunks were generated.")

    # Preserve page and source metadata from the original documents.
    stem = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:20]
    ids = [f"{stem}-{i}" for i in range(len(chunks))]

    store = await asyncio.to_thread(get_vectorstore)
    registry = await asyncio.to_thread(load_registry)
    old_entry = registry.get(filename, {})
    old_ids = old_entry.get("ids", [])

    # Upsert first so a failed embedding or upload does not remove the old file.
    await store.aadd_documents(chunks, ids=ids)

    # Remove obsolete chunks left by a previous version of the file.
    obsolete_ids = list(set(old_ids) - set(ids))

    if obsolete_ids:
        await asyncio.to_thread(store.delete, obsolete_ids)

    registry[filename] = {
        "ids": ids,
        "chunks": len(chunks),
        "uploaded": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    await asyncio.to_thread(_save_registry, registry)

    return {"name": filename, "chunks": len(chunks)}


# ---------- Deletion ----------

async def delete_file(filename: str) -> bool:
    registry = await asyncio.to_thread(load_registry)
    entry = registry.get(filename)

    if not entry:
        return False

    store = await asyncio.to_thread(get_vectorstore)

    # Delete vectors before removing the registry entry.
    await asyncio.to_thread(store.delete, entry["ids"])

    registry.pop(filename)
    await asyncio.to_thread(_save_registry, registry)

    return True


# Import used by the PDF parser.
from pypdf import PdfReader
