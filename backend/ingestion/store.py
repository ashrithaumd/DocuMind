"""Hybrid chunk store: dense (Chroma) + sparse (BM25) over the same chunks,
combined with Reciprocal Rank Fusion.

Why hybrid: dense embeddings catch paraphrases ("remuneration" ~ "salary"),
BM25 catches exact terms embeddings dilute (part numbers, acronyms, names).
RRF combines the two rankings without having to calibrate their incomparable
score scales — only ranks matter.

Persistence: Chroma persists chunks + embeddings + metadata. The BM25 index is
in-memory only and is rebuilt from Chroma at startup. At this project's scale
(hundreds of chunks) a rebuild takes milliseconds, so persisting a second
index would be complexity with no payoff.
"""
import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

import chromadb
from rank_bm25 import BM25Okapi

from config import Settings
from embeddings import EmbeddingProvider
from ingestion.chunking import Chunk

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    filename: str
    page_number: int
    score: float              # fused RRF score (used for the final ranking)
    dense_rank: int | None    # rank in the dense leg, None if it didn't place
    sparse_rank: int | None   # rank in the BM25 leg, None if it didn't place
    similarity: float | None  # cosine similarity, only known for the dense leg


def rrf_fuse(
    dense: dict[str, tuple[int, float, str, dict]],
    sparse: dict[str, tuple[int, str, dict]],
    k: int,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion: score(c) = sum over legs of 1 / (k + rank).

    `dense` maps chunk_id -> (rank, similarity, text, meta), `sparse` maps
    chunk_id -> (rank, text, meta); meta has filename and page_number. Every
    candidate from either leg comes back, sorted by fused score, keeping its
    rank in each leg. Ties keep insertion order (dense candidates first).
    """
    fused: dict[str, RetrievedChunk] = {}
    for cid, (rank, similarity, text, meta) in dense.items():
        fused[cid] = RetrievedChunk(
            chunk_id=cid,
            text=text,
            filename=meta["filename"],
            page_number=meta["page_number"],
            score=1.0 / (k + rank),
            dense_rank=rank,
            sparse_rank=None,
            similarity=round(similarity, 4),
        )
    for cid, (rank, text, meta) in sparse.items():
        if cid in fused:
            fused[cid].score += 1.0 / (k + rank)
            fused[cid].sparse_rank = rank
        else:
            fused[cid] = RetrievedChunk(
                chunk_id=cid,
                text=text,
                filename=meta["filename"],
                page_number=meta["page_number"],
                score=1.0 / (k + rank),
                dense_rank=None,
                sparse_rank=rank,
                similarity=None,
            )
    return sorted(fused.values(), key=lambda c: c.score, reverse=True)


class HybridStore:
    def __init__(self, settings: Settings, embeddings: EmbeddingProvider):
        """`embeddings` is the configured provider (embeddings.py); the store
        only calls embed_documents/embed_query and reads its dimension."""
        self._settings = settings
        self._embeddings = embeddings

        if settings.chroma_host:
            client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        else:
            client = chromadb.PersistentClient(path=settings.chroma_dir)
        self._collection = client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._check_embedding_dimension()

        # BM25 corpus mirror: list of (chunk_id, text, metadata), guarded by a
        # lock because ingestion can race with queries.
        self._lock = threading.Lock()
        self._corpus: list[tuple[str, str, dict]] = []
        self._bm25: BM25Okapi | None = None
        self._rebuild_from_chroma()

    def _check_embedding_dimension(self) -> None:
        """Fail fast if the collection holds vectors from a different model.

        Chroma fixes a collection's dimension at first insert. A collection
        built with a previous embedding model (e.g. 1536-dim OpenAI vectors
        vs 384-dim MiniLM) would otherwise fail cryptically deep inside a
        query. Mixed-model vectors are never comparable anyway, so the only
        correct fix is re-ingesting — say so explicitly.
        """
        peek = self._collection.get(limit=1, include=["embeddings"])
        embeddings = peek.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return  # empty collection: dimension gets set by our first insert
        existing_dim = len(embeddings[0])
        model_dim = self._embeddings.dimension
        if existing_dim != model_dim:
            raise RuntimeError(
                f"Chroma collection '{self._settings.collection_name}' holds "
                f"{existing_dim}-dim vectors, but embedding model "
                f"'{self._embeddings.model}' produces {model_dim}-dim. "
                "It was built with a different embedding model — wipe the "
                "Chroma data (delete ./chroma_db, or `docker compose down -v`) "
                "and re-upload your documents."
            )

    # --- index maintenance -------------------------------------------------

    def _rebuild_from_chroma(self) -> None:
        data = self._collection.get(include=["documents", "metadatas"])
        with self._lock:
            self._corpus = list(zip(data["ids"], data["documents"], data["metadatas"]))
            self._rebuild_bm25_locked()
        logger.info("BM25 index rebuilt from Chroma: %d chunks", len(self._corpus))

    def _rebuild_bm25_locked(self) -> None:
        # BM25Okapi is immutable, so adding documents means rebuilding.
        # Fine at this scale; see module docstring.
        self._bm25 = (
            BM25Okapi([_tokenize(text) for _, text, _ in self._corpus])
            if self._corpus
            else None
        )

    def count(self) -> int:
        with self._lock:
            return len(self._corpus)

    def list_documents(self) -> list[dict]:
        """Distinct uploaded documents, newest first.

        Derived from chunk metadata rather than kept in a separate table —
        Chroma already persists everything needed, so a registry would be a
        second source of truth to keep in sync. Chunks ingested before the
        `uploaded_at` field existed sort last.
        """
        docs: dict[str, dict] = {}
        with self._lock:
            for _, _, meta in self._corpus:
                doc = docs.setdefault(
                    meta["doc_id"],
                    {
                        "doc_id": meta["doc_id"],
                        "filename": meta["filename"],
                        "uploaded_at": meta.get("uploaded_at"),
                        "chunks": 0,
                    },
                )
                doc["chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["uploaded_at"] or "", reverse=True)

    # --- ingestion ----------------------------------------------------------

    async def add_chunks(self, chunks: list[Chunk]) -> None:
        texts = [c.text for c in chunks]
        embeddings = await self._embeddings.embed_documents(texts)
        ids = [c.chunk_id for c in chunks]
        uploaded_at = datetime.now(timezone.utc).isoformat()
        metadatas = [
            {
                "doc_id": c.doc_id,
                "filename": c.filename,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "uploaded_at": uploaded_at,
            }
            for c in chunks
        ]
        self._collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        with self._lock:
            self._corpus.extend(zip(ids, texts, metadatas))
            self._rebuild_bm25_locked()

    # --- retrieval ----------------------------------------------------------

    async def embed_query(self, query: str) -> list[float]:
        return await self._embeddings.embed_query(query)

    async def search(self, query: str, doc_ids: list[str] | None = None) -> list[RetrievedChunk]:
        """Hybrid search: top candidates from each leg, fused with RRF.

        `doc_ids` scopes both legs to those documents: the dense leg via a
        Chroma `where` filter, the sparse leg by skipping out-of-scope chunks
        after scoring (a per-scope BM25 index isn't worth it at this scale).
        """
        if self.count() == 0:
            return []
        return await self.search_with_vector(query, await self.embed_query(query), doc_ids)

    async def search_with_vector(
        self, query: str, query_vector: list[float], doc_ids: list[str] | None = None
    ) -> list[RetrievedChunk]:
        """search() with the query already embedded, so the benchmark can
        time embedding and search separately."""
        return (await self.fused_ranking(query, query_vector, doc_ids))[: self._settings.top_k]

    async def fused_ranking(
        self, query: str, query_vector: list[float], doc_ids: list[str] | None = None
    ) -> list[RetrievedChunk]:
        """Every candidate from either leg, fused and sorted, before the top_k
        cut. Each chunk keeps its rank in each leg (the benchmark reports
        per-leg ranks of the gold page from this)."""
        if self.count() == 0:
            return []

        n = min(self._settings.rrf_candidates, self.count())
        scope = set(doc_ids) if doc_ids else None

        # Dense leg: cosine similarity over the provider's embeddings in Chroma.
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=n,
            where={"doc_id": {"$in": doc_ids}} if doc_ids else None,
            include=["documents", "metadatas", "distances"],
        )
        dense: dict[str, tuple[int, float, str, dict]] = {}
        for rank, (cid, text, meta, dist) in enumerate(
            zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]),
            start=1,
        ):
            dense[cid] = (rank, 1.0 - dist, text, meta)  # cosine distance -> similarity

        # Sparse leg: BM25 over the same chunks. Ranks are assigned within the
        # scope so a filtered search behaves like BM25 over just those docs.
        sparse: dict[str, tuple[int, str, dict]] = {}
        with self._lock:
            scores = self._bm25.get_scores(_tokenize(query))
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            rank = 0
            for i in order:
                if scores[i] <= 0 or rank >= n:
                    break  # zero score = no term overlap at all; not a real match
                cid, text, meta = self._corpus[i]
                if scope is not None and meta["doc_id"] not in scope:
                    continue
                rank += 1
                sparse[cid] = (rank, text, meta)

        return rrf_fuse(dense, sparse, self._settings.rrf_k)
