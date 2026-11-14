"""Cross-encoder reranking (RERANK=true), through ONNX Runtime via fastembed,
so torch isn't loaded at query time.

The default model, Xenova/ms-marco-MiniLM-L-6-v2 (an ONNX export of the
MS MARCO MiniLM cross-encoder, about 80 MB), is downloaded on first use into
fastembed's cache. It reads (query, passage) pairs and scores each passage's
relevance; higher is better. The model loads on the first rerank, not at
startup, so a store with reranking off never pays for it.
"""
import threading


class Reranker:
    def __init__(self, model: str):
        self.model = model
        self._encoder = None
        self._lock = threading.Lock()

    def scores(self, query: str, texts: list[str]) -> list[float]:
        with self._lock:  # one load, even when the first queries arrive together
            if self._encoder is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self._encoder = TextCrossEncoder(model_name=self.model)
        return [float(s) for s in self._encoder.rerank(query, texts)]
