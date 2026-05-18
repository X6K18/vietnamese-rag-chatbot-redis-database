import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.config import INDEX_PATH, DATA_PATH, EMBED_MODEL
from backend.services.redis_client import cache_embedding, get_cached_embedding

class RAG:
    def __init__(self):
        self.index = faiss.read_index(INDEX_PATH)
        with open(DATA_PATH, "rb") as f:
            self.data = pickle.load(f)
        self.model = None

    def _get_embedding(self, text: str):
        cached = get_cached_embedding(text)
        if cached:
            return np.array(cached, dtype=np.float32)
        if self.model is None:
            self.model = SentenceTransformer(EMBED_MODEL)
        emb = self.model.encode([text])[0].astype(np.float32)
        cache_embedding(text, emb.tolist())
        return emb

    def search(self, query: str, category: str = None, k: int = 3):
        q_emb = self._get_embedding(query).reshape(1, -1)
        D, I = self.index.search(q_emb, k * 5)  # lấy nhiều hơn để lọc

        results = []
        for idx, dist in zip(I[0], D[0]):
            item = self.data[idx]
            if category and item["category"] != category:
                continue
            results.append({
                "text": item["text"],
                "title": item["title"],
                "url": item["url"],
                "source": item["source"],
                "score": float(1 / (1 + dist))
            })
            if len(results) == k:
                break
        return results