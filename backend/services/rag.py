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

    def _mmr_diversity(self, indices: list, distances: list, q_emb: np.ndarray, k: int, lambda_: float = 0.5) -> list:
        if len(indices) <= k:
            return list(zip(indices, distances))

        selected = []
        candidate_pool = list(zip(indices, distances))

        all_embs = []
        for idx, _ in candidate_pool[:50]:
            all_embs.append(self._get_embedding(self.data[idx]["text"]))
        all_embs = np.array(all_embs)

        q_norm = q_emb / np.linalg.norm(q_emb)

        for _ in range(min(k, len(candidate_pool))):
            if not candidate_pool:
                break

            best_score = -1
            best_item = None
            best_idx = -1

            for i, (cand_idx, cand_dist) in enumerate(candidate_pool[:100]):
                sim_to_query = q_norm @ all_embs[len(selected)].T if len(selected) < len(all_embs) else 0
                sim_to_selected = 0
                if selected:
                    sel_embs = np.array([self._get_embedding(self.data[s[0]]["text"]) for s in selected])
                    sel_norms = sel_embs / np.linalg.norm(sel_embs, axis=1, keepdims=True)
                    cand_emb = self._get_embedding(self.data[cand_idx]["text"])
                    cand_norm = cand_emb / np.linalg.norm(cand_emb)
                    sims = sel_norms @ cand_norm
                    sim_to_selected = float(np.max(sims))

                mmr_score = lambda_ * (1 / (1 + cand_dist)) - (1 - lambda_) * sim_to_selected

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_item = (cand_idx, cand_dist)
                    best_idx = i

            if best_item:
                selected.append(best_item)
                candidate_pool.pop(best_idx)

        return selected

    def search(self, query: str, category: str = None, k: int = 3, use_mmr: bool = True):
        q_emb = self._get_embedding(query).reshape(1, -1)
        D, I = self.index.search(q_emb, k * 5)

        category_filtered = []
        for idx, dist in zip(I[0], D[0]):
            item = self.data[idx]
            if category and item.get("category") and item["category"] != category:
                continue
            category_filtered.append((idx, dist, item))

        if not category_filtered:
            category_filtered = [
                (idx, dist, self.data[idx])
                for idx, dist in zip(I[0], D[0])
            ]

        if use_mmr and len(category_filtered) > k:
            indices = [x[0] for x in category_filtered]
            distances = [x[1] for x in category_filtered]
            selected = self._mmr_diversity(indices, distances, q_emb, k)
            category_filtered = []
            for idx, dist in selected:
                item = self.data[idx]
                category_filtered.append((idx, dist, item))

        results = []
        for idx, dist, item in category_filtered[:k]:
            results.append({
                "text": item["text"],
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "source": item.get("source", ""),
                "score": float(1 / (1 + dist))
            })

        return results
