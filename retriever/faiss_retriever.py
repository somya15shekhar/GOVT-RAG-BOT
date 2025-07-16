import faiss
import numpy as np
import json

class FaissRetriever:
    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.text_chunks = []
        self.model = None  # Will be set externally
        self.loaded = False

    def load_embeddings(self, json_path="data/embeddings.json"):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        embeddings = []
        for item in data:
            self.text_chunks.append(item["chunk"])
            embeddings.append(np.array(item["embedding"], dtype=np.float32))

        embedding_matrix = np.vstack(embeddings)
        self.index.add(embedding_matrix)
        self.loaded = True

    def search(self, query_embedding, top_k=3):
        if not self.loaded:
            raise RuntimeError("You must call `load_embeddings()` before `search()`.")

        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        seen = set()
        print("\n🔍 Retrieved Chunks:")
        for idx in indices[0]:
            chunk = self.text_chunks[idx]
            if chunk not in seen:
                print(f"--- Chunk ---\n{chunk[:500]}")
                results.append(chunk)
                seen.add(chunk)
            else:
                print("⚠️ Duplicate chunk skipped")

        return results
