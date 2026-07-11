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

    def load_index(self, index_path="data/index.faiss", metadata_path="data/metadata.json"):
        # Load FAISS index from disk
        self.index = faiss.read_index(index_path)

        # Load chunk text and source metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.text_chunks = [item["chunk"] for item in metadata]
        self.loaded = True

    def search(self, query_embedding, top_k=3):
        if not self.loaded:
            raise RuntimeError("You must call `load_index()` before `search()`.")

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
