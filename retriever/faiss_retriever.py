import faiss
import numpy as np
import json

class FaissRetriever:
    def __init__(self, embedding_dim=384, json_path=r"data\embeddings.json"):
        self.embedding_dim = embedding_dim

        # Create FAISS index for L2 (Euclidean) similarity search
        self.index = faiss.IndexFlatL2(embedding_dim)

        self.text_chunks = []  # Store text chunks linked to embeddings

        # Load data and populate the index
        self.load_embeddings(json_path)

    def load_embeddings(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        embeddings = []

        for item in data:
            self.text_chunks.append(item["chunk"])
            embeddings.append(np.array(item["embedding"], dtype=np.float32))  # Convert to float32 for FAISS

        embedding_matrix = np.vstack(embeddings)  # Stack into a 2D matrix
        self.index.add(embedding_matrix)  # Add to FAISS index

    def search(self, query_embedding, top_k=3):
        # Prepare query vector in correct shape and type
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        distances, indices = self.index.search(query_embedding, top_k)  # Perform search

        results = [self.text_chunks[i] for i in indices[0]]  # Get matching text chunks
        return results 