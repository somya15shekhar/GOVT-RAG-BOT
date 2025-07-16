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
    ''' 
    def search(self, query_embedding, top_k=3):
        # Prepare query vector in correct shape and type
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        distances, indices = self.index.search(query_embedding, top_k)  # Perform search

        results = [self.text_chunks[i] for i in indices[0]]  # Get matching text chunks
        return results  '''

    def search(self, query_embedding, top_k=3):
    # Ensure the query embedding is 2D and float32
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        seen = set()

        print("\n🔍 Retrieved Chunks:")
        for idx in indices[0]:
            chunk = self.text_chunks[idx]
            if chunk not in seen:
                print(f"--- Chunk ---\n{chunk[:500]}")  # Print first 500 chars
                results.append(chunk)
                seen.add(chunk)
            else:
                print("⚠️ Duplicate chunk skipped")

        return results


    