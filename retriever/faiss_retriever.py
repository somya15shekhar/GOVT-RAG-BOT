import faiss
import numpy as np
import json

class FaissRetriever:
    def __init__(self, embedding_dim = 384, json_path = "data/embeddings.json"):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim) #Create a simple FAISS index that uses Euclidean distance to find similar vectors of length dim.
        self.text_chunks =[]

        self.load_embeddings(json_path)

    def load_embeddings(self, json_path):
        with open(json_path, "r", encoding= "utf-8") as f:
            data = json.load(f)
        
        embeddings = []
        for item in data:
            self.text_chunks.append(item["chunk"])
            embeddings.append(np.array(item["embedding"], dtype=np.float32))  # Ensure embeddings are

        embedding_matrix = np.vstack(embeddings)
        self.index.add(embedding_matrix)  # Add embeddings to the FAISS index

        print(f"✅ Loaded {len(self.text_chunks)} text chunks and embeddings from {json_path}")

    def search(self, query, top_k=5):
        query_embedding = np.array(query, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i in indices[0]:
            results.append(self.text_chunks[i])
        return results
        