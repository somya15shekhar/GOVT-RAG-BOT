from ocr.extract_text import extract_text
from chunking.chunk_text import chunk_text
from embedding.embed_chunks import get_embedding_model, embed_chunks
from retriever.faiss_retriever import FaissRetriever
from rag.rag_chain import RAGChain
import numpy as np
import os
import json

if __name__ == "__main__":
    r''' COMMENTED THIS BLOCK SINCE RAN IT ONCE AND SAVED THE EMBEDDINGS TO JSON
    pdf_path = r"C:\Users\Somya Shekhar\Desktop\chatbot-rag\data\compendium_of_govt._of_india_schemes_programmes.pdf"
    text = extract_text(pdf_path)
    
    print("\n📄 Extracted Text Preview:")
    print("-" * 50)
    print(text[:1500])

    print("\n🔪 Chunking Text...")
    chunks = chunk_text(text)

    print(f"✅ Total Chunks Created: {len(chunks)}")
    print("\n🧩 First Chunk:")
    print("-" * 50)
    print(chunks[0])
    print("-" * 50)

    print("\n📐 Generating Embeddings...")
    model = get_embedding_model()
    embeddings = embed_chunks(model, chunks)

    data_to_save = [
        {"chunk": chunk, "embedding": embedding.tolist()}
        for chunk, embedding in zip(chunks, embeddings)
    ]
    output_path = "data/embeddings.json"
    with open(output_path, "w") as f:
        json.dump(data_to_save, f, indent=4)
    print(f"✅ Embeddings saved to {output_path}")
    '''

# ON TERMINAL/POWERSHELL: $env:TOGETHER_API_KEY = "your-api-key-here"
# python main.py

    # ✅ Step 1: Get Together API key
    together_api_key = os.getenv("TOGETHER_API_KEY")
    if not together_api_key:
        raise ValueError("Please set TOGETHER_API_KEY as an environment variable.")

    # ✅ Step 2: Load retriever
    print("🔎 Initializing model and retriever...")
    model = get_embedding_model()
    retriever = FaissRetriever()
    retriever.model = model

    # ✅ Step 3: Initialize RAG pipeline
    rag = RAGChain(retriever, together_api_key)

    # ✅ Step 4: Ask your question
    question = "What are the key features of the Pradhan Mantri Awas Yojana?"
    answer = rag.answer_question(question, top_k=2)

    print("\n🤖 Chatbot Answer:")
    print("-" * 50)
    print(answer)
