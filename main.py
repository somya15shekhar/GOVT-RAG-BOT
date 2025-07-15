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

    # ✅ Step 5: Run full RAG chatbot: 
    #  before starting-> on command prompt run(for permanent) :setx OPENAI_API_KEY "your_openai_api_key_" , run this in terminal to see key : echo $Env:OPENAI_API_KEY

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    print("\n🔎 Initializing Retriever & Model...")
    model = get_embedding_model()
    retriever = FaissRetriever()
    retriever.model = model  # Needed for embedding the query

    rag = RAGChain(retriever, openai_api_key)

    question = "What is the eligibility criteria for Pradhan Mantri Kaushal Vikas Yojana?"
    answer = rag.answer_question(question)

    print("\n🤖 Chatbot Answer:")
    print("-" * 50)
    print(answer)
