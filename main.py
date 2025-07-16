import streamlit as st
from retriever.faiss_retriever import FaissRetriever
from embedding.embed_chunks import get_embedding_model
from rag.rag_chain import RAGChain
import os
import gdown
from together import Together
# Preserved commented code for embedding generation of one pdf (test run once)
comments = r'''
from ocr.extract_text import extract_text 
from chunking.chunk_text import chunk_text 
from embedding.embed_chunks import embed_chunks 
import numpy as np 
import json  

# COMMENTED THIS BLOCK SINCE RAN IT ONCE AND SAVED THE EMBEDDINGS TO JSON     
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

def download_embeddings_if_missing():
    """Download embeddings.json from Google Drive if it doesn't exist locally"""
    path = r"data/embeddings.json"
    if not os.path.exists(path):
        st.info("📥 Downloading embeddings.json from Google Drive...")
        gdown.download(r"https://drive.google.com/uc?id=YOUR_FILE_ID&confirm=t", path, quiet=False)

if __name__ == "__main__":
    # ON TERMINAL/POWERSHELL: $env:TOGETHER_API_KEY = "your-api-key-here"
    
    # ✅ Step 1: Download embeddings if missing
    download_embeddings_if_missing()
    
    # ✅ Step 2: Get Together API key
    from dotenv import load_dotenv
    load_dotenv()
    #import os

    together_api_key = os.getenv("TOGETHER_API_KEY") or st.secrets.get("TOGETHER_API_KEY")
    
    if not together_api_key:
        st.error("Please set TOGETHER_API_KEY as an environment variable or in Streamlit secrets.")
        st.stop()
    
    # ✅ Step 3: Initialize model and retriever
    print("🔎 Initializing model and retriever...")
    model = get_embedding_model()
    retriever = FaissRetriever()
    retriever.model = model
    
    print(f"Debug - API key being used: {together_api_key[:15]}...")
    print(f"Debug - API key length: {len(together_api_key)}")
    print(f"Debug - API key type: {type(together_api_key)}")

# Test the client creation
    try:
        test_client = Together(api_key=together_api_key)
        print("✅ Together client created successfully")
    except Exception as e:
        print(f"❌ Error creating Together client: {e}")

    # ✅ Step 4: Initialize RAG pipeline
    rag = RAGChain(retriever, together_api_key)
    
    # ✅ Step 5: Streamlit UI
    st.title("🤖 Sarkari Scheme Chatbot")
    st.caption("Ask about Indian govt schemes in natural language (English/Hindi)")
    
    query = st.text_input("Ask your question:")
    if query:
        with st.spinner("Thinking..."):
            # Use the same question format as your old code
            answer = rag.answer_question(query, top_k=2)
            
            print("\n🤖 Chatbot Answer:")
            print("-" * 50)
            print(answer)
            
            st.success(answer)