import streamlit as st
from retriever.faiss_retriever import FaissRetriever
from embedding.embed_chunks import get_embedding_model
from rag.rag_chain import RAGChain
import os
import requests
from together import Together
from dotenv import load_dotenv
from pathlib import Path  # ✅ Cross-platform path handling
import platform

from translate.translator import translate_hi_to_en, translate_en_to_hi
import langdetect  # Optional: to auto-detect Hindi input


# --- Optional: Preserved commented code for reference ---
comments = r'''
from ocr.extract_text import extract_text 
from chunking.chunk_text import chunk_text 
from embedding.embed_chunks import embed_chunks 
import numpy as np 
import json  

pdf_path = r"C:\Users\Somya Shekhar\Desktop\chatbot-rag\data\compendium_of_govt._of_india_schemes_programmes.pdf"     
text = extract_text(pdf_path)           
chunks = chunk_text(text)      
model = get_embedding_model()     
embeddings = embed_chunks(model, chunks)      

data_to_save = [         
    {"chunk": chunk, "embedding": embedding.tolist()}         
    for chunk, embedding in zip(chunks, embeddings)     
]     
output_path = "data/embeddings.json"     
with open(output_path, "w") as f:         
    json.dump(data_to_save, f, indent=4)     
'''

if "android" in platform.platform().lower():
    st.warning("⚠️ Best viewed on desktop for full performance.")


# --- Step 1: Download Embeddings from Hugging Face if missing ---
def download_embeddings_if_missing():
    path = Path("data/embeddings.json")
    if not path.exists():
        st.info("📥 Downloading embeddings.json from Hugging Face...")
        url = "https://huggingface.co/datasets/somya15shekhar/govt-schemes-embeddings/resolve/main/embeddings.json"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(response.content)
                st.success("✅ Embeddings downloaded successfully.")
            else:
                st.error("❌ Failed to download embeddings. Check the Hugging Face URL or permissions.")
                st.stop()
        except Exception as e:
            st.error(f"❌ Download failed: {e}")
            st.stop()

download_embeddings_if_missing()

# --- Step 2: Load Together API Key ---
load_dotenv()
together_api_key = os.getenv("TOGETHER_API_KEY") or st.secrets.get("TOGETHER_API_KEY")

if not together_api_key:
    st.error("Please set TOGETHER_API_KEY as an environment variable or in Streamlit secrets.")
    st.stop()

# --- Step 3: Cached Model & Retriever ---
@st.cache_resource(show_spinner="🔄 Loading model...")
def load_model():
    return get_embedding_model()

@st.cache_resource(show_spinner="🔄 Loading retriever...")
def load_retriever(_model):
    retriever = FaissRetriever()
    retriever.load_embeddings("data/embeddings.json")
    retriever.model = _model
    return retriever

try:
    model = load_model()
    retriever = load_retriever(model)
except Exception as e:
    st.error(f"❌ Error loading model or retriever: {e}")
    st.stop()

# --- Step 4: Test Together client ---
try:
    test_client = Together(api_key=together_api_key)
    st.success("✅ Together client created successfully")
except Exception as e:
    st.error(f"❌ Error creating Together client: {e}")
    st.stop()

# --- Step 5: Streamlit UI ---
st.title("🤖 Sarkari Scheme Chatbot")
st.caption("Ask about Indian government schemes in English or Hindi")

query = st.text_input("Ask your question:")
if query:
    with st.spinner("Thinking..."):
        try:
            # Detect language (basic check)
            is_hindi = False
            try:
                is_hindi = detect(query) == 'hi'
            except:
                pass  # In case language detection fails

            if is_hindi:
                query_translated = translate_hi_to_en(query)
            else:
                query_translated = query

            rag = RAGChain(retriever, together_api_key)
            answer_en = rag.answer_question(query_translated, top_k=4)

            final_answer = translate_en_to_hi(answer_en) if is_hindi else answer_en
            st.success(final_answer)

        except Exception as e:
            st.error(f"❌ Failed to generate answer: {e}")
