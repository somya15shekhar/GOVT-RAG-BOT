import streamlit as st
from retriever.qdrant_retriever import QdrantRetriever
from embedding.embed_chunks import get_embedding_model
from rag.rag_chain import RAGChain
import os
from together import Together
from dotenv import load_dotenv
import platform

if "android" in platform.platform().lower():
    st.warning("⚠️ Best viewed on desktop for full performance.")

# --- Step 1: Load API Keys ---
load_dotenv()
together_api_key = os.getenv("TOGETHER_API_KEY") or st.secrets.get("TOGETHER_API_KEY")

if not together_api_key:
    st.error("Please set TOGETHER_API_KEY as an environment variable or in Streamlit secrets.")
    st.stop()

# --- Step 2: Load Model & Retriever ---
@st.cache_resource(show_spinner="🔄 Loading model...")
def load_model():
    return get_embedding_model()

try:
    @st.cache_resource(show_spinner="🔄 Connecting to Qdrant...")
    def load_retriever(model):
        retriever = QdrantRetriever()
        retriever.model = model
        return retriever
    model = load_model()
    retriever = load_retriever(model)
    
except Exception as e:
    st.error(f"❌ Error loading model or retriever: {e}")
    st.stop()

# --- Step 3: Test Together client ---
try:
    test_client = Together(api_key=together_api_key)
    st.success("✅ Together client created successfully")
except Exception as e:
    st.error(f"❌ Error creating Together client: {e}")
    st.stop()

# --- Step 4: Streamlit UI ---
st.title("🤖 Sarkari Scheme Chatbot")
st.caption("Ask about Indian government schemes in English or Hindi")

query = st.text_input("Ask your question:")
if query:
    with st.spinner("Thinking..."):
        try:
            rag = RAGChain(retriever, together_api_key)
            answer = rag.answer_question(query, top_k=4)
            st.success(answer)
        except Exception as e:
            st.error(f"❌ Failed to generate answer: {e}")