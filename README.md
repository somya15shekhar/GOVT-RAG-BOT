# 🤖 Sarkari Scheme Chatbot

A chatbot that helps users discover Indian government schemes by asking questions in **English or Hindi**. Powered by a Retrieval-Augmented Generation (RAG) pipeline using 700+ real government PDFs.

---

## 🔍 What It Does

- Answers user queries about welfare schemes (education, jobs, health, etc.)
- Multilingual input (English/Hindi), AI output
- Retrieves responses from real policy data (not hallucinated)
- No fine-tuning, no model hosting — runs fast via API + FAISS

---

## 🧠 Tech Stack

- **Data**: 723+ government scheme PDFs from [MyScheme.gov.in](https://www.myscheme.gov.in/), hosted on Hugging Face  
- **Embedding Model**: `all-MiniLM-L6-v2` (SentenceTransformers)  
- **Vector Store**: FAISS (for semantic search)  
- **LLM API**: `meta-llama/Llama-3.3-70B-Instruct-Turbo` via [Together.ai](https://platform.together.xyz/)  
  - 🆓 Together.ai provides **free API access** with generous token limits for personal and research use  
- **Frontend**: Streamlit  

---

## 🚀 Deployment Notes

- App is hosted on [Streamlit Community Cloud](https://streamlit.io/cloud) — 100% free  
- Embedding file (`embeddings.json`, ~500MB) auto-downloads from Hugging Face at runtime  
- Together API key is stored in `.env` (or `st.secrets` when deployed)  
- No paid OpenAI API needed — uses Together.ai’s **free tier**  

---

## 🧪 Local Setup

```bash
git clone https://github.com/somya15shekhar/govt-rag-bot.git
cd govt-rag-bot
pip install -r requirements.txt
