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

- **Data**: 723+ government scheme PDFs from [MyScheme.gov.in](https://www.myscheme.gov.in/), hosted on [Hugging Face](https://huggingface.co/datasets/somya15shekhar/govt-schemes-embeddings)
- **Embedding Model**: `all-MiniLM-L6-v2` (SentenceTransformers)
- **Vector Store**: FAISS (for semantic search)
- **LLM API**: `meta-llama/Llama-3.3-70B-Instruct-Turbo` via [Together.ai](https://platform.together.xyz/)
  - 🆓 Together.ai provides **free API access** with generous token limits — no payment required
- **Frontend**: Streamlit

---

## 🚀 Deployment Notes

- App is hosted on [Streamlit Community Cloud](https://streamlit.io/cloud) — 100% free
- Embedding file (`embeddings.json`, ~500MB) is **not stored in GitHub**, it auto-downloads from Hugging Face at runtime
- Together API key is stored in `.env` (locally) or `st.secrets` (on Streamlit Cloud)
- Uses no OpenAI API — just **Together.ai’s free tier**

---

## 🧪 Local Setup

```bash
git clone https://github.com/somya15shekhar/govt-rag-bot.git
cd govt-rag-bot
pip install -r requirements.txt

## 🔁 Updating Data (Adding More PDFs)
You have two options:

Local:
Add new PDFs to text_data/ and run process_all_pdfs.py to regenerate embeddings.json.

Remote:
Upload processed embeddings.json to Hugging Face. When Streamlit restarts, it re-downloads this file automatically — no code change needed.