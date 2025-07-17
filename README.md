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
To keep my chatbot’s knowledge up-to-date with new government schemes, follow this pipeline:

Add New PDFs Locally
Place any new scheme PDFs inside the text_data/ folder on local machine.

Regenerate Embeddings
Run the process_all_pdfs.py script locally to extract text, chunk it, and generate an updated embeddings.json file that includes all PDFs.

Upload Updated Embeddings
Upload the new embeddings.json file to remote storage— [HUGGING FACE REPO](https://huggingface.co/datasets/somya15shekhar/govt-schemes-embeddings/tree/main)

Deploy or Restart Your App
Deployed chatbot app will automatically download the latest embeddings.json at startup and use it without requiring any code changes.