````markdown
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
````

Create a `.env` file:

```
TOGETHER_API_KEY=your_key_here
```

Run the app:

```bash
streamlit run main.py
```

---

🚀 Future Work
Improve Hindi Answer Quality: Add a multilingual Together.ai model or post-process answers using translation libraries for better Hindi responses.

Scheme Name Extraction: Implement Named Entity Recognition (NER) or keyword highlighting to ensure the bot names specific schemes in its responses.

Search Filtering: Add filters to let users search by state, category (education, finance, women), or target group (farmers, students, senior citizens).

Better UI: Introduce collapsible answers, feedback thumbs-up/down, or even voice-to-text queries.

📥 Easy Data Updates – Two options to keep your scheme data fresh:

Local Update
➤ Drop new PDFs into the text_data/ folder
➤ Run process_all_pdfs.py to regenerate embeddings.json
➤ Replace the old JSON file in data/

Cloud Update (Hugging Face Repo)
➤ Upload your updated embeddings.json to your Hugging Face dataset repo
➤ On app restart, Streamlit auto-downloads the latest JSON file and loads it — no manual update needed!

---

## 🗣️ Feedback

* Have suggestions? Raise an issue or connect via [LinkedIn](www.linkedin.com/in/somya-shekhar-93299a27a)
* Want to expand to state-specific schemes or more Indian languages? Fork it!

---

## 👤 Author

Made with ❤️ by **Somya Shekhar**
🔗 📩 [somya15shekhar@gmail.com](mailto:somyashekhar61@gmail.com)


