from sentence_transformers import SentenceTransformer

def get_embedding_model():
    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    return model

def embed_chunks(model, chunks):
    embeddings = model.encode(chunks, show_progress_bar= True)
    return embeddings 