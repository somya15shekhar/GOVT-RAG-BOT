import os
import json
from ocr.extract_text import extract_text
from chunking.chunk_text import chunk_text
from embedding.embed_chunks import get_embedding_model, embed_chunks

PDF_DIR = r"data/text_data"
OUTPUT_PATH = r"data/embeddings.json"

def process_all_pdfs():
    model = get_embedding_model()
    all_data = []

    for filename in os.listdir(PDF_DIR):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_DIR, filename)
        print(f"📄 Processing: {filename}")

        try:
            text = extract_text(pdf_path)
            chunks = chunk_text(text)
            embeddings = embed_chunks(model, chunks)

            for chunk, emb in zip(chunks, embeddings):
                all_data.append({
                    "chunk": chunk,
                    "embedding": emb.tolist(),
                    "source": filename  # Optional: useful in RAG
                })
        except Exception as e:
            print(f"⚠️ Error with {filename}: {e}")
            continue

    # Save all embeddings
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)
    print(f"\n✅ Saved {len(all_data)} chunks to {OUTPUT_PATH}")

if __name__ == "__main__":
    process_all_pdfs()
