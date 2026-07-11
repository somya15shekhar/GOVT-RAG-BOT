import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
from ocr.extract_text import extract_text
from chunking.chunk_text import chunk_text
from embedding.embed_chunks import get_embedding_model, embed_chunks

load_dotenv()

PDF_DIR = "data/text_data"
COLLECTION_NAME = "govt-schemes"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
RESET_COLLECTION = True  # Set False to skip re-upload if collection exists

def process_all_pdfs():
    model = get_embedding_model()
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    all_pdfs = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    total_pdfs = len(all_pdfs)

    # Detect embedding dim from a test embed
    sample_embedding = model.encode("test")
    embedding_dim = len(sample_embedding)

    # Handle collection reset or creation
    if RESET_COLLECTION and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"🗑️ Deleted existing collection: {COLLECTION_NAME}")

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
        print(f"✅ Created collection: {COLLECTION_NAME} (dim={embedding_dim})")
    else:
        print(f"ℹ️ Collection already exists, appending to it: {COLLECTION_NAME}")

    # Start point_id after existing points to avoid conflicts when appending
    if not RESET_COLLECTION and client.collection_exists(COLLECTION_NAME):
        point_id = client.count(collection_name=COLLECTION_NAME).count
    else:
        point_id = 0

    total_uploaded = 0

    for pdf_index, filename in enumerate(all_pdfs, start=1):
        pdf_path = os.path.join(PDF_DIR, filename)
        print(f"\n📄 Processing {pdf_index} / {total_pdfs}: {filename}")

        try:
            text = extract_text(pdf_path)
            chunks = chunk_text(text)
            embeddings = embed_chunks(model, chunks)

            points = [
                PointStruct(
                    id=point_id + i,
                    vector=emb.tolist(),
                    payload={
                        "chunk": chunk,
                        "source": filename,
                        "pdf": filename.replace(".pdf", ""),
                        "chunk_id": i,
                    },
                )
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
            ]

            point_id += len(points)

            # Batch upload
            batch_size = 100
            for i in range(0, len(points), batch_size):
                client.upload_points(
                    collection_name=COLLECTION_NAME,
                    points=points[i: i + batch_size],
                )

            print(f"  ↳ Uploaded {len(points)} chunks | Progress: {pdf_index}/{total_pdfs}")
            total_uploaded += len(points)

        except Exception as e:
            print(f"⚠️ Error with {filename}: {e}")
            continue

    print(f"\n✅ Done. Total chunks uploaded to Qdrant: {total_uploaded}")

if __name__ == "__main__":
    process_all_pdfs()