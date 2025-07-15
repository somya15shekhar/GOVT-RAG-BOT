from ocr.extract_text import extract_text  #ocr folder -> extract_text file -> extract_text function
from chunking.chunk_text import chunk_text
from embedding.embed_chunks import get_embedding_model, embed_chunks
import json
import numpy as np

if __name__ == "__main__":
    pdf_path = r"C:\Users\Somya Shekhar\Desktop\chatbot-rag\data\compendium_of_govt._of_india_schemes_programmes.pdf"  # Example: Pradhan Mantri Kaushal Vikas Yojana
    text = extract_text(pdf_path)
    
    print("\n📄 Extracted Text Preview:")
    print("-" * 50)
    print(text[:1500])  # Preview first 1500 chars

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

    print(f"✅ Generated {len(embeddings)} embeddings.")
    print("\n📊 Sample Embedding Vector (First Chunk) (Show only first 10 dimensions) :")
    print(embeddings[0][:10])  # Show only first 10 dimensions

    data_to_save = [
        {
            "chunk": chunk,
            "embedding": embedding.tolist()  # Convert numpy array to list for JSON serialization
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]
    output_path = "data/embeddings.json"
    with open(output_path, "w") as f:
        json.dump(data_to_save, f, indent=4) # Write the data as JSON with indentation for pretty printing
    print(f"✅ Embeddings saved to {output_path}")