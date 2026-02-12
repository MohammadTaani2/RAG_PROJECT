'''
RAG system project

GEN AI course

Mohammad AlTa'any

12/27/2025
'''

# Import required libraries

import os                          # To access environment variables
from PyPDF2 import PdfReader       # Read and extract text from PDF files
from pinecone import Pinecone, ServerlessSpec  # Pinecone vector database client
import requests                    # Make HTTP requests to OpenAI API

# Load API keys from environment

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# Split long text into overlapping chunks

def chunk_text(text, chunk_size=500, overlap=100):
    
    #Splits a long text into smaller overlapping chunks.
    words = text.split()
    chunks = []
    # Move forward by (chunk_size - overlap) words each time
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

# Generate embeddings using OpenAI

def get_embedding(text, api_key):
    """
    Converts text into a numerical vector using OpenAI embeddings.

    These vectors allow Pinecone to:
    - Perform semantic similarity search
    - Retrieve relevant document chunks during queries
    """
    response = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "text-embedding-3-small",
            "input": text
        },
        timeout=30
    )

    # Handle API failure
    if response.status_code != 200:
        print(f"Error: {response.text}")
        return None
    
    return response.json()["data"][0]["embedding"]

def load_pdfs(pdf_folder=r"C:\Users\taani\OneDrive\Desktop\RAG_PROJECT\PDFs"):
    """
    Loads all PDF files from a folder and extracts text.

    Each PDF is treated as a single document before chunking.
    """

    if not os.path.exists(pdf_folder):
        print(f"❌ Folder '{pdf_folder}' not found!")
        return []
    
    docs = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            path = os.path.join(pdf_folder, filename)
            try:
                reader = PdfReader(path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                docs.append(text)
                print(f"  ✅ Loaded: {filename}")
            except Exception as e:
                print(f"  ❌ Error loading {filename}: {e}")
    
    print(f"\nTotal loaded: {len(docs)} PDFs")
    return docs

def main():
    print("=== PDF Ingestion to Pinecone ===\n")

    """
    End-to-end pipeline:
    1. Load PDFs
    2. Chunk text
    3. Generate embeddings
    4. Store vectors in Pinecone
    """

    # Load PDFs
    docs = load_pdfs()
    if not docs:
        print("No PDFs found. Exiting.")
        return
    
    # Chunk documents
    print("\nChunking documents...")
    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)
    print(f"✅ Created {len(all_chunks)} chunks")
    
    # Setup Pinecone
    print("\nConnecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "myind"
    
    # Create index if needed
    if index_name not in [idx.name for idx in pc.list_indexes()]:
        print(f"Creating new index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"✅ Created index")
    else:
        print(f"✅ Using existing index '{index_name}'")
    
    index = pc.Index(index_name)
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    vectors = []
    for i, chunk in enumerate(all_chunks):
        emb = get_embedding(chunk, OPENAI_API_KEY)
        if emb:
            vectors.append((str(i), emb, {"text": chunk}))
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(all_chunks)}")
    
    print(f"✅ Generated {len(vectors)} embeddings")
    
    # Upload to Pinecone in batches
    print("\nUploading to Pinecone...")
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)
        print(f"  Uploaded {min(i+batch_size, len(vectors))}/{len(vectors)}")
    
    print("\nThe PDFs are now indexed and ready to query.")

# Script entry point

if __name__ == "__main__":
    main()