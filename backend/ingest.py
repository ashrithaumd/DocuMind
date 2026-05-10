import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def ingest_pdf(pdf_path):
    print(f" Reading {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)
    
    print(f" Chunking text...")
    chunks = chunk_text(text)
    
    print(f" {len(chunks)} chunks created")
    
    embeddings = model.encode(chunks).tolist()
    
    ids = [f"{pdf_path}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": pdf_path, "chunk_index": i} for i in range(len(chunks))]
    
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    
    print(f" Done! {len(chunks)} chunks stored in ChromaDB")

# Test it
ingest_pdf("test.pdf")