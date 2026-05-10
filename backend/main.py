import os
import chromadb
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
claude = Anthropic()
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")

# ─── Helper Functions ───────────────────────────────────────

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def search_chunks(question, n_results=5):
    question_embedding = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=n_results
    )
    return results["documents"][0], results["metadatas"][0]

# ─── Endpoints ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Extract text
    text = extract_text_from_pdf(file.file)
    
    # Chunk it
    chunks = chunk_text(text)
    
    # Embed it
    embeddings = model.encode(chunks).tolist()
    
    # Store in ChromaDB
    ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": file.filename, "chunk_index": i} for i in range(len(chunks))]
    
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    
    return {
        "message": f"✅ {file.filename} ingested successfully",
        "chunks_created": len(chunks)
    }

@app.post("/chat")
def chat(request: ChatRequest):
    # Step 1 — find relevant chunks
    chunks, metadatas = search_chunks(request.message)
    context = "\n".join(chunks)
    
    # Step 2 — send to Claude with context
    response = claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system="You are a helpful assistant. Answer questions using only the provided context. If the answer isn't in the context, say 'I don't have that information.'",
        messages=[
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.message}"}
        ]
    )
    
    # Step 3 — return answer with sources
    sources = list(set([m["source"] for m in metadatas]))
    
    return {
        "answer": response.content[0].text,
        "sources": sources
    }