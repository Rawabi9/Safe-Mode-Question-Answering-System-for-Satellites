import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

# 1.Data Extraction 
def get_pdf_text():
    all_text = ""
    pdf_files = [f for f in os.listdir() if f.endswith(".pdf")]
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    all_text += extracted_text + "\n"
        except Exception as e:
            print(f"Error reading {pdf}: {e}")
    return all_text

text = get_pdf_text()

# 2.Text Chunking
chunk_size = 600
overlap = 150 
chunks = []
for i in range(0, len(text), chunk_size - overlap):
    chunks.append(text[i:i + chunk_size])

# 3. Vector Embedding
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

print(f"System Ready: Loaded {len(chunks)} text chunks from PDFs.")

# 4.Retraival 
while True:
    query = input("\nDescribe the Satellite Status or ask a question: ")

    if len(query.strip()) < 3:
        print("\nPlease provide more details.")
        continue


    query_embedding = embedding_model.encode([query]).astype("float32")
    k = 5  
    distances, indices = index.search(query_embedding, k)

    retrieved_text = ""
    for i in indices[0]:
        if i != -1: 
            retrieved_text += chunks[i].replace("\n", " ").strip() + "\n---\n"

    prompt = f"""
You are a Satellite Operational Intelligence System. Your goal is to diagnose issues and provide actions based on technical manuals.

TECHNICAL CONTEXT:
{retrieved_text}

USER SCENARIO / QUESTION:
"{query}"

DIAGNOSTIC RULES:
1. If the user describes a problem (e.g., 'low power', 'high heat', 'no signal'), identify the corresponding Safe Mode or recovery procedure from the context.
2. If the exact words aren't there, use the context to provide the closest logical technical advice.
3. Be professional and prioritize "Action Steps" if a failure is detected.

RESPONSE:
"""
    # 5.AI Generation Ollama 
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:0.5b",
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        
        result = response.json().get("response", "No response returned from Ollama")
        print("\n" + "="*50)
        print("DIAGNOSIS & ACTION PLAN:")
        print("="*50)
        print(result)
        print("="*50)
        
    except Exception as e:
        print(f"\nError connecting to Ollama: {e}")
