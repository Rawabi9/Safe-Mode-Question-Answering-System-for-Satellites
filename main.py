from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

reader = PdfReader("nasaSafeMode.pdf")

text = ""
for page in reader.pages:
    extracted_text = page.extract_text()
    if extracted_text:
        text += extracted_text

chunk_size = 500
chunks = []

for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

while True:

    query = input("\nAsk a question about satellite safe mode: ")

    

    if len(query.strip()) < 5:
        print("\nInvalid input, Please enter a valid question.")
        continue

    query_embedding = embedding_model.encode([query]).astype("float32")
    k = 3
    distances, indices = index.search(query_embedding, k)

    if distances[0][0] > 1.2:
        
        print("\nOut of Scope, Please enter a valid question.")
        continue

    retrieved_text = ""
    for i in indices[0]:
        retrieved_text += chunks[i].replace("\n", " ").strip() + "\n"

    prompt = f"""
You are a strict NASA document assistant.

Rules:
- Answer ONLY using the provided documentation.
- Do NOT use any external knowledge.
- Do NOT guess.
- If the answer is not clearly found in the text, reply exactly with:
"The answer is not available in the NASA document."

User question:
{query}

Retrieved documentation:
{retrieved_text}

Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:0.5b",
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()
    result = data.get("response", "No response returned from Ollama")

    print("\nUser question:", query)
    print("\nGenerated Answer:\n")
    print(result)