from fastapi import FastAPI, Form
import hashlib, openai, os
from uuid import uuid4
import pinecone
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# API Keys
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))

def generate_hash(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

@app.post("/query")
async def query(prompt: str = Form(...)):
    prompt_hash = generate_hash(prompt)

    # Exact hash search
    try:
        matches = index.query(vector=[], filter={"hash": {"$eq": prompt_hash}}, top_k=1, include_metadata=True)
        if matches and matches.matches:
            return {"answer": matches.matches[0].metadata["answer"]}
    except:
        pass

    # Semantic search
    embedding = openai.embeddings.create(model="text-embedding-3-small", input=prompt)["data"][0]["embedding"]
    matches = index.query(vector=embedding, top_k=1, include_metadata=True)
    if matches and matches.matches and matches.matches[0].score > 0.90:
        return {"answer": matches.matches[0].metadata["answer"]}

    # Generate new answer
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response["choices"][0]["message"]["content"]

    # Store new in Pinecone
    vector_id = str(uuid4())
    metadata = {"answer": answer, "hash": prompt_hash}
    index.upsert([(vector_id, embedding, metadata)])

    return {"answer": answer}
