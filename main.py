from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List

import concurrent.futures
import random
import time

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from google import genai
from google.genai import types
from google.genai.errors import ServerError

# ==== SETUP ====

app = FastAPI()

# Allow CORS for local React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Change if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

# === Global variables (initialized on startup) ===
client = None
vector_store = None

def safe_generate_content(client, **kwargs):
    max_retries = 5
    backoff_base = 1.0

    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_content(**kwargs)
        except ServerError as e:
            if e.status_code == 503:
                sleep_time = backoff_base * (2 ** (attempt - 1)) + random.uniform(0, backoff_base)
                time.sleep(sleep_time)
            else:
                raise
    raise RuntimeError("Max retries exceeded.")

def load_and_split(csv_path: Path, chunk_size: int = 1000, chunk_overlap: int = 200):
    loader = CSVLoader(file_path=csv_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def init_vector_store(docs, url: str, collection_name: str, embeddings):
    try:
        return QdrantVectorStore.from_existing_collection(
            url=url, collection_name=collection_name, embedding=embeddings
        )
    except Exception:
        return QdrantVectorStore.from_documents(
            documents=docs, url=url, collection_name=collection_name, embedding=embeddings
        )

def generate_parallel_queries(client, original_query: str, n_variants: int = 3) -> List[str]:
    system_instructions = f'''
You are a helpful AI assistant tasked with reformulating user queries to improve retrieval in a RAG system.
Generate {n_variants} distinct, detailed reformulations of "{original_query}". Return only the bullet-or-numbered list.
'''
    response = safe_generate_content(
        client,
        model='gemini-2.0-flash-001',
        config=types.GenerateContentConfig(system_instruction=system_instructions),
        contents=[original_query]
    )
    lines = [ln.strip() for ln in response.text.splitlines() if ln.strip()]
    return [ln.split(maxsplit=1)[-1].lstrip(".- ") for ln in lines]

def retrieve_for_queries(store, queries: List[str], top_k: int = 3):
    all_docs = []
    with concurrent.futures.ThreadPoolExecutor() as pool:
        futures = { pool.submit(store.similarity_search, q, top_k): q for q in queries }
        for fut in concurrent.futures.as_completed(futures):
            try:
                all_docs.extend(fut.result())
            except Exception as e:
                print(f"[Warning] retrieval failed for '{futures[fut]}': {e}")
    seen = set()
    unique_docs = []
    for d in all_docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            unique_docs.append(d)
    return unique_docs

def answer_query(client, user_query: str, context_docs) -> str:
    context = "\n---\n".join(d.page_content for d in context_docs)
    prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    system_instruction = (
        "You are a helpful assistant that answers based on provided context. "
        "If the answer isn't in the context, reply 'There is no information about this.'"
    )
    response = safe_generate_content(
        client,
        model='gemini-2.0-flash-001',
        config=types.GenerateContentConfig(system_instruction=system_instruction),
        contents=[prompt]
    )
    return response.text.strip()

# ==== FastAPI Startup ====

@app.on_event("startup")
def startup_event():
    global client, vector_store
    api_key = ""
    csv_path = Path(__file__).parent / "data.csv"
    qdrant_url = "http://localhost:6333"
    collection = "parallel_queries"

    client = genai.Client(api_key=api_key)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )
    docs = load_and_split(csv_path)
    vector_store = init_vector_store(docs, qdrant_url, collection, embeddings)
    print("Backend initialized.")

# ==== Route ====

@app.post("/chat", response_model=QueryResponse)
def chat(req: QueryRequest):
    variants = generate_parallel_queries(client, req.query)
    retrieved_docs = retrieve_for_queries(vector_store, variants)
    answer = answer_query(client, req.query, retrieved_docs)
    return QueryResponse(response=answer)
