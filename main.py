from pinecone import Pinecone,ServerlessSpec
from dotenv import load_dotenv
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

load_dotenv()
# Initialize FastAPI
app = FastAPI()


# load the api keys 
pc_apikey=os.getenv("pinecone_api")
groq_api=os.getenv("grok_api")

# pincone database details like the index name 
index_name="hybrid-search-uhi-mitigation"

# the hugging face embeddings is used 
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# the spare matrix encoder used is loaded
bm25_encoder = BM25Encoder().load("bm25_values.json")


## initialize the pinecone client and load the index 
pc=Pinecone(api_key=pc_apikey)

# load the index 
index=pc.Index(index_name)

# load the retriever
retriever=PineconeHybridSearchRetriever(embeddings=embeddings,sparse_encoder=bm25_encoder,index=index)

# defining the prompt template 
template="""
    You are an expert in solving problems and giving solutions on UHI.
    Use the given context:
    <context>{context}</context>
    and your own knowledge.

    Provide 5 bullet points on UHI mitigation strategies based on the question below. Each point should be 1–2 lines.

    Question: {question}

    """
prompt = ChatPromptTemplate.from_template(template)

# Initialize components
client = OpenAI(
    api_key=groq_api,
    base_url="https://api.groq.com/openai/v1"
)

# Request model
class QueryRequest(BaseModel):
    query: str
    model: str = "llama3-70b-8192"
    temperature: float = 0.1

# Response model
class RAGResponse(BaseModel):
    answer: str
    sources: List[str] = []

# RAG prompt builder
def build_rag_prompt(query: str, docs: List[str]) -> List[Dict]:
    context = "\n\n".join(docs)
    return [
        {
            "role": "system",
            "content": """You are an expert in solving problems and giving solutions on UHI.
            Provide 5 bullet points on UHI mitigation strategies based on the question below. Each point should be 1–2 lines."""
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }
    ]

# RAG endpoint
@app.post("/query", response_model=RAGResponse)
async def rag_query(request: QueryRequest):
    try:
        # Retrieve documents
        docs = retriever.invoke(request.query)
        doc_texts = [doc.page_content for doc in docs]
        sources = list(set(doc.metadata.get("source", "unknown") for doc in docs))
        
        # Generate response
        response = client.chat.completions.create(
            model=request.model,
            messages=build_rag_prompt(request.query, doc_texts),
            temperature=request.temperature
        )
        
        return RAGResponse(
            answer=response.choices[0].message.content,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}