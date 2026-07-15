import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from markdownify import markdownify as md
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate



load_dotenv()

app = FastAPI(title="Documentation RAG-Agent")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
cohere_api_key = os.getenv("COHERE_API_KEY")

bm25_store = {}


class ScrapeModel(BaseModel):
    url: str
    session_id: str

class QueryModel(BaseModel):
    question: str
    session_id: str


#API Endpoints

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/scrape")
async def scrape_web(data: ScrapeModel):
    try:
        #use jina to avoid getting blocked by website
        jina_url = f"https://r.jina.ai/{data.url}"
        headers = {
            "User-Agent": "Docu-Agent"
        }

        #access the website
        response = requests.get(jina_url, headers=headers, timeout=30)

        if response.status_code != 200:
            return {"Error": f"blocked by website {jina_url} with status code: {response.status_code}"}

        #get markdown representation
        markdown_text = response.text

        if not markdown_text.strip():
            return {"Error": "Website was empty after cleaning"}

        #create langchain document
        doc = Document(page_content=markdown_text, metadata={"source": data.url})

        
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language = Language.MARKDOWN,
            chunk_size = 1000,
            chunk_overlap = 200
        )

        #split text into chunks
        chunks = text_splitter.split_documents([doc])

        collection_name = f"session_{data.session_id}"

        #store embeddings on Qdrant
        QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name,
            force_recreate=True
        )

        #store embeddings locally
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 10
        bm25_store[data.session_id] = bm25_retriever

        return {"Status": "Upload successfull",
                "URL": jina_url,
                "Session_ID": data.session_id,
                "Chunks": len(chunks)}
    except Exception as e:
        return {"ERROR": f"could not scrape: {str(e)}"}

@app.post("/ask")
async def ask_question(query: QueryModel):
    try:
        if query.session_id not in bm25_store:
            return {"load a website first"}
        
        collection_name = f"session_{query.session_id}"
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=embeddings
        )

        vector_retriever = vector_store.as_retriever(search_kwargs={"k":10})
        bm25_retriever = bm25_store[query.session_id]

        ensamble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )

        cohere_reranker = CohereRerank(top_n=3, model="rerank-multilingual-v3.0", cohere_api_key=cohere_api_key)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=cohere_reranker,
            base_retriever=ensamble_retriever
        )

        best_results = compression_retriever.invoke(query.question)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in best_results])

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)

        promt_template = ChatPromptTemplate.from_template("""
        You are a highly technical AI assistant responsible for explaining technical code documentations.
        Explain the question ONLY based on the following context.
        If you can find relevant code in the context, show it!
        If you can not find an answer from the given context, state that clearly and do NOT hallucinate something!
        
        Context:
        {context}
        
        Question:
        {question}
        """)

        chain = promt_template | llm
        response = chain.invoke({"context": context_text, "question": query.question})

        return {"Question": query.question,
                "Response": response.content,
                "Session_ID": query.session_id,
                "Chunks": len(best_results)}
    except Exception as e:
        return {"Error": f"Error in RAG: {str(e)}"}

@app.post("/clear")
async def clear_session(data: ScrapeModel):
    return {"status": f"Cleared Session Data: {data.session_id}"}

