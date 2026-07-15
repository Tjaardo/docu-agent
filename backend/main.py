from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


load_dotenv()

app = FastAPI(title="Documentation RAG-Agent")

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
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        response = requests.get(data.url, headers=headers, timeout=10)

        if response.status_code != 200:
            return {"Error": f"blocked by website with status code: {response.status_code}"}

        soup = BeautifulSoup(response.text, "html.parser")

        main_content = soup.find("main") or soup.find("article") or soup.find(id="content") or soup.find(class_="content")

        if main_content:
            soup = main_content
        else:
            for elem in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "form"]):
                elem.extract()

        markdown_text = md(str(soup), heading_style="ATX")

        if not markdown_text.strip():
            return {"Error": "Website was empty after cleaning"}

        doc = Document(page_content=markdown_text, metadata={"source": data.url})

        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language = Language.MARKDOWN,
            chunk_size = 1000,
            chunk_overlap = 200
        )

        chunks = text_splitter.split_documents([doc])

        return {"URL": data.url,
                "Session_ID": data.session_id,
                "Chunks": len(chunks)}
    except Exception as e:
        return {"ERROR": f"could not scrape: {str(e)}"}

@app.post("/ask")
async def ask_question(query: QueryModel):
    return {"Status": "Successful",
            "question": query.question,
            "Session_ID": query.session_id}

@app.post("/clear")
async def clear_session(data: ScrapeModel):
    return {"status": f"Cleared Session Data: {data.session_id}"}

