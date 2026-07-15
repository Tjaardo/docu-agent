from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

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
    return {"URL": data.url,
            "Session_ID": data.session_id}

@app.post("/ask")
async def ask_question(query: QueryModel):
    return {"Status": "Successful",
            "question": query.question,
            "Session_ID": query.session_id}

@app.post("/clear")
async def clear_session(data: ScrapeModel):
    return {"status": f"Cleared Session Data: {data.session_id}"}

