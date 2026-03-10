import os
import pathlib
from typing import Optional, TypedDict
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

app = FastAPI(title="Projeto Harpia AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=GOOGLE_API_KEY)

# --- CLASSE DE EMBEDDING INFALÍVEL (Usando embedding-001) ---
class GoogleCustomEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # O modelo 'embedding-001' é o mais compatível com a API atual
        result = genai.embed_content(
            model="models/embedding-001",
            content=texts,
            task_type="retrieval_document"
        )
        return result['embedding']

    def embed_query(self, text: str) -> list[float]:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']

# MODELO DE CHAT (gemini-3-pro-preview)
llm = ChatGoogleGenerativeAI(
    model='gemini-3-pro-preview', 
    temperature=0.1, 
    api_key=GOOGLE_API_KEY
)

def inicializar_vectorstore():
    import os
    docs = []
    caminho_pdfs = pathlib.Path(os.getcwd()) / "documentos"
    
    if not caminho_pdfs.exists():
        return None

    # Carregamento robusto de PDFs
    for arquivo in os.listdir(str(caminho_pdfs)):
        if arquivo.lower().endswith(".pdf"):
            try:
                loader = PyMuPDFLoader(str(caminho_pdfs / arquivo))
                docs.extend(loader.load())
            except:
                continue
            
    if not docs:
        return None
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    try:
        # Criando a base com o modelo estável
        return FAISS.from_documents(chunks, GoogleCustomEmbeddings())
    except Exception as e:
        print(f"ERRO NOS EMBEDDINGS: {e}")
        return None

# Inicialização da base de conhecimento
vectorstore = inicializar_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) if vectorstore else None

class UserQuery(BaseModel):
    message: str

@app.post("/chat")
async def chat(query: UserQuery):
    try:
        msg = query.message
        
        if retriever:
            docs_rel = retriever.invoke(msg)
            contexto = "\n\n".join([d.page_content for d in docs_rel])
            prompt = f"Você é o consultor CÓDIGO HARPIA. Use o contexto:\n{contexto}\n\nPergunta: {msg}"
        else:
            prompt = f"Você é o consultor CÓDIGO HARPIA. Responda: {msg}"

        resposta = llm.invoke(prompt)
        return {"resposta": resposta.content, "status": "success"}
    except Exception as e:
        # Retorna o erro real nos logs do Render para diagnóstico
        print(f"ERRO NO CHAT: {e}")
        return {"resposta": "Desculpe, tive um problema técnico. Tente novamente.", "erro": str(e)}

@app.get("/")
async def root():
    return {"status": "Online", "conhecimento": "Pronto" if retriever else "Vazio"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
