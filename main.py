import os
import pathlib
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# IA e Documentos
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # A MUDANÇA REAL

app = FastAPI(title="Projeto Harpia")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# LLM para as respostas (Gemini continua aqui)
llm = ChatGoogleGenerativeAI(
    model='gemini-3-pro-preview', 
    temperature=0.1, 
    api_key=GOOGLE_API_KEY
)

def criar_base_conhecimento():
    print("--- OPERAÇÃO HARPIA: CARREGAMENTO LOCAL ---")
    docs = []
    caminho_pdfs = pathlib.Path(__file__).parent.resolve() / "documentos"
    
    if not caminho_pdfs.exists():
        return None

    pdfs = list(caminho_pdfs.glob("*.pdf"))
    for arquivo in pdfs:
        try:
            print(f"Processando arquivo: {arquivo.name}")
            loader = PyMuPDFLoader(str(arquivo))
            docs.extend(loader.load())
        except Exception as e:
            print(f"Erro no PDF: {e}")
            
    if not docs:
        return None
        
    try:
        print("Iniciando Embedding Local (HuggingFace)...")
        # Este modelo roda NO RENDER. Não usa API do Google. Zero erro 404.
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
        chunks = splitter.split_documents(docs)
        
        db = FAISS.from_documents(chunks, embeddings)
        print("SUCESSO ABSOLUTO: Base de conhecimento pronta!")
        return db
    except Exception as e:
        print(f"ERRO NOS EMBEDDINGS LOCAIS: {e}")
        return None

# Variáveis globais
vectorstore = criar_base_conhecimento()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) if vectorstore else None

class UserQuery(BaseModel):
    message: str

@app.post("/chat")
async def chat(query: UserQuery):
    try:
        if retriever:
            docs_rel = retriever.invoke(query.message)
            contexto = "\n\n".join([d.page_content for d in docs_rel])
            prompt = f"Você é o consultor oficial CÓDIGO HARPIA. Use o contexto:\n{contexto}\n\nPergunta: {query.message}"
        else:
            prompt = f"Consultor CÓDIGO HARPIA (Base Offline). Responda: {query.message}"

        resposta = llm.invoke(prompt)
        return {"resposta": resposta.content}
    except Exception as e:
        return {"resposta": "Erro técnico.", "erro": str(e)}

@app.get("/")
async def root():
    return {
        "status": "Online",
        "conhecimento": "Pronto" if retriever else "Vazio",
        "motor": "HuggingFace Local"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
