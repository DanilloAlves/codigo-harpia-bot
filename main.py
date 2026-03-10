import os
import pathlib
from typing import Optional, TypedDict
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
from langchain_community.embeddings import HuggingFaceEmbeddings # MUDANÇA AQUI

app = FastAPI(title="Projeto Harpia - IA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODELO DE CHAT (Mantemos o seu Gemini 3 Pro Preview)
llm = ChatGoogleGenerativeAI(
    model='gemini-3-pro-preview', 
    temperature=0.1, 
    api_key=GOOGLE_API_KEY
)

def inicializar_vectorstore():
    import os
    docs = []
    diretorio_atual = os.getcwd()
    caminho_pdfs = pathlib.Path(diretorio_atual) / "documentos"
    
    print(f"--- INICIANDO CARREGAMENTO COM EMBEDDING LOCAL ---")

    if not caminho_pdfs.exists():
        return None

    for arquivo in os.listdir(str(caminho_pdfs)):
        if arquivo.lower().endswith(".pdf"):
            try:
                print(f"Lendo PDF: {arquivo}")
                loader = PyMuPDFLoader(str(caminho_pdfs / arquivo))
                docs.extend(loader.load())
            except Exception as e:
                print(f"Erro ao ler {arquivo}: {e}")
            
    if not docs:
        return None
        
    try:
        # --- A GRANDE MUDANÇA ---
        # Esse modelo roda DENTRO do Render. Não usa API do Google. 
        # É impossível dar Erro 404 aqui.
        print("Carregando modelo de embedding local (HuggingFace)...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        
        print(f"Criando Vectorstore com {len(chunks)} trechos...")
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        print(f"ERRO NOS EMBEDDINGS LOCAIS: {e}")
        return None

# Inicialização
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
            prompt = f"Consultor CÓDIGO HARPIA. Use o contexto:\n{contexto}\n\nPergunta: {msg}"
        else:
            prompt = f"Consultor CÓDIGO HARPIA (Base Offline). Responda: {msg}"

        resposta = llm.invoke(prompt)
        return {"resposta": resposta.content, "status": "success"}
    except Exception as e:
        return {"resposta": "Erro no processamento.", "detalhe": str(e)}

@app.get("/")
async def root():
    return {"status": "Online", "conhecimento": "Pronto" if retriever else "Vazio"}

if __name__ == "__main__":
    import uvicorn
    # O Render usa a variável de ambiente PORT. Se não achar, usa 8000.
    port = int(os.environ.get("PORT", 8000))
    # Rodamos o app garantindo que o host seja 0.0.0.0
    uvicorn.run(app, host="0.0.0.0", port=port)
