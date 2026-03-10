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

app = FastAPI(title="Projeto Harpia - IA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=GOOGLE_API_KEY)

# --- CLASSE DE EMBEDDING COM SDK OFICIAL (Sem LangChain intermediário) ---
class GoogleCustomEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Chamada direta ao motor do Google
        # Usamos o nome purista 'embedding-001' que é o mais aceito na v1beta
        response = genai.embed_content(
            model="models/embedding-001",
            content=texts,
            task_type="retrieval_document"
        )
        return response['embedding']

    def embed_query(self, text: str) -> list[float]:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        return response['embedding']

# MODELO DE CHAT (gemini-3-pro-preview)
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
    
    print(f"--- INICIANDO CARREGAMENTO CÓDIGO HARPIA ---")

    if not caminho_pdfs.exists():
        return None

    for arquivo in os.listdir(str(caminho_pdfs)):
        if arquivo.lower().endswith(".pdf"):
            try:
                loader = PyMuPDFLoader(str(caminho_pdfs / arquivo))
                docs.extend(loader.load())
                print(f"Sucesso: {len(docs)} páginas lidas de {arquivo}")
            except Exception as e:
                print(f"Erro na leitura: {e}")
            
    if not docs:
        return None
        
    try:
        # AQUI É ONDE O "VAZIO" VIRA "PRONTO"
        embeddings = GoogleCustomEmbeddings() 
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
        chunks = splitter.split_documents(docs)
        
        print(f"Gerando embeddings para {len(chunks)} trechos...")
        # Criamos o FAISS usando nossa classe que fala direto com o Google
        vector_db = FAISS.from_documents(chunks, embeddings)
        print("BASE DE CONHECIMENTO CRIADA COM SUCESSO!")
        return vector_db
    except Exception as e:
        print(f"ERRO FINAL NOS EMBEDDINGS: {e}")
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
            prompt = f"Consultor CÓDIGO HARPIA (Modo Geral). Responda: {msg}"

        resposta = llm.invoke(prompt)
        return {"resposta": resposta.content, "status": "success"}
    except Exception as e:
        return {"resposta": "Erro ao processar sua pergunta.", "detalhe": str(e)}

@app.get("/")
async def root():
    return {"status": "Online", "conhecimento": "Pronto" if retriever else "Vazio"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
