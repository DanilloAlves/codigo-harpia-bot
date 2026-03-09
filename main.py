import os
import pathlib
from typing import Optional, TypedDict
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Bibliotecas de IA
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings # Interface base

app = FastAPI(title="Projeto Harpia - IA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuração do SDK Oficial da Google
genai.configure(api_key=GOOGLE_API_KEY)

# --- CLASSE DE EMBEDDING CUSTOMIZADA (Para evitar o Erro 404) ---
class GoogleCustomEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Usa o SDK oficial diretamente, que é imune ao erro de rota do LangChain
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=texts,
            task_type="retrieval_document"
        )
        return result['embedding']

    def embed_query(self, text: str) -> list[float]:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding'][0] if isinstance(result['embedding'][0], list) else result['embedding']

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
        print("Pasta documentos não encontrada.")
        return None

    for n in caminho_pdfs.glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(n))
            docs.extend(loader.load())
            print(f"Carregado: {n.name}")
        except Exception as e:
            print(f"Erro no PDF {n.name}: {e}")
            
    if not docs:
        return None
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    try:
        # Usamos nossa classe customizada que usa o SDK direto da Google
        embeddings = GoogleCustomEmbeddings()
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        print(f"ERRO NOS EMBEDDINGS: {e}")
        return None

# Inicialização
vectorstore = inicializar_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) if vectorstore else None

@app.post("/chat")
async def chat(query: BaseModel): # Simplificado para teste
    try:
        msg = query.message if hasattr(query, 'message') else str(query)
        if retriever:
            docs_rel = retriever.invoke(msg)
            contexto = "\n\n".join([d.page_content for d in docs_rel])
            prompt = f"Consultor CÓDIGO HARPIA. Contexto:\n{contexto}\n\nPergunta: {msg}"
        else:
            prompt = f"Consultor CÓDIGO HARPIA. Responda: {msg}"

        resposta = llm.invoke(prompt)
        return {"resposta": resposta.content, "status": "success"}
    except Exception as e:
        print(f"Erro: {e}")
        return {"resposta": "Erro interno no servidor.", "erro": str(e)}

@app.get("/")
async def root():
    return {"status": "Online", "conhecimento": "Pronto" if retriever else "Vazio"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
