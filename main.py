import os
import pathlib
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Nova biblioteca sugerida pelo log do Render
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

app = FastAPI(title="Projeto Harpia")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CLASSE DE EMBEDDING CORRIGIDA (Forçando v1 estável) ---
class GoogleCustomEmbeddings(Embeddings):
    def __init__(self):
        # AQUI ESTÁ O SEGREDO: Forçamos a versão da API para 'v1'
        self.client = genai.Client(
            api_key=GOOGLE_API_KEY,
            http_options={'api_version': 'v1'} 
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self.client.models.embed_content(
                model="text-embedding-004",
                contents=texts,
                config={'task_type': 'RETRIEVAL_DOCUMENT'}
            )
            # Extrai os valores numéricos dos embeddings
            return [e.values for e in response.embeddings]
        except Exception as e:
            print(f"Erro ao gerar embedding de documentos: {e}")
            raise e

    def embed_query(self, text: str) -> list[float]:
        try:
            response = self.client.models.embed_content(
                model="text-embedding-004",
                contents=text,
                config={'task_type': 'RETRIEVAL_QUERY'}
            )
            return response.embeddings[0].values
        except Exception as e:
            print(f"Erro ao gerar embedding de query: {e}")
            raise e

llm = ChatGoogleGenerativeAI(
    model='gemini-3-pro-preview', 
    temperature=0.1, 
    api_key=GOOGLE_API_KEY
)

def inicializar_vectorstore():
    docs = []
    caminho_pdfs = pathlib.Path(os.getcwd()) / "documentos"
    
    if not caminho_pdfs.exists():
        print("Pasta documentos não encontrada.")
        return None

    for arquivo in caminho_pdfs.glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(arquivo))
            docs.extend(loader.load())
        except Exception as e:
            print(f"Erro: {e}")
            
    if not docs:
        return None
        
    try:
        embeddings = GoogleCustomEmbeddings()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
        chunks = splitter.split_documents(docs)
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        print(f"Erro Embedding: {e}")
        return None

vectorstore = inicializar_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) if vectorstore else None

class UserQuery(BaseModel):
    message: str

@app.post("/chat")
async def chat(query: UserQuery):
    try:
        if retriever:
            docs_rel = retriever.invoke(query.message)
            contexto = "\n\n".join([d.page_content for d in docs_rel])
            prompt = f"Consultor CÓDIGO HARPIA. Contexto:\n{contexto}\n\nPergunta: {query.message}"
        else:
            prompt = f"Consultor CÓDIGO HARPIA. Pergunta: {query.message}"

        resposta = llm.invoke(prompt)
        return {"resposta": resposta.content}
    except Exception as e:
        return {"resposta": "Erro técnico.", "detalhe": str(e)}

@app.get("/")
async def root():
    return {"status": "Online", "conhecimento": "Pronto" if retriever else "Vazio"}

# --- INICIALIZAÇÃO FIXA PARA O RENDER ---
if __name__ == "__main__":
    import uvicorn
    # Forçamos a leitura da porta do ambiente
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

