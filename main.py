import os
import pathlib
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Bibliotecas de IA atualizadas
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

class GoogleCustomEmbeddings(Embeddings):
    def __init__(self):
        # Usamos a v1 fixa que já funcionou para as portas
        self.client = genai.Client(
            api_key=GOOGLE_API_KEY,
            http_options={'api_version': 'v1'}
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # O modelo 'text-embedding-004' deu 404, então usamos o 'embedding-001'
        # que é o padrão universal e infalível
        response = self.client.models.embed_content(
            model="embedding-001", 
            contents=texts,
            config={'task_type': 'RETRIEVAL_DOCUMENT'}
        )
        return [e.values for e in response.embeddings]

    def embed_query(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model="embedding-001",
            contents=text,
            config={'task_type': 'RETRIEVAL_QUERY'}
        )
        return response.embeddings[0].values

# Inicialização do LLM
llm = ChatGoogleGenerativeAI(
    model='gemini-3-pro-preview', 
    temperature=0.1, 
    api_key=GOOGLE_API_KEY
)

def criar_base_conhecimento():
    """Função que força o carregamento dos PDFs no início"""
    print("--- INICIANDO CARREGAMENTO DO CÓDIGO HARPIA ---")
    docs = []
    caminho_base = pathlib.Path(__file__).parent.resolve()
    caminho_pdfs = caminho_base / "documentos"
    
    if not caminho_pdfs.exists():
        print(f"ALERTA: Pasta {caminho_pdfs} não encontrada.")
        return None

    pdfs = list(caminho_pdfs.glob("*.pdf"))
    print(f"PDFs localizados: {[f.name for f in pdfs]}")

    for arquivo in pdfs:
        try:
            print(f"Lendo: {arquivo.name}")
            loader = PyMuPDFLoader(str(arquivo))
            docs.extend(loader.load())
        except Exception as e:
            print(f"Erro ao ler PDF: {e}")
            
    if not docs:
        print("ALERTA: Nenhum conteúdo extraído.")
        return None
        
    try:
        print("Criando embeddings v1 (estável)...")
        embeddings = GoogleCustomEmbeddings()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
        chunks = splitter.split_documents(docs)
        db = FAISS.from_documents(chunks, embeddings)
        print("SUCESSO: Base de conhecimento criada!")
        return db
    except Exception as e:
        print(f"ERRO CRÍTICO NO EMBEDDING: {e}")
        return None

# Variáveis globais carregadas no Startup
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
        return {"resposta": "Erro no servidor.", "erro": str(e)}

@app.get("/")
async def root():
    return {
        "status": "Online",
        "conhecimento": "Pronto" if retriever else "Vazio",
        "versao": "1.2.0"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

