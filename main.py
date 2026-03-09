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

# --- CLASSE DE EMBEDDING REVISADA (Versão Estável v1) ---
class GoogleCustomEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Forçamos a versão da API para 'v1' para evitar o erro 404
        client = genai.Client(api_key=GOOGLE_API_KEY, http_options={'api_version': 'v1'})
        result = client.models.embed_content(
            model="models/text-embedding-004",
            contents=texts,
            config=genai.types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        return result.embeddings

    def embed_query(self, text: str) -> list[float]:
        client = genai.Client(api_key=GOOGLE_API_KEY, http_options={'api_version': 'v1'})
        result = client.models.embed_content(
            model="models/text-embedding-004",
            contents=text,
            config=genai.types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        return result.embeddings

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
    
    print(f"--- INICIANDO CARREGAMENTO SEGURO ---")

    if not caminho_pdfs.exists():
        print(f"ERRO: Pasta {caminho_pdfs} não encontrada.")
        return None

    # Lista todos os arquivos para garantir que não perderemos nada
    for arquivo in os.listdir(str(caminho_pdfs)):
        # Verifica se o arquivo termina com .pdf (independente de ser maiúsculo ou minúsculo)
        if arquivo.lower().endswith(".pdf"):
            caminho_completo = caminho_pdfs / arquivo
            try:
                print(f"Lendo PDF: {arquivo}")
                loader = PyMuPDFLoader(str(caminho_completo))
                docs.extend(loader.load())
                print(f"Sucesso: {len(docs)} páginas carregadas até agora.")
            except Exception as e:
                print(f"Erro ao ler o arquivo {arquivo}: {e}")
            
    if not docs:
        print("RESULTADO: Nenhum conteúdo extraído dos PDFs.")
        return None
        
    try:
        # Usando o SDK oficial da Google para evitar o erro 404
        embeddings = GoogleCustomEmbeddings() 
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        print(f"Criando Vectorstore com {len(chunks)} trechos de conhecimento...")
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



