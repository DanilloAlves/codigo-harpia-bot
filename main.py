import os
import pathlib
from typing import Optional, TypedDict

# 1. Configurações de Ambiente e API
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Inicialização do App FastAPI com o Título do seu Projeto
app = FastAPI(title="Projeto Harpia - Inteligência Artificial")

# Configuração de CORS para o seu site na Hostinger e GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Importações de Inteligência Artificial
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END

# --- CONFIGURAÇÃO DO MODELO DE CHAT (gemini-3-pro-preview) ---
llm = ChatGoogleGenerativeAI(
    model='gemini-3-pro-preview', 
    temperature=0.1, 
    api_key=GOOGLE_API_KEY,
    version="v1"  # Força o uso da versão estável
)

def inicializar_vectorstore():
    """Lógica para carregar os PDFs e criar a base de conhecimento (RAG)"""
    import os
    docs = []
    diretorio_atual = os.getcwd()
    caminho_pdfs = pathlib.Path(diretorio_atual) / "documentos"
    
    print(f"--- DIAGNÓSTICO DO PROJETO HARPIA ---")
    print(f"Diretório: {diretorio_atual}")

    if not caminho_pdfs.exists():
        print(f"ERRO: Pasta 'documentos' não encontrada no servidor.")
        return None

    # Tenta carregar os arquivos PDF
    arquivos_pdf = list(caminho_pdfs.glob("*.pdf"))
    if not arquivos_pdf:
        print("AVISO: Nenhum arquivo .pdf encontrado na pasta 'documentos'.")
        return None

    for n in arquivos_pdf:
        try:
            print(f"Lendo documento: {n.name}")
            loader = PyMuPDFLoader(str(n))
            docs.extend(loader.load())
        except Exception as e:
            print(f"Falha ao processar {n.name}: {e}")
            
    if not docs:
        return None
        
    # Divide o conhecimento em blocos para busca eficiente
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    # --- SOLUÇÃO DEFINITIVA PARA O ERRO 404 DE EMBEDDING ---
    try:
        # Usamos o text-embedding-004 forçando a v1 (versão estável)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=GOOGLE_API_KEY,
            version="v1"
        )
        vector_db = FAISS.from_documents(chunks, embeddings)
        print("SUCESSO: Base de conhecimento criada com text-embedding-004 (v1)")
        return vector_db
    except Exception as e:
        print(f"ERRO CRÍTICO NOS EMBEDDINGS: {e}")
        return None

# Inicialização global da base de dados (Carregada ao ligar o servidor no Render)
vectorstore = inicializar_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) if vectorstore else None

# --- LÓGICA DO AGENTE (LangGraph) ---
class AgentState(TypedDict):
    pergunta: str
    resposta: Optional[str]

def node_responder(state: AgentState):
    pergunta = state["pergunta"]
    
    if retriever:
        # Recupera os trechos do seu e-book
        docs_rel = retriever.invoke(pergunta)
        contexto = "\n\n".join([doc.page_content for doc in docs_rel])
        
        prompt = f"""Você é o consultor oficial do CÓDIGO HARPIA. 
        Responda ao empresário com base no conhecimento do e-book abaixo.
        Se a informação não estiver no contexto, use seu conhecimento geral para ajudar.

        CONTEXTO:
        {contexto}
        
        PERGUNTA: {pergunta}"""
    else:
        prompt = f"Você é o consultor oficial do CÓDIGO HARPIA. Responda ao empresário: {pergunta}"

    resposta = llm.invoke(prompt)
    return {"resposta": resposta.content}

# Montagem do fluxo de resposta
workflow = StateGraph(AgentState)
workflow.add_node("responder", node_responder)
workflow.add_edge(START, "responder")
workflow.add_edge("responder", END)
grafo = workflow.compile()

# --- ENDPOINTS DA API ---
class UserQuery(BaseModel):
    message: str

@app.post("/chat")
async def chat(query: UserQuery):
    try:
        resultado = grafo.invoke({"pergunta": query.message})
        return {"resposta": resultado.get("resposta"), "status": "success"}
    except Exception as e:
        print(f"Erro na rota /chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    status_conhecimento = "CARREGADO" if retriever else "NÃO CARREGADO"
    return {
        "projeto": "CÓDIGO HARPIA",
        "status": "Online",
        "modelo": "Gemini 3 Pro Preview",
        "base_conhecimento": status_conhecimento
    }

if __name__ == "__main__":
    import uvicorn
    # O Render define a porta automaticamente
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
