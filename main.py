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

# Inicialização do App FastAPI
app = FastAPI(title="CÓDIGO HARPIA AI")

# Configuração de CORS para aceitar requisições do seu site na Hostinger
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

# --- MODELO PRINCIPAL (Gemini 3 Pro Preview) ---
llm = ChatGoogleGenerativeAI(
    model='gemini-3-pro-preview', 
    temperature=0.1, 
    api_key=GOOGLE_API_KEY
)

def inicializar_vectorstore():
    """Carrega os PDFs da pasta 'documentos' e cria a base de conhecimento."""
    import os
    docs = []
    diretorio_atual = os.getcwd()
    caminho_pdfs = pathlib.Path(diretorio_atual) / "documentos"
    
    print(f"--- INICIANDO BUSCA DE CONHECIMENTO ---")
    print(f"Pasta alvo: {caminho_pdfs}")

    if not caminho_pdfs.exists():
        print(f"ERRO: Pasta {caminho_pdfs} não encontrada no servidor.")
        return None

    # Lista arquivos para conferência nos logs do Render
    arquivos = list(caminho_pdfs.glob("*.pdf"))
    print(f"Arquivos encontrados: {[f.name for f in arquivos]}")

    for n in arquivos:
        try:
            print(f"Lendo documento: {n.name}")
            loader = PyMuPDFLoader(str(n))
            docs.extend(loader.load())
        except Exception as e:
            print(f"Falha ao processar {n.name}: {e}")
            
    if not docs:
        print("AVISO: Nenhum PDF foi carregado com sucesso.")
        return None
        
    # Divide o texto em blocos menores para a IA processar melhor
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    # --- EMBEDDING (Versão Estável para evitar Erro 404) ---
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_db = FAISS.from_documents(chunks, embeddings)
        print("Base de conhecimento (FAISS) criada com sucesso!")
        return vector_db
    except Exception as e:
        print(f"Erro ao criar Embeddings: {e}")
        return None

# Inicialização global da base de dados
vectorstore = inicializar_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) if vectorstore else None

# --- LÓGICA DO AGENTE DE CHAT ---
class AgentState(TypedDict):
    pergunta: str
    resposta: Optional[str]

def node_responder(state: AgentState):
    pergunta = state["pergunta"]
    
    if retriever:
        # Busca trechos relevantes nos seus PDFs
        docs_rel = retriever.invoke(pergunta)
        contexto = "\n\n".join([doc.page_content for doc in docs_rel])
        
        prompt = f"""Você é o consultor oficial do projeto CÓDIGO HARPIA. 
        Sua missão é ajudar empresários a dominar a IA e automação.
        Responda de forma profissional e direta, usando o contexto abaixo.

        CONTEXTO DO E-BOOK:
        {contexto}
        
        PERGUNTA DO EMPRESÁRIO: {pergunta}"""
    else:
        # Resposta caso os PDFs não tenham sido carregados
        prompt = f"Você é o consultor oficial do CÓDIGO HARPIA. Responda ao empresário: {pergunta}"

    resposta = llm.invoke(prompt)
    return {"resposta": resposta.content}

# Construção do Grafo de Decisão (LangGraph)
workflow = StateGraph(AgentState)
workflow.add_node("responder", node_responder)
workflow.add_edge(START, "responder")
workflow.add_edge("responder", END)
grafo = workflow.compile()

# --- ENDPOINT DA API ---
class UserQuery(BaseModel):
    message: str

@app.post("/chat")
async def chat(query: UserQuery):
    try:
        resultado = grafo.invoke({"pergunta": query.message})
        return {"resposta": resultado.get("resposta"), "status": "success"}
    except Exception as e:
        print(f"Erro no processamento da rota /chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "API Código Harpia está Online!", "modelo": "Gemini 3 Pro Preview"}

if __name__ == "__main__":
    import uvicorn
    # O Render fornece a porta automaticamente pela variável de ambiente PORT
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
