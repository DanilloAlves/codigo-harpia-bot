import os
import pathlib
from typing import List, Dict, Optional, Literal, TypedDict

# 1. Configurações de Ambiente e Framework
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="API Código Harpia")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Importações de IA (Removido o que causava erro)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# --- MODELO E BASE DE CONHECIMENTO ---
llm = ChatGoogleGenerativeAI(
    model='gemini-3-pro-preview', 
    temperature=0.0, 
    api_key=GOOGLE_API_KEY
)

def inicializar_vectorstore():
    import os
    docs = []
    
    # Pega o caminho onde o main.py está rodando
    diretorio_atual = os.getcwd()
    print(f"--- DIAGNÓSTICO DE PASTA ---")
    print(f"Diretório atual de execução: {diretorio_atual}")
    print(f"Conteúdo do diretório atual: {os.listdir(diretorio_atual)}")
    
    # Tenta localizar a pasta documentos de forma flexível
    caminho_pdfs = pathlib.Path(diretorio_atual) / "documentos"
    
    if not caminho_pdfs.exists():
        print(f"ERRO CRÍTICO: A pasta {caminho_pdfs} NÃO EXISTE no Render.")
        return None

    print(f"Arquivos na pasta documentos: {os.listdir(str(caminho_pdfs))}")

    for n in caminho_pdfs.glob("*.pdf"):
        try:
            print(f"Tentando carregar: {n.name}")
            loader = PyMuPDFLoader(str(n))
            docs.extend(loader.load())
        except Exception as e:
            print(f"Falha no arquivo {n.name}: {e}")
            
    if not docs:
        print("RESULTADO: Nenhum documento válido foi processado.")
        return None
        
    print(f"TOTAL: {len(docs)} páginas carregadas com sucesso.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    # Voltando para o modelo mais compatível do mercado para evitar o erro 404
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(chunks, embeddings)

vectorstore = inicializar_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) if vectorstore else None

# --- LÓGICA DO AGENTE (FORMA DIRETA) ---
class AgentState(TypedDict):
    pergunta: str
    resposta: Optional[str]

def node_responder(state: AgentState):
    if not retriever:
        return {"resposta": "Conhecimento não carregado."}
    
    # Busca os documentos
    docs_rel = retriever.invoke(state["pergunta"])
    contexto = "\n\n".join([doc.page_content for doc in docs_rel])
    
    # Monta o prompt manualmente (Evita o erro de importar chains)
    prompt = f"""Você é o consultor oficial da Código Harpia. 
    Use o contexto abaixo para responder à pergunta do empresário. 
    Se não souber, diga que não encontrou essa informação e peça para entrar em contato.

    Contexto:
    {contexto}

    Pergunta: {state['pergunta']}"""

    # Chama o Gemini diretamente
    resposta = llm.invoke(prompt)
    return {"resposta": resposta.content}

workflow = StateGraph(AgentState)
workflow.add_node("responder", node_responder)
workflow.add_edge(START, "responder")
workflow.add_edge("responder", END)
grafo = workflow.compile()

# --- ENDPOINT ---
class UserQuery(BaseModel):
    message: str

@app.post("/chat")
async def chat(query: UserQuery):
    try:
        resultado = grafo.invoke({"pergunta": query.message})
        return {"resposta": resultado.get("resposta"), "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)








