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
    model='gemini-2.0-flash', 
    temperature=0.0, 
    api_key=GOOGLE_API_KEY
)

def inicializar_vectorstore():
    docs = []
    caminho_pdfs = pathlib.Path("./documentos")
    if not caminho_pdfs.exists():
        return None
    for n in caminho_pdfs.glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(n))
            docs.extend(loader.load())
        except Exception:
            continue
    if not docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
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

# --- INICIALIZAÇÃO ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
