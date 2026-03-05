import os
import re
import pathlib
from typing import List, Dict, Optional, Literal, TypedDict

# 1. Inicialização de Ambiente
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# 2. Framework Web e Segurança (CORS para Hostinger)
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

# 3. Importação Blindada de IA (Resolução do erro de módulo)
import langchain
import langchain.chains.combine_documents # Força o carregamento do submódulo
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langgraph.graph import StateGraph, START, END

# --- MOTOR DE INTELIGÊNCIA ---
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash', 
    temperature=0.0, 
    api_key=GOOGLE_API_KEY
)

# --- BASE DE CONHECIMENTO (RAG) ---
def inicializar_vectorstore():
    docs = []
    caminho_pdfs = pathlib.Path("./documentos")
    
    if not caminho_pdfs.exists():
        print("Pasta 'documentos' não encontrada.")
        return None

    for n in caminho_pdfs.glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(n))
            docs.extend(loader.load())
        except Exception as e:
            print(f"Erro no arquivo {n}: {e}")
    
    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=GOOGLE_API_KEY
    )
    return FAISS.from_documents(chunks, embeddings)

vectorstore = inicializar_vectorstore()
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={"score_threshold": 0.3, "k": 4}
) if vectorstore else None

# --- LÓGICA DO AGENTE (LANGGRAPH) ---
class AgentState(TypedDict, total=False):
    pergunta: str
    triagem: dict
    resposta: Optional[str]
    rag_sucesso: bool

# Definição do Prompt de Resposta
prompt_rag = ChatPromptTemplate.from_messages([
    ("system", "Você é o consultor oficial da Código Harpia. Use o contexto para ajudar empresários a economizar tempo com IA. Se não souber, diga 'Não sei'."),
    ("human", "Pergunta: {input}\n\nContexto:\n{context}")
])

def node_auto_resolver(state: AgentState):
    if not retriever:
        return {"resposta": "Conhecimento técnico não carregado.", "rag_sucesso": False}
    
    docs_rel = retriever.invoke(state["pergunta"])
    if not docs_rel:
        return {"resposta": "Ainda não tenho essa informação nos meus manuais.", "rag_sucesso": False}
    
    chain = create_stuff_documents_chain(llm, prompt_rag)
    resposta = chain.invoke({"input": state["pergunta"], "context": docs_rel})
    return {"resposta": resposta, "rag_sucesso": True}

# Fluxo Simplificado para Produção
workflow = StateGraph(AgentState)
workflow.add_node("auto_resolver", node_auto_resolver)
workflow.add_edge(START, "auto_resolver")
workflow.add_edge("auto_resolver", END)
grafo = workflow.compile()

# --- INTERFACE DE COMUNICAÇÃO (API) ---
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
    import os
    # O Render define a porta automaticamente na variável de ambiente PORT
    port = int(os.environ.get("PORT", 8000)) 
    # Log para confirmar no console do Render
    print(f"Iniciando o servidor na porta: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


