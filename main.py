import os
import re
import pathlib
from typing import List, Dict, Optional, Literal, TypedDict

# 1. Carregar variáveis de ambiente primeiro
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# 2. Framework Web
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="API Código Harpia")

# Configuração de CORS para permitir acesso do seu site na Hostinger
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Importação das bibliotecas de IA (Garante que os submódulos sejam carregados)
import langchain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langgraph.graph import StateGraph, START, END

# --- CONFIGURAÇÃO DO MODELO ---
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash', 
    temperature=0.0, 
    api_key=GOOGLE_API_KEY
)

# --- CONFIGURAÇÃO DO RAG (Conhecimento dos PDFs) ---
def inicializar_vectorstore():
    docs = []
    # Busca os arquivos na pasta 'documentos' que você subiu no GitHub
    caminho_pdfs = pathlib.Path("./documentos")
    for n in caminho_pdfs.glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(n))
            docs.extend(loader.load())
        except Exception as e:
            print(f"Erro ao carregar {n}: {e}")
    
    if not docs:
        print("Aviso: Nenhum PDF encontrado na pasta 'documentos'.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=GOOGLE_API_KEY
    )
    return FAISS.from_documents(chunks, embeddings)

vectorstore = inicializar_vectorstore()

if vectorstore:
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={"score_threshold": 0.3, "k": 4}
    )
else:
    retriever = None

# --- LÓGICA DE TRIAGEM ---
class TriagemOut(BaseModel):
    decisao: Literal['AUTO_RESOLVER', 'PEDIR_INFO', 'ABRIR_CHAMADO']
    urgencia: Literal['BAIXA', 'MEDIA', 'ALTA']
    campos_faltantes: List[str] = []

triagem_chain = llm.with_structured_output(TriagemOut)

prompt_rag = ChatPromptTemplate.from_messages([
    ("system", "Você é o Assistente da Código Harpia. Responda apenas com base no contexto. Se não souber, diga 'Não sei'."),
    ("human", "Pergunta: {input}\n\nContexto:\n{context}")
])

if retriever:
    document_chain = create_stuff_documents_chain(llm, prompt_rag)

# --- LANGGRAPH (ESTADO E NÓS) ---
class AgentState(TypedDict, total=False):
    pergunta: str
    triagem: dict
    resposta: Optional[str]
    rag_sucesso: bool
    acao_final: str

def node_triagem(state: AgentState):
    prompt_sistema = "Você é um triador de Service Desk para a Código Harpia."
    res = triagem_chain.invoke([
        SystemMessage(content=prompt_sistema), 
        HumanMessage(content=state["pergunta"])
    ])
    return {"triagem": res.model_dump()}

def node_auto_resolver(state: AgentState):
    if not retriever:
        return {"resposta": "Sistema de conhecimento não carregado.", "rag_sucesso": False}
    
    docs_relacionados = retriever.invoke(state["pergunta"])
    if not docs_relacionados:
        return {"resposta": "Não sei dizer sobre isso com base nos meus documentos.", "rag_sucesso": False}
    
    resposta = document_chain.invoke({"input": state["pergunta"], "context": docs_relacionados})
    return {"resposta": resposta, "rag_sucesso": True, "acao_final": "AUTO_RESOLVER"}

# --- CONSTRUÇÃO DO FLUXO ---
workflow = StateGraph(AgentState)
workflow.add_node("triagem", node_triagem)
workflow.add_node("auto_resolver", node_auto_resolver)
workflow.add_edge(START, "triagem")

def decidir(state):
    if state["triagem"]["decisao"] == "AUTO_RESOLVER": return "auto"
    return END

workflow.add_conditional_edges("triagem", decidir, {"auto": "auto_resolver", END: END})
workflow.add_edge("auto_resolver", END)
grafo = workflow.compile()

# --- ENDPOINT API ---
class UserQuery(BaseModel):
    message: str

@app.post("/chat")
async def chat(query: UserQuery):
    try:
        resultado = grafo.invoke({"pergunta": query.message})
        return {
            "resposta": resultado.get("resposta") or "Desculpe, não consegui processar sua dúvida.",
            "status": "success"
        }
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

