import os
import re
import pathlib
from typing import List, Dict, Optional, Literal, TypedDict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Bibliotecas de IA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langgraph.graph import StateGraph, START, END

# Carrega variáveis de ambiente (API KEY)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI(title="API Código Harpia")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Em produção, substitua pelo seu domínio exato para mais segurança
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. CONFIGURAÇÃO DA IA E RAG ---

llm = ChatGoogleGenerativeAI(model='gemini-3-pro-preview', temperature=0.0, api_key=GOOGLE_API_KEY)

# Carregamento de Documentos
def inicializar_vectorstore():
    docs = []
    caminho_pdfs = pathlib.Path("./documentos")
    for n in caminho_pdfs.glob("*.pdf"):
        loader = PyMuPDFLoader(str(n))
        docs.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    return FAISS.from_documents(chunks, embeddings)

# Inicializa o banco de dados (em produção, você salvaria localmente para ser mais rápido)
vectorstore = inicializar_vectorstore()
retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.3, "k": 4})

# --- 2. LÓGICA DE TRIAGEM E RAG ---

class TriagemOut(BaseModel):
    decisao: Literal['AUTO_RESOLVER', 'PEDIR_INFO', 'ABRIR_CHAMADO']
    urgencia: Literal['BAIXA', 'MEDIA', 'ALTA']
    campos_falantes: List[str] = []

triagem_chain = llm.with_structured_output(TriagemOut)

prompt_rag = ChatPromptTemplate.from_messages([
    ("system", "Você é um Assistente da Código Harpia. Responda com base no contexto. Se não souber, diga 'Não sei'."),
    ("human", "Pergunta: {input}\n\nContexto:\n{context}")
])
document_chain = create_stuff_documents_chain(llm, prompt_rag)

# --- 3. FUNÇÕES AUXILIARES (Limpas do Colab) ---

def perguntar_politica_RAG(pergunta: str) -> Dict:
    docs_relacionados = retriever.invoke(pergunta)
    if not docs_relacionados:
        return {"answer": "Não sei.", "contexto_encontrado": False}
    
    answer = document_chain.invoke({"input": pergunta, "context": docs_relacionados})
    return {"answer": answer, "contexto_encontrado": True}

# --- 4. GRAFO (LANGGRAPH) ---

class AgentState(TypedDict, total=False):
    pergunta: str
    triagem: dict
    resposta: Optional[str]
    rag_sucesso: bool
    acao_final: str

def node_triagem(state: AgentState):
    res = triagem_chain.invoke([SystemMessage(content="Você é um triador..."), HumanMessage(content=state["pergunta"])])
    return {"triagem": res.model_dump()}

def node_auto_resolver(state: AgentState):
    res_rag = perguntar_politica_RAG(state["pergunta"])
    return {"resposta": res_rag["answer"], "rag_sucesso": res_rag["contexto_encontrado"], "acao_final": "AUTO_RESOLVER"}

# ... (Mantenha as outras funções de nós e decisões que você já tem no seu código original) ...
# O código do workflow permanece o mesmo, apenas compile-o aqui:
workflow = StateGraph(AgentState)
workflow.add_node("triagem", node_triagem)
workflow.add_node("auto_resolver", node_auto_resolver)
# ... adicione os outros nós e edges ...
workflow.add_edge(START, "triagem")
# ...
grafo = workflow.compile()

# --- 5. ENDPOINTS DA API ---

class UserQuery(BaseModel):
    message: str

@app.post("/chat")
async def chat(query: UserQuery):
    try:
        inputs = {"pergunta": query.message}
        resultado = grafo.invoke(inputs)
        return {
            "status": "success",
            "resposta": resultado.get("resposta"),
            "decisao": resultado.get("triagem", {}).get("decisao"),
            "urgencia": resultado.get("triagem", {}).get("urgencia")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))