import os
import pathlib
import pdfplumber
from typing import Optional, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

# 1. Configurações de Ambiente
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI(title="CÓDIGO HARPIA - IA ESTRATÉGICA")

# 2. Configuração de Acesso (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Inicialização da IA
client = genai.Client(api_key=GOOGLE_API_KEY)
MODELO_ALVO = "gemini-3-pro-preview"
TEMPERATURE = 0.2 

# Variável global para o conteúdo do e-book
CONTEUDO_EBOOK = ""

def carregar_conhecimento_harpia():
    global CONTEUDO_EBOOK
    caminho_base = pathlib.Path(__file__).parent.resolve()
    caminho_pdfs = caminho_base / "documentos"
    
    pdf_files = list(caminho_pdfs.glob("*.pdf"))
    if not pdf_files:
        print("Aviso: Nenhum PDF encontrado.")
        return False

    texto_extraido = ""
    for pdf_path in pdf_files:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    texto_extraido += f"\n--- PÁGINA {i+1} ---\n"
                    texto_extraido += page.extract_text() or ""
            print(f"Sucesso ao ler: {pdf_path.name}")
        except Exception as e:
            print(f"Erro ao processar PDF: {e}")
    
    CONTEUDO_EBOOK = texto_extraido
    return len(CONTEUDO_EBOOK) > 0

# Executa o carregamento
EBOOK_PRONTO = carregar_conhecimento_harpia()

# 4. Modelos de Dados
class ChatMessage(BaseModel):
    role: str  # "user" ou "model"
    content: str

class UserQuery(BaseModel):
    message: str
    history: List[ChatMessage] = []

# 5. Rota Principal de Inteligência
@app.post("/chat")
async def chat(query: UserQuery):
    if not EBOOK_PRONTO:
        return {"resposta": "Sistema em manutenção: base de conhecimento offline."}

    try:
        # Formata o histórico para o prompt
        memoria_texto = ""
        if query.history:
            memoria_texto = "\n".join([
                f"{'Usuário' if m.role == 'user' else 'Estrategista'}: {m.content}" 
                for m in query.history
            ])

        # Prompt Refinado com as Fases de Triagem e Consultoria
        prompt_sistema = f"""
        Você é o ESTRATEGISTA HARPIA.

        ### REGRA DE VELOCIDADE E MEMÓRIA:
        - Na FASE DE TRIAGEM, seja ultra-direto (máximo 25 palavras).
        - Analise o Histórico: {memoria_texto if memoria_texto else "Início de conversa."}
        - Se o usuário já identificou o gargalo (Vendas, Mkt ou Atendimento), você está PROIBIDO de perguntar novamente. Avance para a Consultoria.

        ### FASE 1: TRIAGEM (Início):
        - Se o histórico for vazio ou apenas saudações, responda apenas: "Olá! Sou o Estrategista Harpia. Para eu te dar o caminho exato, qual o maior gargalo operacional da sua empresa hoje: Atendimento, Vendas ou Marketing?"
        - Não cite módulos ou e-book nesta fase.

        ### FASE 2: CONSULTORIA (Após identificação):
        - Indique o Módulo do E-book (ex: "Isso está no Módulo 2").
        - Explique O QUE a automação faz e o BENEFÍCIO generalista.
        - **PROIBIÇÃO:** Não entregue o passo a passo técnico. Use: "O tutorial técnico de configuração deste fluxo é exclusivo do nosso E-book".
        - Se a dúvida for muito específica/fora do e-book, sugira a Mentoria (Upsell).

        ### CONTEXTO DO E-BOOK:
        {CONTEUDO_EBOOK}
        
        ### PERGUNTA ATUAL: 
        {query.message}

        ### RESPOSTA DO ESTRATEGISTA:
        """

        # Configurações de resposta rápida
        response = client.models.generate_content(
            model=MODELO_ALVO,
            contents=prompt_sistema,
            config=types.GenerateContentConfig(
                temperature=TEMPERATURE,
                top_p=0.8,
                top_k=40,
                max_output_tokens=2048
            )
        )
        
        return {"resposta": response.text}

    except Exception as e:
        print(f"Erro na geração: {e}")
        return {"resposta": "Erro técnico no motor de IA.", "detalhe": str(e)}

@app.get("/")
async def status():
    return {"status": "Online", "conhecimento": "Pronto" if EBOOK_PRONTO else "Vazio"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
