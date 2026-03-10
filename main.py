import os
import pathlib
import pdfplumber
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

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

# 3. Inicialização do Cliente Gemini (v1 estável)
client = genai.Client(api_key=GOOGLE_API_KEY)
MODELO_ALVO = "gemini-3-pro-preview"
TEMPERATURE = 0.2

# Variável global para o cérebro do consultor
CONTEUDO_EBOOK = ""

def carregar_conhecimento_harpia():
    """Lógica de extração técnica do PDF usando pdfplumber"""
    global CONTEUDO_EBOOK
    print("\n--- [OPERAÇÃO HARPIA] CARREGANDO CÉREBRO ESTRATÉGICO ---")
    
    # Localiza a pasta documentos de forma segura no Render
    caminho_base = pathlib.Path(__file__).parent.resolve()
    caminho_pdfs = caminho_base / "documentos"
    
    pdf_files = list(caminho_pdfs.glob("*.pdf"))
    if not pdf_files:
        print("!!! ALERTA: Nenhum PDF localizado na pasta documentos !!!")
        return False

    texto_extraido = ""
    for pdf_path in pdf_files:
        try:
            print(f"-> Analisando conteúdo de: {pdf_path.name}")
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    texto_extraido += f"\n--- PÁGINA {i+1} ---\n"
                    texto_extraido += page.extract_text() or ""
            print(f"-> Sucesso: {pdf_path.name} processado com precisão.")
        except Exception as e:
            print(f"!!! Erro ao processar {pdf_path.name}: {e}")
    
    CONTEUDO_EBOOK = texto_extraido
    return len(CONTEUDO_EBOOK) > 0

# Carregamento imediato no boot do servidor
EBOOK_PRONTO = carregar_conhecimento_harpia()

# 4. Modelos de Dados
class UserQuery(BaseModel):
    message: str

# 5. Rota Principal de Inteligência
@app.post("/chat")
async def chat(query: UserQuery):
    if not EBOOK_PRONTO:
        return {"resposta": "Sistema em manutenção: base de conhecimento não carregada."}

    try:
        # PROMPT COM TRIAGEM E SEGURANÇA DE INFORMAÇÃO
        prompt_sistema = f"""
        Você é o ESTRATEGISTA HARPIA, o consultor de elite do método CÓDIGO HARPIA.
        
        Sua missão atual é fazer uma TRIAGEM técnica. Você NÃO deve entregar conteúdos profundos, 
        passo-a-passos detalhados ou trechos do e-book na primeira interação.

        ### REGRA DE OURO DA TRIAGEM:
        - Se o usuário for genérico (ex: "oi", "como funciona", "me ajude"), você deve APENAS identificar a dor dele.
        - Faça perguntas curtas e diretas: "Qual é o seu processo operacional específico (vendas, marketing ou atendimento) que gera mais gargalo?" ou "Qual dúvida específica você tem sobre automação?".
        - Não revele detalhes dos Módulos ou da Metodologia até que o usuário responda sobre o cenário dele.

        ### DIRETRIZES DE RESPOSTA:
        1. **Fase de Diagnóstico:** Se a dúvida não estiver clara, responda com uma saudação assertiva e peça o contexto operacional.
        2. **Fase de Entrega:** Somente após o usuário identificar o problema (ex: "tenho problema no meu atendimento"), você usa o CONTEÚDO DO E-BOOK para dar uma pílula de solução e sugerir o próximo passo.
        3. **Tom de Voz:** Profissional, assertivo e focado em identificar o gargalo.

        ### CONTEXTO DO E-BOOK (PARA USO APENAS APÓS TRIAGEM):
        {CONTEUDO_EBOOK}
        
        ### MENSAGEM DO USUÁRIO: 
        {query.message}

        ### RESPOSTA DO ESTRATEGISTA:
        """
        
        response = client.models.generate_content(
            model=MODELO_ALVO,
            contents=prompt_sistema
        )
        
        return {"resposta": response.text}

    except Exception as e:
        print(f"Erro na geração de resposta: {e}")
        return {"resposta": "Erro técnico no motor de IA.", "detalhe": str(e)}

# 6. Rota de Status (Health Check)
@app.get("/")
async def status_sistema():
    return {
        "projeto": "CÓDIGO HARPIA",
        "consultor": "Estrategista Ativo",
        "conhecimento": "Pronto" if EBOOK_PRONTO else "Vazio",
        "paginas_processadas": "18",
        "status": "Online"
    }

# 7. Inicialização do Servidor
if __name__ == "__main__":
    import uvicorn
    # O Render define a porta via variável de ambiente PORT
    port = int(os.environ.get("PORT", 10000))
    print(f"--- Servidor Harpia Live na Porta {port} ---")
    uvicorn.run(app, host="0.0.0.0", port=port)



