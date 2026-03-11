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

        prompt_sistema = f"""
        Você é o ESTRATEGISTA HARPIA, o consultor de elite do método CÓDIGO HARPIA.

        ### REGRAS DE MEMÓRIA (PRIORIDADE MÁXIMA):
        1. **Analise o Histórico com Rigor:** {memoria_texto if memoria_texto else "Início de conversa."}
        
        2. **PROIBIÇÃO DE REPETIÇÃO:** Se o histórico acima já mostra que o usuário identificou o gargalo (Vendas, Marketing ou Atendimento), você está PROIBIDO de perguntar isso novamente. 
        3. **NÃO REINICIE O FLUXO:** Se o usuário respondeu uma pergunta sua, avance para a próxima solução do e-book. Não use saudações de "primeiro contato" se a conversa já começou.

        ### LÓGICA DE ATUAÇÃO:
        - Se o usuário respondeu "WhatsApp" ou "Dúvidas", e já sabemos que é "Atendimento", procure no E-BOOK a solução específica para 'Automação de WhatsApp' ou 'Chatbots'.
        ###- Se a resposta não estiver no e-book, direcione para o UPSELL (Mentoria/Consultoria).

        ### CONTEXTO DO E-BOOK:
        {CONTEUDO_EBOOK}
        
        ### PERGUNTA ATUAL: 
        {query.message}

        ### RESPOSTA DO ESTRATEGISTA (DIRETA E SEM REPETIÇÃO):
        """

        # Linha 133 corrigida aqui:
        response = client.models.generate_content(
            model=MODELO_ALVO,
            contents=prompt_sistema,
            config=types.GenerateContentConfig(
                temperature=TEMPERATURE,
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


