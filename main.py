import os
import pathlib
import pdfplumber
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI(title="CÓDIGO HARPIA - IA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa o Cliente Gemini novo (conforme o código que você trouxe)
client = genai.Client(api_key=GOOGLE_API_KEY)
MODELO = "gemini-3-pro-preview"

# Variável global para armazenar o conteúdo do e-book
CONTEUDO_EBOOK = ""

def carregar_ebook():
    global CONTEUDO_EBOOK
    print("--- INICIANDO LEITURA TÉCNICA DO PDF ---")
    caminho_pdfs = pathlib.Path(__file__).parent.resolve() / "documentos"
    
    pdf_files = list(caminho_pdfs.glob("*.pdf"))
    if not pdf_files:
        print("ALERTA: PDF não encontrado na pasta documentos.")
        return False

    texto_acumulado = ""
    for pdf_path in pdf_files:
        try:
            print(f"Extraindo texto de: {pdf_path.name}")
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    texto_acumulado += page.extract_text() + "\n"
            print(f"Sucesso: {pdf_path.name} lido completamente.")
        except Exception as e:
            print(f"Erro ao ler PDF {pdf_path.name}: {e}")
    
    CONTEUDO_EBOOK = texto_acumulado
    return len(CONTEUDO_EBOOK) > 0

# Carrega o conhecimento ao iniciar o script
EBOOK_CARREGADO = carregar_ebook()

class UserQuery(BaseModel):
    message: str

@app.post("/chat")
async def chat(query: UserQuery):
    try:
        # Montamos o prompt com o contexto direto
        prompt_sistema = f"""Você é o consultor oficial do CÓDIGO HARPIA. 
        Use o CONTEÚDO DO E-BOOK abaixo para responder às dúvidas do empresário.
        Se a resposta não estiver no texto, use seu conhecimento geral para complementar, 
        mas priorize sempre a metodologia do Código Harpia.

        CONTEÚDO DO E-BOOK:
        {CONTEUDO_EBOOK if EBOOK_CARREGADO else "O e-book não foi carregado corretamente."}
        
        PERGUNTA DO EMPRESÁRIO: {query.message}"""

        response = client.models.generate_content(
            model=MODELO,
            contents=prompt_sistema
        )
        
        return {"resposta": response.text}
    except Exception as e:
        print(f"Erro na IA: {e}")
        return {"resposta": "Desculpe, tive um problema técnico. Pode repetir?", "erro": str(e)}

@app.get("/")
async def root():
    return {
        "status": "Online",
        "conhecimento": "Pronto" if EBOOK_CARREGADO else "Vazio",
        "paginas_lidas": "18" if EBOOK_CARREGADO else "0"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
