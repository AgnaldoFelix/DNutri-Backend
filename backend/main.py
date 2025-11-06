# main.py
from fastapi import FastAPI, HTTPException
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Any
import os
import textwrap
import logging
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Carrega .env
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dr.Nutri - Especialista em Nutrição Esportiva 🤖")

# CORS (ajuste allow_origins em produção)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
model: Optional[Any] = None
GENAI_CONFIGURED: bool = False
MODEL_NAME = os.getenv("GENAI_MODEL", "gemini-2.5-flash")

# Data models
class ChatMessage(BaseModel):
    sender: Literal["user", "model"] = Field(..., alias="from")
    text: str

class UserMessage(BaseModel):
    message: str
    chat_history: Optional[List[ChatMessage]] = None

# Nutrition tips
NUTRITION_TIPS = {
    "proteína": "Se quer aumentar ingestão de proteína: priorize fontes magras (frango, peixe, ovos, whey), distribua proteína nas refeições e use 20–40 g após o treino para recuperação.",
    "calorias": "Para ganhar massa aumente ~250-500 kcal/dia acima do seu gasto; para perder gordura, reduza ~300-500 kcal/dia com atenção à proteína para preservar massa magra.",
    "gordura": "Gorduras saudáveis (azeite, abacate, oleaginosas) são importantes, mas atenção à densidade calórica. Evite exagero se seu objetivo é perda de gordura.",
    "lactose": "Se houver suspeita de intolerância à lactose, prefira versões sem lactose ou alternativas vegetais; observe sintomas como inchaço e desconforto após ingestão.",
    "sódio": "Reduzir sódio é importante para quem tem hipertensão; evite alimentos ultraprocessados e tempere com ervas e limão.",
    "hidratação": "Manter-se hidratado é essencial — regra prática: beber ao menos 30–35 mL/kg/dia (varia por atividade física e clima).",
    "jejum": "Jejum intermitente pode funcionar para alguns, mas não é obrigatório para resultados — avalie tolerância, performance nos treinos e ingestão proteica dentro da janela.",
    "carboidrato": "Carboidratos são importantes para treinos intensos; priorize carboidratos complexos antes do treino e reposição após treinos longos.",
    "vitamina d": "Vitamina D importante para saúde óssea; exposição solar moderada e, se indicado por exame, suplementação orientada por profissional.",
}

SYSTEM_PROMPT = textwrap.dedent("""
Você é o Dr.Nutri, um assistente virtual especialista em NUTRIÇÃO ESPORTIVA e planejamento alimentar.
Suas principais funções:
1) Calcular estimativas de macros e calorias com base em dados fornecidos.
2) Sugerir ajustes de refeições para objetivos (hipertrofia, perda de gordura, manutenção).
3) Explicar escolhas alimentares (timing de nutrientes, suplementos, hidratação) de forma clara e prática.
4) Oferecer alternativas alimentares para restrições (intolerância, alergias, vegano).
5) Quando necessário, pedir dados faltantes (peso, altura, objetivo, treino) para maior precisão.
Mantenha respostas objetivas, com tom amigável e use emojis quando fizer sentido. Sempre destaque quando for apenas uma sugestão e recomende procurar um profissional presencial para diagnósticos/condições médicas.
""").strip()

# Startup: configurar cliente GenAI
@app.on_event("startup")
def startup_event():
    global model, GENAI_CONFIGURED, MODEL_NAME
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY ausente — GenAI NÃO configurado (defina GOOGLE_API_KEY).")
        GENAI_CONFIGURED = False
        return
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)
        GENAI_CONFIGURED = True
        logger.info(f"GenAI configurado com sucesso. Modelo: {MODEL_NAME}")
    except Exception as e:
        GENAI_CONFIGURED = False
        logger.exception("Falha ao configurar GenAI: %s", e)

def build_prompt(system_prompt: str, history: Optional[List[ChatMessage]], user_message: str) -> str:
    lines = []
    lines.append("SYSTEM INSTRUCTIONS:")
    lines.append(system_prompt)
    lines.append("")
    if history:
        lines.append("CONVERSATION HISTORY:")
        for msg in history:
            speaker = "USER" if msg.sender == "user" else "ASSISTANT"
            text = msg.text.replace("\n", " ").strip()
            lines.append(f"{speaker}: {text}")
        lines.append("")
    else:
        lines.append("CONVERSATION HISTORY: (vazio)")
        lines.append("")
    lines.append("CURRENT USER QUESTION:")
    lines.append(user_message.strip())
    lines.append("")
    lines.append("INSTRUCTIONS: Responda de forma clara, prática e com base em evidências. Use emojis quando apropriado.")
    return "\n".join(lines)

@app.post("/avaliar")
async def avaliar_nutricional(user_msg: UserMessage) -> Dict[str, str]:
    texto = (user_msg.message or "").strip()
    if not texto:
        raise HTTPException(status_code=400, detail="Campo 'message' obrigatório.")

    # quick keyword answers
    lower = texto.lower()
    for key, tip in NUTRITION_TIPS.items():
        if key in lower:
            logger.info(f"Palavra-chave nutricional detectada: {key}")
            return {"reply": f"Dr.Nutri: {tip}"}

    if not GENAI_CONFIGURED or model is None:
        logger.error("GenAI não configurado — chamada recusada.")
        raise HTTPException(status_code=500, detail="GenAI não está configurado no servidor (GOOGLE_API_KEY ausente ou inválida).")

    try:
        prompt_text = build_prompt(SYSTEM_PROMPT, user_msg.chat_history, texto)
        contents = [{"parts": [prompt_text]}]

        # CHAMADA: sem parâmetros não suportados (removemos 'temperature' porque sua versão do SDK não aceita)
        # Abaixo, tentamos formas compatíveis com diferentes versões do SDK:
        response = None
        try:
            # versão recomendada: passar contents como argumento nomeado
            response = model.generate_content(contents=contents)
        except TypeError:
            # fallback para algumas versões antigas que aceitam só a lista
            response = model.generate_content(contents)
        except Exception:
            # re-raise para ser tratado
            raise

        # Extract text
        reply_text: Optional[str] = None
        if hasattr(response, "text") and isinstance(getattr(response, "text"), str):
            reply_text = response.text
        elif hasattr(response, "response") and hasattr(response.response, "text"):
            reply_text = response.response.text
        elif isinstance(response, dict):
            if isinstance(response.get("reply"), str):
                reply_text = response["reply"]
            elif "output" in response and isinstance(response["output"], list) and response["output"]:
                first = response["output"][0]
                if isinstance(first, dict):
                    content = first.get("content")
                    if isinstance(content, dict) and isinstance(content.get("parts"), list):
                        for p in content["parts"]:
                            if isinstance(p, str) and p.strip():
                                reply_text = p
                                break
                    if not reply_text:
                        for v in first.values():
                            if isinstance(v, str) and v.strip():
                                reply_text = v
                                break
            if not reply_text:
                reply_text = str(response)
        else:
            reply_text = str(response)

        clean_reply = reply_text.strip() if isinstance(reply_text, str) else str(reply_text)
        return {"reply": f"Dr.Nutri: {clean_reply}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erro ao processar mensagem nutricional")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def health_check():
    return {"status": "healthy", "service": "Dr.Nutri", "version": "1.0.0"}
