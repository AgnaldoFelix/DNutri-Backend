# main.py
from fastapi import FastAPI, HTTPException
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Any
import os
import textwrap
import logging
import json
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import re

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
    intent: Optional[Literal["chat", "calculate_nutrition"]] = "chat"

# Nutrition tips
NUTRITION_TIPS = {
    "proteína": "Priorize fontes magras: frango, peixe, ovos, whey.",
    "calorias": "Para ganhar massa: +300–500 kcal/dia. Para perder: –300–500 kcal/dia.",
    "hidratação": "Beba cerca de 30–35 mL de água por kg de peso por dia.",
}

SYSTEM_PROMPT_CHAT = textwrap.dedent("""
Você é o **Dr.Nutri**, um assistente de nutrição esportiva rápido e direto.  
Sua principal função é **estimar a quantidade média de proteína e calorias** dos alimentos informados pelo usuário.  

💡 Regras:
- Responda **de forma curta e objetiva** (máximo 2 frases).
- Sempre informe **proteína e calorias aproximadas**.
- Pode dar **um pequeno conselho prático** para enriquecer a refeição em proteína (ex: adicionar ovo, frango, iogurte, whey, etc.).
- Evite explicações longas, listas extensas ou textos motivacionais.
- Se não conhecer o alimento, diga: "⚠️ Não encontrei dados suficientes."
- Formato sugerido:
  "🍳 2 ovos + 1 pão integral ≈ 18g proteína | ~200 kcal. Dica: adicione iogurte pra reforçar a proteína."

Seu foco é **responder rápido, com precisão média e utilidade prática.**
""").strip()

# PROMPT específico para cálculo nutricional (JSON apenas) - ATUALIZADO
SYSTEM_PROMPT_NUTRITION = textwrap.dedent("""
Você é um assistente nutricional especializado em calcular valores nutricionais.
Sua ÚNICA tarefa é retornar dados nutricionais em formato JSON.

INSTRUÇÕES ESTRITAS:
1. Analise o alimento e quantidade fornecidos
2. Calcule proteína e calorias totais para a quantidade especificada
3. Retorne APENAS um objeto JSON válido, sem nenhum texto adicional
4. Formato obrigatório: {"protein": número, "calories": número}

EXEMPLOS EM GRAMAS:
- Usuário: "Frango grelhado 150g" → {"protein": 31.5, "calories": 165}
- Usuário: "Arroz branco 200g" → {"protein": 4.8, "calories": 260}
- Usuário: "Ovo cozido 100g" → {"protein": 13, "calories": 155}

EXEMPLOS EM UNIDADES:
- Usuário: "1 ovo" → {"protein": 6, "calories": 70}
- Usuário: "2 ovos" → {"protein": 12, "calories": 140}
- Usuário: "5 ovos" → {"protein": 30, "calories": 350}
- Usuário: "1 maçã" → {"protein": 0.3, "calories": 52}
- Usuário: "2 maçãs" → {"protein": 0.6, "calories": 104}
- Usuário: "1 pão francês" → {"protein": 4, "calories": 150}
- Usuário: "3 pães franceses" → {"protein": 12, "calories": 450}
- Usuário: "1 banana" → {"protein": 1.3, "calories": 89}
- Usuário: "2 bananas" → {"protein": 2.6, "calories": 178}

EXEMPLOS EM ML (LÍQUIDOS):
- Usuário: "Leite 200ml" → {"protein": 6.4, "calories": 124}
- Usuário: "Iogurte natural 150ml" → {"protein": 5, "calories": 90}

NUNCA adicione texto explicativo, emojis ou formatação.
NUNCA use markdown ou blocos de código.
SEMPRE retorne apenas o JSON puro.

Se não encontrar dados para o alimento, retorne: {"protein": 0, "calories": 0}
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
    
    # Instruções específicas baseadas no tipo de prompt
    if system_prompt == SYSTEM_PROMPT_NUTRITION:
        lines.append("INSTRUCTIONS: Retorne APENAS JSON. Nada mais.")
    else:
        lines.append("INSTRUCTIONS: Responda de forma clara, prática e com base em evidências. Use emojis quando apropriado.")
    
    return "\n".join(lines)

def extract_json_from_response(text: str) -> Optional[Dict[str, float]]:
    """Extrai JSON da resposta da IA de forma robusta"""
    try:
        # Remove possíveis blocos de código tipo ```json ... ```
        clean_text = text.replace('```json', '').replace('```', '').strip()

        # Procura um objeto JSON no texto
        json_match = re.search(r'\{[\s\S]*\}', clean_text)
        if json_match:
            return json.loads(json_match.group(0))

        # Se não achou, tenta fazer parse direto
        return json.loads(clean_text)
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        logger.exception("Falha ao extrair JSON: %s", e)
        return None

@app.post("/avaliar")
async def avaliar_nutricional(user_msg: UserMessage) -> Dict[str, str]:
    texto = (user_msg.message or "").strip()
    if not texto:
        raise HTTPException(status_code=400, detail="Campo 'message' obrigatório.")

    # Verifica a intenção da requisição
    intent = user_msg.intent or "chat"
    
    # Se for cálculo nutricional, usa prompt específico
    if intent == "calculate_nutrition":
        logger.info(f"Cálculo nutricional solicitado: {texto}")
        
        if not GENAI_CONFIGURED or model is None:
            raise HTTPException(status_code=500, detail="Serviço de IA não disponível.")

        try:
            # Usa prompt específico para cálculo nutricional
            prompt_text = build_prompt(SYSTEM_PROMPT_NUTRITION, None, texto)
            response = model.generate_content(prompt_text)
            
            # Extrai o texto da resposta
            reply_text = response.text.strip() if hasattr(response, 'text') else str(response)
            
            # Tenta extrair JSON
            nutrition_data = extract_json_from_response(reply_text)
            
            if nutrition_data and isinstance(nutrition_data, dict):
                protein = nutrition_data.get("protein", 0)
                calories = nutrition_data.get("calories", 0)
                
                # Valida os tipos
                if isinstance(protein, (int, float)) and isinstance(calories, (int, float)):
                    logger.info(f"Dados calculados: {protein}g proteína, {calories} kcal")
                    return {
                        "reply": json.dumps({
                            "protein": round(float(protein), 1),
                            "calories": int(calories)
                        })
                    }
            
            # Se não conseguiu extrair JSON válido
            logger.warning(f"Resposta da IA não contém JSON válido: {reply_text}")
            return {
                "reply": json.dumps({
                    "protein": 0,
                    "calories": 0
                })
            }

        except Exception as e:
            logger.exception("Erro ao processar cálculo nutricional")
            return {
                "reply": json.dumps({
                    "protein": 0,
                    "calories": 0
                })
            }

    # Comportamento normal do chat
    else:
        # quick keyword answers (apenas para chat normal)
        lower = texto.lower()
        for key, tip in NUTRITION_TIPS.items():
            if key in lower:
                logger.info(f"Palavra-chave nutricional detectada: {key}")
                return {"reply": f"Dr.Nutri: {tip}"}

        if not GENAI_CONFIGURED or model is None:
            logger.error("GenAI não configurado — chamada recusada.")
            raise HTTPException(status_code=500, detail="GenAI não está configurado no servidor (GOOGLE_API_KEY ausente ou inválida).")

        try:
            prompt_text = build_prompt(SYSTEM_PROMPT_CHAT, user_msg.chat_history, texto)
            response = model.generate_content(prompt_text)
            
            # Extract text
            reply_text = response.text.strip() if hasattr(response, 'text') else str(response)
            clean_reply = reply_text.strip()
            return {"reply": f"Dr.Nutri: {clean_reply}"}

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Erro ao processar mensagem nutricional")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def health_check():
    return {"status": "healthy", "service": "Dr.Nutri", "version": "1.0.0"}