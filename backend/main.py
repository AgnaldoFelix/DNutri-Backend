# main.py - ATUALIZADO com reconhecimento de voz gratuito
import socket
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
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
import asyncio
from datetime import datetime, timedelta
import uuid
import base64
import speech_recognition as sr
from io import BytesIO
import wave

# Carrega .env
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_local_ip():
    """Obtém o IP local da máquina de forma confiável"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            return ip
    except:
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            if local_ip.startswith("127."):
                return "localhost"
            return local_ip
        except:
            return "localhost"

def get_network_ips():
    """Obtém IPs da rede local de forma simples"""
    ips = []
    try:
        local_ip = get_local_ip()
        if local_ip and local_ip != "localhost":
            ips.append(local_ip)
        
        common_prefixes = ['192.168.', '10.0.', '172.16.', '172.17.', '172.18.', '172.19.', 
                          '172.20.', '172.21.', '172.22.', '172.23.', '172.24.', '172.25.',
                          '172.26.', '172.27.', '172.28.', '172.29.', '172.30.', '172.31.']
        
        for prefix in common_prefixes:
            test_ips = [
                f"{prefix}1.100", f"{prefix}0.2", f"{prefix}1.1", 
                f"{prefix}0.1", f"{prefix}1.2", f"{prefix}0.100"
            ]
            for test_ip in test_ips:
                if test_ip not in ips:
                    ips.append(test_ip)
                    
    except Exception as e:
        logger.debug(f"Erro ao obter IPs de rede: {e}")
    
    return ips[:6]

app = FastAPI(
    title="Dr.Nutri - Sistema Completo de Nutrição e Comunidade 🤖",
    description="API para chat nutricional + comunidade em tempo real + reconhecimento de voz", 
    version="2.5.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# SISTEMA DE RECONHECIMENTO DE VOZ (Gratuito)
# =============================================================================

class VoiceRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.supported_languages = {
            'pt-BR': 'portuguese',
            'en-US': 'english',
            'es-ES': 'spanish'
        }
    
    def transcribe_audio(self, audio_data: bytes, language: str = 'pt-BR') -> str:
        """
        Transcreve áudio usando speech_recognition com Google Web Speech API (gratuito)
        ou fallback para Whisper (OpenAI) se disponível
        """
        try:
            # Converte bytes para AudioData do speech_recognition
            audio_file = BytesIO(audio_data)
            
            # Tenta detectar se é WAV ou outro formato
            try:
                # Se for WAV, carrega diretamente
                with sr.AudioFile(audio_file) as source:
                    audio = self.recognizer.record(source)
            except:
                # Se não for WAV, tenta converter
                audio = sr.AudioData(audio_data, sample_rate=16000, sample_width=2)
            
            # Tenta com Google Web Speech API (gratuito)
            try:
                logger.info(f"Tentando reconhecimento com Google Web Speech API ({language})")
                text = self.recognizer.recognize_google(
                    audio, 
                    language=language,
                    show_all=False
                )
                logger.info(f"Transcrição bem-sucedida: {text[:100]}...")
                return text
                
            except sr.UnknownValueError:
                logger.warning("Google Web Speech API não conseguiu entender o áudio")
                raise HTTPException(status_code=400, detail="Não foi possível entender o áudio")
                
            except sr.RequestError as e:
                logger.warning(f"Erro no Google Web Speech API: {e}")
                # Fallback para reconhecimento offline se disponível
                try:
                    logger.info("Tentando reconhecimento offline com Sphinx")
                    text = self.recognizer.recognize_sphinx(audio, language='pt-BR')
                    return text
                except:
                    raise HTTPException(
                        status_code=503, 
                        detail="Serviço de reconhecimento de voz temporariamente indisponível"
                    )
                    
        except Exception as e:
            logger.error(f"Erro no processamento de áudio: {e}")
            raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")

# Instância do reconhecedor de voz
voice_recognizer = VoiceRecognizer()

# =============================================================================
# MODELOS DE DADOS PARA VOZ
# =============================================================================

class VoiceTranscriptionRequest(BaseModel):
    audio_base64: Optional[str] = None
    language: str = "pt-BR"

class VoiceFoodDescription(BaseModel):
    description: str
    language: str = "pt-BR"

class ParsedFoodData(BaseModel):
    name: str
    amount: str
    unit: str
    protein: Optional[float] = 0
    calories: Optional[float] = 0

# =============================================================================
# SISTEMA DE CHAT EM TEMPO REAL (ConnectionManager)
# =============================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.online_users = []
        self.chat_messages = []
        self.user_cache = {}
        self.message_cache = []
        self.max_cached_messages = 1000
        
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"Usuário {user_id} conectado. Total: {len(self.active_connections)}")
        
    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            self.online_users = [u for u in self.online_users if u.get('id') != user_id]
            logger.info(f"Usuário {user_id} desconectado. Total: {len(self.active_connections)}")
            
    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_text(message)
            except Exception as e:
                logger.error(f"Erro ao enviar mensagem para {user_id}: {e}")
                self.disconnect(user_id)
                
    async def broadcast(self, message: str):
        disconnected = []
        for user_id, connection in self.active_connections.items():
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Erro ao broadcast para {user_id}: {e}")
                disconnected.append(user_id)
                
        for user_id in disconnected:
            self.disconnect(user_id)
            
    def add_online_user(self, user_data: Dict):
        self.online_users = [u for u in self.online_users if u.get('id') != user_data.get('id')]
        user_data['last_seen'] = datetime.now().isoformat()
        self.online_users.append(user_data)
        self.clean_old_users()
        
    def clean_old_users(self):
        cutoff_time = datetime.now() - timedelta(minutes=5)
        self.online_users = [
            u for u in self.online_users 
            if datetime.fromisoformat(u.get('last_seen', '2000-01-01')) > cutoff_time
        ]
        
    def remove_online_user(self, user_id: str):
        self.online_users = [u for u in self.online_users if u.get('id') != user_id]
        
    def add_chat_message(self, message_data: Dict):
        message_data['id'] = str(uuid.uuid4())
        message_data['timestamp'] = datetime.now().isoformat()
        
        self.chat_messages.append(message_data)
        self.message_cache.append(message_data)
        
        if len(self.message_cache) > self.max_cached_messages:
            self.message_cache = self.message_cache[-self.max_cached_messages:]
            
        if len(self.chat_messages) > 2000:
            self.chat_messages = self.chat_messages[-1000:]
            
    def get_recent_messages(self, limit: int = 100):
        return self.chat_messages[-limit:] if self.chat_messages else []
    
    def get_online_users(self):
        self.clean_old_users()
        return self.online_users

# Instância global do gerenciador de conexões
connection_manager = ConnectionManager()

# =============================================================================
# MODELOS DE DADOS
# =============================================================================

class OnlineUser(BaseModel):
    id: str
    name: str
    avatar: str
    isOnline: bool = True
    lastSeen: Optional[str] = None
    profileEnabled: bool = True

class ChatMessage(BaseModel):
    id: Optional[str] = None
    userId: str
    userName: str
    userAvatar: str
    message: str
    timestamp: Optional[str] = None
    type: Literal["text", "system"] = "text"

class UserMessage(BaseModel):
    message: str
    chat_history: Optional[List[ChatMessage]] = None
    intent: Optional[Literal["chat", "calculate_nutrition", "parse_food_description"]] = "chat"

# =============================================================================
# SISTEMA DR. NUTRI
# =============================================================================

model: Optional[Any] = None
GENAI_CONFIGURED: bool = False
MODEL_NAME = os.getenv("GENAI_MODEL", "gemini-2.5-flash")

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

NUNCA adicione texto explicativo, emojis ou formatação.
NUNCA use markdown ou blocos de código.
SEMPRE retorne apenas o JSON puro.

Se não encontrar dados para o alimento, retorne: {"protein": 0, "calories": 0}
""").strip()

SYSTEM_PROMPT_PARSE_FOOD = textwrap.dedent("""
Você é um especialista em processar descrições de alimentos em linguagem natural.
Sua tarefa é extrair informações estruturadas de descrições de alimentos faladas.

INSTRUÇÕES:
1. Analise a descrição do alimento em português
2. Extraia: nome do alimento, quantidade, unidade (g, ml, unit)
3. Se possível, estime proteína e calorias
4. Retorne APENAS JSON válido no formato:
{
  "name": "nome do alimento",
  "amount": "quantidade em número",
  "unit": "g, ml ou unit",
  "protein": número ou 0,
  "calories": número ou 0
}

EXEMPLOS:
- "duzentos gramas de frango grelhado" → {"name": "frango grelhado", "amount": "200", "unit": "g", "protein": 46, "calories": 330}
- "dois ovos cozidos" → {"name": "ovos cozidos", "amount": "2", "unit": "unit", "protein": 12, "calories": 140}
- "uma xícara de arroz" → {"name": "arroz", "amount": "1", "unit": "unit", "protein": 4.5, "calories": 240}
- "trezentos mililitros de leite" → {"name": "leite", "amount": "300", "unit": "ml", "protein": 9.6, "calories": 186}

Para unidades:
- gramas, g, grama → "g"
- mililitros, ml, mililitro → "ml"
- unidades, unidade, un, uni → "unit"

Se não conseguir identificar quantidade ou unidade, use valores padrão:
- amount: "100"
- unit: "g"

NUNCA adicione texto extra, apenas o JSON.
""").strip()

@app.on_event("startup")
def startup_event():
    global model, GENAI_CONFIGURED, MODEL_NAME
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Detectar IP local
    local_ip = get_local_ip()
    network_ips = get_network_ips()
    
    print("🌐 IPs detectados:")
    print(f"   Local: {local_ip}")
    if network_ips:
        print(f"   Rede: {', '.join(network_ips[:3])}")
    
    if not api_key:
        logger.warning("GOOGLE_API_KEY ausente — GenAI NÃO configurado (defina GOOGLE_API_KEY).")
        GENAI_CONFIGURED = False
        return
        
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)
        GENAI_CONFIGURED = True
        logger.info(f"GenAI configurado com sucesso. Modelo: {MODEL_NAME}")
        
        # Usuários de exemplo
        connection_manager.add_online_user({
            "id": "user_ana",
            "name": "Ana Silva", 
            "avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=Ana",
            "isOnline": True,
            "profileEnabled": True
        })
        
        connection_manager.add_online_user({
            "id": "user_carlos",
            "name": "Carlos Santos",
            "avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=Carlos", 
            "isOnline": True,
            "profileEnabled": True
        })
        
        # Mensagem de boas-vindas
        connection_manager.add_chat_message({
            "userId": "system",
            "userName": "Sistema",
            "userAvatar": "/Essentia.png",
            "message": "🎤 Funcionalidade de voz ativada! Agora você pode adicionar alimentos falando.",
            "type": "system"
        })
        
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
    
    if system_prompt == SYSTEM_PROMPT_NUTRITION or system_prompt == SYSTEM_PROMPT_PARSE_FOOD:
        lines.append("INSTRUCTIONS: Retorne APENAS JSON. Nada mais.")
    else:
        lines.append("INSTRUCTIONS: Responda de forma clara, prática e com base em evidências. Use emojis quando apropriado.")
    
    return "\n".join(lines)

def extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    try:
        clean_text = text.replace('```json', '').replace('```', '').strip()
        json_match = re.search(r'\{[\s\S]*\}', clean_text)
        if json_match:
            return json.loads(json_match.group(0))
        return json.loads(clean_text)
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        logger.exception("Falha ao extrair JSON: %s", e)
        return None

# =============================================================================
# ENDPOINTS DE VOZ
# =============================================================================

@app.post("/transcrever")
async def transcrever_audio(file: UploadFile = File(...), language: str = "pt-BR"):
    """
    Transcreve áudio para texto usando reconhecimento de voz gratuito
    """
    try:
        # Lê o arquivo de áudio
        audio_data = await file.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Arquivo de áudio vazio")
        
        logger.info(f"Transcrevendo áudio ({len(audio_data)} bytes) em {language}")
        
        # Transcreve usando o reconhecedor de voz
        texto = voice_recognizer.transcribe_audio(audio_data, language)
        
        return {
            "transcript": texto,
            "language": language,
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na transcrição: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.post("/transcrever-base64")
async def transcrever_audio_base64(request: VoiceTranscriptionRequest):
    """
    Transcreve áudio em base64 para texto
    """
    try:
        if not request.audio_base64:
            raise HTTPException(status_code=400, detail="audio_base64 é obrigatório")
        
        # Decodifica base64
        audio_data = base64.b64decode(request.audio_base64)
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Áudio vazio")
        
        logger.info(f"Transcrevendo áudio base64 ({len(audio_data)} bytes) em {request.language}")
        
        # Transcreve
        texto = voice_recognizer.transcribe_audio(audio_data, request.language)
        
        return {
            "transcript": texto,
            "language": request.language,
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na transcrição base64: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.post("/parse-food-description")
async def parse_food_description(desc: VoiceFoodDescription):
    """
    Processa descrição de alimento em linguagem natural e retorna dados estruturados
    """
    if not GENAI_CONFIGURED or model is None:
        raise HTTPException(status_code=500, detail="Serviço de IA não disponível.")
    
    try:
        prompt_text = build_prompt(SYSTEM_PROMPT_PARSE_FOOD, None, desc.description)
        response = model.generate_content(prompt_text)
        
        reply_text = response.text.strip() if hasattr(response, 'text') else str(response)
        parsed_data = extract_json_from_response(reply_text)
        
        if not parsed_data:
            # Fallback: regex simples para extração básica
            parsed_data = simple_food_parser(desc.description)
        
        # Garante que os campos obrigatórios existam
        if "name" not in parsed_data:
            parsed_data["name"] = desc.description
        
        if "amount" not in parsed_data:
            parsed_data["amount"] = "100"
        
        if "unit" not in parsed_data:
            parsed_data["unit"] = "g"
        
        if "protein" not in parsed_data:
            parsed_data["protein"] = 0
            
        if "calories" not in parsed_data:
            parsed_data["calories"] = 0
        
        logger.info(f"Dados processados da descrição: {parsed_data}")
        return parsed_data
        
    except Exception as e:
        logger.error(f"Erro ao processar descrição de alimento: {e}")
        # Retorna fallback
        return {
            "name": desc.description,
            "amount": "100",
            "unit": "g",
            "protein": 0,
            "calories": 0
        }

def simple_food_parser(description: str) -> Dict[str, Any]:
    """
    Parser simples de regex para fallback quando a IA falha
    """
    description = description.lower()
    
    # Padrões para quantidades
    number_words = {
        'um': '1', 'uma': '1', 'dois': '2', 'duas': '2', 'três': '3',
        'quatro': '4', 'cinco': '5', 'seis': '6', 'sete': '7', 'oito': '8',
        'nove': '9', 'dez': '10', 'cem': '100', 'duzentos': '200',
        'trezentos': '300', 'quatrocentos': '400', 'quinhentos': '500'
    }
    
    # Extrair número
    amount = "100"
    unit = "g"
    name = description
    
    # Procura por números e unidades
    for word, num in number_words.items():
        if word in description:
            amount = num
            break
    
    # Procura por números digitais
    match = re.search(r'(\d+)\s*(g|gramas|ml|mililitros|unidades|unidade)', description)
    if match:
        amount = match.group(1)
        unit_str = match.group(2)
        if unit_str in ['g', 'gramas']:
            unit = 'g'
        elif unit_str in ['ml', 'mililitros']:
            unit = 'ml'
        elif unit_str in ['unidades', 'unidade']:
            unit = 'unit'
    
    # Remove a parte da quantidade/unit do nome
    name = re.sub(r'\d+\s*(g|gramas|ml|mililitros|unidades|unidade)', '', name).strip()
    for word in number_words:
        name = name.replace(word, '').strip()
    
    return {
        "name": name if name else description,
        "amount": amount,
        "unit": unit,
        "protein": 0,
        "calories": 0
    }

# =============================================================================
# WEBSOCKETS E ENDPOINTS EXISTENTES
# =============================================================================

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await connection_manager.connect(websocket, user_id)
    try:
        initial_data = {
            "type": "sync_data",
            "data": {
                "online_users": connection_manager.get_online_users(),
                "chat_messages": connection_manager.get_recent_messages(50),
                "server_time": datetime.now().isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
        await connection_manager.send_personal_message(json.dumps(initial_data), user_id)
        
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            message_type = message_data.get("type")
            
            if message_type == "user_join":
                user_data = message_data.get("data", {})
                connection_manager.add_online_user(user_data)
                
                broadcast_msg = {
                    "type": "user_join",
                    "data": user_data,
                    "timestamp": datetime.now().isoformat()
                }
                await connection_manager.broadcast(json.dumps(broadcast_msg))
                
                system_msg = {
                    "userId": "system",
                    "userName": "Sistema",
                    "userAvatar": "/Essentia.png",
                    "message": f"🎉 {user_data.get('name', 'Usuário')} entrou na comunidade!",
                    "type": "system"
                }
                connection_manager.add_chat_message(system_msg)
                
                chat_broadcast = {
                    "type": "chat_message", 
                    "data": system_msg,
                    "timestamp": datetime.now().isoformat()
                }
                await connection_manager.broadcast(json.dumps(chat_broadcast))
                
            elif message_type == "user_leave":
                leave_user_id = message_data.get("data", {}).get("userId")
                if leave_user_id:
                    user = next((u for u in connection_manager.online_users if u.get('id') == leave_user_id), None)
                    if user:
                        connection_manager.remove_online_user(leave_user_id)
                        
                        broadcast_msg = {
                            "type": "user_leave",
                            "data": {"userId": leave_user_id, "userName": user.get('name')},
                            "timestamp": datetime.now().isoformat()
                        }
                        await connection_manager.broadcast(json.dumps(broadcast_msg))
                        
                        system_msg = {
                            "userId": "system",
                            "userName": "Sistema", 
                            "userAvatar": "/Essentia.png",
                            "message": f"👋 {user.get('name', 'Usuário')} saiu da comunidade",
                            "type": "system"
                        }
                        connection_manager.add_chat_message(system_msg)
                        
                        chat_broadcast = {
                            "type": "chat_message",
                            "data": system_msg,
                            "timestamp": datetime.now().isoformat()
                        }
                        await connection_manager.broadcast(json.dumps(chat_broadcast))
                        
            elif message_type == "chat_message":
                chat_data = message_data.get("data", {})
                connection_manager.add_chat_message(chat_data)
                
                broadcast_msg = {
                    "type": "chat_message",
                    "data": chat_data,
                    "timestamp": datetime.now().isoformat()
                }
                await connection_manager.broadcast(json.dumps(broadcast_msg))
                
            elif message_type == "user_update":
                user_data = message_data.get("data", {})
                connection_manager.add_online_user(user_data)
                
                broadcast_msg = {
                    "type": "user_update",
                    "data": user_data,
                    "timestamp": datetime.now().isoformat()
                }
                await connection_manager.broadcast(json.dumps(broadcast_msg))
                
    except WebSocketDisconnect:
        connection_manager.disconnect(user_id)
    except Exception as e:
        logger.error(f"Erro no WebSocket para {user_id}: {e}")
        connection_manager.disconnect(user_id)

@app.post("/avaliar")
async def avaliar_nutricional(user_msg: UserMessage) -> Dict[str, str]:
    texto = (user_msg.message or "").strip()
    if not texto:
        raise HTTPException(status_code=400, detail="Campo 'message' obrigatório.")

    intent = user_msg.intent or "chat"
    
    if intent == "calculate_nutrition":
        logger.info(f"Cálculo nutricional solicitado: {texto}")
        
        if not GENAI_CONFIGURED or model is None:
            raise HTTPException(status_code=500, detail="Serviço de IA não disponível.")

        try:
            prompt_text = build_prompt(SYSTEM_PROMPT_NUTRITION, None, texto)
            response = model.generate_content(prompt_text)
            
            reply_text = response.text.strip() if hasattr(response, 'text') else str(response)
            nutrition_data = extract_json_from_response(reply_text)
            
            if nutrition_data and isinstance(nutrition_data, dict):
                protein = nutrition_data.get("protein", 0)
                calories = nutrition_data.get("calories", 0)
                
                if isinstance(protein, (int, float)) and isinstance(calories, (int, float)):
                    logger.info(f"Dados calculados: {protein}g proteína, {calories} kcal")
                    return {
                        "reply": json.dumps({
                            "protein": round(float(protein), 1),
                            "calories": int(calories)
                        })
                    }
            
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
    
    elif intent == "parse_food_description":
        # Reutiliza a função parse_food_description
        try:
            parsed_data = await parse_food_description(VoiceFoodDescription(
                description=texto,
                language="pt-BR"
            ))
            return {"reply": json.dumps(parsed_data)}
        except Exception as e:
            logger.error(f"Erro no parse_food_description via /avaliar: {e}")
            return {
                "reply": json.dumps({
                    "name": texto,
                    "amount": "100",
                    "unit": "g",
                    "protein": 0,
                    "calories": 0
                })
            }

    else:
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
            
            reply_text = response.text.strip() if hasattr(response, 'text') else str(response)
            clean_reply = reply_text.strip()
            return {"reply": f"Dr.Nutri: {clean_reply}"}

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Erro ao processar mensagem nutricional")
            raise HTTPException(status_code=500, detail=str(e))

# Endpoints de status
@app.get("/status")
async def health_check():
    return {
        "status": "healthy", 
        "service": "Dr.Nutri + Comunidade Essentia + Voz", 
        "version": "2.5.0",
        "online_users": len(connection_manager.active_connections),
        "total_messages": len(connection_manager.chat_messages),
        "genai_configured": GENAI_CONFIGURED,
        "voice_available": True,
        "server_time": datetime.now().isoformat(),
        "performance": {
            "cached_messages": len(connection_manager.message_cache),
            "active_connections": len(connection_manager.active_connections)
        }
    }

# =============================================================================
# INICIALIZAÇÃO
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    local_ip = get_local_ip()
    
    print("=" * 60)
    print("🎤 Dr.Nutri com Reconhecimento de Voz Gratuito")
    print("=" * 60)
    print(f"📡 Host: {host}")
    print(f"🔢 Porta: {port}")
    print("🌐 URLs de acesso:")
    print(f"   • Local: http://localhost:{port}")
    print(f"   • IP Local: http://{local_ip}:{port}")
    
    network_ips = get_network_ips()
    if network_ips:
        print("   • Possíveis IPs de rede:")
        for ip in network_ips[:3]:
            print(f"     - http://{ip}:{port}")
    
    print("")
    print("🎤 Endpoints de voz:")
    print(f"   • POST /transcrever - Upload de arquivo de áudio")
    print(f"   • POST /transcrever-base64 - Áudio em base64")
    print(f"   • POST /parse-food-description - Processar descrição de alimento")
    print("")
    print("📋 Endpoints principais:")
    print(f"   • API Docs: http://{local_ip}:{port}/docs")
    print(f"   • Status: http://{local_ip}:{port}/status")
    print("=" * 60)
    
    uvicorn.run(
        "main:app", 
        host=host, 
        port=port, 
        reload=True,
        log_level="info"
    )