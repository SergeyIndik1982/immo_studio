import os
import httpx
import stripe
import asyncio
from fastapi import FastAPI, Request, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Boolean, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# --- DATABASE SETUP ---
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./immo_studio.db" # Переименовали базу данных

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    email = Column(String, primary_key=True, index=True)
    is_premium = Column(Boolean, default=False)
    credits = Column(Integer, default=7) # Семь тестовых генераций для маклеров

Base.metadata.create_all(bind=engine)

# --- APP SETUP ---
app = FastAPI(title="AIES Real Estate Marketing Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
stripe.api_key = STRIPE_SECRET_KEY
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# --- ADAPTED PYDANTIC MODELS ---
class GenerateRequest(BaseModel):
    email: str
    object_type: str        # Wohnung, Haus, Penthouse, Gewerbe
    location: str           # Stadt / Region (e.g., Berlin, Wolfsburg)
    rooms: float            # Количество комнат (e.g., 3.5)
    size_sqm: int           # Метраж в кв.м.
    key_features: str       # Фишки: "Einbauküche, Fußbodenheizung, Balkon, Sanierter Altbau"
    target_audience: str    # Kapitalanleger (Инвесторы), Familien (Семьи), Singles / Studenten
    marketing_tone: str     # Professional, Luxury, Emotional, Witty
    language: str = "German" # По умолчанию строгий Hochdeutsch

class CheckoutRequest(BaseModel):
    email: str
    plan: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- THE MIND OF ENGINE: ADVANCED REAL ESTATE PROMPT ---
def build_prompt(data: GenerateRequest):
    # Психологические триггеры для немецкого рынка недвижимости
    audience_triggers = {
        "Kapitalanleger": "Focus on high ROI (Rendite), solid cash flow, stress-free property management, and long-term value retention in stable German regions.",
        "Familien": "Focus on security, local schools (Schulen in der Nähe), quiet green neighborhoods (Grüne Lage), spacious layouts, and a place to grow.",
        "Singles / Studenten": "Focus on urban lifestyle, proximity to public transport (ÖPNV), smart space utilization, modern design, and vibrant local cafes/workspaces.",
        "General": "Focus on premium German real estate quality, transparent property specifications, and professional transaction standards."
    }

    # Стили подачи контента под немецкий менталитет
    tone_instructions = {
        "Professional": "Tone: Analytical, highly corporate, objective. Focus on facts, technical condition, and legal/financial transparency.",
        "Luxury": "Tone: Exclusive, sophisticated, elegant. Use high-end real estate vocabulary (Exquisites Wohnerlebnis, Premium-Ausstattung, Denkmalschutz-Charme).",
        "Emotional": "Tone: Warm, welcoming, sensory. Focus on the feeling of 'coming home', morning light through large windows, and family dinners.",
        "Witty": "Tone: Modern, slightly bold, catchy. Use the 'Expectation vs Reality' of finding an apartment in Germany. (e.g., No endless lines at mass viewings)."
    }

    trigger = audience_triggers.get(data.target_audience, audience_triggers["General"])
    tone_instr = tone_instructions.get(data.marketing_tone, tone_instructions["Professional"])

    return (
        f"You are a Senior Real Estate Marketing Expert and Copywriter specialized in the German property market (Immobilienmarkt).\n"
        f"{tone_instr}\n"
        f"Language: STRICTLY WRITE IN HIGH GERMAN (Hochdeutsch).\n\n"
        f"PROPERTY DATA:\n"
        f"- Type: {data.object_type}\n"
        f"- Location: {data.location}\n"
        f"- Layout: {data.rooms} Zimmer, {data.size_sqm} m²\n"
        f"- Features: {data.key_features}\n"
        f"- Target Buyer/Tenant: {data.target_audience}\n\n"
        f"STRATEGY:\n"
        f"{trigger}\n\n"
        f"TASK: Generate a complete marketing package for this property. Output EXACTLY two sections separated by '---'. No conversational filler, no meta-text.\n\n"
        f"REQUIRED FORMAT STRUCTURE:\n\n"
        f"SECTION 1: PROFESSIONAL EXPOSÉ TEXT\n"
        f"[Write a high-converting, professional description for ImmobilienScout24/Immowelt here. Include an attention-grabbing Title, an 'Objektbeschreibung' section, and an 'Ausstattung' section based on key features. Use professional paragraph formatting]\n"
        f"--- \n"
        f"SECTION 2: SHORT-FORM VIDEO SCRIPT (Reels / TikTok / Shorts)\n"
        f"[Write a 60-second engaging video script for the agent. Format it line by line like this:\n"
        f"Hook (0-5s): [Catchy German hook line]\n"
        f"Visual 1: [What the agent should film, e.g., Close-up of modern kitchen counters, bokeh background]\n"
        f"Audio 1: [What the agent says in German]\n"
        f"Visual 2: [Next scene]\n"
        f"Audio 2: [Next spoken line]\n"
        f"Call to Action: [Instructions for the viewer to DM or check the link in bio for the Exposé]]\n\n"
        f"CONSTRAINTS:\n"
        f"- Apply sensory marketing: describe textures, space depth, and premium finishes (e.g., high ceilings, natural stone, parquet flooring).\n"
        f"- BANNED GERMAN WORDS: 'schön', 'gemütlich', 'nett', 'wunderbar'. Replace with precise descriptive property terms (e.g., lichtdurchflutet, durchdachter Grundriss, hochwertig saniert).\n"
        f"- Limit emojis to a maximum of 3 in the entire script."
    )

# --- ENDPOINTS ---

@app.get("/get-credits")
def get_credits(email: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(email=email, credits=7, is_premium=False)
        db.add(user); db.commit(); db.refresh(user)
    return {"credits": user.credits, "is_premium": user.is_premium}

@app.post("/generate")
async def generate(data: GenerateRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        user = User(email=data.email, is_premium=False, credits=7)
        db.add(user); db.commit(); db.refresh(user)
    
    if not user.is_premium and user.credits < 1:
        return {"error": "credits_depleted"}

    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": build_prompt(data)},
                {"role": "user", "content": "Generate professional real estate content package now."}
            ],
            "temperature": 0.6 # Чуть снизили температуру для большей строгости текста
        }
        
        try:
            response = await client.post(GROQ_URL, headers=headers, json=payload, timeout=30)
            if response.status_code != 200:
                return {"error": f"Groq Error: {response.text}"}
            
            content = response.json()["choices"][0]["message"]["content"]
            # Разделяем Exposé и Видео-скрипт по нашему сепаратору '---'
            sections = [section.strip() for section in content.split('---') if section.strip()]

            if sections:
                if not user.is_premium:
                    user.credits -= 1
                    db.commit()
                return {
                    "expose_text": sections[0] if len(sections) > 0 else "Error generating expose",
                    "video_script": sections[1] if len(sections) > 1 else "Error generating script",
                    "remaining_credits": user.credits
                }
            
            return {"error": "AI failed to split real estate package"}
        except Exception as e:
            return {"error": str(e)}

@app.post("/create-checkout-session")
async def create_checkout_session(data: CheckoutRequest):
    DOMAIN = os.getenv("BASE_URL", "https://aies-immo.de") # Домен твоей новой студии
    # B2B цены: €49 в месяц или €490 в год за безлимитную генерацию объектов
    amount = 4900 if data.plan == "monthly" else 49000
    mode = "subscription" if data.plan == "monthly" else "payment"
    
    try:
        session = stripe.checkout.Session.create(
            mode=mode,
            customer_email=data.email,
            line_items=[{
                "price_data": {
                    "currency": "eur", # Перевели B2B на Евро
                    "product_data": {"name": f"AIES Real Estate Marketer {data.plan.capitalize()} Pro"},
                    "unit_amount": amount,
                    "recurring": {"interval": "month"} if mode == "subscription" else None,
                },
                "quantity": 1,
            }],
            success_url=f"{DOMAIN}/?success=true",
            cancel_url=f"{DOMAIN}/?canceled=true",
        )
        return {"url": session.url}
    except Exception as e:
        return {"error": str(e)}

@app.post("/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db), stripe_signature: str = Header(None)):
    payload = await request.body()
    try:
        event = stripe.Webhook.construct_event(payload, stripe_signature, STRIPE_WEBHOOK_SECRET)
    except Exception:
        raise HTTPException(status_code=400)

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        email = session.get("customer_email")
        amount = session.get("amount_total")

        if email:
            user = db.query(User).filter(User.email == email).first()
            if user:
                user.is_premium = True
                if amount == 4900: user.credits += 50 # Выдаем пакеты лимитов
                elif amount >= 49000: user.credits += 600
                db.commit()
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def index():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "Frontend index.html for Real Estate Studio not found in /static"