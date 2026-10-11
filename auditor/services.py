import os
import cv2
import base64
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "yolov8_ppe.pt")
RULES_PATH = os.path.join(BASE_DIR, "data", "mining_safety_rules.txt")

# Load YOLO model
model = YOLO(MODEL_PATH)

def detect_ppe(image_bytes: bytes):
    """Processes image bytes with YOLOv8, returns annotated base64 image and detected classes."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = model.predict(source=img_rgb, conf=0.25) # Changed from 0.50
    res = results[0]

    detections = []
    for box in res.boxes:
        cls_id = int(box.cls[0].item())
        class_name = res.names[cls_id]
        detections.append(class_name)

    annotated_bgr = res.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    # Encode annotated image to base64 for API delivery
    pil_img = Image.fromarray(annotated_rgb)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return img_base64, detections

def audit_ppe_compliance(detections: list) -> str:
    """Invokes LangChain + Gemini to audit detected violations."""
    
    # If the AI sees absolutely nothing, throw a fail-safe immediately.
    if not detections:
        return "### STATUS: NO ACTIVITY DETECTED\n\nVision system did not detect any active personnel or relevant entities in the current frame."

    violations = [d for d in detections if d in ["no-helmet", "no-hard-hat", "no-hardhat", "no_vest", "no-vest"]]
    
    # Fail-safe heuristic: If a person is seen, but NO hat is seen, it is a violation.
    if "person" in detections and not any(d in detections for d in ["hard-hat", "hardhat", "helmet"]):
        if "no-hardhat" not in violations:
            violations.append("no-hardhat")
            
    # Stricter Fail-safe: If neither a hat nor a person is seen, but we reached here, 
    # it means it detected something random. Treat as no activity.
    if "person" not in detections and not any(d in detections for d in ["hard-hat", "hardhat", "helmet"]):
         return "### STATUS: NO ACTIVITY DETECTED\n\nVision system did not detect personnel."

    if not violations:
        return "### STATUS: FULLY COMPLIANT\n\nAll active personnel on frame meet mandatory DGMS PPE provisions (IS:2925 industrial headwear & high-visibility protective gear)."

    regulations = ""
    # ... (Keep the rest of your LangChain prompt logic identical from here down)
    if os.path.exists(RULES_PATH):
        with open(RULES_PATH, "r", encoding="utf-8") as f:
            regulations = f.read()

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2
    )

    template = """You are an Autonomous AI Mine Safety Compliance Auditor enforcing statutory regulations in open-cast and underground mines under DGMS guidelines, Coal Mines Regulations (CMR), Metalliferous Mines Regulations (MMR), and the Mines Act, 1952.

The computer vision system detected the following safety infractions: {violations}

Statutory mining rules context:
{regulations}

Generate an official STATUTORY VIOLATION INCIDENT LOG with the following sections in Markdown:
1. Infraction Summary
2. Statutory Reference (Quote MMR/CMR rules and Section 72C of the Mines Act, 1952)
3. Hazard Risk Level (CRITICAL / HIGH / MODERATE)
4. Immediate Corrective Action (Directed to Shift In-Charge / Mining Sirdar / Overman)
"""
    prompt = PromptTemplate(template=template, input_variables=["violations", "regulations"])
    chain = prompt | llm
    response = chain.invoke({"violations": ", ".join(violations), "regulations": regulations})
    return response.content

def ask_statutory_copilot(query: str) -> str:
    """Statutory Q&A Copilot endpoint query."""
    regulations = ""
    if os.path.exists(RULES_PATH):
        with open(RULES_PATH, "r", encoding="utf-8") as f:
            regulations = f.read()

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3
    )

    template = """You are an expert DGMS Mining Statutory Safety Copilot.
Answer this technical inquiry citing relevant sections of the Mines Act, CMR, and MMR:
Question: {query}
Regulations: {regulations}
"""
    prompt = PromptTemplate(template=template, input_variables=["query", "regulations"])
    chain = prompt | llm
    response = chain.invoke({"query": query, "regulations": regulations})
    return response.content