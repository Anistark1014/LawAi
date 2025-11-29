"""
Simple Legal API - Serves the retrieval-based legal model
This is faster and more reliable than transformer training
"""

import os
import io
import json
import pickle
import numpy as np
import faiss
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from datetime import datetime
from deep_translator import GoogleTranslator
from PyPDF2 import PdfReader
from PIL import Image
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    import pytesseract
except ImportError:  # Optional OCR dependency
    pytesseract = None

    marathi_tokenizer = None
    marathi_model = None
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi"
}

MARATHI_HINTS = {
    "मराठी", "कृपया", "आहे", "कायदा", "माझे", "करा", "तुम्ही", "नाही", "काय", "माझ्या"
}

HINDI_HINTS = {
    "हिंदी", "कृपया", "है", "कानून", "मेरा", "आप", "नहीं", "क्या", "किए", "करें"
}


def detect_language(text):
    """Tiny heuristic detector for supported languages."""
    if not text:
        return "en"

    lowered = text.lower()
    has_devanagari = any('\u0900' <= ch <= '\u097f' for ch in lowered)

    if not has_devanagari:
        return "en"

    if "मराठी" in lowered:
        return "mr"
    if "हिंदी" in lowered:
        return "hi"

    marathi_score = sum(1 for hint in MARATHI_HINTS if hint in lowered)
    hindi_score = sum(1 for hint in HINDI_HINTS if hint in lowered)

    if marathi_score > hindi_score:
        return "mr"
    if hindi_score > marathi_score:
        return "hi"

    # Default to Hindi for Devanagari script when unsure
    return "hi"


def simplify_markdown(text):
    """Convert heavy markdown formatting into plain conversational text."""
    if not isinstance(text, str):
        return text

    cleaned = text.replace('**', '')
    for marker in ['## ', '### ', '#### ', '---']:
        cleaned = cleaned.replace(marker, '')

    cleaned = cleaned.replace('•', '•')  # ensure bullet symbol remains consistent
    return '\n'.join(line.strip() for line in cleaned.splitlines() if line.strip())


ALLOWED_DOCUMENT_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "webp", "bmp"}
MAX_DOCUMENT_CHARACTERS = 6000


def extract_text_from_pdf(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text:
            pages.append(page_text)
    return "\n".join(pages)


def extract_text_from_image(file_bytes):
    if pytesseract is None:
        raise RuntimeError("Image OCR requires pytesseract. Please install pytesseract and the Tesseract engine.")
    image = Image.open(io.BytesIO(file_bytes))
    try:
        return pytesseract.image_to_string(image)
    finally:
        image.close()


def extract_text_from_document(file_bytes, extension):
    if extension == 'pdf':
        return extract_text_from_pdf(file_bytes)
    if extension in {'png', 'jpg', 'jpeg', 'webp', 'bmp'}:
        return extract_text_from_image(file_bytes)
    raise ValueError(f"Unsupported file type: {extension}")


def ensure_marathi_model():
    global marathi_tokenizer, marathi_model
    if marathi_tokenizer is None or marathi_model is None:
        marathi_tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-mr')
        marathi_model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-en-mr')


def translate_to_marathi(text):
    if not text:
        return text

    ensure_marathi_model()

    paragraphs = [para.strip() for para in text.split('\n')]
    translated_paragraphs = []

    buffer = []
    buffer_len = 0

    def flush_buffer():
        nonlocal buffer_len
        if not buffer:
            return
        segment = ' '.join(buffer).strip()
        buffer.clear()
        buffer_len = 0
        if not segment:
            return
        inputs = marathi_tokenizer(segment, return_tensors="pt", padding=True, truncation=True)
        outputs = marathi_model.generate(**inputs, max_length=512)
        translated = marathi_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        translated_paragraphs.append(translated)

    for para in paragraphs:
        if not para:
            flush_buffer()
            translated_paragraphs.append('')
            continue

        if buffer_len + len(para) > 400:
            flush_buffer()

        buffer.append(para)
        buffer_len += len(para)

    flush_buffer()

    return '\n'.join(translated_paragraphs).strip()


class SimpleLegalModelAPI:
    def __init__(self, model_dir="simple_legal_model"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = model_dir if os.path.isabs(model_dir) else os.path.join(base_dir, model_dir)
        self.documents = []
        self.faiss_index = None
        self.embeddings = None
        self.sentence_model = None
        self.model_info = {}
        self.load_model()
    
    def load_model(self):
        """Load the simple legal model"""
        try:
            if not os.path.exists(self.model_dir):
                logger.error(f"Model directory {self.model_dir} not found")
                logger.error("Please run 'python simple_train.py' first")
                return False
            
            logger.info(f"Loading model from {self.model_dir}")
            
            # Load documents
            with open(f"{self.model_dir}/documents.json", "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            
            # Load FAISS index
            self.faiss_index = faiss.read_index(f"{self.model_dir}/legal_index.faiss")
            
            # Load embeddings
            self.embeddings = np.load(f"{self.model_dir}/embeddings.npy")
            
            # Load sentence model
            self.sentence_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            
            # Load model info
            with open(f"{self.model_dir}/model_info.json", "r") as f:
                self.model_info = json.load(f)
            
            logger.info(f"Model loaded successfully! {len(self.documents)} documents available")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def retrieve_relevant_docs(self, query, k=3):
        """Retrieve most relevant documents"""
        if self.faiss_index is None or self.sentence_model is None:
            return []
        
        try:
            # Encode query
            query_embedding = self.sentence_model.encode([query])
            
            # Search
            distances, indices = self.faiss_index.search(
                query_embedding.astype('float32'), k
            )
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    similarity = 1 / (1 + distance)
                    results.append({
                        'text': doc['text'],
                        'source': doc['source'],
                        'similarity': float(similarity)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            return []
    
    def generate_legal_response(self, query):
        """Generate practical legal advice based on retrieved documents"""
        relevant_docs = self.retrieve_relevant_docs(query, k=3)
        
        if not relevant_docs:
            return {
                'response': "I couldn't find specific information in the legal documents for your query. Please consult with a legal professional for personalized advice.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Analyze query for legal issues
        query_lower = query.lower()
        legal_issues = {
            'harassment': ['beat', 'beaten', 'harass', 'bully', 'ragg', 'senior', 'threat', 'intimidation', 'abuse'],
            'cybercrime': ['hack', 'unauthorized access', 'breach', 'intrusion', 'cyber'],
            'fraud': ['fraud', 'cheat', 'scam', 'deceive', 'fake', 'payment'],
            'privacy': ['privacy', 'data', 'breach', 'leak', 'information'],
            'workplace': ['work', 'office', 'colleague', 'boss', 'employee']
        }
        
        detected_issues = []
        for issue, keywords in legal_issues.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_issues.append(issue)
        
        # Generate practical advice based on the issue type
        advice = self._generate_practical_advice(query_lower, detected_issues, relevant_docs)
        
        # Minimal source information - focus on practical advice
        avg_confidence = sum(doc['similarity'] for doc in relevant_docs) / len(relevant_docs)
        
        # Just return basic info, not detailed sources
        supporting_snippets = []
        for doc in relevant_docs[:2]:
            supporting_snippets.append({
                'source': doc.get('source', 'Legal Document'),
                'snippet': doc['text'][:500]
            })

        return {
            'response': advice,
            'supporting_snippets': supporting_snippets,
            'source_count': len(relevant_docs)
        }
    
    def _generate_practical_advice(self, query_lower, detected_issues, relevant_docs):
        """Generate AI-powered practical advice based on actual legal document content"""
        
        if not relevant_docs:
            return "I couldn't find relevant legal information for your specific situation. Please consult with a legal professional."
        
        # Extract key information from the user's query
        query_context = self._analyze_user_situation(query_lower)
        
        # Use actual legal document content to build personalized advice
        legal_knowledge = self._extract_legal_knowledge(relevant_docs, query_context)
        
        # Generate personalized response based on the specific situation
        return self._create_personalized_advice(query_context, legal_knowledge, relevant_docs)
    
    def _analyze_user_situation(self, query):
        """Analyze the user's specific situation using AI"""
        situation = {
            'urgency': 'normal',
            'crime_type': 'general',
            'victim_type': 'individual',
            'evidence_available': False,
            'financial_loss': False,
            'reputation_damage': False,
            'physical_harm': False,
            'platform': 'unknown',
            'witness_available': False,
            'criminal_seen': False
        }
        
        # Priority crime detection (most serious first)
        if any(word in query for word in ['murder', 'killed', 'dead', 'death', 'murdered', 'homicide']):
            situation['crime_type'] = 'murder'
            situation['urgency'] = 'critical'
            situation['physical_harm'] = True
        
        elif any(word in query for word in ['rape', 'sexual assault', 'molest', 'sexual abuse']):
            situation['crime_type'] = 'sexual_assault'
            situation['urgency'] = 'critical'
            situation['physical_harm'] = True
        
        elif any(word in query for word in ['kidnapped', 'abducted', 'missing person', 'kidnapping']):
            situation['crime_type'] = 'kidnapping'
            situation['urgency'] = 'critical'
        
        elif any(word in query for word in ['stolen', 'theft', 'stole', 'robbed', 'robbery', 'purse', 'bag', 'wallet', 'phone stolen']):
            situation['crime_type'] = 'theft'
            situation['urgency'] = 'high'
            situation['financial_loss'] = True
        
        elif any(word in query for word in ['dowry', 'dowry harassment', 'dowry death']):
            situation['crime_type'] = 'dowry_harassment'
            situation['urgency'] = 'high'
        
        elif any(word in query for word in ['domestic violence', 'husband beats', 'wife beating', 'family violence']):
            situation['crime_type'] = 'domestic_violence'
            situation['urgency'] = 'high'
            situation['physical_harm'] = True
        
        elif any(word in query for word in ['cheating', 'fraud', 'scam', 'fake', 'deceived']):
            situation['crime_type'] = 'fraud'
            situation['financial_loss'] = True
        
        elif any(word in query for word in ['hacked', 'hack', 'unauthorized access', 'account compromised']):
            situation['crime_type'] = 'cybercrime'
        
        elif any(word in query for word in ['beaten', 'hit', 'hurt', 'injured', 'physical', 'violence']):
            situation['crime_type'] = 'assault'
            situation['physical_harm'] = True
        
        # Context analysis
        if any(word in query for word in ['urgent', 'emergency', 'immediately', 'right now', 'asap', 'help']):
            if situation['urgency'] == 'normal':
                situation['urgency'] = 'high'
        
        if any(word in query for word in ['saw', 'witnessed', 'seen', 'found the person', 'know who']):
            situation['witness_available'] = True
            situation['criminal_seen'] = True
        
        if any(word in query for word in ['screenshot', 'evidence', 'proof', 'messages', 'chat', 'video', 'photo']):
            situation['evidence_available'] = True
        
        if any(word in query for word in ['instagram', 'facebook', 'twitter', 'whatsapp', 'social media']):
            situation['platform'] = 'social_media'
        
        if any(word in query for word in ['money', 'rupees', 'payment', 'bank', 'transaction']):
            situation['financial_loss'] = True
        
        return situation
    
    def _extract_legal_knowledge(self, relevant_docs, situation):
        """Extract relevant legal information from documents based on situation"""
        legal_info = {
            'applicable_laws': [],
            'penalties': [],
            'procedures': [],
            'time_limits': [],
            'compensation': []
        }
        
        # AI-powered extraction from legal documents
        for doc in relevant_docs:
            text = doc['text'].lower()
            
            # Extract specific legal sections and penalties
            if 'section' in text and any(num in text for num in ['66', '420', '323', '325', '499', '500']):
                if 'punishment' in text or 'penalty' in text or 'imprisonment' in text:
                    legal_info['penalties'].append(doc['text'][:300])
            
            # Extract procedures and time limits
            if any(word in text for word in ['complaint', 'fir', 'report', 'file', 'within']):
                legal_info['procedures'].append(doc['text'][:200])
            
            # Extract compensation information
            if any(word in text for word in ['compensation', 'damages', 'fine', 'refund']):
                legal_info['compensation'].append(doc['text'][:200])
        
        return legal_info
    
    def _create_personalized_advice(self, situation, legal_knowledge, relevant_docs):
        """Create personalized advice with improved 3-section format"""
        
        # Generate situation summary
        situation_summary = self._generate_situation_summary(situation, relevant_docs)
        
        # Header with urgency indicator
        advice = f"## ⚖️ **AI LEGAL ANALYSIS & GUIDANCE**\n\n"
        
        if situation['urgency'] == 'high':
            advice += "⚠️ **URGENT SITUATION DETECTED** - Time-sensitive actions required\n\n"
        
        advice += "---\n\n"
        
        # SECTION 1: WHAT HAPPENED (Understanding the Situation)
        advice += "### 📋 **WHAT HAPPENED - Legal Situation Analysis**\n\n"
        advice += situation_summary
        advice += "\n"
        
        # Extract relevant legal context
        if legal_knowledge['penalties']:
            legal_context = legal_knowledge['penalties'][0][:250].strip()
            advice += f"**� Legal Context:** {legal_context}...\n\n"
        
        advice += "---\n\n"
        
        # SECTION 2: WHAT YOU SHOULD DO (Action Steps)
        advice += "### ✅ **WHAT YOU SHOULD DO - Action Plan**\n\n"
        
        # Immediate Actions (Priority 1)
        advice += "#### 🚨 **Immediate Actions (Next 24-48 Hours):**\n\n"
        immediate_actions = self._get_immediate_actions(situation)
        for i, action in enumerate(immediate_actions, 1):
            advice += f"{i}. **{action}**\n"
        advice += "\n"
        
        # Legal Actions (Priority 2)
        advice += "#### ⚖️ **Legal Actions You Can Take:**\n\n"
        legal_actions = self._get_legal_actions(situation, legal_knowledge)
        for action in legal_actions:
            advice += f"• **{action}**\n"
        advice += "\n"
        
        # Documentation & Evidence
        advice += "#### 📸 **Evidence & Documentation:**\n\n"
        evidence_steps = self._get_evidence_steps(situation)
        for step in evidence_steps:
            advice += f"• {step}\n"
        advice += "\n"
        
        advice += "---\n\n"
        
        # SECTION 3: WHAT YOU SHOULDN'T DO (Critical Mistakes to Avoid)
        advice += "### ❌ **WHAT YOU SHOULDN'T DO - Critical Mistakes to Avoid**\n\n"
        
        mistakes_to_avoid = self._get_mistakes_to_avoid(situation)
        for i, mistake in enumerate(mistakes_to_avoid, 1):
            advice += f"{i}. **❌ {mistake}**\n"
        advice += "\n"
        
        advice += "---\n\n"
        
        # Footer with confidence and next steps
        confidence = sum(doc['similarity'] for doc in relevant_docs) / len(relevant_docs)
        advice += f"### 📊 **AI Analysis Summary**\n\n"
        advice += f"**🎯 Confidence Level:** {confidence*100:.1f}% (Based on {len(relevant_docs)} legal documents)\n"
        advice += f"**🏛️ Legal Jurisdiction:** Indian Laws (IT Act, IPC, Constitution)\n"
        advice += f"**⏰ Analysis Time:** Real-time AI processing\n\n"
        
        advice += "**📞 Emergency Contacts:**\n"
        advice += "• Police Emergency: **100**\n"
        advice += "• Cyber Crime Helpline: **1930**\n"
        advice += "• Women's Helpline: **1091**\n"
        advice += "• Legal Aid: Contact local Bar Association\n\n"
        
        advice += "*💡 This AI analysis is for guidance only. Always consult qualified lawyers for complex legal matters.*"
        
        return advice
    
    def _generate_situation_summary(self, situation, relevant_docs):
        """Generate intelligent situation summary"""
        summary = ""
        
        if situation['crime_type'] == 'murder':
            summary += f"**🔍 Analysis:** You've witnessed or have information about a **MURDER** - the most serious crime under Indian law. "
            summary += "This falls under **IPC Section 302** (punishment for murder) with **life imprisonment or death penalty**. "
            summary += "This is a **non-bailable offense** requiring immediate police intervention.\n\n"
        
        elif situation['crime_type'] == 'sexual_assault':
            summary += f"**🔍 Analysis:** This involves **sexual assault/rape** - a heinous crime under IPC Sections 375-376. "
            summary += "This carries **minimum 7 years to life imprisonment**. Time-sensitive medical and legal evidence collection is critical.\n\n"
        
        elif situation['crime_type'] == 'kidnapping':
            summary += f"**🔍 Analysis:** This involves **kidnapping/abduction** under IPC Sections 359-369. "
            summary += "This is a **serious offense** with up to **7 years imprisonment**. Every minute counts in rescue operations.\n\n"
        
        elif situation['crime_type'] == 'theft':
            summary += f"**🔍 Analysis:** You're victim of **theft** under IPC Section 378-382. "
            if situation['criminal_seen']:
                summary += "Since you've identified the perpetrator, this strengthens your case significantly. "
            summary += "This can result in **imprisonment up to 3 years** for the thief.\n\n"
        
        elif situation['crime_type'] == 'domestic_violence':
            summary += f"**🔍 Analysis:** This involves **domestic violence** under Domestic Violence Act 2005 and IPC. "
            summary += "You have legal protection and can get **immediate restraining orders**.\n\n"
        
        elif situation['crime_type'] == 'dowry_harassment':
            summary += f"**🔍 Analysis:** This involves **dowry harassment** under IPC Section 498A and Dowry Prohibition Act. "
            summary += "This carries **imprisonment up to 3 years** and is a **cognizable offense**.\n\n"
        
        elif situation['crime_type'] == 'cybercrime':
            summary += f"**🔍 Analysis:** Your digital accounts/data have been compromised. "
            if situation['financial_loss']:
                summary += "This involves **cyber fraud** with financial extortion components. "
            summary += "This falls under **IT Act 2000** with potential **3 years imprisonment**.\n\n"
        
        elif situation['crime_type'] == 'fraud':
            summary += f"**🔍 Analysis:** You're victim of **cheating and fraud** under IPC Section 420. "
            summary += "This constitutes criminal breach of trust with **imprisonment up to 7 years**.\n\n"
        
        elif situation['crime_type'] == 'assault':
            summary += f"**🔍 Analysis:** You've experienced **physical assault** under IPC Sections 323/325. "
            summary += "This can result in **imprisonment up to 2 years** (simple hurt) or **7 years** (grievous hurt).\n\n"
        
        else:
            summary += f"**🔍 Analysis:** Based on your description, this appears to be a legal matter requiring "
            summary += f"immediate attention and proper legal remedies.\n\n"
        
        # Add legal severity assessment
        if situation['urgency'] == 'critical':
            summary += "**⚡ Severity:** CRITICAL - Call Police 100 IMMEDIATELY\n"
        elif situation['urgency'] == 'high':
            summary += "**⚡ Severity:** HIGH - Immediate legal intervention required\n"
        else:
            summary += "**⚡ Severity:** MODERATE - Timely legal action recommended\n"
        
        return summary
    
    def _get_immediate_actions(self, situation):
        """Generate immediate action steps based on situation"""
        actions = []
        
        if situation['crime_type'] == 'murder':
            actions.append("CALL POLICE 100 IMMEDIATELY - Do not delay even by minutes")
            actions.append("Do not touch or disturb the crime scene")
            actions.append("Preserve your safety - do not confront anyone")
            actions.append("Note down exact time, location, and all details you witnessed")
            actions.append("Identify yourself as witness and provide statement to police")
        
        elif situation['crime_type'] == 'sexual_assault':
            actions.append("CALL POLICE 100 and Women's Helpline 1091 immediately")
            actions.append("Do not wash, change clothes, or clean up (preserves evidence)")
            actions.append("Go to nearest hospital for medical examination within 24 hours")
            actions.append("Contact trusted family member or friend for support")
            actions.append("Request female police officer for statement recording")
        
        elif situation['crime_type'] == 'kidnapping':
            actions.append("CALL POLICE 100 IMMEDIATELY - Every minute is critical")
            actions.append("Provide last known location and description of victim")
            actions.append("Share recent photos of victim with police")
            actions.append("Check victim's phone location, social media activity")
            actions.append("Contact all friends/family who might have information")
        
        elif situation['crime_type'] == 'theft':
            actions.append("File FIR at nearest police station within 24 hours")
            if situation['criminal_seen']:
                actions.append("Provide detailed description of the thief to police")
                actions.append("Check CCTV footage from the area immediately")
            actions.append("Block all cards/phones and inform bank/telecom immediately")
            actions.append("List all stolen items with approximate values")
            actions.append("Inform insurance company if items were insured")
        
        elif situation['crime_type'] == 'domestic_violence':
            actions.append("Call Women's Helpline 1091 for immediate assistance")
            actions.append("Document all injuries with photographs")
            actions.append("File complaint at nearest police station or women's cell")
            actions.append("Get medical treatment and preserve medical records")
            actions.append("Contact Protection Officer for immediate restraining order")
        
        elif situation['crime_type'] == 'dowry_harassment':
            actions.append("File FIR under IPC Section 498A immediately")
            actions.append("Document all dowry demands with evidence")
            actions.append("Contact women's helpline and local NGOs")
            actions.append("Preserve all gifts/money transaction records")
            actions.append("Get medical examination if physically harmed")
        
        elif situation['crime_type'] == 'cybercrime':
            actions.append("Try account recovery through official platform channels")
            if situation['reputation_damage']:
                actions.append("Alert all contacts about the compromise immediately")
            actions.append("Document unauthorized posts/messages with screenshots")
            actions.append("Report hacked account to platform security team")
            actions.append("File online complaint at cybercrime.gov.in")
        
        elif situation['crime_type'] == 'fraud':
            actions.append("Contact bank/financial institution to freeze accounts")
            actions.append("File FIR at nearest police station")
            actions.append("Report to cyber crime cell within 24 hours")
            actions.append("Preserve all transaction records and communications")
        
        elif situation['crime_type'] == 'assault':
            actions.append("Seek immediate medical attention and get medical certificate")
            actions.append("File FIR at police station with injury documentation")
            actions.append("Collect witness statements and contact information")
            actions.append("Report to HR/administration if workplace incident")
        
        else:
            actions.append("Document all evidence related to the incident")
            actions.append("Report to appropriate law enforcement authorities")
            actions.append("Consult with a qualified legal professional")
        
        return actions
    
    def _get_legal_actions(self, situation, legal_knowledge):
        """Generate legal action options"""
        actions = []
        
        if situation['crime_type'] == 'account_hacking':
            actions.append("Criminal case under IT Act 2000 Section 66 (imprisonment up to 3 years)")
            if situation['reputation_damage']:
                actions.append("Defamation case under IPC Section 499/500 (compensation + imprisonment)")
            if situation['financial_loss']:
                actions.append("Civil suit for financial recovery and damages")
        
        elif situation['crime_type'] == 'financial_fraud':
            actions.append("Criminal case under IPC Section 420 (Cheating - up to 7 years jail)")
            actions.append("Civil recovery suit for full refund of defrauded amount")
            actions.append("Complaint to Banking Ombudsman for institutional recovery")
        
        elif situation['crime_type'] == 'assault':
            actions.append("Criminal case under IPC Section 323/325 (Simple/Grievous Hurt)")
            actions.append("Civil suit for compensation (medical expenses + mental trauma)")
            actions.append("Workplace complaint under relevant labor laws")
        
        else:
            actions.append("Appropriate criminal proceedings based on specific offense")
            actions.append("Civil remedies for recovery of damages")
        
        return actions
    
    def _get_evidence_steps(self, situation):
        """Generate evidence collection steps"""
        steps = []
        
        if situation['crime_type'] == 'account_hacking':
            steps.append("Screenshot all unauthorized posts before they're deleted")
            steps.append("Save conversation threads and threatening messages")
            steps.append("Get platform's incident report/security logs if available")
        
        elif situation['crime_type'] == 'financial_fraud':
            steps.append("Preserve all banking/transaction records")
            steps.append("Save email/SMS communications from fraudsters")
            steps.append("Document financial losses with bank statements")
        
        elif situation['crime_type'] == 'assault':
            steps.append("Photograph all visible injuries immediately")
            steps.append("Get medical reports and treatment records")
            steps.append("Collect witness contact information and statements")
        
        else:
            steps.append("Document incident with photos/videos if applicable")
            steps.append("Preserve all relevant communications and records")
            steps.append("Maintain chronological record of events")
        
        return steps
    
    def _get_mistakes_to_avoid(self, situation):
        """Generate critical mistakes to avoid"""
        mistakes = []
        
        if situation['crime_type'] == 'account_hacking':
            mistakes.append("Don't pay any ransom or respond to extortion demands")
            mistakes.append("Don't delete evidence or create new accounts immediately")
            mistakes.append("Don't handle this privately - involve authorities")
            mistakes.append("Don't wait too long - digital evidence disappears quickly")
        
        elif situation['crime_type'] == 'financial_fraud':
            mistakes.append("Don't make additional payments hoping to recover money")
            mistakes.append("Don't share more personal/financial information")
            mistakes.append("Don't accept 'settlement' offers without legal consultation")
            mistakes.append("Don't delay reporting - time limits apply for recovery")
        
        elif situation['crime_type'] == 'assault':
            mistakes.append("Don't retaliate with violence - it weakens your legal case")
            mistakes.append("Don't 'settle' privately without legal complaint first")
            mistakes.append("Don't clean injuries before documentation")
            mistakes.append("Don't discuss details on social media before legal proceedings")
        
        else:
            mistakes.append("Don't handle complex legal matters without professional help")
            mistakes.append("Don't destroy or tamper with evidence")
            mistakes.append("Don't accept informal settlements without legal review")
            mistakes.append("Don't delay action due to legal time limitations")
        
        return mistakes

# Initialize the model
legal_api = SimpleLegalModelAPI()

@app.route('/')
def home():
    """API information"""
    return jsonify({
        "service": "Simple Legal AI API",
        "model_loaded": legal_api.faiss_index is not None,
        "documents_count": len(legal_api.documents),
        "model_info": legal_api.model_info,
        "endpoints": {
            "POST /api/legal-advice": "Get legal advice",
            "GET /api/health": "Health check",
            "GET /api/search": "Search legal documents"
        }
    })

@app.route('/api/legal-advice', methods=['POST'])
def legal_advice():
    """Main endpoint for your React website"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        query = data.get('message', '').strip()
        if not query:
            return jsonify({"error": "Message is required"}), 400
        
        if not legal_api.faiss_index:
            return jsonify({
                "error": "Model not loaded",
                "message": "Please run 'python simple_train.py' first",
                "status": "error"
            }), 500
        
        auto_language = detect_language(query)
        requested_language = data.get('language', auto_language)
        if isinstance(requested_language, str):
            requested_language = requested_language.lower()
        else:
            requested_language = auto_language

        if requested_language not in SUPPORTED_LANGUAGES:
            requested_language = 'en'

        detected_language = auto_language if auto_language in SUPPORTED_LANGUAGES else 'en'

        processed_query = query
        if detected_language != 'en' or not query.isascii():
            try:
                processed_query = GoogleTranslator(source='auto', target='en').translate(query)
            except Exception as translate_error:
                logger.error(f"Query translation failed: {translate_error}")
                processed_query = query

        # Generate response
        result = legal_api.generate_legal_response(processed_query)
        english_response = simplify_markdown(result['response'])

        translated_response = english_response
        snippet_translator = None
        translation_failed = False
        response_language = requested_language

        if response_language == 'mr':
            try:
                translated_response = translate_to_marathi(english_response)
            except Exception as translate_error:
                logger.error(f"Marathi translation failed: {translate_error}")
                translated_response = english_response
                translation_failed = True
                response_language = 'en'
        elif response_language != 'en':
            try:
                snippet_translator = GoogleTranslator(source='auto', target=requested_language)
                translated_response = snippet_translator.translate(english_response)
            except Exception as translate_error:
                logger.error(f"Response translation failed: {translate_error}")
                translated_response = english_response
                translation_failed = True
                response_language = 'en'
                snippet_translator = None

        translated_input = query
        if response_language == 'mr':
            try:
                translated_input = translate_to_marathi(query)
            except Exception as translate_error:
                logger.error(f"Input Marathi translation failed: {translate_error}")
                translated_input = query
        elif response_language != 'en':
            translator_for_input = snippet_translator
            if translator_for_input is None and response_language != 'en':
                try:
                    translator_for_input = GoogleTranslator(source='auto', target=response_language)
                except Exception as translate_error:
                    logger.error(f"Input translation setup failed: {translate_error}")
                    translator_for_input = None

            if translator_for_input:
                try:
                    translated_input = translator_for_input.translate(query)
                except Exception as translate_error:
                    logger.error(f"Input translation failed: {translate_error}")
                    translated_input = query

        translated_snippets = []
        for snippet in result.get('supporting_snippets', []):
            snippet_text = snippet.get('snippet', '')
            display_text = snippet_text

            if response_language == 'mr' and snippet_text:
                try:
                    display_text = translate_to_marathi(snippet_text)
                except Exception as snippet_error:
                    logger.error(f"Snippet Marathi translation failed: {snippet_error}")
                    display_text = snippet_text
            elif response_language != 'en' and snippet_translator and snippet_text:
                try:
                    display_text = snippet_translator.translate(snippet_text)
                except Exception as snippet_error:
                    logger.error(f"Snippet translation failed: {snippet_error}")
                    display_text = snippet_text

            snippet_payload = {
                'source': snippet.get('source', 'Legal Document'),
                'snippet': display_text
            }

            if response_language != 'en' and display_text != snippet_text:
                snippet_payload['original_snippet'] = snippet_text

            translated_snippets.append(snippet_payload)

        response_payload = {
            "response": translated_response,
            "original_response": english_response,
            "language": response_language,
            "detected_language": detected_language,
            "model": "Simple Legal AI",
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "query": query,
            "translated_input": translated_input,
            "supporting_snippets": translated_snippets,
            "source_count": result.get('source_count', 0)
        }

        if translation_failed:
            response_payload["translation_error"] = True

        if processed_query != query:
            response_payload["processed_query"] = processed_query
        
        return jsonify(response_payload)
        
    except Exception as e:
        logger.error(f"Error in legal_advice: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
            "status": "error"
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        "status": "healthy",
        "model_loaded": legal_api.faiss_index is not None,
        "documents_count": len(legal_api.documents),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/search', methods=['POST'])
def search_documents():
    """Search legal documents directly"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        k = data.get('results', 5)
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        results = legal_api.retrieve_relevant_docs(query, k=k)
        
        return jsonify({
            "query": query,
            "results": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


    @app.route('/api/analyze-document', methods=['POST'])
    def analyze_document():
        """Analyze uploaded legal documents (PDF or image) for guidance."""
        try:
            if 'file' not in request.files:
                return jsonify({
                    "status": "error",
                    "message": "No file provided. Please upload a PDF or image."
                }), 400

            uploaded_file = request.files['file']
            if not uploaded_file or uploaded_file.filename == '':
                return jsonify({
                    "status": "error",
                    "message": "Empty file upload."
                }), 400

            filename = uploaded_file.filename
            extension = filename.rsplit('.', 1)[-1].lower()
            if extension not in ALLOWED_DOCUMENT_EXTENSIONS:
                return jsonify({
                    "status": "error",
                    "message": f"Unsupported file type: {extension}. Allowed: {', '.join(sorted(ALLOWED_DOCUMENT_EXTENSIONS))}."
                }), 400

            file_bytes = uploaded_file.read()
            if not file_bytes:
                return jsonify({
                    "status": "error",
                    "message": "Uploaded file is empty."
                }), 400

            try:
                extracted_text = extract_text_from_document(file_bytes, extension)
            except RuntimeError as ocr_error:
                return jsonify({
                    "status": "error",
                    "message": str(ocr_error)
                }), 400
            except Exception as extraction_error:
                logger.error(f"Document extraction failed: {extraction_error}")
                return jsonify({
                    "status": "error",
                    "message": "Unable to read the uploaded document."
                }), 400

            if not extracted_text or not extracted_text.strip():
                return jsonify({
                    "status": "error",
                    "message": "Could not extract readable text from the document."
                }), 400

            trimmed_text = extracted_text.strip()
            if len(trimmed_text) > MAX_DOCUMENT_CHARACTERS:
                trimmed_text = trimmed_text[:MAX_DOCUMENT_CHARACTERS]

            auto_language = detect_language(trimmed_text)
            requested_language = request.form.get('language', auto_language)
            if isinstance(requested_language, str):
                requested_language = requested_language.lower()
            else:
                requested_language = auto_language

            if requested_language not in SUPPORTED_LANGUAGES:
                requested_language = auto_language if auto_language in SUPPORTED_LANGUAGES else 'en'

            detected_language = auto_language if auto_language in SUPPORTED_LANGUAGES else 'en'

            processed_text = trimmed_text
            if detected_language != 'en':
                try:
                    processed_text = GoogleTranslator(source='auto', target='en').translate(trimmed_text)
                except Exception as translate_error:
                    logger.error(f"Document translation failed: {translate_error}")
                    processed_text = trimmed_text

            result = legal_api.generate_legal_response(processed_text)
            english_response = simplify_markdown(result['response'])

            translated_response = english_response
            snippet_translator = None
            translation_failed = False
            response_language = requested_language

            if response_language == 'mr':
                try:
                    translated_response = translate_to_marathi(english_response)
                except Exception as translate_error:
                    logger.error(f"Document Marathi translation failed: {translate_error}")
                    translated_response = english_response
                    translation_failed = True
                    response_language = 'en'
            elif response_language != 'en':
                try:
                    snippet_translator = GoogleTranslator(source='auto', target=response_language)
                    translated_response = snippet_translator.translate(english_response)
                except Exception as translate_error:
                    logger.error(f"Document response translation failed: {translate_error}")
                    translated_response = english_response
                    translation_failed = True
                    response_language = 'en'
                    snippet_translator = None

            translated_snippets = []
            for snippet in result.get('supporting_snippets', []):
                snippet_text = snippet.get('snippet', '')
                display_text = snippet_text

                if response_language == 'mr' and snippet_text:
                    try:
                        display_text = translate_to_marathi(snippet_text)
                    except Exception as snippet_error:
                        logger.error(f"Document snippet Marathi translation failed: {snippet_error}")
                        display_text = snippet_text
                elif response_language != 'en' and snippet_translator and snippet_text:
                    try:
                        display_text = snippet_translator.translate(snippet_text)
                    except Exception as snippet_error:
                        logger.error(f"Document snippet translation failed: {snippet_error}")
                        display_text = snippet_text

                snippet_payload = {
                    'source': snippet.get('source', 'Legal Document'),
                    'snippet': display_text
                }

                if response_language != 'en' and display_text != snippet_text:
                    snippet_payload['original_snippet'] = snippet_text

                translated_snippets.append(snippet_payload)

            translated_input = trimmed_text
            if response_language == 'mr':
                try:
                    translated_input = translate_to_marathi(trimmed_text)
                except Exception as translate_error:
                    logger.error(f"Document input Marathi translation failed: {translate_error}")
                    translated_input = trimmed_text
            elif response_language != 'en':
                translator_for_input = snippet_translator
                if translator_for_input is None:
                    try:
                        translator_for_input = GoogleTranslator(source='auto', target=response_language)
                    except Exception as translate_error:
                        logger.error(f"Document input translation setup failed: {translate_error}")
                        translator_for_input = None

                if translator_for_input:
                    try:
                        translated_input = translator_for_input.translate(trimmed_text)
                    except Exception as translate_error:
                        logger.error(f"Document input translation failed: {translate_error}")
                        translated_input = trimmed_text

            response_payload = {
                "status": "success",
                "filename": filename,
                "language": response_language,
                "detected_language": detected_language,
                "response": translated_response,
                "original_response": english_response,
                "translated_input": translated_input,
                "original_input": trimmed_text,
                "supporting_snippets": translated_snippets,
                "source_count": result.get('source_count', 0),
                "document_preview": trimmed_text[:600],
                "document_length": len(trimmed_text),
                "timestamp": datetime.now().isoformat()
            }

            if translation_failed:
                response_payload['translation_error'] = True

            if processed_text != trimmed_text:
                response_payload['processed_query'] = processed_text

            return jsonify(response_payload)

        except Exception as e:
            logger.error(f"Error in analyze_document: {str(e)}")
            return jsonify({
                "status": "error",
                "message": "Internal server error"
            }), 500

# Legacy endpoint
@app.route('/chat', methods=['POST'])
def chat():
    """Legacy endpoint"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    query = data.get('message', '')
    if not legal_api.faiss_index:
        return jsonify({"response": "Model not loaded. Please train the model first."})
    
    result = legal_api.generate_legal_response(query)
    return jsonify({"response": result['response']})

if __name__ == '__main__':
    print("\n🏛️ Simple Legal AI API Server")
    print("=" * 50)
    
    if legal_api.faiss_index:
        print("✅ Legal model loaded successfully!")
        print(f"📚 Documents: {len(legal_api.documents)}")
        print(f"🔍 Model: {legal_api.model_info.get('model_type', 'Unknown')}")
    else:
        print("❌ Model not loaded!")
        print("📋 To train the model:")
        print("   1. Run: python simple_train.py")
        print("   2. Then restart this API server")
    
    print("\n🚀 API Endpoints:")
    print("   POST /api/legal-advice  (main endpoint)")
    print("   GET  /api/health        (health check)")
    print("   POST /api/search        (document search)")
    
    print(f"\n🌐 Server starting on http://localhost:5000")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)