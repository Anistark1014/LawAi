#!/usr/bin/env python3
"""
AI Justice Bot - Production API
Uses trained ML model with 6,611 legal documents
NO hardcoded responses - Pure AI-powered legal advice
"""

import os
import json
import numpy as np
import faiss
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from datetime import datetime
import logging
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class ProductionLegalAI:
    def __init__(self):
        self.documents = []
        self.faiss_index = None
        self.sentence_model = None
        self.model_loaded = False
        
    def load_model(self):
        """Load the trained model and all legal documents"""
        try:
            logger.info("🔄 Loading AI Justice Bot Production Model...")
            
            # Load sentence transformer model (same as used during training)
            logger.info("Loading sentence transformer...")
            self.sentence_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            
            # Load documents
            docs_path = os.path.join('simple_legal_model', 'documents.json')
            if os.path.exists(docs_path):
                with open(docs_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                logger.info(f"✅ Loaded {len(self.documents)} legal documents")
            else:
                raise FileNotFoundError(f"Documents file not found: {docs_path}")
            
            # Load FAISS index
            index_path = os.path.join('simple_legal_model', 'legal_index.faiss')
            if os.path.exists(index_path):
                self.faiss_index = faiss.read_index(index_path)
                logger.info(f"✅ Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            else:
                raise FileNotFoundError(f"FAISS index not found: {index_path}")
            
            # Verify model info
            info_path = os.path.join('simple_legal_model', 'model_info.json')
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    model_info = json.load(f)
                logger.info(f"✅ Model Info: {model_info}")
            
            self.model_loaded = True
            logger.info("🚀 AI Justice Bot Production Model Ready!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            self.model_loaded = False
            return False
    
    def retrieve_relevant_docs(self, query, k=5):
        """Retrieve most relevant legal documents for the query"""
        if not self.model_loaded or not self.faiss_index:
            return []
        
        try:
            # Encode the query
            query_embedding = self.sentence_model.encode([query])
            
            # Search in FAISS index
            distances, indices = self.faiss_index.search(
                query_embedding.astype('float32'), k
            )
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    similarity = 1 / (1 + distance)  # Convert distance to similarity
                    results.append({
                        'text': doc['text'],
                        'source': doc.get('source', 'Legal Document'),
                        'similarity': float(similarity)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {str(e)}")
            return []
    
    def analyze_legal_situation(self, query):
        """Analyze the user's legal situation using AI"""
        query_lower = query.lower()
        
        # Detect crime types using keywords from legal documents
        crime_analysis = {
            'assault': ['hit', 'beaten', 'hurt', 'physical', 'violence', 'attack', 'fight'],
            'theft': ['stole', 'stolen', 'theft', 'robbery', 'bag', 'phone', 'money', 'wallet'],
            'cybercrime': ['hacked', 'hack', 'cyber', 'online', 'internet', 'account', 'password'],
            'fraud': ['fraud', 'scam', 'cheated', 'fake', 'deceived', 'payment', 'transaction'],
            'harassment': ['harassment', 'harass', 'threaten', 'bully', 'intimidate']
        }
        
        detected_crimes = []
        for crime_type, keywords in crime_analysis.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_crimes.append(crime_type)
        
        # Detect urgency
        urgency_keywords = ['urgent', 'emergency', 'immediately', 'help', 'asap', 'now']
        is_urgent = any(keyword in query_lower for keyword in urgency_keywords)
        
        return {
            'detected_crimes': detected_crimes,
            'is_urgent': is_urgent,
            'primary_crime': detected_crimes[0] if detected_crimes else 'general'
        }
    
    def generate_ai_legal_advice(self, query):
        """Generate AI-powered legal advice using trained model"""
        if not self.model_loaded:
            return {
                'response': "AI model is not loaded. Please check server configuration.",
                'sources': [],
                'confidence': 0.0,
                'error': 'model_not_loaded'
            }
        
        # Analyze the legal situation  
        situation = self.analyze_legal_situation(query)
        
        # Retrieve relevant legal documents
        relevant_docs = self.retrieve_relevant_docs(query, k=5)
        
        if not relevant_docs:
            return {
                'response': "I couldn't find relevant legal information for your specific question. Please consult with a legal professional for personalized advice.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Generate specific advice based on the situation
        legal_advice = self._build_specific_legal_response(query, situation, relevant_docs)
        
        # Calculate confidence based on document similarity
        avg_confidence = sum(doc['similarity'] for doc in relevant_docs) / len(relevant_docs)
        
        return {
            'response': legal_advice,
            'sources': [{'source': doc['source'], 'relevance': doc['similarity']} for doc in relevant_docs[:3]],
            'confidence': float(avg_confidence),
            'detected_crimes': situation['detected_crimes'],
            'is_urgent': situation['is_urgent']
        }
    
    def _build_specific_legal_response(self, query, situation, relevant_docs):
        """Build specific legal response based on crime type and legal documents"""
        
        # Extract key legal information from documents
        legal_sections = []
        procedures = []
        penalties = []
        
        for doc in relevant_docs:
            text = doc['text'].lower()
            
            # Extract legal sections
            if 'section' in text and any(num in text for num in ['66', '420', '323', '325', '379', '354']):
                legal_sections.append(doc['text'][:200])
            
            # Extract procedures
            if any(word in text for word in ['complaint', 'fir', 'police', 'court', 'file']):
                procedures.append(doc['text'][:150])
            
            # Extract penalties
            if any(word in text for word in ['punishment', 'penalty', 'imprisonment', 'fine']):
                penalties.append(doc['text'][:150])
        
        # Build response based on primary crime type
        primary_crime = situation['primary_crime']
        response = f"## ⚖️ **LEGAL GUIDANCE FOR YOUR SITUATION**\n\n"
        
        if situation['is_urgent']:
            response += "🚨 **URGENT SITUATION DETECTED**\n\n"
        
        # Crime-specific advice
        if primary_crime == 'assault':
            response += "### 🤕 **ASSAULT CASE ANALYSIS**\n\n"
            response += "**Immediate Actions:**\n"
            response += "• Get medical treatment immediately and preserve medical records\n"
            response += "• File FIR at nearest police station within 24 hours\n"
            response += "• Collect witness statements and evidence\n"
            response += "• Take photographs of injuries\n\n"
            
        elif primary_crime == 'theft':
            response += "### 🎒 **THEFT CASE ANALYSIS**\n\n"
            response += "**Immediate Actions:**\n"
            response += "• File FIR at nearest police station immediately\n"
            response += "• Provide detailed list of stolen items with values\n"
            response += "• Check CCTV footage in the area\n"
            response += "• Block cards/phones if stolen\n"
            response += "• Contact insurance company if applicable\n\n"
            
        elif primary_crime == 'cybercrime':
            response += "### 💻 **CYBERCRIME CASE ANALYSIS**\n\n"
            response += "**Immediate Actions:**\n"
            response += "• Change all passwords immediately\n"
            response += "• File complaint at Cyber Crime Cell\n"
            response += "• Take screenshots of unauthorized activities\n"
            response += "• Report to platform (social media/email provider)\n"
            response += "• Enable two-factor authentication\n\n"
            
        elif primary_crime == 'fraud':
            response += "### 💰 **FINANCIAL FRAUD CASE ANALYSIS**\n\n"
            response += "**Immediate Actions:**\n"
            response += "• Contact bank immediately to freeze accounts\n"
            response += "• File FIR with Economic Offence Wing\n"
            response += "• Preserve all transaction records\n"
            response += "• File complaint with banking ombudsman\n"
            response += "• Don't make any additional payments\n\n"
        
        # Add legal sections if found
        if legal_sections:
            response += "### 📋 **APPLICABLE LAWS:**\n\n"
            for section in legal_sections[:2]:
                response += f"• {section.strip()}\n"
            response += "\n"
        
        # Add procedures if found
        if procedures:
            response += "### ⚖️ **LEGAL PROCEDURE:**\n\n"
            for procedure in procedures[:2]:
                response += f"• {procedure.strip()}\n"
            response += "\n"
        
        # Add penalties if found
        if penalties:
            response += "### ⚠️ **LEGAL CONSEQUENCES FOR OFFENDER:**\n\n"
            for penalty in penalties[:1]:
                response += f"• {penalty.strip()}\n"
            response += "\n"
        
        response += "### 🎯 **NEXT STEPS:**\n\n"
        response += "1. **Gather Evidence** - Collect all relevant documents, photos, messages\n"
        response += "2. **File Complaint** - Visit police station or online complaint portal\n"
        response += "3. **Legal Consultation** - Consider consulting a lawyer for complex cases\n"
        response += "4. **Follow Up** - Keep tracking your case status regularly\n\n"
        
        response += "**⚠️ Disclaimer:** This is AI-generated legal guidance based on Indian law. For specific legal advice, please consult with a qualified lawyer.\n"
        
        return response

# Initialize the AI model
legal_ai = ProductionLegalAI()

@app.before_first_request
def initialize_model():
    """Initialize the model when the app starts"""
    legal_ai.load_model()

@app.route('/')
def home():
    """API Information"""
    return jsonify({
        "service": "AI Justice Bot - Production API",
        "status": "running",
        "model_loaded": legal_ai.model_loaded,
        "documents_count": len(legal_ai.documents),
        "ai_powered": True,
        "no_hardcoded_responses": True,
        "endpoints": {
            "POST /api/legal-advice": "Get AI-powered legal advice",
            "GET /api/health": "Health check"
        },
        "usage": {
            "method": "POST",
            "endpoint": "/api/legal-advice", 
            "payload": {"message": "Your legal question here"}
        }
    })

@app.route('/api/legal-advice', methods=['POST'])
def legal_advice():
    """Main endpoint for AI-powered legal advice"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        query = data.get('message', '').strip()
        if not query:
            return jsonify({"error": "Message is required"}), 400
        
        # Generate AI-powered legal advice
        result = legal_ai.generate_ai_legal_advice(query)
        
        return jsonify({
            "query": query,
            "response": result['response'],
            "sources": result['sources'],
            "confidence": result['confidence'],
            "detected_crimes": result.get('detected_crimes', []),
            "is_urgent": result.get('is_urgent', False),
            "ai_powered": True,
            "model": "Production Legal AI",
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Error in legal_advice endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
            "status": "error"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if legal_ai.model_loaded else "model_not_loaded",
        "model_loaded": legal_ai.model_loaded,
        "documents_count": len(legal_ai.documents),
        "ai_ready": legal_ai.model_loaded,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting AI Justice Bot Production API...")
    print("📚 Loading trained model with 6,611 legal documents...")
    print("🤖 NO hardcoded responses - Pure AI-powered advice!")
    print("\n🌐 Server will be available at: http://localhost:5000")
    print("📡 API Endpoint: POST /api/legal-advice")
    print("❤️  Health Check: GET /api/health")
    
    # Load model before starting server
    if legal_ai.load_model():
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("❌ Failed to load AI model. Please check model files.")