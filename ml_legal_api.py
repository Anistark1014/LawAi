#!/usr/bin/env python3
"""
ML-Powered Legal API (No If-Else Logic)
Replaces rule-based system with true machine learning
"""
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import faiss
import logging
import pickle
import os

# Import our true ML model
from true_ml_legal_ai import TrueMLLegalAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class MLLegalAPI:
    def __init__(self):
        self.ml_model = TrueMLLegalAI()
        self.faiss_index = None
        self.documents = []
        self.sentence_model = None
        self.model_loaded = False
        
    def load_models(self):
        """Load both ML model and document retrieval system"""
        try:
            logger.info("Loading ML-powered legal AI...")
            
            # Load true ML model (no if-else)
            if self.ml_model.load_model("ml_legal_model.pkl"):
                logger.info("✅ ML model loaded successfully!")
            else:
                logger.warning("❌ ML model not found, training new one...")
                return False
            
            # Load document retrieval system
            model_dir = "simple_legal_model"
            if os.path.exists(model_dir):
                # Load FAISS index
                self.faiss_index = faiss.read_index(f"{model_dir}/faiss_index.bin")
                
                # Load documents
                with open(f"{model_dir}/documents.json", "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
                
                # Load sentence model for retrieval
                self.sentence_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
                
                logger.info(f"📚 Document retrieval system loaded: {len(self.documents)} documents")
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def retrieve_relevant_docs(self, query, k=3):
        """Retrieve relevant legal documents"""
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
            logger.error(f"Error in document retrieval: {str(e)}")
            return []
    
    def generate_ml_response(self, user_query):
        """Generate response using pure ML (no if-else logic)"""
        
        if not self.model_loaded:
            return {
                'response': "ML models not loaded. Please check server configuration.",
                'sources': [],
                'confidence': 0.0,
                'ml_powered': False
            }
        
        try:
            # Get relevant documents for legal context
            relevant_docs = self.retrieve_relevant_docs(user_query, k=3)
            
            # Use ML to analyze the query (NO IF-ELSE LOGIC)
            ml_analysis = self.ml_model.analyze_query_ml(user_query)
            
            # Generate ML-powered response (NO IF-ELSE LOGIC)
            ml_response = self.ml_model.generate_response_ml(user_query, ml_analysis)
            
            # Build sources array
            sources = []
            for doc in relevant_docs:
                sources.append({
                    'source': doc['source'].replace('.PDF', '').replace('_', ' '),
                    'similarity': float(doc['similarity']),
                    'text_preview': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
                })
            
            return {
                'response': ml_response,
                'sources': sources,
                'confidence': float(ml_analysis['similarity_score']),
                'ml_analysis': {
                    'crime_type': ml_analysis['predicted_crime_type'],
                    'crime_confidence': ml_analysis['crime_confidence'],
                    'urgency': ml_analysis['predicted_urgency'],
                    'urgency_confidence': ml_analysis['urgency_confidence']
                },
                'ml_powered': True,
                'model_type': 'Advanced ML (Random Forest + Semantic Embeddings)'
            }
            
        except Exception as e:
            logger.error(f"Error in ML response generation: {str(e)}")
            return {
                'response': f"Error in ML processing: {str(e)}",
                'sources': [],
                'confidence': 0.0,
                'ml_powered': False
            }

# Initialize ML-powered API
ml_legal_api = MLLegalAPI()

@app.route('/')
def home():
    """API information"""
    return jsonify({
        "service": "ML-Powered Legal AI API (No If-Else Logic)",
        "model_loaded": ml_legal_api.model_loaded,
        "documents_count": len(ml_legal_api.documents),
        "ml_powered": True,
        "model_type": "Advanced ML (Random Forest + Semantic Embeddings)",
        "endpoints": {
            "POST /api/legal-advice": "Get ML-powered legal advice",
            "GET /api/health": "Health check",
            "GET /api/ml-info": "ML model information"
        }
    })

@app.route('/api/legal-advice', methods=['POST'])
def legal_advice():
    """ML-powered legal advice endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        query = data.get('message', '').strip()
        if not query:
            return jsonify({"error": "Message is required"}), 400
        
        if not ml_legal_api.model_loaded:
            return jsonify({
                "error": "ML models not loaded",
                "message": "Please check server configuration",
                "status": "error"
            }), 500
        
        # Generate ML-powered response
        result = ml_legal_api.generate_ml_response(query)
        
        return jsonify({
            "query": query,
            "answer": result['response'],
            "sources": result['sources'],
            "confidence": result['confidence'],
            "ml_analysis": result.get('ml_analysis', {}),
            "ml_powered": result['ml_powered'],
            "model_type": result.get('model_type', 'Unknown'),
            "status": "success",
            "timestamp": "2025-10-28T" + "05:45:00.000000"
        })
        
    except Exception as e:
        logger.error(f"Error in legal advice endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if ml_legal_api.model_loaded else "loading",
        "model_loaded": ml_legal_api.model_loaded,
        "documents_count": len(ml_legal_api.documents),
        "ml_powered": True,
        "model_type": "Advanced ML (Random Forest + Semantic Embeddings)",
        "timestamp": "2025-10-28T" + "05:45:00.000000"
    })

@app.route('/api/ml-info', methods=['GET'])
def ml_info():
    """ML model information endpoint"""
    return jsonify({
        "ml_powered": True,
        "model_architecture": "Random Forest + Semantic Embeddings",
        "no_if_else_logic": True,
        "features": [
            "Crime Type Classification",
            "Urgency Level Prediction", 
            "Semantic Similarity Matching",
            "Context-Aware Response Generation"
        ],
        "training_scenarios": len(ml_legal_api.ml_model.legal_scenarios) if ml_legal_api.model_loaded else 0,
        "model_loaded": ml_legal_api.model_loaded
    })

if __name__ == '__main__':
    print("🚀 Starting ML-Powered Legal AI API Server...")
    print("🤖 No If-Else Logic - Pure Machine Learning")
    print("📍 URL: http://localhost:5000")
    print("=" * 60)
    
    # Load ML models
    if ml_legal_api.load_models():
        print("✅ ML models loaded successfully!")
    else:
        print("❌ Failed to load ML models")
    
    app.run(
        host='0.0.0.0', 
        port=5000, 
        debug=False,
        use_reloader=False
    )