"""
Simple Legal API - Serves the retrieval-based legal model
This is faster and more reliable than transformer training
"""

import os
import json
import pickle
import numpy as np
import faiss
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class SimpleLegalModelAPI:
    def __init__(self, model_dir="simple_legal_model"):
        self.model_dir = model_dir
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
        """Generate legal response based on retrieved documents"""
        relevant_docs = self.retrieve_relevant_docs(query, k=3)
        
        if not relevant_docs:
            return {
                'response': "I couldn't find specific information in the legal documents for your query. Please consult with a legal professional.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Analyze query for cyber crime keywords
        query_lower = query.lower()
        crime_keywords = {
            'hacking': ['hack', 'unauthorized access', 'breach', 'intrusion'],
            'fraud': ['fraud', 'cheat', 'scam', 'deceive', 'fake'],
            'cyberbullying': ['bully', 'harass', 'threat', 'intimidation'],
            'identity_theft': ['identity', 'impersonation', 'stolen details'],
            'data_breach': ['data', 'privacy', 'leak', 'information']
        }
        
        detected_crimes = []
        for crime, keywords in crime_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_crimes.append(crime)
        
        # Build response
        response = "Based on the legal documents, here's what I found:\n\n"
        
        if detected_crimes:
            response += f"**Potential Legal Issues:** {', '.join(detected_crimes).title()}\n\n"
        
        response += "**Relevant Legal Information:**\n"
        sources = []
        
        for i, doc in enumerate(relevant_docs, 1):
            source_name = doc['source'].replace('.PDF', '').replace('_', ' ')
            response += f"\n{i}. **From {source_name}:**\n"
            
            # Get most relevant part of the document
            text = doc['text']
            if len(text) > 400:
                text = text[:400] + "..."
            
            response += f"   {text}\n"
            
            sources.append({
                'source': doc['source'],
                'similarity': doc['similarity'],
                'text_preview': text[:100] + "..."
            })
        
        response += "\n**Important:**\n"
        response += "• This information is based on legal documents in our database\n"
        response += "• For specific legal advice, please consult with a qualified lawyer\n"
        response += "• Consider reporting cyber crimes to local police cyber crime cell\n"
        
        avg_confidence = sum(doc['similarity'] for doc in relevant_docs) / len(relevant_docs)
        
        return {
            'response': response,
            'sources': sources,
            'confidence': float(avg_confidence),
            'detected_issues': detected_crimes
        }

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
        
        # Generate response
        result = legal_api.generate_legal_response(query)
        
        return jsonify({
            "response": result['response'],
            "sources": result['sources'],
            "confidence": result['confidence'],
            "detected_issues": result.get('detected_issues', []),
            "model": "Simple Legal AI",
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "query": query
        })
        
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