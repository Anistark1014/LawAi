"""
ML Inference API for Trained Legal Model
This serves the trained legal model via REST API for your React website
"""

import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React

class LegalModelInference:
    def __init__(self, model_path="./trained_legal_model"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """Load the trained legal model and tokenizer"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model not found at {self.model_path}")
                logger.error("Please run 'python train_legal_model.py' first to train the model")
                return False
                
            logger.info(f"Loading trained legal model from {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def generate_response(self, user_query, max_length=300, temperature=0.7):
        """Generate response using the trained legal model"""
        if not self.model or not self.tokenizer:
            return "Error: Model not loaded. Please train the model first."
        
        try:
            # Format input as conversation
            input_text = f"User: {user_query}\nAssistant:"
            
            # Tokenize input
            inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            response_parts = full_response.split("Assistant:")
            if len(response_parts) > 1:
                response = response_parts[-1].strip()
                # Clean up the response
                response = response.split("User:")[0].strip()  # Remove any follow-up user text
                return response
            else:
                return "I need more information to provide accurate legal guidance."
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

# Initialize the model inference
legal_model = LegalModelInference()

@app.route('/')
def home():
    """API information page"""
    return jsonify({
        "service": "AI Justice Bot - Trained Legal Model API",
        "status": "running",
        "model_loaded": legal_model.model is not None,
        "endpoints": {
            "POST /api/legal-advice": "Main endpoint for legal queries",
            "GET /api/health": "Health check",
            "GET /api/model-info": "Model information"
        },
        "usage": {
            "endpoint": "/api/legal-advice",
            "method": "POST",
            "payload": {
                "message": "Your legal question here",
                "max_length": 300,
                "temperature": 0.7
            }
        }
    })

@app.route('/api/legal-advice', methods=['POST'])
def legal_advice():
    """Main endpoint for your React website"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        # Optional parameters
        max_length = data.get('max_length', 300)
        temperature = data.get('temperature', 0.7)
        
        # Check if model is loaded
        if not legal_model.model:
            return jsonify({
                "error": "Model not loaded",
                "message": "Please run 'python train_legal_model.py' first to train the model",
                "status": "error"
            }), 500
        
        # Generate response using trained model
        response = legal_model.generate_response(
            user_message, 
            max_length=max_length, 
            temperature=temperature
        )
        
        return jsonify({
            "response": response,
            "model": "trained_legal_model",
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "query": user_message
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
        "status": "healthy",
        "service": "AI Justice Bot - Trained Model API",
        "model_loaded": legal_model.model is not None,
        "device": str(legal_model.device),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if not legal_model.model:
        return jsonify({
            "model_loaded": False,
            "message": "No model loaded. Please train the model first."
        })
    
    return jsonify({
        "model_loaded": True,
        "model_path": legal_model.model_path,
        "device": str(legal_model.device),
        "model_type": "Legal Document Trained Model",
        "parameters": {
            "max_length": "300 (default)",
            "temperature": "0.7 (default)"
        }
    })

# Legacy endpoint for backward compatibility
@app.route('/chat', methods=['POST'])
def chat():
    """Legacy chat endpoint"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
        
    user_message = data.get('message', '')
    if not legal_model.model:
        return jsonify({"response": "Model not loaded. Please train the model first."})
    
    response = legal_model.generate_response(user_message)
    return jsonify({"response": response})

if __name__ == '__main__':
    print("\n🏛️ AI Justice Bot - Trained Legal Model API")
    print("=" * 50)
    
    if legal_model.model:
        print("✅ Legal model loaded successfully!")
        print(f"🔧 Device: {legal_model.device}")
        print(f"📁 Model path: {legal_model.model_path}")
    else:
        print("❌ Model not loaded!")
        print("📋 To train the model:")
        print("   1. Add your legal PDF documents to 'legal_documents' folder")
        print("   2. Run: python train_legal_model.py")
        print("   3. Then restart this API server")
    
    print("\n🚀 API Endpoints for your React website:")
    print("   POST /api/legal-advice  (main endpoint)")
    print("   GET  /api/health        (health check)")
    print("   GET  /api/model-info    (model details)")
    
    print(f"\n🌐 Server starting on http://localhost:5000")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)