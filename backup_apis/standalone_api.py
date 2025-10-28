"""
AI Justice Bot - Standalone API Endpoint
This provides a REST API for legal guidance without external LLM dependencies.
Your React frontend can POST to /api/legal-advice with user queries.
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import re
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Simple knowledge base of cyber crimes and responses
CYBER_CRIMES_DB = {
    "online_fraud": {
        "keywords": ["fraud", "scam", "fake", "money", "payment", "cheat", "deceive"],
        "crime_type": "Online Financial Fraud",
        "sections": ["IT Act Section 66D", "IPC Section 420"],
        "punishment": "Up to 3 years imprisonment and fine up to Rs. 1 lakh",
        "process": [
            "File FIR at local police station",
            "Report to Cyber Crime Cell", 
            "Preserve all transaction records",
            "Report to bank/payment gateway"
        ]
    },
    "identity_theft": {
        "keywords": ["identity", "personal", "details", "stolen", "misuse", "impersonation"],
        "crime_type": "Identity Theft/Impersonation",
        "sections": ["IT Act Section 66C", "IPC Section 419"],
        "punishment": "Up to 3 years imprisonment and fine up to Rs. 1 lakh",
        "process": [
            "File complaint with Cyber Crime Cell",
            "Report to identity verification agencies",
            "Change passwords and secure accounts",
            "Monitor credit reports"
        ]
    },
    "hacking": {
        "keywords": ["hack", "account", "unauthorized", "access", "breach", "login"],
        "crime_type": "Unauthorized Access/Hacking",
        "sections": ["IT Act Section 66", "IT Act Section 43"],
        "punishment": "Up to 3 years imprisonment and fine up to Rs. 5 lakh",
        "process": [
            "Change all passwords immediately",
            "Enable two-factor authentication",
            "File complaint with Cyber Crime Cell",
            "Preserve logs and evidence"
        ]
    },
    "cyberbullying": {
        "keywords": ["bully", "harass", "threat", "abuse", "intimidation", "troll"],
        "crime_type": "Cyberbullying/Online Harassment", 
        "sections": ["IT Act Section 67", "IPC Section 507"],
        "punishment": "Up to 3 years imprisonment and fine",
        "process": [
            "Screenshot/preserve evidence",
            "Block the harasser",
            "Report to platform administrators",
            "File complaint with local police"
        ]
    },
    "data_theft": {
        "keywords": ["data", "information", "stolen", "leaked", "privacy", "breach"],
        "crime_type": "Data Theft/Privacy Violation",
        "sections": ["IT Act Section 72", "IT Act Section 43A"],
        "punishment": "Up to 3 years imprisonment and fine up to Rs. 5 lakh",
        "process": [
            "Report to Data Protection Officer",
            "File complaint with Cyber Crime Cell",
            "Notify affected parties",
            "Implement security measures"
        ]
    },
    "assault": {
        "keywords": ["hit", "beat", "beaten", "attack", "hurt", "violence", "enemy", "fight", "punch", "kick"],
        "crime_type": "Physical Assault",
        "sections": ["IPC Section 323 (Simple Hurt)", "IPC Section 325 (Grievous Hurt)"],
        "punishment": "Simple Hurt: Up to 1 year imprisonment or fine up to Rs. 1000 | Grievous Hurt: Up to 7 years imprisonment",
        "process": [
            "Get immediate medical attention if injured",
            "File FIR at nearest police station immediately",
            "Preserve medical records as evidence",
            "Get witness statements if available",
            "Take photographs of injuries"
        ]
    },
    "theft": {
        "keywords": ["stole", "stolen", "theft", "bag", "purse", "wallet", "phone", "laptop", "money", "robbed", "took"],
        "crime_type": "Theft/Robbery",
        "sections": ["IPC Section 378 (Theft)", "IPC Section 392 (Robbery)"],
        "punishment": "Theft: Up to 3 years imprisonment or fine | Robbery: Up to 10 years imprisonment",  
        "process": [
            "File FIR at nearest police station immediately",
            "Provide detailed list of stolen items with values",
            "Check for CCTV footage in the area",
            "Block all stolen cards/phones immediately",
            "Contact insurance company if items were insured"
        ]
    }
}

def analyze_query(user_message):
    """Analyze user query and determine the most likely cyber crime type."""
    message_lower = user_message.lower()
    
    # Score each crime type based on keyword matches
    scores = {}
    for crime_type, info in CYBER_CRIMES_DB.items():
        score = 0
        for keyword in info["keywords"]:
            if keyword in message_lower:
                score += 1
        scores[crime_type] = score
    
    # Return the crime type with highest score, or None if no matches
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return None

def generate_response(user_message, conversation_history=None):
    """Generate structured legal guidance based on user query."""
    
    crime_type = analyze_query(user_message)
    
    if not crime_type:
        return {
            "type": "CLARIFY",
            "response": "I need more specific details to help you better. Could you tell me:\n- What exactly happened?\n- Which platform or service was involved?\n- When did this occur?\n- What kind of loss or damage occurred?"
        }
    
    crime_info = CYBER_CRIMES_DB[crime_type]
    
    # Generate structured guidance
    guidance = f"""GUIDE: Based on your description, this appears to be a case of {crime_info['crime_type']}.

**Type of Cyber Crime:**
{crime_info['crime_type']}

**Applicable Laws:**
- {' | '.join(crime_info['sections'])}

**Potential Consequences & Punishment:**
{crime_info['punishment']}

**Immediate Steps for You:**
"""
    
    for i, step in enumerate(crime_info['process'], 1):
        guidance += f"\n{i}. {step}"
    
    guidance += "\n\n**Additional Advice:**\n- Keep all evidence (screenshots, emails, transaction records)\n- Do not delay in reporting\n- Consider consulting a cyber law advocate"
    
    return {
        "type": "GUIDE", 
        "response": guidance,
        "crime_type": crime_info['crime_type'],
        "sections": crime_info['sections'],
        "punishment": crime_info['punishment']
    }

# API Endpoints

@app.route('/')
def home():
    """Basic info page about the API."""
    return render_template('api_info.html')

@app.route('/api/legal-advice', methods=['POST'])
def legal_advice():
    """Main endpoint for legal advice - designed for React frontend."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
            
        conversation_history = data.get('history', [])
        language = data.get('language', 'English')
        
        # Generate response
        result = generate_response(user_message, conversation_history)
        
        # Add metadata
        result.update({
            "timestamp": datetime.now().isoformat(),
            "language": language,
            "status": "success"
        })
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
            "status": "error"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for React frontend."""
    return jsonify({
        "status": "healthy",
        "service": "AI Justice Bot API",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/crimes', methods=['GET'])  
def list_crimes():
    """List available crime types for React frontend."""
    crimes = []
    for crime_type, info in CYBER_CRIMES_DB.items():
        crimes.append({
            "id": crime_type,
            "name": info["crime_type"],
            "keywords": info["keywords"][:3]  # First 3 keywords only
        })
    return jsonify({"crimes": crimes})

# Legacy endpoint for backward compatibility with the existing template
@app.route('/chat', methods=['POST'])
def chat():
    """Legacy chat endpoint - redirects to new API."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
        
    user_message = data.get('message', '')
    result = generate_response(user_message)
    
    return jsonify({"response": result["response"]})

if __name__ == '__main__':
    print("Starting AI Justice Bot API Server...")
    print("API Endpoints:")
    print("- POST /api/legal-advice (main endpoint for React)")
    print("- GET /api/health (health check)")
    print("- GET /api/crimes (list crime types)")
    print("- POST /chat (legacy endpoint)")
    print("\nServer running on http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)