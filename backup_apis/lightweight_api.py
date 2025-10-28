#!/usr/bin/env python3
"""
Lightweight Legal AI API Server - No Heavy Dependencies
"""
import os
import json
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Store the legal knowledge base
legal_kb = {
    "cybercrime": {
        "hacking": "Under Section 66 of the Information Technology Act, 2000, unauthorized access to computer systems is punishable with imprisonment up to 3 years or fine up to Rs. 5 lakh. For protection: Report to Cyber Crime Police Station immediately, preserve evidence, change passwords, enable 2FA.",
        "identity_theft": "Identity theft falls under Section 66C of IT Act 2000, punishable with imprisonment up to 3 years and fine up to Rs. 1 lakh. Immediate steps: File FIR at nearest police station, contact banks/credit agencies, monitor accounts regularly.",
        "online_fraud": "Online fraud is covered under Section 66D of IT Act 2000 with imprisonment up to 3 years and fine up to Rs. 1 lakh. Also applicable: Section 420 IPC for cheating. Report to cybercrime.gov.in portal immediately."
    },
    "consumer_protection": {
        "defective_products": "Under Consumer Protection Act 2019, consumers can file complaints for defective products. Compensation includes refund, replacement, or damages. File complaint within 2 years at District Consumer Forum.",
        "service_deficiency": "Service deficiency is covered under Consumer Protection Act 2019. Consumers can claim compensation for mental agony and harassment. Evidence required: bills, correspondence, witness statements."
    },
    "family_law": {
        "divorce": "Divorce in India is governed by personal laws. Hindu Marriage Act 1955 provides grounds like cruelty, desertion, adultery. Muslim Personal Law allows divorce through various methods. Consult family court for proceedings.",
        "domestic_violence": "Protection of Women from Domestic Violence Act 2005 provides legal remedies. File complaint with Magistrate, seek protection order, residence order, and compensation. 24x7 helpline: 181."
    },
    "property_law": {
        "property_disputes": "Property disputes are handled under Transfer of Property Act 1882 and Registration Act 1908. File civil suit for declaration of title, possession recovery. Ensure proper documentation and registration.",
        "tenant_issues": "Rent control laws vary by state. Generally, tenants have rights against arbitrary eviction. Landlords can evict for non-payment, subletting without consent. Check local rent control act."
    }
}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Legal AI API is running"})

@app.route('/legal-advice', methods=['POST'])
def get_legal_advice():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "No message provided"}), 400
        
        user_message = data['message'].lower()
        
        # Simple keyword-based matching with legal advice
        response = "I understand you need legal assistance. "
        
        # Check for specific legal issues
        if any(word in user_message for word in ['hack', 'hacked', 'cyber attack', 'computer', 'unauthorized access']):
            response += legal_kb["cybercrime"]["hacking"]
        elif any(word in user_message for word in ['identity theft', 'identity stolen', 'fake identity']):
            response += legal_kb["cybercrime"]["identity_theft"]
        elif any(word in user_message for word in ['online fraud', 'scam', 'fake website', 'phishing']):
            response += legal_kb["cybercrime"]["online_fraud"]
        elif any(word in user_message for word in ['defective product', 'faulty product', 'product issue']):
            response += legal_kb["consumer_protection"]["defective_products"]
        elif any(word in user_message for word in ['service problem', 'bad service', 'service deficiency']):
            response += legal_kb["consumer_protection"]["service_deficiency"]
        elif any(word in user_message for word in ['divorce', 'separation', 'marriage problem']):
            response += legal_kb["family_law"]["divorce"]
        elif any(word in user_message for word in ['domestic violence', 'abuse', 'harassment']):
            response += legal_kb["family_law"]["domestic_violence"]
        elif any(word in user_message for word in ['property dispute', 'property problem', 'land issue']):
            response += legal_kb["property_law"]["property_disputes"]
        elif any(word in user_message for word in ['tenant', 'landlord', 'rent', 'eviction']):
            response += legal_kb["property_law"]["tenant_issues"]
        else:
            response += """Based on your query, here are some general legal guidance steps:

1. **Document Everything**: Keep records of all relevant documents, communications, and evidence.

2. **Know Your Rights**: Research the applicable laws for your situation. Key Indian laws include:
   - Indian Penal Code (IPC) for criminal matters
   - Information Technology Act 2000 for cyber crimes
   - Consumer Protection Act 2019 for consumer issues
   - Civil Procedure Code for civil disputes

3. **Seek Professional Help**: For complex legal matters, consult with a qualified lawyer who specializes in the relevant area of law.

4. **Time Limitations**: Be aware of limitation periods for filing cases (usually 3 years for most civil matters).

5. **Legal Aid**: If you cannot afford a lawyer, you may be eligible for free legal aid through the Legal Services Authority.

For specific legal advice tailored to your situation, please provide more details about your legal issue."""

        return jsonify({
            "response": response,
            "confidence": 0.85,
            "legal_areas": ["general_guidance"],
            "next_steps": [
                "Consult with a qualified lawyer",
                "Gather relevant documents and evidence",
                "Research applicable laws and regulations",
                "Consider alternative dispute resolution if applicable"
            ]
        })
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/analyze', methods=['POST'])
def analyze_legal_case():
    try:
        data = request.get_json()
        if not data or 'case_details' not in data:
            return jsonify({"error": "No case details provided"}), 400
        
        case_details = data['case_details'].lower()
        
        # Simple analysis based on keywords
        urgency = "medium"
        case_type = "general"
        
        if any(word in case_details for word in ['urgent', 'emergency', 'immediate', 'asap']):
            urgency = "high"
        elif any(word in case_details for word in ['when possible', 'sometime', 'eventually']):
            urgency = "low"
            
        if any(word in case_details for word in ['cyber', 'hack', 'online', 'internet']):
            case_type = "cybercrime"
        elif any(word in case_details for word in ['consumer', 'product', 'service']):
            case_type = "consumer_protection"
        elif any(word in case_details for word in ['family', 'marriage', 'divorce', 'domestic']):
            case_type = "family_law"
        elif any(word in case_details for word in ['property', 'land', 'tenant', 'landlord']):
            case_type = "property_law"
        
        return jsonify({
            "urgency": urgency,
            "case_type": case_type,
            "recommended_actions": [
                "Document all relevant evidence",
                "Consult with a specialist lawyer",
                "Research applicable laws",
                "Consider time limitations for legal action"
            ],
            "estimated_complexity": "medium"
        })
        
    except Exception as e:
        return jsonify({"error": f"Analysis error: {str(e)}"}), 500

if __name__ == '__main__':
    print("🚀 Starting Lightweight Legal AI API Server...")
    print("📍 URL: http://localhost:5000")
    print("🔗 Available endpoints:")
    print("   - POST /legal-advice (for legal questions)")
    print("   - POST /analyze (for case analysis)")
    print("   - GET /health (health check)")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0', 
        port=5000, 
        debug=False,
        use_reloader=False
    )