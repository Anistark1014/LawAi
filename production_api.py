#!/usr/bin/env python3
"""
Legal AI API Server - Production Ready for Render.com
Lightweight, fast, and optimized for cloud deployment
"""
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Configure CORS for production
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"], 
     allow_headers=["Content-Type", "Authorization"])

# Legal Knowledge Base - Comprehensive Indian Law Coverage
legal_kb = {
    "cybercrime": {
        "hacking": {
            "law": "Section 66 of Information Technology Act, 2000",
            "punishment": "Imprisonment up to 3 years or fine up to Rs. 5 lakh",
            "advice": "Report to Cyber Crime Police Station immediately. Preserve evidence including screenshots, logs, and system information. Change all passwords and enable 2FA on all accounts. File FIR within 24 hours for best results.",
            "helpline": "Cybercrime helpline: 1930"
        },
        "identity_theft": {
            "law": "Section 66C of IT Act 2000",
            "punishment": "Imprisonment up to 3 years and fine up to Rs. 1 lakh",
            "advice": "File FIR at nearest police station immediately. Contact banks and credit agencies. Monitor all financial accounts. Report to cybercrime.gov.in portal. Keep all evidence of identity misuse.",
            "helpline": "Banking fraud: 1930, RBI helpline: 14440"
        },
        "online_fraud": {
            "law": "Section 66D of IT Act 2000 & Section 420 IPC",
            "punishment": "Imprisonment up to 3 years and fine up to Rs. 1 lakh",
            "advice": "Report immediately to cybercrime.gov.in. File FIR with transaction details. Contact bank to freeze accounts if needed. Keep all evidence including emails, messages, transaction receipts.",
            "helpline": "Cyber fraud: 1930"
        }
    },
    "consumer_protection": {
        "defective_products": {
            "law": "Consumer Protection Act 2019",
            "remedy": "Refund, replacement, or compensation for damages",
            "advice": "File complaint at District Consumer Forum within 2 years. Keep purchase receipt, warranty card, and product photos. Consumer can claim compensation for mental agony up to Rs. 1 lakh.",
            "procedure": "File online at edaakhil.nic.in or visit consumer forum"
        },
        "service_deficiency": {
            "law": "Consumer Protection Act 2019",
            "remedy": "Compensation including mental agony and harassment",
            "advice": "Document all interactions with service provider. Keep bills, emails, and complaint numbers. File complaint within 2 years. Evidence required: bills, correspondence, witness statements.",
            "procedure": "District Consumer Forum for claims up to Rs. 1 crore"
        }
    },
    "family_law": {
        "divorce": {
            "law": "Hindu Marriage Act 1955, Muslim Personal Law, Indian Christian Marriage Act 1872",
            "grounds": "Cruelty, desertion, adultery, conversion, mental disorder",
            "advice": "Consult family court lawyer. Gather evidence of grounds for divorce. Consider mediation first. Maintenance and child custody issues need separate consideration.",
            "procedure": "File petition in family court with jurisdiction"
        },
        "domestic_violence": {
            "law": "Protection of Women from Domestic Violence Act 2005",
            "remedy": "Protection order, residence order, maintenance, compensation",
            "advice": "File complaint with Magistrate immediately. Seek medical treatment and keep records. Contact Protection Officer. Available remedies include staying in matrimonial home and financial support.",
            "helpline": "Women helpline: 181, Domestic violence: 1091"
        }
    },
    "property_law": {
        "property_disputes": {
            "law": "Transfer of Property Act 1882, Registration Act 1908",
            "remedy": "Declaration of title, possession recovery, damages",
            "advice": "Ensure proper title documents and registration. File civil suit for title declaration. Get property survey done. Check encumbrance certificate. Limitation period is usually 12 years.",
            "procedure": "Civil court with territorial jurisdiction"
        },
        "tenant_issues": {
            "law": "State Rent Control Acts (varies by state)",
            "rights": "Protection against arbitrary eviction, fair rent",
            "advice": "Check state-specific rent control laws. Tenants cannot be evicted without proper notice and grounds. Landlords can evict for non-payment, unauthorized subletting, or personal necessity.",
            "procedure": "Rent Controller or Civil Court depending on state"
        }
    }
}

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Legal AI API - Powered by Indian Law Knowledge Base",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "legal_advice": "/legal-advice (POST)",
            "case_analysis": "/analyze (POST)"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "message": "Legal AI API is running smoothly",
        "server": "Render.com",
        "legal_areas": list(legal_kb.keys())
    })

@app.route('/legal-advice', methods=['POST', 'OPTIONS'])
def get_legal_advice():
    if request.method == 'OPTIONS':
        return jsonify({"status": "OK"})
    
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "No message provided"}), 400
        
        user_message = data['message'].lower()
        
        # Enhanced keyword matching with multiple legal areas
        response = "Based on your legal query, here's the relevant information:\n\n"
        legal_areas = []
        confidence = 0.5
        next_steps = []
        
        # Cybercrime detection
        if any(word in user_message for word in ['hack', 'hacked', 'cyber attack', 'unauthorized access', 'computer breach']):
            crime_info = legal_kb["cybercrime"]["hacking"]
            response += f"**CYBERCRIME - HACKING**\n"
            response += f"• **Law**: {crime_info['law']}\n"
            response += f"• **Punishment**: {crime_info['punishment']}\n"
            response += f"• **Immediate Action**: {crime_info['advice']}\n"
            response += f"• **Helpline**: {crime_info['helpline']}\n\n"
            legal_areas.append("cybercrime")
            confidence = 0.9
            next_steps.extend(["File FIR immediately", "Preserve digital evidence", "Contact cyber crime cell"])
            
        elif any(word in user_message for word in ['identity theft', 'identity stolen', 'fake identity', 'impersonation']):
            crime_info = legal_kb["cybercrime"]["identity_theft"]
            response += f"**CYBERCRIME - IDENTITY THEFT**\n"
            response += f"• **Law**: {crime_info['law']}\n"
            response += f"• **Punishment**: {crime_info['punishment']}\n"
            response += f"• **Action Required**: {crime_info['advice']}\n"
            response += f"• **Helplines**: {crime_info['helpline']}\n\n"
            legal_areas.append("cybercrime")
            confidence = 0.9
            next_steps.extend(["File police complaint", "Alert banks", "Monitor credit reports"])
            
        else:
            response += """**GENERAL LEGAL GUIDANCE**

**🏛️ KNOW YOUR LEGAL RIGHTS:**
• Constitution of India guarantees fundamental rights (Articles 14-32)
• Right to legal aid under Article 39A
• Right to speedy trial and fair hearing

**📋 DOCUMENTATION CHECKLIST:**
• Keep all relevant documents, receipts, agreements
• Maintain timeline of events with dates
• Collect witness contact information
• Take photographs/screenshots as evidence"""

            legal_areas.append("general_guidance")
            confidence = 0.7
            next_steps = [
                "Consult with qualified lawyer for personalized advice",
                "Gather and organize all relevant documents",
                "Research applicable laws and time limitations"
            ]

        return jsonify({
            "response": response,
            "confidence": confidence,
            "legal_areas": legal_areas,
            "next_steps": next_steps,
            "urgency_level": "high" if any(word in user_message for word in ['urgent', 'emergency', 'immediate']) else "medium"
        })
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_legal_case():
    if request.method == 'OPTIONS':
        return jsonify({"status": "OK"})
    
    try:
        data = request.get_json()
        if not data or 'case_details' not in data:
            return jsonify({"error": "No case details provided"}), 400
        
        case_details = data['case_details'].lower()
        
        urgency = "medium"
        case_type = "general"
        
        if any(word in case_details for word in ['urgent', 'emergency', 'immediate']):
            urgency = "high"
        elif any(word in case_details for word in ['cyber', 'hack', 'online']):
            case_type = "cybercrime"
        elif any(word in case_details for word in ['consumer', 'product']):
            case_type = "consumer_protection"
        
        return jsonify({
            "urgency": urgency,
            "case_type": case_type,
            "recommended_actions": [
                "Consult with a specialist lawyer",
                "Document all relevant evidence",
                "Research applicable laws"
            ]
        })
        
    except Exception as e:
        return jsonify({"error": f"Analysis error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)