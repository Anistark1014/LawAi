#!/usr/bin/env python3
"""
Expand Training Data for Legal AI Model
Add more legal scenarios and real-world questions
"""
import json
import os
from datetime import datetime

def create_expanded_training_data():
    """Create expanded training dataset with real legal scenarios"""
    
    expanded_data = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "description": "Expanded Legal AI Training Data",
            "version": "2.0",
            "total_scenarios": 0
        },
        "training_scenarios": []
    }
    
    # Add real legal scenarios with context-aware responses
    legal_scenarios = [
        # Murder/Violence Cases
        {
            "user_query": "I witnessed a murder last night near my apartment",
            "legal_context": "Murder under IPC Section 302",
            "situation_analysis": {
                "crime_type": "murder",
                "urgency": "critical",
                "evidence_available": True,
                "witness_role": "eyewitness"
            },
            "expected_response_type": "critical_emergency",
            "legal_sections": ["IPC 302", "IPC 161", "Evidence Act"],
            "priority_actions": ["call_police_immediately", "preserve_scene", "witness_statement"]
        },
        
        # Theft Cases
        {
            "user_query": "Someone stole my laptop from office, I saw who did it on CCTV",
            "legal_context": "Theft under IPC Section 378",
            "situation_analysis": {
                "crime_type": "theft",
                "urgency": "high",
                "evidence_available": True,
                "perpetrator_identified": True,
                "location": "workplace"
            },
            "expected_response_type": "immediate_action",
            "legal_sections": ["IPC 378", "IPC 379", "IPC 411"],
            "priority_actions": ["file_fir", "preserve_cctv", "inform_employer"]
        },
        
        # Cybercrime Cases
        {
            "user_query": "My Instagram was hacked, they posted inappropriate photos and demanding 50000 rupees",
            "legal_context": "Cybercrime and Extortion under IT Act and IPC",
            "situation_analysis": {
                "crime_type": "cybercrime_extortion",
                "urgency": "high",
                "financial_demand": True,
                "reputation_damage": True,
                "platform": "social_media"
            },
            "expected_response_type": "cyber_response",
            "legal_sections": ["IT Act 66", "IT Act 66A", "IPC 384", "IPC 509"],
            "priority_actions": ["report_platform", "cybercrime_portal", "preserve_evidence"]
        },
        
        # Domestic Violence
        {
            "user_query": "My husband beats me regularly and demands more dowry money",
            "legal_context": "Domestic Violence and Dowry Harassment",
            "situation_analysis": {
                "crime_type": "domestic_violence_dowry",
                "urgency": "high",
                "physical_harm": True,
                "financial_harassment": True,
                "victim_type": "spouse"
            },
            "expected_response_type": "protection_focused",
            "legal_sections": ["DV Act 2005", "IPC 498A", "Dowry Prohibition Act"],
            "priority_actions": ["women_helpline", "protection_order", "medical_evidence"]
        },
        
        # Fraud Cases
        {
            "user_query": "Someone called pretending to be from my bank and stole 25000 rupees",
            "legal_context": "Financial Fraud and Cheating",
            "situation_analysis": {
                "crime_type": "financial_fraud",
                "urgency": "high",
                "financial_loss": True,
                "method": "phone_fraud",
                "amount": 25000
            },
            "expected_response_type": "financial_recovery",
            "legal_sections": ["IPC 420", "IT Act 66D", "RBI Guidelines"],
            "priority_actions": ["inform_bank", "cyber_complaint", "transaction_dispute"]
        },
        
        # Sexual Harassment
        {
            "user_query": "My boss is sending inappropriate messages and touching me inappropriately",
            "legal_context": "Sexual Harassment at Workplace",
            "situation_analysis": {
                "crime_type": "sexual_harassment",
                "urgency": "high",
                "location": "workplace",
                "evidence_available": True,
                "power_dynamics": True
            },
            "expected_response_type": "workplace_protection",
            "legal_sections": ["POSH Act 2013", "IPC 354", "IPC 509"],
            "priority_actions": ["internal_committee", "evidence_preservation", "legal_notice"]
        },
        
        # Property Disputes
        {
            "user_query": "My neighbor built a wall on my property and refuses to remove it",
            "legal_context": "Property Dispute and Trespass",
            "situation_analysis": {
                "crime_type": "property_dispute",
                "urgency": "moderate",
                "civil_matter": True,
                "documentation_needed": True
            },
            "expected_response_type": "civil_remedy",
            "legal_sections": ["IPC 441", "Transfer of Property Act", "Civil Procedure Code"],
            "priority_actions": ["survey_documents", "legal_notice", "civil_suit"]
        },
        
        # Online Harassment
        {
            "user_query": "Someone is posting my personal photos on fake social media profiles",
            "legal_context": "Cyber Harassment and Privacy Violation",
            "situation_analysis": {
                "crime_type": "cyber_harassment",
                "urgency": "high",
                "privacy_violation": True,
                "reputation_damage": True,
                "platform": "multiple_social_media"
            },
            "expected_response_type": "privacy_protection",
            "legal_sections": ["IT Act 66E", "IPC 509", "Right to Privacy"],
            "priority_actions": ["report_platforms", "cyber_complaint", "takedown_notice"]
        },
        
        # Employment Issues
        {
            "user_query": "My company fired me without notice and is not paying my salary",
            "legal_context": "Employment Law Violation",
            "situation_analysis": {
                "crime_type": "employment_violation",
                "urgency": "moderate",
                "financial_loss": True,
                "labor_rights": True
            },
            "expected_response_type": "labor_remedy",
            "legal_sections": ["Industrial Disputes Act", "Payment of Wages Act", "Labor Laws"],
            "priority_actions": ["labor_commissioner", "legal_notice", "conciliation"]
        },
        
        # Consumer Fraud
        {
            "user_query": "I bought a phone online, they sent a fake product and refuse to refund",
            "legal_context": "Consumer Protection and E-commerce Fraud",
            "situation_analysis": {
                "crime_type": "consumer_fraud",
                "urgency": "moderate",
                "financial_loss": True,
                "e_commerce": True,
                "defective_product": True
            },
            "expected_response_type": "consumer_remedy",
            "legal_sections": ["Consumer Protection Act 2019", "IPC 420", "E-commerce Rules"],
            "priority_actions": ["consumer_forum", "platform_complaint", "refund_claim"]
        }
    ]
    
    expanded_data["training_scenarios"] = legal_scenarios
    expanded_data["metadata"]["total_scenarios"] = len(legal_scenarios)
    
    # Save expanded training data
    with open("expanded_training_data.json", "w", encoding="utf-8") as f:
        json.dump(expanded_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created expanded training data with {len(legal_scenarios)} scenarios")
    print("📁 Saved as: expanded_training_data.json")
    
    return expanded_data

if __name__ == "__main__":
    create_expanded_training_data()