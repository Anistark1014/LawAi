"""
Test the trained legal model API
"""
import requests
import json

def test_trained_legal_api():
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Trained Legal Model API")
    print("=" * 50)
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/api/health")
        print(f"✅ Health Check: {response.status_code}")
        data = response.json()
        print(f"   Model Loaded: {data.get('model_loaded')}")
        print(f"   Documents: {data.get('documents_count')}")
        print()
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test 2: Legal advice - Hacking case
    print("🔍 Test Case 1: Hacking")
    try:
        payload = {
            "message": "Someone hacked my email account and is sending spam messages to my contacts"
        }
        response = requests.post(f"{base_url}/api/legal-advice", json=payload)
        data = response.json()
        
        print(f"Status: {response.status_code}")
        print(f"Response: {data.get('response')[:300]}...")
        print(f"Confidence: {data.get('confidence', 0):.2f}")
        print(f"Detected Issues: {data.get('detected_issues', [])}")
        print(f"Sources: {len(data.get('sources', []))} documents")
        print()
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    # Test 3: Legal advice - Fraud case
    print("🔍 Test Case 2: Online Fraud")
    try:
        payload = {
            "message": "I was cheated in an online payment, someone took my money through a fake website"
        }
        response = requests.post(f"{base_url}/api/legal-advice", json=payload)
        data = response.json()
        
        print(f"Status: {response.status_code}")
        print(f"Response: {data.get('response')[:300]}...")
        print(f"Confidence: {data.get('confidence', 0):.2f}")
        print(f"Detected Issues: {data.get('detected_issues', [])}")
        print()
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_trained_legal_api()