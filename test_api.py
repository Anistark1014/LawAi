"""
Test script for the AI Justice Bot API
"""
import requests
import json

def test_api():
    base_url = "http://localhost:5000"
    
    print("Testing AI Justice Bot API...")
    print("=" * 50)
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/api/health")
        print(f"Health Check: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Test 2: Legal advice endpoint with a hacking query
    test_message = "Someone hacked my email account and is using it to send spam messages"
    try:
        payload = {
            "message": test_message,
            "history": [],
            "language": "English"
        }
        response = requests.post(f"{base_url}/api/legal-advice", json=payload)
        print(f"Legal Advice Test: {response.status_code}")
        data = response.json()
        print(f"Type: {data.get('type')}")
        print(f"Crime Type: {data.get('crime_type')}")
        print(f"Response Preview: {data.get('response')[:200]}...")
        print()
    except Exception as e:
        print(f"Legal advice test failed: {e}")
    
    # Test 3: Test with fraud query
    test_message2 = "I was scammed online, someone took my money through fake payment"
    try:
        payload = {
            "message": test_message2,
            "history": [],
            "language": "English"
        }
        response = requests.post(f"{base_url}/api/legal-advice", json=payload)
        print(f"Fraud Test: {response.status_code}")
        data = response.json()
        print(f"Type: {data.get('type')}")
        print(f"Crime Type: {data.get('crime_type')}")
        print(f"Sections: {data.get('sections')}")
        print()
    except Exception as e:
        print(f"Fraud test failed: {e}")
    
    # Test 4: List available crimes
    try:
        response = requests.get(f"{base_url}/api/crimes")
        print(f"Crimes List: {response.status_code}")
        data = response.json()
        print(f"Available crimes: {len(data.get('crimes', []))}")
        for crime in data.get('crimes', [])[:2]:  # Show first 2
            print(f"  - {crime['name']} (keywords: {', '.join(crime['keywords'])})")
        print()
    except Exception as e:
        print(f"Crimes list test failed: {e}")

if __name__ == "__main__":
    test_api()