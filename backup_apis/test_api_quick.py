import requests
import json

# Test the API
try:
    # Test health endpoint
    health_response = requests.get("http://localhost:5000/health")
    print("Health Check:", health_response.json())
    
    # Test legal advice endpoint
    legal_data = {
        "message": "I got hacked and someone stole my personal information"
    }
    
    advice_response = requests.post(
        "http://localhost:5000/legal-advice",
        json=legal_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print("\nLegal Advice Response:")
    print(json.dumps(advice_response.json(), indent=2))
    
except Exception as e:
    print(f"Error testing API: {e}")