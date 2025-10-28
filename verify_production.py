"""
Simple production test - Verifies the AI model gives different responses
Run this ONLY for local testing, NOT for deployment
"""
import requests
import time

def test_production_api():
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Production API with Different Legal Scenarios")
    print("=" * 60)
    
    # Wait for server to start
    print("⏳ Waiting for server to initialize...")
    time.sleep(3)
    
    # Test cases with different legal situations
    test_cases = [
        {
            "name": "Assault Case",
            "query": "My enemy hit me and I got injured"
        },
        {
            "name": "Theft Case", 
            "query": "Someone stole my bag with phone and money"
        },
        {
            "name": "Cybercrime Case",
            "query": "My Instagram account was hacked"
        }
    ]
    
    # Health check first
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ API Status: {health['status']}")
            print(f"📚 Documents: {health['documents_count']}")
            print(f"🤖 AI Ready: {health['ai_ready']}")
            print()
        else:
            print("❌ Health check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("🔧 Make sure to run: python production_api.py")
        return
    
    # Test different scenarios
    for i, test_case in enumerate(test_cases, 1):
        print(f"🔍 Test {i}: {test_case['name']}")
        print(f"📝 Query: {test_case['query']}")
        
        try:
            payload = {"message": test_case['query']}
            response = requests.post(f"{base_url}/api/legal-advice", json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                response_preview = data['response'][:200].replace('\n', ' ')
                
                print(f"✅ Status: Success")
                print(f"🎯 Detected Crimes: {data.get('detected_crimes', [])}")
                print(f"📊 Confidence: {data.get('confidence', 0):.2f}")
                print(f"💬 Response Preview: {response_preview}...")
                print(f"🤖 AI Powered: {data.get('ai_powered', False)}")
                
                # Check if response is specific (not generic)
                if any(keyword in data['response'].lower() for keyword in ['assault', 'theft', 'cybercrime', 'hack']):
                    print("✅ Response is SPECIFIC (not generic template)")
                else:
                    print("⚠️  Response might be generic")
                    
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
        
        print("-" * 40)
        print()

if __name__ == "__main__":
    test_production_api()