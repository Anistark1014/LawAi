#!/usr/bin/env python3
"""
Test the new API with the failing queries
"""
import requests
import json

def test_query(query, description):
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {description}")
    print(f"Query: '{query}'")
    print(f"{'='*60}")
    
    try:
        response = requests.post(
            'http://localhost:5000/api/legal-advice',
            json={'message': query},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS!")
            print(f"Response: {data.get('response', '')[:300]}...")
            print(f"Crime Type: {data.get('crime_type', 'Not detected')}")
            print(f"Sections: {data.get('sections', [])}")
        else:
            print(f"❌ ERROR: Status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")

if __name__ == "__main__":
    print("🏛️ AI Justice Bot - Fixed API Test")
    
    # Test the failing queries
    test_query("My enemy hit me", "Assault Case")
    test_query("someone stole my bag", "Theft Case") 
    test_query("someone hacked my account", "Cybercrime Case")
    
    print(f"\n{'='*60}")
    print("🎉 Test Complete!")