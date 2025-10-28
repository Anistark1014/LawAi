"""
Test with a more specific legal question
"""
import requests
import json

def test_specific_question():
    base_url = "http://localhost:5000"
    
    # More specific question about IT Act sections
    question = "What are the penalties under IT Act Section 66 for unauthorized access to computer systems?"
    
    print("🏛️ Testing Specific Legal Question")
    print("=" * 50)
    print(f"📝 QUESTION: {question}")
    print("\n🤖 AI RESPONSE:")
    print("-" * 50)
    
    try:
        payload = {"message": question}
        response = requests.post(f"{base_url}/api/legal-advice", json=payload, timeout=20)
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ LEGAL GUIDANCE:")
            print(data.get('response', 'No response'))
            
            print(f"\n🎯 CONFIDENCE: {data.get('confidence', 0)*100:.1f}%")
            
            if data.get('detected_issues'):
                print(f"🚨 ISSUES: {', '.join(data.get('detected_issues', []))}")
            
            sources = data.get('sources', [])
            print(f"\n📚 SOURCES ({len(sources)}):")
            for source in sources:
                print(f"   • {source['source']} ({source['similarity']*100:.1f}% relevant)")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_specific_question()