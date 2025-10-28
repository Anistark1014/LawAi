import os
import httpx
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

async def list_available_models():
    """
    Calls the Gemini API to list all available models for your key.
    """
    apiKey = os.getenv("GEMINI_API_KEY")
    if not apiKey:
        print("ERROR: GEMINI_API_KEY not found in .env file.")
        return

    # This is the endpoint to list models
    list_models_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={apiKey}"
    
    print("Fetching available models...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(list_models_url)
            response.raise_for_status()
            data = response.json()
            
            print("\n--- Models Available for Your API Key ---")
            for model in data.get('models', []):
                # Check if the 'generateContent' method is supported
                is_supported = any(
                    method == 'generateContent' for method in model.get('supportedGenerationMethods', [])
                )
                if is_supported:
                    print(f"- {model['name']} (Display Name: {model.get('displayName')})")
            print("-----------------------------------------")

        except httpx.HTTPStatusError as e:
            print(f"API Error: {e.response.status_code} {e.response.text}")
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(list_available_models())

    
