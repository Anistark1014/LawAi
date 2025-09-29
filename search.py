import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import asyncio
import os
import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 1. --- Load the saved components ---
print("Loading the RAG model components...")
try:
    retrieval_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    index = faiss.read_index("cyber_law_index.bin")
    with open("cyber_law_chunks.pkl", "rb") as f:
        all_chunks = pickle.load(f)
    print("Retrieval components loaded.")
except FileNotFoundError:
    print("\nERROR: Database files not found. Please run 'python main.py' first.")
    exit()

# 2. --- The RAG retrieval function ---
def retrieve_from_rag(query, k=10): # Increased k to 10 for more context
    query_embedding = retrieval_model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding), k)
    results = []
    seen_content = set()
    for i, idx in enumerate(indices[0]):
        if idx < len(all_chunks):
            chunk_content = all_chunks[idx]['content']
            if chunk_content not in seen_content:
                results.append({"chunk": all_chunks[idx], "distance": distances[0][i]})
                seen_content.add(chunk_content)
    return results

# 3. --- The Gemini Generation Function ---
async def get_gemini_response(conversation_history, context_chunks, language_choice):
    """Uses the Gemini API to synthesize an answer, emphasizing RAG."""
    print("\nAnalyzing retrieved context and preparing response...")
    
    # --- THIS IS THE KEY CHANGE FOR GITHUB ---
    # Safely load the API key from the .env file
    apiKey = os.getenv("GEMINI_API_KEY")
    if not apiKey:
        print("ERROR: GEMINI_API_KEY not found in .env file. Please create it and add your key.")
        return "API Key not configured."
    # -----------------------------------------

    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={apiKey}"

    context_str = "\n".join([f"Source Document Snippet:\n\"{chunk['chunk']['content']}\"\n" for chunk in context_chunks])

    prompt = f"""
    You are a highly precise AI legal assistant for the Indian Judiciary. Your ONLY task is to act as a summarizer and synthesizer of the provided legal context. You MUST NOT use any external knowledge. Your entire response must be derived directly from the "Retrieved Legal Context" below.

    **Conversation History:**
    {conversation_history}

    **Retrieved Legal Context (Your ONLY source of information):**
    ---
    {context_str}
    ---

    **Your Task:**
    Analyze the retrieved context to answer the user's latest query.
    1.  If the context does not contain enough information to form a conclusive answer, ask a single, clarifying question to get the missing details. Start your response with "CLARIFY:".
    2.  If the context is sufficient, provide a detailed, structured legal guidance. Start your response with "GUIDE:". Your guidance MUST ONLY contain information present in the retrieved context. The structure must be:
        - **Type of Cyber Crime:** (Identify the crime based on the context)
        - **Applicable Laws:** (List the specific IT Act and IPC sections mentioned in the context)
        - **Potential Consequences & Punishment:** (Detail the punishments as described in the context)
        - **Further Process for the Victim:** (Outline the steps for reporting as described in the context)

    **CRITICAL RULES:**
    - Your entire response MUST be in the {language_choice} language.
    - If the user's latest message is a command like "stop", "answer now", or "give me the answer", you MUST skip asking more questions and immediately provide a final GUIDE response based on the information you have.
    """

    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(apiUrl, json=payload)
            response.raise_for_status()
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        except httpx.HTTPStatusError as e:
            return f"API Error: {e.response.status_code} {e.response.text}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"

# 4. --- Main execution block ---
async def main():
    print("\nPlease select your preferred language:")
    print("1: English")
    print("2: हिन्दी (Hindi)")
    print("3: मराठी (Marathi)")
    
    lang_map = {"1": "English", "2": "Hindi", "3": "Marathi"}
    choice = input("Enter the number for your choice (1/2/3): ")
    
    language_choice = lang_map.get(choice, "English")
    print(f"Language set to: {language_choice}")
    
    conversation_history = ""
    print("\n--- Cyber Crime Legal Assistant ---")
    
    initial_prompt = input(f"Please describe your situation in {language_choice}: ")
    conversation_history += f"User: {initial_prompt}\n"

    while True:
        retrieved_chunks = retrieve_from_rag(conversation_history)
        
        ai_response = await get_gemini_response(conversation_history, retrieved_chunks, language_choice)
        
        if ai_response.strip().startswith("CLARIFY:"):
            clarifying_question = ai_response.replace("CLARIFY:", "").strip()
            print(f"\nAssistant: {clarifying_question}")
            conversation_history += f"Assistant: {clarifying_question}\n"
            
            user_answer = input("Your answer: ")
            conversation_history += f"User: {user_answer}\n"
        
        elif ai_response.strip().startswith("GUIDE:"):
            final_guidance = ai_response.replace("GUIDE:", "").strip()
            print("\n--- Legal Guidance ---")
            print(final_guidance)
            print("----------------------")
            print("\nDisclaimer: This is AI-generated guidance and not a substitute for professional legal advice.")
            break
        else:
            print(f"\nAssistant: {ai_response}")
            break

if __name__ == "__main__":
    asyncio.run(main())
