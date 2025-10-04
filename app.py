import os
import faiss
import pickle
import httpx
import json
import asyncio
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- 1. Initialize Flask App ---
app = Flask(__name__)

# --- 2. Load RAG Components (Done Once at Startup) ---
print("Loading RAG components...")
try:
    retrieval_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    index = faiss.read_index("cyber_law_index.bin")
    with open("cyber_law_chunks.pkl", "rb") as f:
        all_chunks = pickle.load(f)
    print("All models and database components loaded successfully.")
except FileNotFoundError:
    print("\nFATAL ERROR: Database files (cyber_law_index.bin, cyber_law_chunks.pkl) not found.")
    print("Please run 'python main.py' first to build the database before starting the web server.")
    exit()

# --- 3. Core RAG and LLM Functions ---
def retrieve_from_rag(query, k=10):
    """Searches the vector database and returns the top k most relevant text chunks."""
    query_embedding = retrieval_model.encode([query])
    # Ensure the data type is float32 for FAISS
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k)
    return [{"chunk": all_chunks[idx], "distance": dist} for idx, dist in zip(indices[0], distances[0])]

async def get_gemini_response(conversation_history, context_chunks, language="English"):
    """Gets a response from the Gemini API."""
    apiKey = os.getenv("GEMINI_API_KEY")
    if not apiKey:
        return "ERROR: GEMINI_API_KEY not found. Please check your .env file."
    
    # Use the correct, specific model name
    model_name = "gemini-2.5-flash-preview-05-20"
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={apiKey}"
    
    context_str = "\n".join([f"Context from {res['chunk']['source']}:\n\"{res['chunk']['content']}\"\n---" for res in context_chunks])
    
    prompt = f"""
    You are an expert legal AI assistant for the Indian Judiciary system. Your goal is to help a user identify a potential cyber crime and understand the consequences.

    **Conversation History:**
    {conversation_history}

    **Retrieved Legal Context from Indian Law Documents:**
    ---
    {context_str}
    ---

    **Your Task:**
    1.  Analyze the user's latest message based on the conversation and the retrieved legal context.
    2.  DECIDE: Do you have enough specific information to identify a potential crime?
    3.  ACT:
        * If NO, ask a single, clear clarifying question to get the missing details. Start your response with "CLARIFY:".
        * If YES, provide a detailed, structured legal guidance. Start your response with "GUIDE:". Your guidance MUST be specific and include sections for: Type of Cyber Crime, Applicable Laws (IT Act & IPC), Potential Consequences & Punishment, and Further Process for the Victim.

    **IMPORTANT RULES:**
    - You MUST respond in the chosen language: {language}.
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

# --- 4. Flask Routes (API Endpoints) ---

@app.route("/")
def home(): # Renamed to 'home' to avoid variable conflicts
    """Serves the main HTML page."""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
async def chat():
    """Handles the chat logic."""
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400

    user_message = data.get("message")
    conversation_history = data.get("history", [])
    language = data.get("language", "English")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    retrieved_chunks = retrieve_from_rag(user_message)
    assistant_response = await get_gemini_response(conversation_history, retrieved_chunks, language)
    
    return jsonify({"response": assistant_response})

# --- 5. Run the App ---
if __name__ == "__main__":
    # Ensure the judgments folder exists
    if not os.path.exists('judgments'):
        os.makedirs('judgments')
    
    # Run the web server
    app.run(debug=True, port=5001)

