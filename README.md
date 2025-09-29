AI Justice Bot: Multilingual Legal Assistant for Indian Cybercrime Law
AI Justice Bot is an advanced, conversational AI agent designed to help users understand the complexities of Indian cybercrime law. It uses a Retrieval-Augmented Generation (RAG) architecture to provide accurate, context-aware answers based on a custom knowledge base of legal documents. The agent can interact in English, Hindi, and Marathi, making legal information more accessible.

🌟 Key Features
Conversational Q&A: Engages in a natural, multi-turn dialogue, asking clarifying questions to precisely understand the user's situation.

Multilingual Support: Automatically detects and responds in the user's language (English, Hindi, or Marathi).

Retrieval-Augmented Generation (RAG): Provides factual answers grounded in your specific legal documents, preventing the AI from making up information (hallucination).

Customizable Knowledge Base: Easily update the AI's knowledge by adding your own PDFs, DOCX, or TXT files (like judgments or legal acts) to a folder.

Structured & Actionable Guidance: Delivers a final, detailed response that includes:

The specific type of cybercrime.

Applicable laws from the IT Act, 2000 and Indian Penal Code (IPC).

Potential consequences and punishments for the suspect.

A step-by-step guide for the victim to follow.

Automated Knowledge Expansion: Includes a web scraper to fetch new judgments from sources like Indian Kanoon to continuously grow the knowledge base.

🤖 System Architecture: How It Works
The project uses a RAG pipeline, which separates the "knowledge" from the "reasoning." This makes the system both powerful and reliable.

Think of it like an expert legal team: a diligent paralegal and a brilliant senior lawyer.

1. The Retrieval System (The "Paralegal")
This is your local system that is an expert on your documents.

Sentence Transformer (paraphrase-multilingual-mpnet-base-v2): This model reads all your legal documents and converts their semantic meaning into numerical vectors (embeddings).

Vector Database (FAISS): This is the high-speed library where all the document vectors are stored. When you ask a question, it instantly finds the most relevant document chunks—acting as the perfect, super-fast memory.

2. The Generation System (The "Senior Lawyer")
This is the powerful LLM that handles communication and reasoning.

LLM (Google Gemini): It receives the relevant documents found by the Retrieval System. It has no prior knowledge of your files but is an expert at synthesizing information.

Prompt Engineering: A carefully designed set of instructions guides the LLM to analyze the context, ask clarifying questions, and structure the final answer in the required legal format.

🛠️ Tech Stack
Python 3.8+

AI & NLP:

sentence-transformers: For creating multilingual text embeddings.

langchain: For text processing and chunking.

transformers: For running open-source LLMs locally (optional).

Vector Database: faiss-cpu

Web Scraping: requests, beautifulsoup4

File Handling: pypdf2, python-docx

API Interaction: httpx, python-dotenv

🚀 Getting Started
Follow these steps to set up and run the project on your local machine.

Prerequisites
Python 3.8 or newer

Git installed on your system

1. Clone the Repository
git clone [https://github.com/AmarJogdand/AI-Justice_Bot.git](https://github.com/AmarJogdand/AI-Justice_Bot.git)
cd AI-Justice_Bot

2. Set Up a Virtual Environment (Recommended)
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Set Up the API Key
Create a file named .env in the main project folder.

Get your API key from Google AI Studio.

Add your key to the .env file like this:

GEMINI_API_KEY="YOUR_KEY_HERE"

5. Populate the Knowledge Base
The repository includes a judgments folder.

Add your legal documents (PDFs, DOCX, or TXT files) into this folder. The more high-quality documents you add, the smarter the agent will be.

⚙️ How to Use the Project
The project is run in two main steps.

Step 1: Build the Vector Database
Before you can chat, you must process your documents. This script reads everything in the judgments folder and creates the AI's "brain."

python main.py

This will create cyber_law_index.bin and cyber_law_chunks.pkl. You only need to re-run this when you add or change the documents in the judgments folder.

Step 2: Run the Conversational Agent
Once the database is built, you can start the AI assistant.

python search.py

The application will start, ask for your language preference, and then you can begin describing your situation.

Step 3 (Optional): Scrape New Documents
You can add new judgments from Indian Kanoon to your knowledge base using the scraper.

python scraper.py
