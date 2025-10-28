"""
Lightweight Legal Model Training - Faster Alternative
This uses a simpler approach that should train faster and be more stable
"""

import os
import json
import pickle
from pathlib import Path
import PyPDF2
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleLegalAI:
    def __init__(self, pdf_folder="legal_documents"):
        self.pdf_folder = pdf_folder
        self.documents = []
        self.embeddings = None
        self.sentence_model = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.faiss_index = None
        
    def extract_and_process_documents(self):
        """Extract text from PDFs and process for retrieval"""
        logger.info("Extracting text from legal documents...")
        
        pdf_files = list(Path(self.pdf_folder).glob("*.pdf"))
        if not pdf_files:
            logger.error(f"No PDF files found in {self.pdf_folder}")
            return False
        
        for pdf_path in pdf_files:
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text() + "\n"
                    
                    if text.strip():
                        # Split into chunks
                        chunks = self.split_text_into_chunks(text, max_length=500)
                        for chunk in chunks:
                            if len(chunk.strip()) > 100:
                                self.documents.append({
                                    'source': pdf_path.name,
                                    'text': chunk.strip(),
                                    'length': len(chunk)
                                })
                        
                        logger.info(f"Processed {pdf_path.name}: {len(chunks)} chunks")
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {str(e)}")
        
        logger.info(f"Total documents processed: {len(self.documents)}")
        return len(self.documents) > 0
    
    def split_text_into_chunks(self, text, max_length=500):
        """Split text into manageable chunks"""
        sentences = text.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + "."
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "."
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def build_retrieval_system(self):
        """Build FAISS index for fast retrieval"""
        logger.info("Building retrieval system...")
        
        # Initialize sentence transformer
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        
        # Create embeddings
        texts = [doc['text'] for doc in self.documents]
        logger.info("Creating embeddings...")
        self.embeddings = self.sentence_model.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(self.embeddings.astype('float32'))
        
        # Also build TF-IDF for keyword matching
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        logger.info("Retrieval system built successfully!")
        return True
    
    def save_model(self, model_dir="simple_legal_model"):
        """Save the trained model components"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save documents
        with open(f"{model_dir}/documents.json", "w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, f"{model_dir}/legal_index.faiss")
        
        # Save embeddings
        np.save(f"{model_dir}/embeddings.npy", self.embeddings)
        
        # Save TF-IDF
        with open(f"{model_dir}/tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        with open(f"{model_dir}/tfidf_matrix.pkl", "wb") as f:
            pickle.dump(self.tfidf_matrix, f)
        
        # Save model info
        model_info = {
            "model_type": "Simple Legal AI",
            "num_documents": len(self.documents),
            "embedding_dimension": self.embeddings.shape[1],
            "sentence_model": "paraphrase-multilingual-mpnet-base-v2"
        }
        
        with open(f"{model_dir}/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Model saved to {model_dir}")
        return model_dir
    
    def retrieve_relevant_docs(self, query, k=5):
        """Retrieve most relevant documents for a query"""
        if self.faiss_index is None:
            return []
        
        # Encode query
        query_embedding = self.sentence_model.encode([query])
        
        # Search FAISS index
        distances, indices = self.faiss_index.search(
            query_embedding.astype('float32'), k
        )
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    'text': doc['text'],
                    'source': doc['source'],
                    'similarity': 1 / (1 + distance)  # Convert distance to similarity
                })
        
        return results
    
    def generate_response(self, query):
        """Generate response using retrieval-based approach"""
        # Get relevant documents
        relevant_docs = self.retrieve_relevant_docs(query, k=3)
        
        if not relevant_docs:
            return "I couldn't find relevant information in the legal documents for your query."
        
        # Create response based on retrieved documents
        response = "Based on the legal documents, here's what I found:\n\n"
        
        for i, doc in enumerate(relevant_docs, 1):
            response += f"**Reference {i}** (from {doc['source']}):\n"
            response += f"{doc['text'][:300]}...\n\n"
        
        response += "**Legal Advice:**\nPlease consult with a qualified legal professional for specific advice regarding your situation."
        
        return response
    
    def train_complete_system(self):
        """Run the complete training pipeline"""
        logger.info("Starting Simple Legal AI Training...")
        
        # Step 1: Extract documents
        if not self.extract_and_process_documents():
            logger.error("Failed to extract documents")
            return None
        
        # Step 2: Build retrieval system
        if not self.build_retrieval_system():
            logger.error("Failed to build retrieval system")
            return None
        
        # Step 3: Save model
        model_path = self.save_model()
        
        logger.info(f"Simple Legal AI training completed! Model saved to: {model_path}")
        return model_path

def main():
    print("🏛️ Simple Legal AI - Fast Training")
    print("=" * 50)
    
    trainer = SimpleLegalAI()
    model_path = trainer.train_complete_system()
    
    if model_path:
        print(f"\n✅ SUCCESS: Legal AI model trained and saved to: {model_path}")
        print("\n📋 Next steps:")
        print("1. Run 'python simple_legal_api.py' to start the API server")
        print("2. Your React website can now query the trained model!")
        
        # Quick test
        print("\n🧪 Quick test:")
        test_query = "What are the penalties for hacking?"
        response = trainer.generate_response(test_query)
        print(f"Query: {test_query}")
        print(f"Response: {response[:200]}...")
        
    else:
        print("\n❌ FAILED: Please check the logs above for errors")

if __name__ == "__main__":
    main()