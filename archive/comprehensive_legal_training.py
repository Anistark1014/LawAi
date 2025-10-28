#!/usr/bin/env python3
"""
Comprehensive Legal AI Training with FIRE 2019 Dataset
Combines existing data with massive legal case database for true ML
"""
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
import faiss
from pathlib import Path
import re

class ComprehensiveLegalAI:
    def __init__(self):
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.tfidf_vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 3))
        
        # Multiple ML models for ensemble
        self.crime_classifier = GradientBoostingClassifier(n_estimators=200, random_state=42)
        self.urgency_classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.case_relevance_model = RandomForestClassifier(n_estimators=150, random_state=42)
        
        # Data storage
        self.legal_cases = []
        self.legal_statutes = []
        self.query_examples = []
        self.training_data = []
        
        # Encoders
        self.crime_encoder = LabelEncoder()
        self.urgency_encoder = LabelEncoder()
        
        # FAISS index for fast similarity search
        self.faiss_index = None
        self.embeddings = None
        
    def load_fire_2019_dataset(self):
        """Load the comprehensive FIRE 2019 legal dataset"""
        print("📚 Loading FIRE 2019 Legal Dataset...")
        
        base_path = Path("legal_documents/New DS")
        
        # Load legal queries
        query_file = base_path / "Query_doc.txt"
        if query_file.exists():
            with open(query_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if '||' in line:
                        query_id, query_text = line.strip().split('||', 1)
                        self.query_examples.append({
                            'id': query_id,
                            'text': query_text,
                            'source': 'FIRE_2019_Query'
                        })
        
        print(f"✅ Loaded {len(self.query_examples)} legal queries")
        
        # Load case documents
        casedocs_path = base_path / "Object_casedocs"
        if casedocs_path.exists():
            case_files = list(casedocs_path.glob("C*.txt"))
            print(f"📂 Processing {len(case_files)} case documents...")
            
            for i, case_file in enumerate(case_files[:500]):  # Process first 500 for faster training
                try:
                    with open(case_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        self.legal_cases.append({
                            'id': case_file.stem,
                            'text': content,
                            'source': 'FIRE_2019_Case',
                            'file_path': str(case_file)
                        })
                    
                    if (i + 1) % 100 == 0:
                        print(f"   Processed {i + 1} case documents...")
                        
                except Exception as e:
                    print(f"   Error reading {case_file}: {e}")
        
        print(f"✅ Loaded {len(self.legal_cases)} case documents")
        
        # Load statute documents  
        statutes_path = base_path / "Object_statutes"
        if statutes_path.exists():
            statute_files = list(statutes_path.glob("S*.txt"))
            print(f"📂 Processing {len(statute_files)} statute documents...")
            
            for statute_file in statute_files:
                try:
                    with open(statute_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if len(lines) >= 2:
                            title = lines[0].replace('Title:', '').strip()
                            desc = lines[1].replace('Desc:', '').strip()
                            full_text = f"{title} {desc}"
                            
                            self.legal_statutes.append({
                                'id': statute_file.stem,
                                'title': title,
                                'description': desc,
                                'text': full_text,
                                'source': 'FIRE_2019_Statute'
                            })
                except Exception as e:
                    print(f"   Error reading {statute_file}: {e}")
        
        print(f"✅ Loaded {len(self.legal_statutes)} statute documents")
        
        # Load QA datasets if available
        qa_files = ['constitution_qa.json', 'crpc_qa.json', 'ipc_qa.json']
        for qa_file in qa_files:
            qa_path = base_path / qa_file
            if qa_path.exists():
                try:
                    with open(qa_path, 'r', encoding='utf-8') as f:
                        qa_data = json.load(f)
                        for item in qa_data:
                            if isinstance(item, dict) and 'question' in item:
                                self.training_data.append({
                                    'query': item['question'],
                                    'answer': item.get('answer', ''),
                                    'source': f"QA_{qa_file.replace('.json', '')}",
                                    'legal_area': qa_file.replace('_qa.json', '').upper()
                                })
                except Exception as e:
                    print(f"   Error reading {qa_file}: {e}")
        
        print(f"✅ Loaded {len(self.training_data)} QA pairs")
        
    def create_enhanced_training_data(self):
        """Create enhanced training data with automatic labeling"""
        print("🏷️ Creating enhanced training data with automatic labeling...")
        
        enhanced_data = []
        
        # Process queries with automatic crime type detection
        for query in self.query_examples:
            text = query['text'].lower()
            
            # Automatic crime type detection using keywords
            crime_type = self._detect_crime_type(text)
            urgency = self._detect_urgency(text)
            
            enhanced_data.append({
                'text': query['text'],
                'crime_type': crime_type,
                'urgency': urgency,
                'source': query['source'],
                'id': query['id']
            })
        
        # Add cases with inferred labels
        for case in self.legal_cases[:200]:  # Use subset for training
            text = case['text'].lower()
            
            crime_type = self._detect_crime_type(text)
            urgency = self._detect_urgency(text)
            
            enhanced_data.append({
                'text': case['text'][:1000],  # Truncate for training
                'crime_type': crime_type,
                'urgency': urgency,
                'source': case['source'],
                'id': case['id']
            })
        
        self.training_data_enhanced = enhanced_data
        print(f"✅ Created {len(enhanced_data)} enhanced training samples")
        
    def _detect_crime_type(self, text):
        """Automatically detect crime type from text"""
        crime_patterns = {
            'murder': ['murder', 'killed', 'death', 'homicide', 'deceased'],
            'theft': ['theft', 'stolen', 'stole', 'robbery', 'burglary'],
            'fraud': ['fraud', 'cheat', 'deceive', 'scam', 'embezzle'],
            'assault': ['assault', 'attack', 'beat', 'violence', 'hurt'],
            'cybercrime': ['cyber', 'hacking', 'online', 'internet', 'digital'],
            'domestic_violence': ['domestic', 'wife', 'husband', 'dowry', 'marriage'],
            'corruption': ['bribe', 'corruption', 'illegal gratification'],
            'property_dispute': ['property', 'land', 'boundary', 'possession'],
            'employment': ['employment', 'termination', 'service', 'salary'],
            'constitutional': ['constitutional', 'fundamental rights', 'writ']
        }
        
        for crime, keywords in crime_patterns.items():
            if any(keyword in text for keyword in keywords):
                return crime
        
        return 'general'
    
    def _detect_urgency(self, text):
        """Automatically detect urgency level from text"""
        critical_patterns = ['murder', 'death', 'killed', 'emergency', 'urgent']
        high_patterns = ['assault', 'theft', 'fraud', 'harassment', 'violence']
        
        if any(pattern in text.lower() for pattern in critical_patterns):
            return 'critical'
        elif any(pattern in text.lower() for pattern in high_patterns):
            return 'high'
        else:
            return 'moderate'
    
    def train_comprehensive_model(self):
        """Train comprehensive ML model on all data"""
        print("🤖 Training comprehensive ML model...")
        
        if not self.training_data_enhanced:
            print("❌ No training data available!")
            return
        
        # Prepare features
        texts = [item['text'] for item in self.training_data_enhanced]
        crime_types = [item['crime_type'] for item in self.training_data_enhanced]
        urgency_levels = [item['urgency'] for item in self.training_data_enhanced]
        
        # Create TF-IDF features
        print("🔤 Creating TF-IDF features...")
        tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        
        # Create semantic embeddings
        print("🧠 Creating semantic embeddings...")
        semantic_features = self.sentence_model.encode(texts, show_progress_bar=True)
        
        # Combine features
        combined_features = np.hstack([tfidf_features.toarray(), semantic_features])
        
        # Encode labels
        crime_types_encoded = self.crime_encoder.fit_transform(crime_types)
        urgency_levels_encoded = self.urgency_encoder.fit_transform(urgency_levels)
        
        # Split data
        X_train, X_test, y_crime_train, y_crime_test, y_urgency_train, y_urgency_test = train_test_split(
            combined_features, crime_types_encoded, urgency_levels_encoded, 
            test_size=0.2, random_state=42
        )
        
        # Train crime classifier
        print("🎯 Training crime type classifier...")
        self.crime_classifier.fit(X_train, y_crime_train)
        crime_pred = self.crime_classifier.predict(X_test)
        crime_accuracy = accuracy_score(y_crime_test, crime_pred)
        print(f"   Crime Classification Accuracy: {crime_accuracy:.3f}")
        
        # Train urgency classifier
        print("⚡ Training urgency classifier...")
        self.urgency_classifier.fit(X_train, y_urgency_train)
        urgency_pred = self.urgency_classifier.predict(X_test)
        urgency_accuracy = accuracy_score(y_urgency_test, urgency_pred)
        print(f"   Urgency Classification Accuracy: {urgency_accuracy:.3f}")
        
        # Create FAISS index for fast similarity search
        print("🔍 Creating FAISS index...")
        self.embeddings = semantic_features.astype('float32')
        self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.faiss_index.add(self.embeddings)
        
        print("✅ Comprehensive ML model trained successfully!")
        
    def analyze_query_comprehensive(self, user_query):
        """Comprehensive query analysis using all available data"""
        
        # Extract features
        tfidf_features = self.tfidf_vectorizer.transform([user_query])
        semantic_features = self.sentence_model.encode([user_query])
        combined_features = np.hstack([tfidf_features.toarray(), semantic_features])
        
        # ML predictions
        crime_pred = self.crime_classifier.predict(combined_features)[0]
        crime_proba = max(self.crime_classifier.predict_proba(combined_features)[0])
        crime_type = self.crime_encoder.inverse_transform([crime_pred])[0]
        
        urgency_pred = self.urgency_classifier.predict(combined_features)[0]
        urgency_proba = max(self.urgency_classifier.predict_proba(combined_features)[0])
        urgency_level = self.urgency_encoder.inverse_transform([urgency_pred])[0]
        
        # Find similar cases using FAISS
        query_embedding = semantic_features.astype('float32')
        similarities, indices = self.faiss_index.search(query_embedding, k=3)
        
        similar_cases = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < len(self.training_data_enhanced):
                case = self.training_data_enhanced[idx]
                similar_cases.append({
                    'text': case['text'][:200] + "...",
                    'similarity': float(sim),
                    'source': case['source'],
                    'id': case.get('id', 'unknown')
                })
        
        return {
            'predicted_crime_type': crime_type,
            'crime_confidence': float(crime_proba),
            'predicted_urgency': urgency_level,
            'urgency_confidence': float(urgency_proba),
            'similar_cases': similar_cases,
            'model_type': 'Comprehensive ML (FIRE 2019 + Custom)',
            'dataset_size': len(self.training_data_enhanced)
        }
    
    def save_comprehensive_model(self, filepath="comprehensive_legal_model.pkl"):
        """Save the comprehensive model"""
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'crime_classifier': self.crime_classifier,
            'urgency_classifier': self.urgency_classifier,
            'crime_encoder': self.crime_encoder,
            'urgency_encoder': self.urgency_encoder,
            'training_data_enhanced': self.training_data_enhanced,
            'legal_cases': self.legal_cases[:100],  # Save subset
            'legal_statutes': self.legal_statutes,
            'embeddings': self.embeddings,
            'faiss_index_data': faiss.serialize_index(self.faiss_index) if self.faiss_index else None
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
            
        print(f"💾 Comprehensive model saved to: {filepath}")
    
    def load_comprehensive_model(self, filepath="comprehensive_legal_model.pkl"):
        """Load the comprehensive model"""
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                model_data = pickle.load(f)
                
            self.tfidf_vectorizer = model_data["tfidf_vectorizer"]
            self.crime_classifier = model_data["crime_classifier"]
            self.urgency_classifier = model_data["urgency_classifier"]
            self.crime_encoder = model_data["crime_encoder"]
            self.urgency_encoder = model_data["urgency_encoder"]
            self.training_data_enhanced = model_data["training_data_enhanced"]
            self.legal_cases = model_data.get("legal_cases", [])
            self.legal_statutes = model_data.get("legal_statutes", [])
            self.embeddings = model_data.get("embeddings")
            
            if model_data.get("faiss_index_data"):
                self.faiss_index = faiss.deserialize_index(model_data["faiss_index_data"])
            
            print(f"📂 Comprehensive model loaded from: {filepath}")
            return True
        return False

def train_comprehensive_legal_ai():
    """Main training function"""
    print("🚀 Starting Comprehensive Legal AI Training...")
    print("📊 Dataset: FIRE 2019 + Custom Legal Data")
    print("=" * 60)
    
    ai = ComprehensiveLegalAI()
    
    # Load all data
    ai.load_fire_2019_dataset()
    ai.create_enhanced_training_data()
    
    # Train comprehensive model
    ai.train_comprehensive_model()
    ai.save_comprehensive_model()
    
    # Test the model
    test_queries = [
        "I witnessed a murder last night",
        "Someone stole my purse and I found the thief", 
        "My Instagram was hacked and they want money",
        "My husband beats me and demands dowry",
        "Company terminated me without notice"
    ]
    
    print("\n🧪 Testing Comprehensive Model:")
    print("-" * 40)
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        analysis = ai.analyze_query_comprehensive(query)
        print(f"🤖 Crime Type: {analysis['predicted_crime_type']} ({analysis['crime_confidence']*100:.1f}%)")
        print(f"⚡ Urgency: {analysis['predicted_urgency']} ({analysis['urgency_confidence']*100:.1f}%)")
        print(f"📚 Similar Cases Found: {len(analysis['similar_cases'])}")
    
    print(f"\n✅ Training Complete!")
    print(f"📊 Total Training Samples: {len(ai.training_data_enhanced)}")
    print(f"📂 Legal Cases: {len(ai.legal_cases)}")
    print(f"📜 Statutes: {len(ai.legal_statutes)}")
    print("💾 Model saved as: comprehensive_legal_model.pkl")

if __name__ == "__main__":
    train_comprehensive_legal_ai()