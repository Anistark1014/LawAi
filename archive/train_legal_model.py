"""
ML Pipeline for Training Legal Document Model
This script will:
1. Extract text from PDF legal documents 
2. Preprocess the data for training
3. Train a transformer model on the legal data
4. Save the trained model for inference
"""

import os
import json
import pickle
from pathlib import Path
import PyPDF2
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    TextDataset,
    DataCollatorForLanguageModeling
)
import torch
from datasets import Dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalDocumentProcessor:
    def __init__(self, pdf_folder="legal_documents", model_name="microsoft/DialoGPT-medium"):
        self.pdf_folder = pdf_folder
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.processed_data = []
        
    def extract_text_from_pdfs(self):
        """Extract text from all PDF files in the legal_documents folder"""
        if not os.path.exists(self.pdf_folder):
            os.makedirs(self.pdf_folder)
            logger.info(f"Created {self.pdf_folder} folder. Please add your legal PDF documents here.")
            return []
        
        extracted_texts = []
        pdf_files = list(Path(self.pdf_folder).glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_folder}")
            return []
            
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_path in pdf_files:
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text() + "\n"
                    
                    if text.strip():
                        extracted_texts.append({
                            'filename': pdf_path.name,
                            'text': text.strip(),
                            'length': len(text)
                        })
                        logger.info(f"Extracted {len(text)} characters from {pdf_path.name}")
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {str(e)}")
                
        return extracted_texts
    
    def preprocess_text_for_training(self, extracted_texts):
        """Convert extracted text into question-answer pairs for training"""
        training_data = []
        
        for doc in extracted_texts:
            text = doc['text']
            filename = doc['filename']
            
            # Split text into chunks (simulate Q&A pairs)
            chunks = self.split_into_chunks(text, max_length=512)
            
            for i, chunk in enumerate(chunks):
                # Create synthetic Q&A pairs from legal text
                if len(chunk.strip()) > 100:  # Only process meaningful chunks
                    training_data.append({
                        'source': filename,
                        'text': chunk.strip(),
                        'input_text': f"Legal Question: Based on Indian cyber law, what should I know about this situation?",
                        'target_text': chunk.strip()
                    })
        
        logger.info(f"Created {len(training_data)} training examples from {len(extracted_texts)} documents")
        return training_data
    
    def split_into_chunks(self, text, max_length=512):
        """Split long text into smaller chunks for training"""
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < max_length:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def prepare_model_and_tokenizer(self):
        """Initialize the model and tokenizer"""
        logger.info(f"Loading model and tokenizer: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("Model and tokenizer loaded successfully")
    
    def create_training_dataset(self, training_data):
        """Create dataset for training"""
        # Prepare training texts
        texts = []
        for item in training_data:
            # Format as conversation
            formatted_text = f"User: {item['input_text']}\nAssistant: {item['target_text']}{self.tokenizer.eos_token}"
            texts.append(formatted_text)
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Create dataset
        dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': encodings['input_ids'].clone()
        })
        
        return dataset
    
    def train_model(self, dataset, output_dir="./trained_legal_model"):
        """Train the model on legal documents"""
        logger.info("Starting model training...")
        
        # Split dataset
        train_dataset, val_dataset = train_test_split(
            range(len(dataset)), 
            test_size=0.1, 
            random_state=42
        )
        
        train_data = dataset.select(train_dataset)
        val_data = dataset.select(val_dataset)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            eval_strategy="steps",  # Updated parameter name
            eval_steps=50,
            save_steps=100,
            save_total_limit=2,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save model and tokenizer
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model training completed and saved to {output_dir}")
        return output_dir
    
    def run_full_pipeline(self):
        """Run the complete training pipeline"""
        logger.info("Starting Legal Document ML Pipeline...")
        
        # Step 1: Extract text from PDFs
        extracted_texts = self.extract_text_from_pdfs()
        if not extracted_texts:
            logger.error("No text extracted from PDFs. Please add PDF files to the legal_documents folder.")
            return None
        
        # Step 2: Preprocess for training
        training_data = self.preprocess_text_for_training(extracted_texts)
        
        # Save processed data
        with open("processed_training_data.json", "w", encoding="utf-8") as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        # Step 3: Prepare model
        self.prepare_model_and_tokenizer()
        
        # Step 4: Create dataset
        dataset = self.create_training_dataset(training_data)
        
        # Step 5: Train model
        model_path = self.train_model(dataset)
        
        logger.info(f"Training pipeline completed! Model saved at: {model_path}")
        return model_path

def main():
    # Create processor
    processor = LegalDocumentProcessor()
    
    # Run training pipeline
    model_path = processor.run_full_pipeline()
    
    if model_path:
        print(f"\n✅ SUCCESS: Legal model trained and saved to: {model_path}")
        print("\n📋 Next steps:")
        print("1. Run 'python ml_inference_api.py' to start the API server")
        print("2. Your React website can now query the trained model!")
    else:
        print("\n❌ FAILED: Please add PDF documents to the 'legal_documents' folder and try again")

if __name__ == "__main__":
    main()