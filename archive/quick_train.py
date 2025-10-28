"""
Quick Start Script for Legal Model Training
Run this after adding your PDF documents to legal_documents folder
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check if legal_documents folder exists and has PDFs
    legal_docs_path = Path("legal_documents")
    if not legal_docs_path.exists():
        print("❌ legal_documents folder not found!")
        return False
    
    pdf_files = list(legal_docs_path.glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found in legal_documents folder!")
        print("📋 Please add your legal PDF documents to the legal_documents folder")
        return False
    
    print(f"✅ Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files[:5]:  # Show first 5
        print(f"   - {pdf.name}")
    if len(pdf_files) > 5:
        print(f"   ... and {len(pdf_files) - 5} more")
    
    return True

def main():
    print("🏛️ AI Justice Bot - Legal Model Training")
    print("=" * 50)
    
    if not check_requirements():
        print("\n❌ Requirements not met. Please:")
        print("1. Add your legal PDF documents to 'legal_documents' folder")
        print("2. Run this script again")
        return
    
    print("\n🚀 Starting model training...")
    print("This may take 15-30 minutes depending on:")
    print("- Number of PDF documents")
    print("- Size of documents") 
    print("- Your computer's processing power")
    
    # Import and run training
    try:
        from train_legal_model import LegalDocumentProcessor
        
        processor = LegalDocumentProcessor()
        model_path = processor.run_full_pipeline()
        
        if model_path:
            print("\n" + "=" * 50)
            print("🎉 SUCCESS! Your legal model is trained and ready!")
            print("=" * 50)
            print(f"📁 Model saved at: {model_path}")
            print("\n📋 Next steps:")
            print("1. Run: python ml_inference_api.py")
            print("2. Your React website can now query the trained model at:")
            print("   POST http://localhost:5000/api/legal-advice")
            print("\n💡 Test it with queries like:")
            print("   'Someone hacked my email account'")
            print("   'I was cheated in an online transaction'")
            
        else:
            print("\n❌ Training failed. Check the logs above for errors.")
            
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        print("Please check that all dependencies are installed:")
        print("pip install torch transformers datasets scikit-learn pandas PyPDF2")

if __name__ == "__main__":
    main()