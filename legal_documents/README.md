# Legal Documents Folder

This folder is where you should place your legal PDF documents for training the ML model.

## Instructions:

1. **Add your PDF files here** - Place all your legal documents (court judgments, legal acts, cyber law documents) in this folder
2. **Supported formats**: PDF files only
3. **File naming**: Use descriptive names like:
   - `IT_Act_2000.pdf`
   - `Cyber_Crime_Judgments_2023.pdf`
   - `IPC_Sections_Cyber_Crime.pdf`

## Example structure:
```
legal_documents/
├── IT_Act_2000.pdf
├── Cyber_Crime_Cases_2023.pdf
├── Supreme_Court_Judgments.pdf
├── High_Court_Rulings.pdf
└── Legal_Precedents.pdf
```

## What happens next:
1. After adding your PDFs, run: `python train_legal_model.py`
2. The system will extract text from all PDFs
3. Train a custom ML model on your legal data
4. Save the trained model for use in the API

## Tips:
- More documents = better model performance
- Include diverse legal documents (acts, judgments, rulings)
- Ensure PDFs contain extractable text (not just images)
- Remove any confidential/sensitive documents before training