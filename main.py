import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from PyPDF2 import PdfReader
import docx

# --- Helper functions to extract text from files ---

def get_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def get_text_from_docx(file_path):
    """Extracts text from a DOCX file."""
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
    return text

def build_vector_database():
    """
    Loads documents, creates embeddings, and saves the FAISS index and chunks.
    """
    print("Loading multilingual embedding model...")
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    print("Model loaded.")

    # --- Enhanced Knowledge Base with Specific Legal Details ---
    # IMPORTANT: This is still example data. Replace with your real judgment files.
    documents = [
        {
            "source": "Cyber_Fraud_IPC.txt",
            "content": """
            Online Financial Fraud: When an individual is deceived into transferring money through digital means under false pretenses, it constitutes online cheating.
            This is primarily covered under Section 420 of the Indian Penal Code (IPC), which deals with 'Cheating and dishonestly inducing delivery of property'.
            Consequences: The punishment for an offense under Section 420 IPC can be imprisonment for a term which may extend to seven years, and shall also be liable to a fine.
            The IT Act 2000 complements this through Section 66D, which pertains to cheating by personation using a computer resource, punishable by imprisonment up to three years and a fine up to one lakh rupees.
            """
        },
        {
            "source": "Identity_Theft_IT_Act.txt",
            "content": """
            Identity Theft and Impersonation: The act of fraudulently using someone else's electronic signature, password, or other unique identification feature is defined as identity theft.
            This is specifically addressed by Section 66C of the Information Technology Act, 2000.
            Punishment: A person found guilty under Section 66C shall be punished with imprisonment of either description for a term which may extend to three years and shall also be liable to a fine which may extend to rupees one lakh.
            This can often be linked with Section 419 of the IPC for 'Punishment for cheating by personation'.
            """
        },
        {
            "source": "Data_Theft_Hacking.txt",
            "content": """
            Hacking and Data Theft: Unauthorized access to a computer, computer system, or network, and subsequently downloading, copying, or extracting data without permission is a serious offense.
            Section 66 of the IT Act, 2000, is the primary provision for such computer-related offenses. If any person dishonestly or fraudulently commits the act referred to in Section 43, they shall be punishable with imprisonment for a term which may extend to three years or with a fine which may extend to five lakh rupees or with both.
            This is often tried alongside Section 379 of the IPC for theft.
            """
        }
    ]

    document_folder = 'judgments'
    print(f"Loading documents from the '{document_folder}' folder and using internal examples...")
    if os.path.exists(document_folder):
        for filename in os.listdir(document_folder):
            file_path = os.path.join(document_folder, filename)
            content = ""
            if filename.lower().endswith('.pdf'): content = get_text_from_pdf(file_path)
            elif filename.lower().endswith('.docx'): content = get_text_from_docx(file_path)
            elif filename.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
            if content: documents.append({"source": filename, "content": content})

    print(f"Total documents to process: {len(documents)}")
    if not documents:
        print("No documents found. Please add files to the 'judgments' folder or check the internal list.")
        return

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc['content'])
        for chunk in chunks:
            all_chunks.append({"source": doc['source'], "content": chunk})
    print(f"Created {len(all_chunks)} chunks.")

    print("Generating embeddings for all chunks...")
    chunk_contents = [chunk['content'] for chunk in all_chunks]
    embeddings = model.encode(chunk_contents, convert_to_tensor=False, show_progress_bar=True)
    print("Embeddings generated.")

    print("Building FAISS index...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings))
    print(f"FAISS index built with {index.ntotal} vectors.")

    print("Saving FAISS index and chunks...")
    faiss.write_index(index, "cyber_law_index.bin")
    with open("cyber_law_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    print("Vector database built and saved successfully!")

if __name__ == "__main__":
    build_vector_database()
