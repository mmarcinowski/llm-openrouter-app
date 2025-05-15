import fitz
import os

def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            with fitz.open(os.path.join(folder_path, filename)) as pdf:
                text = "".join(page.get_text() for page in pdf)
            documents.append({"text": text, "filename": filename})
    return documents