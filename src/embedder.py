from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

def create_index(documents):
    embeddings = OpenAIEmbeddings()
    texts = [doc["text"] for doc in documents]
    metadata = [{"filename": doc["filename"]} for doc in documents]
    return FAISS.from_texts(texts, embeddings, metadata=metadata)

def retrieve_documents(index, query):
    return index.similarity_search(query)