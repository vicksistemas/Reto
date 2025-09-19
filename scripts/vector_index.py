import os
from langchain.embeddings import OpenAIEmbeddings
import chromadb

def chunk_text(text, chunk_size=200, chunk_overlap=50):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - chunk_overlap)
    return chunks

def index_texts(data_path="data"):
    client = chromadb.Client()
    embeddings = OpenAIEmbeddings()
    collections = {}
    results = {}

    for filename in os.listdir(data_path):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(data_path, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().replace("\n", " ")
        chunks = chunk_text(text)
        collection_name = filename.replace(".txt", "")
        # Borra la coleccion si ya existe para limpiar ejecuciones previas
        try:
            client.delete_collection(name=collection_name)
        except:
            pass
        collection = client.create_collection(name=collection_name)
        collections[collection_name] = collection

        metadatas = [{"source": filename} for _ in chunks]
        ids = [f"{collection_name}_{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, metadatas=metadatas, ids=ids, embeddings=embeddings.embed_documents(chunks))
        
        # Report cardinality vector y número vectores
        vector_dim = len(collection.get()["embedding"][0]) if len(collection.get()["embedding"]) > 0 else 0
        vector_count = len(chunks)
        results[filename] = (vector_dim, vector_count)

    return results

if __name__=="__main__":
    results = index_texts()
    for f, (dim, count) in results.items():
        print(f"Archivo: {f} - Dimensión vector: {dim} - Número de vectores: {count}")
