import os
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import chromadb

def load_collection(collection_name):
    client = chromadb.Client()
    collection = client.get_collection(name=collection_name)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(collection_name=collection_name, embedding_function=embeddings, client=client)
    return vectorstore

def create_qa_chain(vectorstore, temperature=0.2):
    llm = OpenAI(temperature=temperature)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k":3}))
    return qa

def ask_questions(qa, questions):
    answers = []
    for question in questions:
        answer = qa.run(question)
        answers.append(answer)
        print(f"Q: {question}\nA: {answer}\n---")
    return answers

if __name__=="__main__":
    import sys
    # load all collections to have unified context
    collections = ["politica_contraseñas", "politica_uso_de_equipos", "politica_uso_de_red",
                   "politica_instalacion_de_programas", "politica_visitantes_red"]
    # For simplicity, merge collections by retrieving from multiple vectorstores in a custom retriever (not implemented here)
    # We'll pick one (ej: politica_contraseñas) for demonstration

    collection_name = "politica_contraseñas"
    vectorstore = load_collection(collection_name)
    temps = [0.2, 0.3, 0.4, 0.7]
    questions = [
        "¿Qué requisitos tiene un usuario visitante para la creación de contraseñas?",
        "¿Cuándo deben cambiar las contraseñas los usuarios administrativos?",
        "¿Qué controles existen para las contraseñas de usuarios desarrolladores?",
        "¿Cómo se protege la confidencialidad de las contraseñas para usuarios gerenciales?",
        "¿Qué acciones se toman en caso de sospecha de compromiso de contraseñas?"
    ]

    for temp in temps:
        print(f"\n------- Respuestas con temperatura {temp} -------")
        qa = create_qa_chain(vectorstore, temperature=temp)
        answers = ask_questions(qa, questions)
