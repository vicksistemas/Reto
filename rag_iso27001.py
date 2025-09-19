#pip install --quiet langchain openai chromadb tiktoken numpy

import os
import glob
#from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import numpy as np

# Variables globales
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

# API Key - el usuario debe configurar en secrets GitHub o en entorno de Colab
OPENAI_API_KEY =OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise ValueError("Debe establecerse la variable OPENAI_API_KEY antes de ejecutar.")

def load_texts_and_chunk(path_txt_folder):
    text_files = glob.glob(os.path.join(path_txt_folder, '*.txt'))
    documents = {}
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    for file in text_files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = text_splitter.split_text(text)
        documents[file] = chunks
    return documents

text_folder = "./texts/"
documents_chunks = load_texts_and_chunk(text_folder)

print(f"Se cargaron y fragmentaron los textos de la carpeta {text_folder}")
for k,v in documents_chunks.items():
    print(f"Archivo: {os.path.basename(k)} - chunks: {len(v)}")

def build_chroma_vectorstore(docs_chunks, embedding_model, persist_directory="./chroma_db"):
    vectorstores = {}
    for filename, chunks in docs_chunks.items():
        # Creamos documentos para Langchain (texto por chunk)
        from langchain.schema import Document
        docs = [Document(page_content=c) for c in chunks]
        # Creamos vectorstore Chroma por archivo, persistiendo por separado
        db = Chroma.from_documents(docs, embedding_model, persist_directory=f"{persist_directory}/{os.path.basename(filename)}")
        db.persist()
        vectorstores[filename] = db
    return vectorstores

embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstores = build_chroma_vectorstore(documents_chunks, embedding_model)

for k,v in vectorstores.items():
    collection = v._collection
    print(f"Vectorstore para {os.path.basename(k)}, vectores total: {collection.count()}")

from langchain.chains import RetrievalQA

# Funcion para crear motor RAG para un vectorstore especifico
def create_retrieval_qa(vectorstore, temperature=0.3):
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=temperature)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

question_sets = [
    [ # Set 1
        "¿Cuáles son los requisitos de contraseñas para usuarios visitantes según la política?",
        "¿Qué permisos tiene un usuario desarrollador para instalación de programas?",
        "¿Cómo deben conectarse los visitantes a la red corporativa?",
        "¿Qué controles de acceso aplica la política para usuarios móviles?",
        "¿Cuándo deben cambiar su contraseña los usuarios administrativos?"
    ],
    [ # Set 2
        "¿Qué restricciones hay para usuarios administrativos en el uso de equipos?",
        "¿Cómo se protege el acceso a internet para usuarios gerenciales?",
        "¿Cuáles son las políticas para uso de aplicaciones móviles en dispositivos móviles?",
        "¿Qué procedimientos siguen los visitantes para conectarse a la red?",
        "¿Qué sanciones se aplican en caso de incumplimiento en la política de contraseñas?",
        "¿Cómo se controla la instalación de software en dispositivos corporativos?"
    ],
    [ # Set 3
        "¿Qué tipo de segmentación de red se usa para visitantes?",
        "¿Qué controles existen para programas usados por usuarios eventuales?",
        "¿Cómo deben registrar su ingreso los usuarios eventuales?",
        "¿Qué accesos especiales tienen los usuarios desarrolladores?",
        "¿Cómo se debe proteger la información en dispositivos móviles?"
    ],
    [ # Set 4
        "¿Cuál es la política para bloqueo en dispositivos móviles?",
        "¿Qué autorización necesitan los usuarios gerenciales para instalar software?",
        "¿Cómo se administran los perfiles temporales para usuarios eventuales?",
        "¿Qué medidas debe tomar un visitante antes de usar su programa en la red?",
        "¿Qué debe contener la capacitación sobre contraseñas para todos los usuarios?"
    ],
    [ # Set 5
        "¿Qué monitoreo existe para detectar intentos de acceso no autorizado?",
        "¿Cómo se manejan las actualizaciones de software para usuarios administrativos?",
        "¿Qué implica la política para la retirada segura de equipos?",
        "¿Cuáles son las características de una contraseña segura para usuarios móviles?",
        "¿Qué mecanismos debe usar un usuario administrativo para autenticarse?",
        "¿Cómo se definen los tiempos de acceso para usuarios eventuales?"
    ],
]

from numpy.linalg import norm

def embedding_for_text(text, embedding_model):
    return np.array(embedding_model.embed_query(text))

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1)*norm(vec2))

def cosine_distance(vec1, vec2):
    # distancia = 1 - similitud
    return 1 - cosine_similarity(vec1, vec2)

# Configuraciones de temperatura a evaluar
temperatures = [0.2, 0.3, 0.4, 0.7]
num_sets = len(question_sets)

def evaluate_sets(vectorstores, question_sets, temperatures):
    # Recordaremos resultados por set
    best_set = -1
    best_score = 1000000

    for set_idx, questions in enumerate(question_sets):
        print(f"\nEvaluando Set de Preguntas #{set_idx+1}:")
        total_distances = []
        # Para simplicidad, uso una consulta RAG combinada para la indicación
        # Selección de vectorstores: Vamos a combinar todas con prioridad la de politica_uso_red.txt por ejemplo
        vectorestore_key = list(vectorstores.keys())[2]  # Ejemplo uso 3er vectorstore: politica_uso_red.txt
        vectorstore = vectorstores[vectorestore_key]

        # Guardar respuestas por pregunta y temperatura
        responses = {temp: [] for temp in temperatures}

        # Crear motores para cada temperatura
        qa_models = {temp: create_retrieval_qa(vectorstore, temperature=temp) for temp in temperatures}

        for q in questions:
            # generar respuestas para cada temperatura
            for temp in temperatures:
                response = qa_models[temp].run(q)
                responses[temp].append(response)

        # Calcular distancia promedio por pregunta / pares temperaturas
        # Evaluamos distancias por temperatura (ejemplo todas contra 0.2)
        base_temp = 0.2
        for i in range(len(questions)):
            base_emb = embedding_for_text(responses[base_temp][i], embedding_model)
            for temp in temperatures:
                if temp == base_temp:
                    continue
                comp_emb = embedding_for_text(responses[temp][i], embedding_model)
                dist = cosine_distance(base_emb, comp_emb)
                total_distances.append(dist)
                criterio = "Buena" if dist <= 0.3 else "Regular" if dist <= 0.5 else "Mala"
                print(f"\nPregunta #{i+1}: '{questions[i]}'")
                print(f"Temperatura {base_temp} vs {temp} - Distancia: {dist:.3f} - Criterio: {criterio}")

        avg_distance = np.mean(total_distances)
        print(f"\nDistancia promedio para Set #{set_idx+1}: {avg_distance:.3f}")

        if avg_distance < best_score:
            best_set = set_idx
            best_score = avg_distance

    print("\n--------------------------------")
    print(f"Mejor Set de Preguntas: #{best_set+1} con distancia promedio {best_score:.3f}")
    print("--------------------------------")
    return best_set, best_score

best_set, best_score = evaluate_sets(vectorstores, question_sets, temperatures)