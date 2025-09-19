from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def text_to_vecs(texts):
    vectorizer = TfidfVectorizer().fit(texts)
    vectors = vectorizer.transform(texts).toarray()
    return vectors

def eval_similarity(answers_list):
    # answers_list: list[str] de respuestas por pregunta, en diferentes temperaturas (4 por cada pregunta)
    # Evaluar distancia cosine entre respuestas a distintas temperaturas para cada pregunta
    # Devuelve lista de calificaciones (buena, regular, mala) para cada respuesta comparando con respuesta temperatura 0.2
    results = []
    base_ans = answers_list[0]
    for idx in range(1, len(answers_list)):
        vectors = text_to_vecs([base_ans, answers_list[idx]])
        dist = cosine_distances([vectors[0]], [vectors[1]])[0][0]
        if dist <= 0.3:
            results.append(("buena", dist))
        elif dist <= 0.5:
            results.append(("regular", dist))
        else:
            results.append(("mala", dist))
    return results

# Nota: Para evaluación extendida se requiere integración con preguntas y múltiples sets

if __name__=="__main__":
    # Ejemplo dummy
    answers = [
        "La contraseña debe ser compleja y renovada cada 90 días.",
        "Se debe cambiar la contraseña cada 90 días.",
        "Es obligatorio usar contraseñas fuertes y actualizar regularmente.",
        "Las contraseñas se actualizan trimestralmente.",
        "Usuarios deben cambiar las claves periódicamente para seguridad."
    ]
    results = eval_similarity(answers)
    for i, (cat, dist) in enumerate(results):
        print(f"Temperatura vs 0.2 respuesta {i+1}: {cat} con distancia {dist:.3f}")
