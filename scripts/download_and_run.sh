#!/bin/bash

# Clonar repo
git clone https://github.com/usuario/iso27001-rag.git
cd iso27001-rag

# Crear entorno y activar (opcional en colab)
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar scripts
python scripts/vector_index.py
python scripts/rag_qa.py
python scripts/evaluate_sets.py
