#!/bin/bash

# Instalar dependencias
pip install -r requirements.txt

# Crear directorios si no existen
mkdir -p static/js templates

# Copiar archivos a las ubicaciones correctas
cp index.html templates/
cp main.js static/js/

# Ejecutar la aplicación
uvicorn api:app --reload --host 0.0.0.0 --port 8000 