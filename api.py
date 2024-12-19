"""API FastAPI para generación y gestión de datos de tráfico 5G."""
from typing import Dict, List, Optional, Any
import os
import io
import json
import time
from datetime import datetime, timedelta
import uuid

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from pydantic import BaseModel
from pymongo import MongoClient
from fastapi.responses import StreamingResponse
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Configuración
MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017/")
DB_NAME = os.getenv("MONGO_DB", "network_5g")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "traffic_data")
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

app = FastAPI(title="API de Datos 5G")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configurar templates
templates = Jinja2Templates(directory="templates")


class DatasetResponse(BaseModel):
    """Modelo de respuesta para datasets."""
    dataset_id: str
    preview: List[dict]
    summary: dict


class RAGHelper:
    """Clase auxiliar para gestionar RAG."""
    def __init__(self) -> None:
        self.llm = OllamaLLM(
            model="qwen2.5:latest",
            temperature=0.5,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
        self.vector_store = None

    def create_vector_store(self, dataset_df: pd.DataFrame) -> None:
        """Crea vector store desde DataFrame."""
        documents = []
        for _, row in dataset_df.iterrows():
            doc_text = self._format_document(row)
            documents.append(doc_text)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        texts = text_splitter.create_documents(documents)
        self.vector_store = FAISS.from_documents(texts, self.embeddings)

    @staticmethod
    def _format_document(row: pd.Series) -> str:
        """Formatea una fila como documento."""
        return f"""
        Device ID: {row['device_id']}
        Bandwidth: {row['bandwidth_mbps']} Mbps
        Latency: {row['latency_ms']} ms
        Packet Loss: {row['packet_loss']}
        Signal Strength: {row['signal_strength_dbm']} dBm
        Connection Type: {row['connection_type']}
        """

    def query_dataset(self, query: str) -> str:
        """Consulta el dataset usando RAG."""
        if not self.vector_store:
            raise ValueError("Vector store no inicializado")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )
        return qa_chain.run(query)


# Inicializar RAG helper como variable global
rag_helper = RAGHelper()


def generate_synthetic_data(num_samples: int = 10000, patterns: dict = None):
    """Genera datos sintéticos y retorna un DataFrame"""
    if patterns is None:
        # Usar valores por defecto con límites seguros
        data = {
            'device_id': np.random.randint(1000, 9999, size=num_samples),
            'timestamp': [datetime.now() + timedelta(hours=x) for x in range(num_samples)],
            'bandwidth_mbps': np.clip(np.random.normal(500, 150, size=num_samples), 0, 1000),
            'latency_ms': np.clip(np.random.normal(10, 3, size=num_samples), 0, 100),
            'packet_loss': np.clip(np.random.beta(2, 50, size=num_samples), 0, 1),
            'signal_strength_dbm': np.clip(np.random.normal(-95, 10, size=num_samples), -120, -70),
            'cell_id': np.random.randint(1, 100, size=num_samples),
            'connection_type': np.random.choice(['MIMO', 'Beamforming', 'Carrier Aggregation'], size=num_samples)
        }
    else:
        # Usar patrones del modelo con límites seguros
        bandwidth_min, bandwidth_max = patterns.get('bandwidth_range', [0, 1000])
        latency_min, latency_max = patterns.get('latency_range', [0, 100])
        packet_loss_mean, packet_loss_std = patterns.get('packet_loss_pattern', [0.02, 0.01])
        connection_types = patterns.get('connection_types', ['MIMO', 'Beamforming', 'Carrier Aggregation'])
        
        data = {
            'device_id': np.random.randint(1000, 9999, size=num_samples),
            'timestamp': [datetime.now() + timedelta(hours=x) for x in range(num_samples)],
            'bandwidth_mbps': np.clip(np.random.uniform(bandwidth_min, bandwidth_max, size=num_samples), 0, 1000),
            'latency_ms': np.clip(np.random.uniform(latency_min, latency_max, size=num_samples), 0, 100),
            'packet_loss': np.clip(np.random.normal(packet_loss_mean, packet_loss_std, size=num_samples), 0, 1),
            'signal_strength_dbm': np.clip(np.random.normal(-95, 10, size=num_samples), -120, -70),
            'cell_id': np.random.randint(1, 100, size=num_samples),
            'connection_type': np.random.choice(connection_types, size=num_samples)
        }
    
    # Convertir timestamps a strings ISO para asegurar serialización JSON
    df = pd.DataFrame(data)
    df['timestamp'] = df['timestamp'].apply(lambda x: x.isoformat())
    return df

@app.post("/generate-dataset", response_model=DatasetResponse)
async def generate_dataset(num_samples: int = 10000):
    """Genera un nuevo dataset, lo guarda en MongoDB y retorna su ID y una vista previa"""
    try:
        # Generar datos
        df = generate_synthetic_data(num_samples)
        
        # Generar ID único
        dataset_id = str(uuid.uuid4())
        
        # Convertir las primeras 5 filas a diccionario para la vista previa
        preview = df.head().to_dict('records')
        
        # Generar resumen estadístico
        summary = df.describe().to_dict()
        
        # Preparar los documentos para MongoDB
        records = df.to_dict('records')
        for doc in records:
            doc['dataset_id'] = dataset_id
        
        # Guardar en MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Insertar registros
        result = collection.insert_many(records)
        
        # Crear índice para mejor rendimiento
        collection.create_index("dataset_id")
        
        client.close()
        
        return DatasetResponse(
            dataset_id=dataset_id,
            preview=preview,
            summary=summary
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transform-to-csv/{dataset_id}")
async def dataset_to_csv(dataset_id: str, preview: bool = True):
    """
    Transforma un dataset a CSV y permite descargarlo.
    Si preview=True, muestra y descarga las primeras 5 filas.
    Si preview=False, descarga el CSV completo.
    """
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Buscar documentos con el dataset_id específico
        documents = list(collection.find(
            {'dataset_id': dataset_id},
            {'_id': 0}  # Excluir el _id de MongoDB
        ))
        
        if not documents:
            raise HTTPException(status_code=404, detail="Dataset no encontrado")
        
        # Convertir a DataFrame
        df = pd.DataFrame(documents)
        
        # Preparar el CSV para descarga
        stream = io.StringIO()
        
        if preview:
            # Usar solo las primeras 5 filas
            preview_df = df.head()
            preview_df.to_csv(stream, index=False)
            filename = f"preview_dataset_{dataset_id}.csv"
        else:
            # Usar el dataset completo
            df.to_csv(stream, index=False)
            filename = f"dataset_{dataset_id}.csv"
        
        # Preparar la respuesta para descarga
        response = StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        return response
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        client.close()


def clean_numeric_values(value):
    """Limpia valores numéricos para asegurar compatibilidad con JSON"""
    if isinstance(value, (float, np.float32, np.float64)):
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    elif isinstance(value, (np.int32, np.int64)):
        return int(value)
    return value


def clean_dict_for_json(d):
    """Limpia un diccionario para asegurar compatibilidad con JSON"""
    return {k: clean_numeric_values(v) for k, v in d.items()}


@app.post("/ollama-data-generate-from-csv")
async def generate_data_from_csv(
    file: UploadFile = File(...),
    num_samples: int = 5,
    context_query: str = "Analiza los patrones y genera datos similares"
):
    """
    Genera datos usando RAG con un CSV como contexto y qwen2.5
    1. Carga el CSV
    2. Crea vector store con FAISS
    3. Genera datos usando el modelo
    4. Devuelve CSV con datos generados
    """
    try:
        # 1. Leer el CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # 2. Crear documentos para FAISS
        documents = []
        for _, row in df.iterrows():
            doc_text = " ".join([f"{col}: {val}" for col, val in row.items()])
            documents.append(doc_text)
        
        # 3. Crear vector store
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        texts = text_splitter.create_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
        vector_store = FAISS.from_documents(texts, embeddings)
        
        # 4. Configurar RAG
        llm = OllamaLLM(
            model="qwen2.5:7b",
            temperature=0.5,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )
        
        # 5. Generar datos
        generated_records = []
        columns = df.columns.tolist()
        
        for i in range(num_samples):
            prompt = f"""
            Basado en el siguiente contexto de datos y valores:
            {context_query}

            Importante:
            - Genera un registro nuevo y unico.
            - El registro debe seguir patrones realistas y contener estos campos:
              {', '.join(columns)}.

            En particular:
            - La columna 'connection_type' debe tener una mayor variación.
            - Aparte de los valores por defecto,a lgunos ejemplos válidos para 'connection_type' son:
              - '4G'
              - '5G'
              - 'LTE'
              - 'WiFi'
              - 'Satellite'
              - 'Edge Network'
            - Escoge uno de estos valores de forma aleatoria o combina con otras alternativas realistas.

            Responde SOLO con el JSON puro, sin markdown, sin comillas triples, sin explicaciones.
            """
            
            response = qa_chain.run(prompt)
            
            try:
                # Limpiar la respuesta de markdown y espacios
                cleaned_response = response.strip()
                if cleaned_response.startswith("```"):
                    cleaned_response = cleaned_response.split("```")[1]
                if cleaned_response.startswith("json"):
                    cleaned_response = cleaned_response[4:]
                cleaned_response = cleaned_response.strip()
                
                # Intentar parsear el JSON limpio
                record = json.loads(cleaned_response)
                
                # Asegurar que el registro tiene todas las columnas y valores válidos
                clean_record = {}
                for col in columns:
                    value = record.get(col, '')
                    # Convertir y validar tipos de datos según la columna
                    if col == 'device_id':
                        clean_record[col] = int(value) if value else 0
                    elif col in ['bandwidth_mbps', 'latency_ms', 'packet_loss']:
                        clean_record[col] = float(value) if value else 0.0
                    elif col == 'signal_strength_dbm':
                        clean_record[col] = max(-120, min(-70, float(value))) if value else -95
                    else:
                        clean_record[col] = str(value)
                
                generated_records.append(clean_record)
                print(f"Registro {i+1} generado exitosamente")
                
            except json.JSONDecodeError as e:
                print(f"Error parsing record {i+1}: {str(e)}\nResponse: {response}")
                continue
            except Exception as e:
                print(f"Error processing record {i+1}: {str(e)}")
                continue
        
        if not generated_records:
            raise HTTPException(
                status_code=500,
                detail="No se pudieron generar registros válidos"
            )
        
        # 6. Crear DataFrame con los nuevos datos
        new_df = pd.DataFrame(generated_records)
        
        # 7. Preparar CSV para descarga
        output = io.StringIO()
        new_df.to_csv(output, index=False)
        
        # 8. Devolver el CSV
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=generated_data_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

@app.post("/ollama-data-generate-from-json")
async def generate_data_from_json(
    file: UploadFile = File(...),
    num_samples: int = 5,
    context_query: str = "Analiza los patrones y genera datos similares"
):
    """
    Genera datos usando RAG con un archivo JSON como contexto y qwen2.5
    1. Carga el JSON
    2. Extrae solo los registros relevantes
    3. Genera datos usando el modelo
    4. Devuelve CSV limpio con datos tabulares
    """
    try:
        # 1. Leer y parsear el archivo JSON
        contents = await file.read()
        json_data = json.loads(contents)
        
        # 2. Extraer registros según la estructura del JSON
        if isinstance(json_data, dict) and "data" in json_data:
            # Si es la estructura de MongoDB
            records = json_data["data"]
        elif isinstance(json_data, list):
            # Si es una lista directa de registros
            records = json_data
        else:
            # Si es un único registro
            records = [json_data]
        
        # 3. Convertir a DataFrame y limpiar
        df = pd.DataFrame(records)
        
        # Eliminar columnas no deseadas si existen
        columns_to_drop = ['_id', 'dataset_id'] if '_id' in df.columns else []
        df = df.drop(columns=columns_to_drop, errors='ignore')
        
        # 4. Crear documentos para FAISS
        documents = []
        for _, row in df.iterrows():
            doc_text = " ".join([f"{col}: {val}" for col, val in row.items()])
            documents.append(doc_text)
        
        # 5. Crear vector store
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        texts = text_splitter.create_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
        vector_store = FAISS.from_documents(texts, embeddings)
        
        # 6. Configurar RAG con el modelo por defecto
        llm = OllamaLLM(
            model="qwen2.5:7b",
            temperature=0.8,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )
        
        # 7. Generar datos
        generated_records = []
        columns = df.columns.tolist()
        
        # Obtener estadísticas del dataset original para el contexto
        # stats_context = df.describe().to_string()
        
        for i in range(num_samples):
            prompt = f"""
            Basado en el siguiente contexto de datos y valores:
            {context_query}

            Importante:
            - Genera un registro nuevo y unico.
            - El registro debe seguir patrones realistas y contener estos campos:
              {', '.join(columns)}.

            En particular:
            - La columna 'connection_type' debe tener una mayor variación.
            - Aparte de los valores por defecto,a lgunos ejemplos válidos para 'connection_type' son:
              - '4G'
              - '5G'
              - 'LTE'
              - 'WiFi'
              - 'Satellite'
              - 'Edge Network'
            - Escoge uno de estos valores de forma aleatoria o combina con otras alternativas realistas.

            Responde SOLO con el JSON puro, sin markdown, sin comillas triples, sin explicaciones.
            """
            
            response = qa_chain.run(prompt)
            
            try:
                # Limpiar la respuesta
                cleaned_response = response.strip()
                if cleaned_response.startswith("```"):
                    cleaned_response = cleaned_response.split("```")[1]
                if cleaned_response.startswith("json"):
                    cleaned_response = cleaned_response[4:]
                cleaned_response = cleaned_response.strip()
                
                # Parsear y validar
                record = json.loads(cleaned_response)
                
                # Limpiar y validar valores según el tipo de dato original
                clean_record = {}
                for col in columns:
                    value = record.get(col, '')
                    original_dtype = df[col].dtype
                    
                    if pd.api.types.is_numeric_dtype(original_dtype):
                        try:
                            clean_record[col] = float(value) if '.' in str(value) else int(value)
                        except (ValueError, TypeError):
                            clean_record[col] = 0
                    else:
                        clean_record[col] = str(value)
                
                generated_records.append(clean_record)
                print(f"Registro {i+1} generado exitosamente")
                
            except Exception as e:
                print(f"Error en registro {i+1}: {str(e)}")
                continue
        
        if not generated_records:
            raise HTTPException(
                status_code=500,
                detail="No se pudieron generar registros válidos"
            )
        
        # 8. Crear DataFrame y CSV
        new_df = pd.DataFrame(generated_records)
        
        # Ordenar columnas para consistencia
        if not new_df.empty:
            standard_columns = [
                'device_id', 'timestamp', 'bandwidth_mbps', 'latency_ms',
                'packet_loss', 'signal_strength_dbm', 'cell_id', 'connection_type'
            ]
            columns_order = [col for col in standard_columns if col in new_df.columns]
            new_df = new_df[columns_order]
        
        # 9. Generar CSV limpio
        output = io.StringIO()
        new_df.to_csv(output, index=False)
        
        # 10. Devolver CSV
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=generated_data_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            }
        )
        
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error al parsear el archivo JSON: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-mongodb-data")
async def get_mongodb_data(
    limit: int = 1000,
    skip: int = 0,
    format: str = "json",  # "json" o "csv"
    dataset_id: str = None  # Opcional: filtrar por dataset_id
):
    """
    Obtiene datos almacenados en MongoDB con opciones de filtrado y paginación.
    """
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Construir filtro
        filter_query = {}
        if dataset_id:
            filter_query["dataset_id"] = dataset_id
        
        # Obtener total de registros que coinciden con el filtro
        total_records = collection.count_documents(filter_query)
        
        # Obtener datos con paginación
        cursor = collection.find(
            filter_query,
            {'_id': 0}  # Excluir el _id de MongoDB
        ).skip(skip).limit(limit)
        
        documents = list(cursor)
        
        if not documents:
            raise HTTPException(
                status_code=404,
                detail="No se encontraron datos"
            )
        
        # Convertir a DataFrame
        df = pd.DataFrame(documents)
        
        # Devolver según el formato solicitado
        if format.lower() == "csv":
            output = io.StringIO()
            df.to_csv(output, index=False)
            
            filename = f"mongodb_data_{dataset_id if dataset_id else 'all'}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}"
                }
            )
        else:  # JSON
            return {
                "total_records": total_records,
                "records_returned": len(documents),
                "skip": skip,
                "limit": limit,
                "dataset_id": dataset_id,
                "data": documents,
                "summary": {
                    k: clean_dict_for_json(v) 
                    for k, v in df.describe().to_dict().items()
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        client.close() 


@app.post("/clean-json-data")
async def clean_json_data(
    file: UploadFile = File(...),
    include_summary: bool = False
):
    """
    Limpia y procesa un JSON con estructura de MongoDB,
    extrayendo los datos relevantes y generando un CSV descargable.
    
    Args:
        file: Archivo JSON a procesar
        include_summary: Si se incluye el resumen estadístico
    """
    try:
        # 1. Leer y parsear el JSON
        contents = await file.read()
        json_data = json.loads(contents)
        
        # 2. Extraer los datos relevantes
        if "data" in json_data:
            records = json_data["data"]
        else:
            raise HTTPException(
                status_code=400,
                detail="El JSON debe contener una clave 'data' con los registros"
            )
            
        # 3. Convertir registros a DataFrame
        df = pd.DataFrame(records)
        
        # 4. Preparar el DataFrame de resumen si se solicita
        if include_summary and "summary" in json_data:
            summary_df = pd.DataFrame()
            for field, stats in json_data["summary"].items():
                for stat_name, value in stats.items():
                    summary_df.loc[stat_name, field] = value
        
        # 5. Preparar el CSV
        output = io.StringIO()
        
        # Si se incluye el resumen, agregarlo al principio
        if include_summary and "summary" in json_data:
            output.write("# Resumen Estadístico\n")
            summary_df.to_csv(output)
            output.write("\n# Registros\n")
        
        # Escribir los registros
        df.to_csv(output, index=False)
        
        # 6. Devolver el CSV
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": (
                    f"attachment; "
                    f"filename=clean_data_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                )
            }
        )
        
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error al parsear el JSON: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar los datos: {str(e)}"
        )


@app.get("/", response_class=HTMLResponse)
async def get_html(request: Request):
    """Sirve la interfaz web"""
    return templates.TemplateResponse("index.html", {"request": request})
