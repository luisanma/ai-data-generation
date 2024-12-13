# API de Datos 5G con RAG y Generación Inteligente

Este proyecto implementa una API para generar y gestionar datos sintéticos de tráfico 5G utilizando RAG (Retrieval Augmented Generation) y modelos de lenguaje a través de Ollama.

## Características

- Generación de datos sintéticos de tráfico 5G
- Generación inteligente usando RAG y modelos de lenguaje
- Almacenamiento en MongoDB
- Exportación a CSV
- Procesamiento de datos desde CSV
- Consulta y filtrado de datos almacenados

## Requisitos

- Docker y Docker Compose
- Python 3.9+
- Ollama instalado con el modelo qwen2.5

# API de Datos 5G - Documentación de Endpoints

## Generación de Datos

### 1. Generar Dataset Base
**Endpoint**: `POST /generate`
- **Descripción**: Genera un nuevo conjunto de datos sintéticos de tráfico 5G
- **Parámetros**:
  - `num_samples` (int, default=10000): Número de muestras a generar
- **Respuesta**:
  ```json
  {
    "dataset_id": "uuid",
    "preview": [...],  // Primeros 5 registros
    "summary": {...}   // Estadísticas del dataset
  }
  ```

### 2. Generar desde JSON
**Endpoint**: `POST /ollama-data-generate-from-json`
- **Descripción**: Genera datos usando RAG basado en un JSON existente
- **Parámetros**:
  - `file` (File): Archivo JSON con datos de ejemplo
  - `num_samples` (int, default=5): Número de muestras a generar
  - `context_query` (string): Consulta para guiar la generación
- **Respuesta**: Archivo CSV con datos generados

### 3. Limpiar JSON
**Endpoint**: `POST /clean-json-data`
- **Descripción**: Limpia y procesa un JSON con estructura MongoDB
- **Parámetros**:
  - `file` (File): Archivo JSON a procesar
  - `include_summary` (bool, default=false): Incluir resumen estadístico
- **Respuesta**: Archivo CSV con datos limpios

## Consulta de Datos

### 4. Consultar MongoDB
**Endpoint**: `GET /get-mongodb-data`
- **Descripción**: Obtiene datos almacenados con opciones de filtrado
- **Parámetros**:
  - `limit` (int, default=1000): Límite de registros
  - `skip` (int, default=0): Registros a saltar
  - `format` (string): "json" o "csv"
  - `dataset_id` (string, opcional): Filtrar por dataset
- **Respuesta**:
  ```json
  {
    "total_records": int,
    "records_returned": int,
    "skip": int,
    "limit": int,
    "dataset_id": string,
    "data": [...],
    "summary": {...}
  }
  ```

### 5. Exportar a CSV
**Endpoint**: `GET /to-csv/{dataset_id}`
- **Descripción**: Exporta un dataset específico a CSV
- **Parámetros**:
  - `dataset_id` (string): ID del dataset
  - `preview` (bool, default=true): Solo primeras 5 filas
- **Respuesta**: Archivo CSV descargable

### 6. Inicializar RAG
**Endpoint**: `POST /initialize-rag/{dataset_id}`
- **Descripción**: Inicializa el sistema RAG con un dataset
- **Parámetros**:
  - `dataset_id` (string): ID del dataset a usar
- **Respuesta**:
  ```json
  {
    "message": "RAG inicializado exitosamente",
    "dataset_id": "uuid"
  }
  ```

## Interfaz Web

### 7. Página Principal
**Endpoint**: `GET /`
- **Descripción**: Sirve la interfaz web interactiva
- **Respuesta**: Página HTML con formularios para todos los endpoints

## Estructura de Datos
```json
{
    "device_id": "int (1000-9999)",
    "timestamp": "datetime",
    "bandwidth_mbps": "float (0-1000)",
    "latency_ms": "float (0-100)",
    "packet_loss": "float (0-1)",
    "signal_strength_dbm": "float (-120 to -70)",
    "cell_id": "int (1-100)",
    "connection_type": "enum ['MIMO', 'Beamforming', 'Carrier Aggregation']"
}
```
## Notas
- Todos los endpoints retornan errores HTTP estándar en caso de fallo
- Los datos generados se almacenan automáticamente en MongoDB
- El sistema RAG utiliza el modelo qwen2.5 a través de Ollama
- Los CSV generados incluyen encabezados de columna