# API de Datos 5G con RAG y Generación Inteligente

Este proyecto implementa una API para generar y gestionar datos sintéticos de tráfico 5G utilizando RAG (Retrieval Augmented Generation) y modelos de lenguaje a través de Ollama.

## Características

- Generación de datos sintéticos de tráfico 5G
- Generación inteligente usando RAG y modelos de lenguaje
- Almacenamiento en MongoDB
- Exportación a CSV
- Procesamiento de datos desde CSV y JSON
- Consulta y filtrado de datos almacenados
- Interfaz web interactiva para todos los endpoints

# Guía de Instalación y Ejecución con Docker Compose

## Prerrequisitos
- Docker instalado
- Docker Compose instalado
- Git instalado
- Ollama instalado (para el modelo qwen2.5)

## 1. Clonar el Repositorio
```bash
git clone https://github.com/luisanma/ai-data-generation.git
cd ai-data-generation
```

## 2. Configuración de Ollama
```bash
# Instalar Ollama (si no está instalado)
curl https://ollama.ai/install.sh | sh

# Descargar el modelo qwen2.5
ollama pull qwen2.5:latest
```

## 3. Estructura del Proyecto
Asegúrate de tener la siguiente estructura:
```
ai-data-generation/
├── api/
│   ├── api.py
│   └── generate_5g_data.py
├── static/
│   └── js/
│       └── main.js
├── templates/
│   └── index.html
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env (opcional)
```

## 4. Variables de Entorno (Opcional)
Crear archivo `.env`:
```bash
MONGO_URI=mongodb://mongodb:27017/
MONGO_DB=network_5g
MONGO_COLLECTION=traffic_data
OLLAMA_BASE_URL=http://localhost:11434
```

## 5. Construir y Ejecutar con Docker Compose

### Iniciar los servicios
```bash
# Construir las imágenes
docker-compose build

# Levantar los servicios
docker-compose up -d
```

### Verificar el estado
```bash
# Ver los logs
docker-compose logs -f

# Verificar servicios en ejecución
docker-compose ps
```

## 6. Acceder a la Aplicación
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 7. Comandos Útiles

### Gestión de Servicios
```bash
# Detener servicios
docker-compose down

# Reiniciar servicios
docker-compose restart

# Ver logs de un servicio específico
docker-compose logs -f api
```

### Limpieza
```bash
# Eliminar contenedores y redes
docker-compose down

# Eliminar todo, incluyendo volúmenes
docker-compose down -v

# Eliminar imágenes
docker-compose down --rmi all
```

## 8. Solución de Problemas

### Verificar Conectividad MongoDB
```bash
# Acceder al contenedor de MongoDB
docker-compose exec mongodb mongo

# Verificar logs de MongoDB
docker-compose logs mongodb
```

### Problemas Comunes
1. **Error de conexión a MongoDB**:
   - Verificar que MongoDB está corriendo: `docker-compose ps`
   - Comprobar logs: `docker-compose logs mongodb`

2. **Error con Ollama**:
   - Verificar que Ollama está corriendo: `ollama list`
   - Comprobar URL en variables de entorno

3. **Problemas de permisos**:
   ```bash
   # Dar permisos a los directorios
   chmod -R 755 static templates
   ```

## 9. Desarrollo Local

### Reconstruir después de cambios
```bash
# Reconstruir servicio específico
docker-compose build api

# Reconstruir y reiniciar
docker-compose up -d --build
```

### Hot Reload
Los cambios en el código se reflejan automáticamente gracias al volumen montado y uvicorn --reload

## 10. Monitoreo

### Recursos del Sistema
```bash
# Ver uso de recursos
docker stats

# Ver logs en tiempo real
docker-compose logs -f
```

### Backup de MongoDB
```bash
# Crear backup
docker-compose exec mongodb mongodump --out /dump

# Restaurar backup
docker-compose exec mongodb mongorestore /dump
```

## 11. Detener el Sistema
```bash
# Detener servicios
docker-compose down

# Detener y eliminar volúmenes
docker-compose down -v
```

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