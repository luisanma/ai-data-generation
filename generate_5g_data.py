from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Crear un DataFrame de ejemplo con la estructura deseada
num_samples = 10000
data = {
    'device_id': np.random.randint(1000, 9999, size=num_samples),
    'timestamp': [datetime.now() + timedelta(hours=x) for x in range(num_samples)],
    'bandwidth_mbps': np.random.normal(500, 150, size=num_samples).clip(100, 1000),
    'latency_ms': np.random.normal(10, 3, size=num_samples).clip(1, 20),
    'packet_loss': np.random.beta(2, 50, size=num_samples),
    'signal_strength_dbm': np.random.normal(-95, 10, size=num_samples).clip(-120, -70),
    'cell_id': np.random.randint(1, 100, size=num_samples),
    'connection_type': np.random.choice(['MIMO', 'Beamforming', 'Carrier Aggregation'], size=num_samples)
}

synthetic_data = pd.DataFrame(data)

# Mostrar las primeras 10 filas del DataFrame
print("\n=== Vista previa de los datos generados ===")
print("\nPrimeras 10 filas del DataFrame:")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(synthetic_data.head(10))

# Mostrar un resumen estadístico básico
print("\n=== Resumen estadístico ===")
print(synthetic_data.describe())

# Configuración de MongoDB desde variables de entorno
MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017/")
DB_NAME = os.getenv("MONGO_DB", "network_5g")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "traffic_data")

def wait_for_mongodb(max_retries=30, delay_seconds=2):
    retries = 0
    while retries < max_retries:
        try:
            client = MongoClient(MONGO_URI)
            # La siguiente línea generará una excepción si no puede conectarse
            client.admin.command('ping')
            client.close()
            print("Conexión a MongoDB establecida exitosamente")
            return True
        except Exception as e:
            print(f"Intento {retries + 1}/{max_retries}: MongoDB no está disponible aún... {str(e)}")
            retries += 1
            time.sleep(delay_seconds)
    return False

def save_to_mongodb(data):
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Convertir DataFrame a lista de diccionarios
        records = data.to_dict('records')
        
        # Crear índice para mejor rendimiento
        collection.create_index("device_id")
        
        # Insertar registros
        result = collection.insert_many(records)
        print(f"Se insertaron {len(result.inserted_ids)} documentos en MongoDB")
        
    except Exception as e:
        print(f"Error al conectar con MongoDB: {str(e)}")
        raise
    finally:
        client.close()

def view_mongodb_data(limit=10):
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        print("\n=== Datos almacenados en MongoDB ===")
        for doc in collection.find().limit(limit):
            print("\nRegistro:")
            # Excluir el _id de MongoDB de la impresión
            doc.pop('_id', None)
            for key, value in doc.items():
                print(f"{key}: {value}")
                
    except Exception as e:
        print(f"Error al consultar MongoDB: {str(e)}")
    finally:
        client.close()

# Esperar a que MongoDB esté disponible
if wait_for_mongodb():
    # Guardar los datos sintéticos en MongoDB
    save_to_mongodb(synthetic_data)
    print("Datos sintéticos generados y guardados en MongoDB exitosamente!")
    
    # Mostrar algunos registros de MongoDB
    view_mongodb_data()
else:
    print("No se pudo establecer conexión con MongoDB después de varios intentos") 