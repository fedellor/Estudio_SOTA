"""
TFG: Optimización Cuántica de Hiperparámetros (HPO) - 15 Qubits
Archivo 01: Benchmark Clásico y Modelo Subrogado (Random Forest)
"""
import json
import random
import os
import gc
import sys
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from itertools import product

ruta_script = os.path.dirname(os.path.abspath(__file__))
os.chdir(ruta_script)
sys.path.append(ruta_script)

import config
from data_processors.event_logs import EventLog, get_window_size
from encoders_and_decoders import EventTransformer
from training import fit
from evaluation import test

# 1. ESPACIO DE BÚSQUEDA (15 QUBITS = 32,768 COMBINACIONES)
ESPACIO_HIPERPARAMETROS = {
    'BATCH_SIZE': [16, 32, 48, 64, 96, 128, 192, 256],           # 3 Qubits
    'LEARNING_RATE': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6], # 3 Qubits
    'EMB_SIZE': [16, 32, 64, 96, 128, 192, 256, 512],            # 3 Qubits
    'ATTN_HEADS': [1, 2, 4, 8],                                  # 2 Qubits
    'ENC_LAYERS': [1, 2, 3, 4],                                  # 2 Qubits
    'DEC_LAYERS': [1, 2, 3, 4]                                   # 2 Qubits
}

def bitstring_a_hiperparametros(bitstring):
    """Convierte 15 bits en valores de hiperparámetros de PyTorch."""
    idx_batch = int(bitstring[0:3], 2)
    idx_lr = int(bitstring[3:6], 2)
    idx_emb = int(bitstring[6:9], 2)
    idx_heads = int(bitstring[9:11], 2)
    idx_enc = int(bitstring[11:13], 2)
    idx_dec = int(bitstring[13:15], 2)
    
    return [
        ESPACIO_HIPERPARAMETROS['BATCH_SIZE'][idx_batch],
        ESPACIO_HIPERPARAMETROS['LEARNING_RATE'][idx_lr],
        ESPACIO_HIPERPARAMETROS['EMB_SIZE'][idx_emb],
        ESPACIO_HIPERPARAMETROS['ATTN_HEADS'][idx_heads],
        ESPACIO_HIPERPARAMETROS['ENC_LAYERS'][idx_enc],
        ESPACIO_HIPERPARAMETROS['DEC_LAYERS'][idx_dec]
    ]

# 2. CARGA GLOBAL DEL DATASET Y CONFIGURACIÓN
print("Cargando el EventLog (env_permit) en memoria...")
# Uso el Fold 0 para la exploración inicial rápida
EVENT_LOG = EventLog('env_permit', '0', start_of_suffix=True)
WINDOW_SIZE = get_window_size('auto', EVENT_LOG)
LOG_DICT = EVENT_LOG.to_dict()

# Convertir Int32 a Int64 (Long) para PyTorch
for data_split in [EVENT_LOG.train_data, EVENT_LOG.val_data, EVENT_LOG.test_data]:
    for key, value in vars(data_split).items():
        if isinstance(value, np.ndarray) and value.dtype == np.int32:
            setattr(data_split, key, value.astype(np.int64))

# 3. CONEXIÓN CON EL ENTRENAMIENTO REAL
def evaluar_transformer_real(params):
    batch, lr, emb, heads, enc, dec = params
    
    # REGLA ARQUITECTÓNICA CRÍTICA:
    # PyTorch falla si la dimensión del embedding no es divisible por las cabezas.
    if emb % heads != 0: 
        return random.uniform(10.0, 20.0) # Precisión muy baja (Penalización)
    
    # 1. Inyectar Hiperparámetros al config del profesor
    config.BATCH_SIZE = int(batch)
    # Configuro el CosineAnnealingLR con el LR inicial y uno final más pequeño
    config.T_LEARNING_RATE = [float(lr), float(lr) / 10] 
    
    try:
        # 2. Instanciar el Transformer con los nuevos bloques
        model = EventTransformer(
            cat_attributes = LOG_DICT['cat_attributes'],
            num_attributes = LOG_DICT['num_attributes'],
            embedding_size = int(emb),
            encoder_layers = int(enc),
            decoder_layers = int(dec),
            encoder_attn_heads = int(heads)
        )
        
        # 3. Entrenamiento Súper Rápido (Solo 2 Epochs para evaluar potencial)
        model = fit(model, EVENT_LOG, WINDOW_SIZE, epochs=5)
        
        # 4. Evaluación de la métrica (Damerau-Levenshtein similarity)
        similarities_dict = test(model, EVENT_LOG, WINDOW_SIZE)
        
        # El archivo devuelve un dict: {'concept:name': 0.83, 'org:group': 0.80...}
        # Hago la media de todas las variables y la paso a porcentaje (%)
        precision_media = (sum(similarities_dict.values()) / len(similarities_dict.values())) * 100
        
        # 5. Limpieza vital de memoria para evitar OutOfMemory (OOM) en bucle
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        return precision_media

    except Exception as e:
        print(f"Error entrenando configuración {params}: {e}")
        return random.uniform(10.0, 20.0)

# 4. PIPELINE DEL MODELO SUBROGADO
def generar_paisaje_cuantico_15q():
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    print("\n=========================================================")
    print(" PIPELINE SUBROGADO (15 QUBITS / 32,768 ESTADOS) ")
    print("=========================================================")
    
    todos_los_estados = ["".join(seq) for seq in product("01", repeat=15)]
    
    # Muestreo Estratégico (250 muestras = ~0.7% del espacio)
    tamano_muestra = 250
    print(f"\nEvaluando {tamano_muestra} configuraciones reales en PyTorch...")
    estados_entrenamiento = random.sample(todos_los_estados, tamano_muestra)
    
    X_train, y_train = [], []
    
    for i, estado in enumerate(estados_entrenamiento):
        print(f"[{i+1}/{tamano_muestra}] Entrenando estado |{estado}>...")
        params = bitstring_a_hiperparametros(estado)
        acc = evaluar_transformer_real(params)
        
        X_train.append([float(p) for p in params])
        y_train.append(acc)
        print(f"   -> Precisión media extraída: {acc:.2f}%\n")
        
    print("Entrenando Random Forest Regressor (Modelo Subrogado)...")
    modelo_subrogado = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo_subrogado.fit(X_train, y_train)
    print(f"   -> R^2 Score (Ajuste): {modelo_subrogado.score(X_train, y_train):.4f}")
    
    print("\nInfiriendo las 32,518 combinaciones restantes en segundos...")
    diccionario_resultados = {}
    
    # Vectorización para inferencia ultrarrápida
    X_pred_str = [estado for estado in todos_los_estados if estado not in estados_entrenamiento]
    X_pred_params = [bitstring_a_hiperparametros(estado) for estado in X_pred_str]
    X_pred_floats = [[float(p) for p in params] for params in X_pred_params]
        
    y_pred = modelo_subrogado.predict(X_pred_floats)
    
    # Calculo el R^2 y lo redondeo a 4 decimales para que quede limpio
    r2_calculado = round(modelo_subrogado.score(X_train, y_train), 4)
    
    # Guardado de reales
    for i, estado in enumerate(estados_entrenamiento):
        diccionario_resultados[estado] = round(y_train[i], 2)
        
    # Guardado de predicciones con penalización estricta por arquitectura
    for i, estado in enumerate(X_pred_str):
        params = X_pred_params[i]
        if params[2] % params[3] != 0: # emb % heads != 0
            diccionario_resultados[estado] = round(random.uniform(10.0, 20.0), 2)
        else:
            diccionario_resultados[estado] = round(y_pred[i], 2)
        
    datos = {
        "dataset_info": "Process Mining (env_permit) - Fold 0",
        "qubits": 15,
        "total_estados": 32768,
        "metodo_generacion": f"Modelo Subrogado (RF entrenado con {tamano_muestra} muestras)",
        "r2_score": r2_calculado, 
        "resultados_precision": diccionario_resultados
    }
    
    with open(ruta_json, 'w') as f:
        json.dump(datos, f, indent=4)
        
    print(f"\n¡ÉXITO! Paisaje de energía guardado en: {ruta_json}")

if __name__ == "__main__":
    generar_paisaje_cuantico_15q()