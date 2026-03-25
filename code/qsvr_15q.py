"""
TFG: Optimizacion Cuantica de Hiperparametros (HPO)
Archivo: qsvr_15q.py (Quantum Support Vector Regression)

Implemento un modelo de regresión basado en Kernels Cuánticos. 
Utilizo el ZZFeatureMap para mapear hiperparámetros clásicos a estados 
cuánticos y calcular su similitud en el espacio de Hilbert.

Referencias:
1. Kundavaram et al. (2025): Uso de ZZFeatureMap para alta dimensionalidad.
2. Chowdhury (2025): Analisis de RMSE y repetibilidad de Kernels.
3. Ahmad et al. (2026): Benchmarking de modelos hibridos.
"""

import json
import os
import time
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from qiskit import transpile
from sklearn.svm import SVR

def ejecutar_qsvr():
    # Localizo y cargo el dataset de precisiones subrogadas
    ruta_json = os.path.join(os.path.dirname(__file__), 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    cadenas_todas = list(resultados.keys())
    
    # Configuro el mapa de características (Feature Map)
    # Según Chowdhury (2025), 8 qubits son suficientes para capturar la estructura local
    n_qubits_kernel = 8
    reps = 2
    feature_map = ZZFeatureMap(feature_dimension=n_qubits_kernel, reps=reps, entanglement='linear')
    
    # --- AUDITORÍA DE HARDWARE ---
    # Calculo las métricas físicas transpilando a las puertas base de IBM
    qc_aud = transpile(feature_map, basis_gates=['cx', 'u'], optimization_level=3)
    cnot_por_eval = qc_aud.count_ops().get('cx', 0)
    profundidad = qc_aud.depth()
    
    # --- PREPARACIÓN DE DATOS ---
    n_muestras = 60
    # La semilla la gestiona el runner globalmente para asegurar repetibilidad
    x_train_bin = np.random.choice(cadenas_todas, n_muestras, replace=False)
    y_train = np.array([resultados[c] for c in x_train_bin])

    def encode_data(bin_str):
        # Mapeo los primeros n bits a ángulos en el intervalo [0, pi]
        return np.array([int(b) for b in bin_str[:n_qubits_kernel]]) * np.pi 

    X_train = np.array([encode_data(c) for c in x_train_bin])
    
    # --- CONSTRUCCIÓN DEL KERNEL CUÁNTICO ---
    def calcular_matriz_gram(X1, X2):
        matriz = np.zeros((X1.shape[0], X2.shape[0]))
        # El kernel cuántico es la fidelidad: |<psi(x1)|psi(x2)>|^2
        for i, x1 in enumerate(X1):
            # Asigno parámetros al mapa de características para x1
            psi1 = Statevector(feature_map.assign_parameters(x1))
            for j, x2 in enumerate(X2):
                psi2 = Statevector(feature_map.assign_parameters(x2))
                # Calculo el producto escalar en el espacio de Hilbert
                matriz[i, j] = np.abs(psi1.inner(psi2))**2
        return matriz

    start_time = time.time()

    # Fase de Entrenamiento (Matriz de Gram simétrica)
    gram_train = calcular_matriz_gram(X_train, X_train)
    evals_train = X_train.shape[0] * X_train.shape[0]
    
    # Entreno el SVR clásico usando el kernel precomputado cuánticamente
    modelo_qsvr = SVR(kernel='precomputed')
    modelo_qsvr.fit(gram_train, y_train)
    
    # Fase de Inferencia (Muestreo de test)
    n_test = 100
    candidatos_test = [c for c in cadenas_todas if c not in x_train_bin]
    x_test_bin = np.random.choice(candidatos_test, n_test, replace=False)
    X_test = np.array([encode_data(c) for c in x_test_bin])
    
    gram_test = calcular_matriz_gram(X_test, X_train)
    evals_test = X_test.shape[0] * X_train.shape[0]
    
    # Realizo las predicciones
    predicciones = modelo_qsvr.predict(gram_test)
    
    # Identifico el mejor bitstring según el QSVR
    idx_mejor = np.argmax(predicciones)
    best_bitstring = x_test_bin[idx_mejor]
    precision_surrogate = resultados[best_bitstring]
    
    tiempo_total = time.time() - start_time
    evaluaciones_totales = evals_train + evals_test

    # Devuelvo la tupla de 6 variables para el runner
    return best_bitstring, precision_surrogate, tiempo_total, cnot_por_eval, profundidad, evaluaciones_totales

if __name__ == "__main__":
    print("Ejecución de prueba del QSVR...")
    print(ejecutar_qsvr())