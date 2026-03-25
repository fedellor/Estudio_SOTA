"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo 32: QMKL (Quantum Multiple Kernel Learning)

Implemento un modelo subrogado basado en la combinación óptima de múltiples 
kernels cuánticos para predecir el paisaje de hiperparámetros.

Referencias implementadas y analizadas:
1. Vedaie, Noori, Oberoi, Sanders & Zahedinejad (2020): "Quantum Multiple 
   Kernel Learning" (Optimización de pesos de combinación de kernels).
2. Miyabe et al. (2023): "Quantum Multiple Kernel Learning in Financial 
   Classification Tasks" (Aplicación híbrida sobre datos tabulares/clásicos).
3. Fu, Zhang, Yang & Qi (2024): "Exploiting A Quantum Multiple Kernel 
   Learning Approach For Low-Resource Spoken Command Recognition" 
   (Mejora de representación mediante proyecciones multi-kernel).
"""

import json
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize
from sklearn.svm import SVR
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit.quantum_info import Statevector

# Ajusto las rutas para mi entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

def ejecutar_qmkl():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    cadenas_todas = list(resultados.keys())
    
    # Utilizo un núcleo reducido (8 qubits) para evitar sobreajuste y ruido,
    # siguiendo la estrategia de Chowdhury (2025) y Miyabe (2023).
    n_qubits_kernel = 8
    
    # 1. Defino mi "Pool" de Kernels Cuánticos (Feature Maps)
    # Diferentes arquitecturas capturan distintas correlaciones de los datos
    feature_maps = [
        ZZFeatureMap(feature_dimension=n_qubits_kernel, reps=1, entanglement='linear'),
        ZZFeatureMap(feature_dimension=n_qubits_kernel, reps=2, entanglement='linear'),
        ZFeatureMap(feature_dimension=n_qubits_kernel, reps=2)
    ]
    
    # Auditoría de hardware: calculo las métricas físicas promediadas
    cnots_totales = 0
    profundidad_max = 0
    for fm in feature_maps:
        qc_aud = transpile(fm, basis_gates=['u', 'cx'], optimization_level=3)
        cnots_totales += qc_aud.count_ops().get('cx', 0)
        profundidad_max = max(profundidad_max, qc_aud.depth())
        
    cnot_media = int(cnots_totales / len(feature_maps))

    # 2. Preparación de Datos (Muestreo Low-Resource)
    n_muestras = 50
    x_train_bin = np.random.choice(cadenas_todas, n_muestras, replace=False)
    y_train = np.array([resultados[c] for c in x_train_bin])
    
    # Centrado del target para el cálculo de alineación (Kernel Alignment)
    y_centered = y_train - np.mean(y_train)
    Y_target = np.outer(y_centered, y_centered)

    def encode_data(bin_str):
        return np.array([int(b) for b in bin_str[:n_qubits_kernel]]) * np.pi 

    X_train = np.array([encode_data(c) for c in x_train_bin])
    
    # 3. Construcción de las Matrices de Gram base
    def calcular_matriz_gram(X1, X2, fm):
        matriz = np.zeros((X1.shape[0], X2.shape[0]))
        evals = 0
        for i, x1 in enumerate(X1):
            psi1 = Statevector(fm.assign_parameters(x1))
            for j, x2 in enumerate(X2):
                psi2 = Statevector(fm.assign_parameters(x2))
                matriz[i, j] = np.abs(psi1.inner(psi2))**2
                evals += 1
        return matriz, evals

    start_time = time.time()
    evaluaciones_cuanticas = 0
    
    print("Calculando matrices de Gram para múltiples kernels...")
    matrices_gram_train = []
    for fm in feature_maps:
        K_train, evals = calcular_matriz_gram(X_train, X_train, fm)
        matrices_gram_train.append(K_train)
        evaluaciones_cuanticas += evals

    # 4. Optimización de los Pesos del MKL (Kernel Alignment)
    # Busco la combinación convexa de mu_i que maximice la similitud con Y_target
    def funcion_coste_mkl(mu):
        K_comb = sum(mu[i] * matrices_gram_train[i] for i in range(len(feature_maps)))
        # Alineamiento del kernel (Frobenius inner product)
        alineamiento = np.sum(K_comb * Y_target)
        norma_K = np.linalg.norm(K_comb, 'fro')
        norma_Y = np.linalg.norm(Y_target, 'fro')
        if norma_K == 0 or norma_Y == 0:
            return 0
        return -(alineamiento / (norma_K * norma_Y)) # Minimizo el negativo del alineamiento

    # Restricciones: suma(mu) = 1, mu >= 0
    restricciones = ({'type': 'eq', 'fun': lambda mu: np.sum(mu) - 1.0})
    limites = [(0, 1) for _ in range(len(feature_maps))]
    mu_init = np.ones(len(feature_maps)) / len(feature_maps)
    
    res_mu = minimize(funcion_coste_mkl, mu_init, method='SLSQP', bounds=limites, constraints=restricciones)
    pesos_optimos = res_mu.x
    print(f"Pesos óptimos QMKL hallados: {pesos_optimos}")

    # 5. Entrenamiento del Regresor SVR Clásico
    K_train_optimo = sum(pesos_optimos[i] * matrices_gram_train[i] for i in range(len(feature_maps)))
    modelo_qsvr = SVR(kernel='precomputed', C=1.0, epsilon=0.1)
    modelo_qsvr.fit(K_train_optimo, y_train)

    # 6. Inferencia (Búsqueda sobre candidatos restantes)
    n_test = 150
    candidatos_test = [c for c in cadenas_todas if c not in x_train_bin]
    x_test_bin = np.random.choice(candidatos_test, n_test, replace=False)
    X_test = np.array([encode_data(c) for c in x_test_bin])
    
    matrices_gram_test = []
    for fm in feature_maps:
        K_test, evals = calcular_matriz_gram(X_test, X_train, fm)
        matrices_gram_test.append(K_test)
        evaluaciones_cuanticas += evals
        
    K_test_optimo = sum(pesos_optimos[i] * matrices_gram_test[i] for i in range(len(feature_maps)))
    
    predicciones = modelo_qsvr.predict(K_test_optimo)
    idx_mejor = np.argmax(predicciones)
    best_bitstring = x_test_bin[idx_mejor]
    precision_surrogate = resultados[best_bitstring]
    
    tiempo_total = time.time() - start_time

    return best_bitstring, precision_surrogate, tiempo_total, cnot_media, profundidad_max, evaluaciones_cuanticas

if __name__ == "__main__":
    print("Ejecutando Quantum Multiple Kernel Learning (QMKL)...")
    print(ejecutar_qmkl())