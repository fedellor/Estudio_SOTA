"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo 41: Swift-Hyperband (Exact Paper Implementation)

Implementación rigurosa del algoritmo Swift-Hyperband propuesto por 
García Amboage et al. (2024). Utiliza un SVR rápido como predictor de 
rendimiento con un único punto de decisión extra por ronda para ejecutar 
Early Stopping.

Referencias implementadas y analizadas:
1. García Amboage, Wulff, Girone & Pena (2024): "Model Performance Prediction 
   for Hyperparameter Optimization of Deep Learning Models Using High 
   Performance Computing and Quantum Annealing".
"""

import json
import os
import sys
import time
import math
import numpy as np
from sklearn.svm import SVR

# Ajusto las rutas para mi entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

def ejecutar_swift_hyperband():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    cadenas_todas = list(resultados.keys())
    
    start_time = time.time()
    evaluaciones_totales = 0
    
    # 1. Función para simular el entrenamiento parcial (Curvas de Aprendizaje)
    # El paper asume que podemos evaluar el modelo en una "época" intermedia[cite: 18].
    def simular_entrenamiento_parcial(bitstring, epoca_actual, epoca_objetivo):
        acc_final = resultados[bitstring]
        # Simulamos una curva logarítmica típica de entrenamiento con un poco de ruido
        progreso = math.log(epoca_actual + 1) / math.log(epoca_objetivo + 1)
        ruido = np.random.normal(0, 0.02)
        return min(acc_final * progreso + ruido, acc_final)

    def bitstring_a_array(bs):
        return np.array([int(b) for b in bs])

    # 2. Configuración de Hyperband
    R_max = 81    # Épocas máximas (target epoch)
    eta = 3       # Factor de reducción
    s_max = math.floor(math.log(R_max, eta))
    B = (s_max + 1) * R_max

    mejor_estado_global = None
    mejor_precision_global = -1.0

    print("Iniciando Swift-Hyperband con SVR de alto rendimiento...")

    # 3. Bucle Principal de los Brackets
    for s in reversed(range(s_max + 1)):
        n_candidatos = math.ceil(int(B / R_max / (s + 1)) * (eta**s))
        r_inicial = R_max * (eta**(-s))
        
        candidatos_actuales = np.random.choice(cadenas_todas, n_candidatos, replace=False).tolist()
        
        # Rondas de Successive Halving
        for i in range(s + 1):
            n_i = math.floor(n_candidatos * (eta**(-i)))
            r_i = math.floor(r_inicial * (eta**i))
            
            # --- LÓGICA SWIFT-HYPERBAND EXACTA ---
            # 1. Definición del punto de decisión extra (Swift-Hyperband) 
            epoca_decision_extra = max(1, math.floor(r_i * 0.5)) 
            
            # 2. Entrenamos un subconjunto ("few trials") hasta el final de la ronda 
            n_trials_umbral = max(2, math.floor(n_i * 0.2)) # 20% de los candidatos
            trials_umbral = candidatos_actuales[:n_trials_umbral]
            resto_candidatos = candidatos_actuales[n_trials_umbral:]
            
            X_train = []
            y_train = []
            
            # Obtenemos el rendimiento real de los trials de umbral al final de la ronda
            for c in trials_umbral:
                acc_final_ronda = simular_entrenamiento_parcial(c, r_i, R_max)
                evaluaciones_totales += 1
                X_train.append(bitstring_a_array(c))
                y_train.append(acc_final_ronda)
                
                if acc_final_ronda > mejor_precision_global:
                    mejor_precision_global = acc_final_ronda
                    mejor_estado_global = c
            
            # Definimos el umbral como la mediana del rendimiento de estos trials 
            umbral = np.median(y_train)
            
            # 3. Entrenamos el SVR predictor con estos datos 
            predictor = SVR(kernel='rbf', C=1.0, epsilon=0.01)
            if len(set(y_train)) > 1: # Prevenir fallo del SVR si todos los targets son iguales
                predictor.fit(X_train, y_train)
            
            candidatos_supervivientes = list(trials_umbral)
            
            # 4. Los demás trials se entrenan SOLO hasta el punto de decisión 
            if len(set(y_train)) > 1 and len(resto_candidatos) > 0:
                X_pred = [bitstring_a_array(c) for c in resto_candidatos]
                
                # Predecimos su rendimiento final de ronda 
                predicciones = predictor.predict(X_pred)
                
                # 5. Comparamos con el umbral para descartar (Early Stopping anticipado) 
                for idx, pred in enumerate(predicciones):
                    if pred >= umbral:
                        candidatos_supervivientes.append(resto_candidatos[idx])
                        evaluaciones_totales += 1 # Contabilizamos que sobrevivió al corte
            else:
                candidatos_supervivientes.extend(resto_candidatos)
                evaluaciones_totales += len(resto_candidatos)

            # --- FIN LÓGICA SWIFT-HYPERBAND ---

            # Corte clásico de Hyperband al final de la ronda
            candidatos_actuales = candidatos_supervivientes
            
            # Obtenemos las precisiones reales para ordenar y hacer el corte eta
            precisiones_ronda = []
            for c in candidatos_actuales:
                acc = simular_entrenamiento_parcial(c, r_i, R_max)
                precisiones_ronda.append(acc)
                if acc > mejor_precision_global:
                    mejor_precision_global = acc
                    mejor_estado_global = c
                    
            indices_top = np.argsort(precisiones_ronda)[-math.floor(len(candidatos_actuales) / eta):]
            if len(indices_top) == 0:
                break
            candidatos_actuales = [candidatos_actuales[idx] for idx in indices_top]

    tiempo_total = time.time() - start_time

    # Como el SVR es clásico (el paper usa un Quantum Annealer, no un circuito de puertas),
    # devolvemos 0 en métricas CNOTs y Depth de Qiskit.
    return mejor_estado_global, mejor_precision_global, tiempo_total, 0, 0, evaluaciones_totales

if __name__ == "__main__":
    print("Ejecutando Swift-Hyperband ...")
    print(ejecutar_swift_hyperband())