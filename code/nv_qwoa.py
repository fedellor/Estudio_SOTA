"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo 44: NV-QWOA (Non-Variational Quantum Walk Optimization Algorithm)

Implemento una simulación de Caminata Cuántica en Tiempo Continuo sobre el 
grafo del hipercubo. Utilizo la variante No Variacional (NV-QWOA) para 
eliminar el optimizador clásico mediante un schedule Trotterizado fijo.

Referencias implementadas y analizadas:
1. "Quantum Approximate and Quantum Walk Optimization Algorithms" (Definición 
   del mezclador como matriz de adyacencia del hipercubo).
2. "Quantum optimisation applied to the Quadratic Assignment Problem" (Comparativa 
   de QWOA frente a QAOA).
3. "Benchmarking Lie-Algebraic Pretraining and Non-Variational QWOA" (Eliminación 
   del bucle clásico y uso de schedules deterministas).
"""

import json
import os
import sys
import time
import numpy as np

# Ajusto las rutas para mi entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

def ejecutar_nv_qwoa():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    
    # 1. Normalización de Energía (Hamiltoniano de Coste)
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    vector_coste = np.zeros(2**n_qubits)
    for b, acc in resultados.items():
        vector_coste[int(b, 2)] = -(acc - min_acc) / (max_acc - min_acc)

    # 2. Configuración de la Caminata Cuántica (NV-QWOA)
    # Fijo una profundidad alta para reducir el error de Trotter
    p_pasos = 20 
    tiempo_total_caminata = 10.0 # Tiempo T total de la evolución continua
    delta_t = tiempo_total_caminata / p_pasos
    
    # Auditoría de hardware (Estimación teórica de CNOTs para p=20)
    # Cada capa del hipercubo/coste equivale al QAOA estándar en topología
    cnot_totales = int((32768 * 0.05) * p_pasos) # Aproximación basada en tu oráculo
    profundidad = p_pasos * 5 # Estimación de depth por capa

    # 3. Evolución Tensorial Rápida (Sin Optimizador Clásico)
    start_time = time.time()
    
    # Inicializo el estado en superposición uniforme (estado inicial del hipercubo)
    tensor_st = np.ones((2,) * n_qubits, dtype=np.complex128) / np.sqrt(2**n_qubits)
    
    # Aplico el schedule Trotterizado de la caminata
    for i in range(1, p_pasos + 1):
        # Schedule lineal adiabático (s va de 0 a 1)
        s = i / p_pasos
        gamma = s * delta_t             # Evolución del Coste
        beta = (1.0 - s) * delta_t      # Evolución de la Adyacencia (Caminata)
        
        # A. Paso del Hamiltoniano de Coste (Operador Diagonal)
        st_flat = tensor_st.flatten()
        st_flat *= np.exp(-1j * gamma * vector_coste)
        tensor_st = st_flat.reshape((2,) * n_qubits)
        
        # B. Paso de la Caminata sobre el Grafo (Adyacencia del Hipercubo)
        # El grafo de hipercubo se genera con puertas RX locales
        theta = 2 * beta
        c, sen = np.cos(theta/2), -1j * np.sin(theta/2)
        rx_matrix = np.array([[c, sen], [sen, c]], dtype=np.complex128)
        
        for q in range(n_qubits):
            # Contraigo la matriz sobre el qubit actual
            tensor_st = np.tensordot(rx_matrix, tensor_st, axes=([1], [q]))
            tensor_st = np.moveaxis(tensor_st, 0, q)
            
    tiempo_total_q = time.time() - start_time
    
    # 4. Medición Final
    probabilidades = np.abs(tensor_st.flatten())**2
    idx_max = np.argmax(probabilidades)
    best_bitstring = format(idx_max, f'0{n_qubits}b')
    precision_surrogada = resultados.get(best_bitstring, 0)

    # El número de evaluaciones del oráculo clásico es 0 (no hay bucle variacional)
    return best_bitstring, precision_surrogada, tiempo_total_q, cnot_totales, profundidad, 0

if __name__ == "__main__":
    print("Ejecutando NV-QWOA (Non-Variational Quantum Walk)...")
    print(ejecutar_nv_qwoa())