"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo 30: QuantumNAS (Quantum Neural Architecture Search)

Implementa un framework One-Shot NAS para circuitos cuánticos.
Entrena un Super-Circuito y utiliza búsqueda evolutiva para encontrar 
la sub-arquitectura más eficiente heredando los pesos.

Referencias implementadas y analizadas:
1. Son & Park (2025): "Q-RLONAS: Towards Efficient Quantum Neural Architecture 
   Search" (Uso de One-Shot NAS y evaluación de sub-arquitecturas).
2. Kulshrestha, Liu, Ushijima-Mwesigwa & Safro (2025): "Neural Architecture 
   Search Algorithms for Quantum Autoencoders" (Automatización del diseño de circuitos).
3. Burugupalli (2025): "Transferable Agentic AI for Accelerating Quantum Algorithm 
   Discovery: Meta-Learning and AutoML for Parameterized Quantum Circuits".
"""

import json
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

# Ajuste de rutas para el entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

def ejecutar_quantumnas():
    # Carga del dataset
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    
    # Normalización del paisaje de energía
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    vector_energias = np.zeros(2**n_qubits)
    for bitstring, acc in resultados.items():
        vector_energias[int(bitstring, 2)] = -(acc - min_acc) / (max_acc - min_acc)

    # 1. Definición del Super-Circuito
    p_capas = 2
    # En cada capa hay n_qubits rotaciones RY y (n_qubits-1) entrelazadores CX
    n_rotaciones_por_capa = n_qubits
    n_cx_por_capa = n_qubits - 1
    total_params = p_capas * n_rotaciones_por_capa

    def construir_y_evaluar_subnet(parametros, mascara_ry, mascara_cx):
        """Construye un sub-circuito aplicando las máscaras arquitectónicas."""
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits)) # Estado inicial
        
        param_idx = 0
        for p in range(p_capas):
            # Capa de rotaciones RY
            for q in range(n_qubits):
                if mascara_ry[p, q] == 1:
                    qc.ry(parametros[param_idx], q)
                param_idx += 1
                
            # Capa de entrelazamiento CX
            for q in range(n_qubits - 1):
                if mascara_cx[p, q] == 1:
                    qc.cx(q, q + 1)
                    
        st = Statevector(qc).data
        energia = np.real(np.dot(np.abs(st)**2, vector_energias))
        return energia, qc

    start_time = time.time()
    evals_totales = 0

    # 2. Fase de Entrenamiento del Super-Circuito (One-Shot NAS)
    print("Fase 1: Entrenando el Super-Circuito (One-Shot NAS)...")
    mascara_ry_full = np.ones((p_capas, n_qubits))
    mascara_cx_full = np.ones((p_capas, n_qubits - 1))
    
    def coste_supercircuito(parametros):
        energia, _ = construir_y_evaluar_subnet(parametros, mascara_ry_full, mascara_cx_full)
        return energia

    params_init = np.random.uniform(-np.pi, np.pi, total_params)
    res_super = minimize(coste_supercircuito, params_init, method='L-BFGS-B', options={'maxiter': 30})
    pesos_heredados = res_super.x
    evals_totales += res_super.nfev

    # 3. Fase de Búsqueda Arquitectónica (Evolutiva / AutoML)
    print("Fase 2: Búsqueda de Sub-Arquitecturas (Subnets)...")
    pop_size = 20
    generaciones = 5
    tasa_mutacion = 0.2
    
    # Inicialización de la población (máscaras aleatorias)
    poblacion_ry = [np.random.choice([0, 1], size=(p_capas, n_qubits), p=[0.4, 0.6]) for _ in range(pop_size)]
    poblacion_cx = [np.random.choice([0, 1], size=(p_capas, n_qubits - 1), p=[0.5, 0.5]) for _ in range(pop_size)]
    
    mejor_energia_eval = float('inf')
    mejor_mascara_ry = None
    mejor_mascara_cx = None

    for gen in range(generaciones):
        fitness = []
        for i in range(pop_size):
            # Evaluamos la subnet usando los pesos heredados (Sin re-entrenar)
            energia, _ = construir_y_evaluar_subnet(pesos_heredados, poblacion_ry[i], poblacion_cx[i])
            # Penalización ligera por número de CNOTs para favorecer circuitos compactos (Hardware-Aware)
            coste_cx = np.sum(poblacion_cx[i]) * 0.001
            fitness.append(energia + coste_cx)
            evals_totales += 1
            
        # Selección de los mejores (Elitismo)
        indices_ordenados = np.argsort(fitness)
        mejores_ry = [poblacion_ry[i] for i in indices_ordenados[:pop_size//2]]
        mejores_cx = [poblacion_cx[i] for i in indices_ordenados[:pop_size//2]]
        
        if fitness[indices_ordenados[0]] < mejor_energia_eval:
            mejor_energia_eval = fitness[indices_ordenados[0]]
            mejor_mascara_ry = mejores_ry[0]
            mejor_mascara_cx = mejores_cx[0]
            
        # Cruce y Mutación para la siguiente generación
        nueva_pob_ry, nueva_pob_cx = [], []
        for _ in range(pop_size):
            p1, p2 = np.random.choice(len(mejores_ry), 2, replace=False)
            # Cruce uniforme
            hijo_ry = np.where(np.random.rand(p_capas, n_qubits) > 0.5, mejores_ry[p1], mejores_ry[p2])
            hijo_cx = np.where(np.random.rand(p_capas, n_qubits - 1) > 0.5, mejores_cx[p1], mejores_cx[p2])
            
            # Mutación
            mut_ry = np.random.rand(p_capas, n_qubits) < tasa_mutacion
            hijo_ry ^= mut_ry
            mut_cx = np.random.rand(p_capas, n_qubits - 1) < tasa_mutacion
            hijo_cx ^= mut_cx
            
            nueva_pob_ry.append(hijo_ry)
            nueva_pob_cx.append(hijo_cx)
            
        poblacion_ry = nueva_pob_ry
        poblacion_cx = nueva_pob_cx

    # 4. Fine-Tuning de la Mejor Sub-Arquitectura
    print("Fase 3: Fine-Tuning de la arquitectura óptima...")
    def coste_finetune(parametros):
        energia, _ = construir_y_evaluar_subnet(parametros, mejor_mascara_ry, mejor_mascara_cx)
        return energia
        
    res_final = minimize(coste_finetune, pesos_heredados, method='L-BFGS-B', options={'maxiter': 20})
    evals_totales += res_final.nfev
    
    tiempo_total = time.time() - start_time

    # 5. Extracción de Resultados y Auditoría de Hardware
    _, qc_optimo = construir_y_evaluar_subnet(res_final.x, mejor_mascara_ry, mejor_mascara_cx)
    st_final = Statevector(qc_optimo).data
    idx_max = np.argmax(np.abs(st_final)**2)
    best_bitstring = format(idx_max, f'0{n_qubits}b')
    precision_final = resultados.get(best_bitstring, 0)
    
    # Transpilación para métricas reales
    qc_trans = transpile(qc_optimo, basis_gates=['u', 'cx'], optimization_level=3)
    cnot_totales = qc_trans.count_ops().get('cx', 0)
    profundidad = qc_trans.depth()

    return best_bitstring, precision_final, tiempo_total, cnot_totales, profundidad, evals_totales

if __name__ == "__main__":
    print("Ejecutando QuantumNAS (One-Shot)...")
    print(ejecutar_quantumnas())