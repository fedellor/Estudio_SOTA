"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo 26: QA (Quantum Annealing con Mitigación SEMO)

Implementa la formulación QUBO del paisaje de hiperparámetros y la resolución 
mediante Temple Cuántico Simulado, seguido de mitigación de errores post-procesada.

Referencias:
1. Fedouaki et al. (2024): "Quantum Computing for Supply Chain Optimization: 
   Algorithms, Hybrid Frameworks, and Industry Applications"
2. Viet et al. (2026): "Advanced Quantum Annealing for the Bi-Objective 
   Traveling Thief Problem: An Epsilon-Constraint-based Approach"
3. Yang et al. (2026): "Toward Solution-Time Advantage With Error-Mitigated 
   Quantum Annealing for Combinatorial Optimization"
"""

import json
import os
import sys
import time
import numpy as np

# Ajusto las rutas para el entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

def ejecutar_qa_qubo():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    N_total = 2**n_qubits
    
    start_time = time.time()
    evals_totales = 0

    # 1. CONSTRUCCIÓN DEL MODELO QUBO (Viet et al., 2026)
    # Extraemos una aproximación cuadrática (2-body) del paisaje de precisión.
    # Minimizamos la energía E = -Precisión
    print("Fase 1: Mapeando el paisaje de hiperparámetros a formulación QUBO...")
    
    # Vectorización rápida para extraer la matriz de características X
    bits = np.array([[int(b) for b in format(i, f'0{n_qubits}b')] for i in range(N_total)])
    quad_terms = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            quad_terms.append(bits[:, i] * bits[:, j])
            
    X_matrix = np.hstack([bits, np.column_stack(quad_terms)])
    y_vals = np.array([-resultados.get(format(i, f'0{n_qubits}b'), 0) for i in range(N_total)])
    
    # Ajuste por mínimos cuadrados para encontrar los coeficientes QUBO
    coeficientes, _, _, _ = np.linalg.lstsq(X_matrix, y_vals, rcond=None)
    evals_totales += N_total # Evaluaciones necesarias para modelar el sistema físico
    
    # Reconstrucción de la matriz Q (Triangular superior)
    Q = np.zeros((n_qubits, n_qubits))
    idx = n_qubits
    for i in range(n_qubits):
        Q[i, i] = coeficientes[i]
        for j in range(i + 1, n_qubits):
            Q[i, j] = coeficientes[idx]
            idx += 1

    # 2. SIMULACIÓN DE QUANTUM ANNEALING (Evolución Térmica)
    print("Fase 2: Ejecutando Quantum Annealing (Simulated)...")
    def anneal_qubo(Q_mat, num_reads=50, num_sweeps=1000):
        best_x = None
        best_E = float('inf')
        
        for _ in range(num_reads):
            # Estado inicial aleatorio (Superposición equivalente)
            x = np.random.randint(0, 2, n_qubits)
            E = np.dot(x, np.dot(Q_mat, x))
            
            # Schedule de recocido (Cooling schedule)
            T = 10.0
            alpha = 0.95
            
            for _ in range(num_sweeps):
                # Fluctuación cuántica/térmica simulada
                flip_idx = np.random.randint(n_qubits)
                x_new = x.copy()
                x_new[flip_idx] = 1 - x_new[flip_idx]
                
                # Diferencia de energía local rápida
                E_new = np.dot(x_new, np.dot(Q_mat, x_new))
                delta_E = E_new - E
                
                # Criterio de aceptación de Metropolis
                if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
                    x = x_new
                    E = E_new
                    
                T *= alpha # Enfriamiento
                
            if E < best_E:
                best_E = E
                best_x = x
                
        return best_x

    estado_qubo_opt = anneal_qubo(Q)
    str_qubo_opt = "".join(map(str, estado_qubo_opt))
    acc_pre_semo = resultados.get(str_qubo_opt, 0)
    print(f" -> Solución cruda QA: |{str_qubo_opt}> (Precisión: {acc_pre_semo}%)")

    # 3. MITIGACIÓN DE ERRORES SEMO (Yang et al., 2026)
    # Spin-Error Mitigation for Optimization: Corrección de bits post-QA
    print("Fase 3: Aplicando Mitigación de Errores SEMO...")
    x_semo = estado_qubo_opt.copy()
    mejora = True
    
    while mejora:
        mejora = False
        acc_actual = resultados.get("".join(map(str, x_semo)), 0)
        
        for i in range(n_qubits):
            x_test = x_semo.copy()
            x_test[i] = 1 - x_test[i] # Flip de 1 bit (Búsqueda de vecindario local)
            str_test = "".join(map(str, x_test))
            acc_test = resultados.get(str_test, 0)
            evals_totales += 1
            
            if acc_test > acc_actual:
                x_semo = x_test
                acc_actual = acc_test
                mejora = True
                print(f"    [SEMO] Error de espín corregido en qubit {i}. Nueva precisión: {acc_actual}%")

    tiempo_total = time.time() - start_time
    mejor_estado_final = "".join(map(str, x_semo))
    precision_final = resultados.get(mejor_estado_final, 0)
    
    # 4. MÉTRICAS DE HARDWARE PARA QA
    # En Quantum Annealing no existen las puertas lógicas (CNOTs) ni la profundidad (Depth)
    # de los circuitos de puertas. El cálculo se hace de forma continua (analógica).
    # Devolvemos 0 para mantener la compatibilidad tabular con el orquestador.
    cnot_qa = 0 
    depth_qa = 0 

    return mejor_estado_final, precision_final, tiempo_total, cnot_qa, depth_qa, evals_totales

if __name__ == "__main__":
    print("Ejecutando Quantum Annealing individual...")
    print(ejecutar_qa_qubo())