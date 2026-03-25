"""
TFG: Optimizacion Cuantica de Hiperparametros (HPO)
Archivo 15: QBO-QNN (Batch Active Learning + SPSA)
Motor de inferencia ultrarrapido SOTA mediante gradiente estocastico.
Referencias: Dai et al. (2023), Kale et al. (2024), Sa et al. (2024).
"""
import json
import os
import sys
import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp

# Ajusto las rutas para mi entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

def ejecutar_qbo_qnn_spsa():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    cadenas_todas = list(resultados.keys())
    
    y_min_global = min(resultados.values())
    y_max_global = max(resultados.values())
    
    def normalizar(y):
        return 2 * (y - y_min_global) / (y_max_global - y_min_global) - 1
        
    def desnormalizar(y_norm):
        return (y_norm + 1) / 2 * (y_max_global - y_min_global) + y_min_global

    # 1. Arquitectura QNN (Ansatz p=1)
    qc_ansatz = QuantumCircuit(n_qubits)
    theta_pesos = ParameterVector('w', n_qubits * 2) 
    
    idx_p = 0
    for i in range(n_qubits):
        qc_ansatz.ry(theta_pesos[idx_p], i)
        idx_p += 1
        qc_ansatz.rz(theta_pesos[idx_p], i)
        idx_p += 1
    for i in range(n_qubits - 1):
        qc_ansatz.cx(i, i+1)

    # Extraigo métricas de hardware mediante transpilación real
    qc_aud = transpile(qc_ansatz, basis_gates=['u', 'cx'], optimization_level=3)
    cnot_totales = qc_aud.count_ops().get('cx', 0)
    profundidad = qc_aud.depth()
                
    terminos_pauli = [("I" * i + "Z" + "I" * (n_qubits - 1 - i), 1.0 / n_qubits) for i in range(n_qubits)]
    observable_global = SparsePauliOp.from_list(terminos_pauli)

    def predecir_qnn_batch(x_strings, pesos):
        # Asigno pesos al circuito una sola vez para el batch
        qc_bound = qc_ansatz.assign_parameters(pesos)
        predicciones_norm = []
        for x_bin in x_strings:
            # Cargo el bitstring como estado inicial
            st_inicial = Statevector.from_label(x_bin[::-1])
            st_final = st_inicial.evolve(qc_bound)
            valor_pauli = np.real(st_final.expectation_value(observable_global))
            predicciones_norm.append(valor_pauli)
        return np.array(predicciones_norm)

    # 2. Inicialización
    n_inicial = 50 
    # El runner gestiona la semilla global para la reproducibilidad
    x_train = list(np.random.choice(cadenas_todas, n_inicial, replace=False))
    y_train_norm = np.array([normalizar(resultados[x]) for x in x_train])
    
    pesos_optimos = np.random.uniform(-0.1, 0.1, n_qubits * 2)
    start_time = time.time()

    # 3. BUCLE DE OPTIMIZACIÓN (Batch Mode + SPSA)
    n_ciclos_bo = 3 
    muestras_exploracion = 250 
    batch_size = 3
    
    for ciclo in range(1, n_ciclos_bo + 1):
        def funcion_perdida(pesos):
            y_preds_norm = predecir_qnn_batch(x_train, pesos)
            mse = np.mean((y_train_norm - y_preds_norm)**2)
            return mse + 0.05 * np.sum(pesos**2) # MSE + Regularización L2

        # OPTIMIZACIÓN SPSA (Simultaneous Perturbation Stochastic Approximation)
        iteraciones_spsa = 25
        a_s, c_s, A_s, alpha_s, gamma_s = 0.5, 0.1, 5, 0.602, 0.101
        
        for k in range(iteraciones_spsa):
            ak = a_s / (k + 1 + A_s)**alpha_s
            ck = c_s / (k + 1)**gamma_s
            delta = np.random.choice([-1, 1], size=len(pesos_optimos))
            
            # SPSA solo requiere dos evaluaciones del gradiente por paso
            f_plus = funcion_perdida(pesos_optimos + ck * delta)
            f_minus = funcion_perdida(pesos_optimos - ck * delta)
            
            g_k = (f_plus - f_minus) / (2 * ck * delta)
            pesos_optimos = pesos_optimos - ak * g_k
        
        # Fase de Inferencia: escaneo candidatos fuera del conjunto de entrenamiento
        candidatos_disponibles = [c for c in cadenas_todas if c not in x_train]
        x_candidatos = np.random.choice(candidatos_disponibles, muestras_exploracion, replace=False)
        preds_candidatos_norm = predecir_qnn_batch(x_candidatos, pesos_optimos)
        
        # Selecciono los mejores basándome en la predicción de la QNN
        top_indices = np.argsort(preds_candidatos_norm)[-batch_size:]
        
        for idx in reversed(top_indices): 
            mejor_candidato = x_candidatos[idx]
            precision_real_ciclo = resultados[mejor_candidato]
            
            x_train.append(mejor_candidato)
            y_train_norm = np.append(y_train_norm, normalizar(precision_real_ciclo))

    tiempo_total_q = time.time() - start_time
    
    # Recupero las precisiones reales para el informe final
    y_train_real = [desnormalizar(y) for y in y_train_norm]
    mejor_historico_idx = np.argmax(y_train_real)
    best_bitstring = x_train[mejor_historico_idx]
    precision_surrogada = y_train_real[mejor_historico_idx]
    evals_totales = len(x_train)

    # Retorno exacto en el formato exigido por runner_experimentos.py
    return best_bitstring, precision_surrogada, tiempo_total_q, cnot_totales, profundidad, evals_totales

if __name__ == "__main__":
    print("Este archivo debe ser llamado desde runner_experimentos.py")