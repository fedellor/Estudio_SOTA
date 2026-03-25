"""
TFG: Optimizacion Cuantica de Hiperparametros (HPO)
Archivo 14: CL-QAOA (Cyclic Layerwise QAOA)
Basado en: Jang et al. (2026 arXiv) - "A Cyclic Layerwise QAOA Training".
Estrategia de entrenamiento por capas para evadir Barren Plateaus.
"""
import json
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Ajusto las rutas para el entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

def ejecutar_cl_qaoa():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    
    # Normalizo el paisaje de energía (Hamiltoniano de Coste)
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    vector_hp = np.zeros(2**n_qubits)
    for b, acc in resultados.items():
        vector_hp[int(b, 2)] = -(acc - min_acc) / (max_acc - min_acc)

    p_max = 4  # Profundidad objetivo
    params_capas = [] 

    def simulacion_estado(gammas, betas):
        """Simulación rápida del estado aplicando p capas de QAOA"""
        psi = np.ones(2**n_qubits, dtype=np.complex128) / np.sqrt(2**n_qubits)
        
        for g, b in zip(gammas, betas):
            # Evolución de Coste (Diagonal)
            psi = psi * np.exp(-1j * g * vector_hp)
            
            # Evolución Mezcladora (RX en cada qubit)
            qc_mix = QuantumCircuit(n_qubits)
            qc_mix.rx(2 * b, range(n_qubits))
            psi = Statevector(psi).evolve(qc_mix).data
            
        return psi

    def funcion_coste_layerwise(params_nuevos, params_anteriores):
        gs = [p[0] for p in params_anteriores] + [params_nuevos[0]]
        bs = [p[1] for p in params_anteriores] + [params_nuevos[1]]
        st = simulacion_estado(gs, bs)
        return np.dot(np.abs(st)**2, vector_hp)

    start_time = time.time()
    total_evals = 0

    # FASE 1: CRECIMIENTO (Layerwise Training)
    for p in range(1, p_max + 1):
        res = minimize(
            funcion_coste_layerwise, 
            [0.1, 0.1], 
            args=(params_capas,), 
            method='COBYLA'
        )
        params_capas.append(res.x)
        total_evals += res.nfev

    # FASE 2: REFINAMIENTO CÍCLICO
    for ciclo in range(1): 
        for i in range(p_max):
            def funcion_coste_ciclo(p_i):
                temp_params = list(params_capas)
                temp_params[i] = p_i
                gs = [p[0] for p in temp_params]
                bs = [p[1] for p in temp_params]
                st = simulacion_estado(gs, bs)
                return np.dot(np.abs(st)**2, vector_hp)

            res = minimize(funcion_coste_ciclo, params_capas[i], method='COBYLA')
            params_capas[i] = res.x
            total_evals += res.nfev

    tiempo_total = time.time() - start_time

    # Resultado Final
    gs_final = [p[0] for p in params_capas]
    bs_final = [p[1] for p in params_capas]
    st_final = simulacion_estado(gs_final, bs_final)
    
    idx_max = np.argmax(np.abs(st_final)**2)
    estado_bin = format(idx_max, f'0{n_qubits}b')
    precision_surrogada = resultados.get(estado_bin, 0)

    # Cálculo de CNOTs y Profundidad para el hardware
    cnot_base = 22698 
    cnot_totales = cnot_base * p_max
    
    # Estimación de profundidad de hardware (asumiendo paralelización máxima de un grafo completo)
    profundidad_estimada = p_max * (n_qubits * 2) 

    # Retorno exacto en el formato exigido por runner_experimentos.py
    return estado_bin, precision_surrogada, tiempo_total, cnot_totales, profundidad_estimada, total_evals

if __name__ == "__main__":
    print("Este archivo debe ser llamado desde runner_experimentos.py")