"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo: vans_15q.py (Variable Ansatz)

Implemento:
- Bilkis et al. (2023): "A semi-agnostic ansatz with variable structure..."
Este algoritmo utiliza reglas de crecimiento y poda para encontrar un ansatz
óptimo y compacto para el problema de HPO.
"""

import json
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

# Me aseguro de que el entorno encuentre los módulos locales
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

def ejecutar_vans():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    
    # Normalizo el paisaje de energía
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    costes = -(acc_vals - min_acc) / (max_acc - min_acc)

    # Estado inicial de la arquitectura
    estructura = []  
    parametros = np.array([])
    
    def construir_circuito(params, struct):
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits)) # Estado inicial uniforme
        
        p_idx = 0
        for op in struct:
            if op[0] == 'ry':
                qc.ry(params[p_idx], op[1])
                p_idx += 1
            elif op[0] == 'cx':
                qc.cx(op[1], op[2])
        return qc

    def funcion_objetivo(params):
        qc = construir_circuito(params, estructura)
        probabilidades = np.abs(Statevector(qc).data)**2
        return np.sum(probabilidades * costes)

    start_time = time.time()
    max_macro_iters = 5
    umbral_poda = 1e-3  
    mejor_coste_global = float('inf')
    evals_totales = 0

    # Bucle de Crecimiento y Poda (Darwinismo Cuántico)
    for macro in range(max_macro_iters):
        # 1. CRECIMIENTO: Inserto nuevas puertas para explorar el espacio
        # 
        nuevos_params = []
        for _ in range(2): # Añado 2 CNOTs aleatorias
            q1, q2 = np.random.choice(n_qubits, 2, replace=False)
            estructura.append(('cx', q1, q2))
            
        for q in range(n_qubits): # Añado una capa de rotación Ry
            estructura.append(('ry', q))
            nuevos_params.append(np.random.normal(0, 0.01))
            
        parametros = np.concatenate([parametros, nuevos_params]) if len(parametros) > 0 else np.array(nuevos_params)
        
        # 2. OPTIMIZACIÓN LOCAL: Ajusto los ángulos del nuevo ansatz
        res = minimize(funcion_objetivo, parametros, method='L-BFGS-B', options={'maxiter': 50})
        parametros = res.x
        evals_totales += res.nfev
        coste_actual = res.fun
        
        # 3. PODA: Elimino puertas Ry cuyo ángulo es despreciable (no aportan)
        estructura_nueva = []
        parametros_nuevos = []
        p_idx = 0
        for op in estructura:
            if op[0] == 'ry':
                if abs(parametros[p_idx]) > umbral_poda:
                    estructura_nueva.append(op)
                    parametros_nuevos.append(parametros[p_idx])
                p_idx += 1
            else: # Mantengo todas las CX de momento
                estructura_nueva.append(op)
        
        estructura = estructura_nueva
        parametros = np.array(parametros_nuevos)
        mejor_coste_global = coste_actual

    # --- EXTRACCIÓN DE RESULTADOS Y MÉTRICAS ---
    qc_final = construir_circuito(parametros, estructura)
    
    # Simulo el estado final para hallar el mejor bitstring
    sv_final = Statevector(qc_final)
    idx_max = np.argmax(np.abs(sv_final.data)**2)
    bitstring = format(idx_max, f'0{n_qubits}b')
    precision_surr = resultados.get(bitstring, 0)
    
    # Auditoría de Hardware con Transpile
    qc_aud = transpile(qc_final, basis_gates=['u', 'cx'], optimization_level=3)
    cnot_finales = qc_aud.count_ops().get('cx', 0)
    profundidad = qc_aud.depth()
    
    tiempo_total = time.time() - start_time

    # Devuelvo la tupla de 6 valores para el runner
    return bitstring, precision_surr, tiempo_total, cnot_finales, profundidad, evals_totales

if __name__ == "__main__":
    print("Ejecutando VAns individual...")
    print(ejecutar_vans())