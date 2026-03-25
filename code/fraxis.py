"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo 36: FRAXIS / FQS (Free-Axis / Free-Quaternion Selection)

Implemento un optimizador secuencial de eje libre para puertas de un solo qubit,
incorporando el método de congelación de puertas (Gate Freezing) para ahorrar recursos.

Referencias implementadas y analizadas:
1. Pankkonen, Raasakka, Marchesin, Tittonen & Ylinen (2025): "Gate Freezing Method 
   for Gradient-Free Variational Quantum Algorithms in Circuit Optimization".
2. Sato et al. (2023): "Variational quantum algorithm for generalized eigenvalue problems".
3. Endo et al. (2023): "Optimal parameter configurations for sequential optimization 
   of the variational quantum eigensolver".
"""

import json
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

# Ajusto las rutas para mi entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

def ejecutar_fraxis():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    
    # 1. Normalización del Paisaje de Energía
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    vector_energias = np.zeros(2**n_qubits)
    for bitstring, acc in resultados.items():
        vector_energias[int(bitstring, 2)] = -(acc - min_acc) / (max_acc - min_acc)

    # 2. Arquitectura del Circuito (Ansatz de Eje Libre)
    # Utilizo puertas U(theta, phi, lambda) que cubren todo el grupo SU(2)
    p_capas = 2
    num_gates = p_capas * n_qubits
    # Cada puerta U tiene 3 parámetros. Total = p_capas * n_qubits * 3
    num_params = num_gates * 3

    def construir_circuito(parametros):
        qc = QuantumCircuit(n_qubits)
        idx = 0
        for _ in range(p_capas):
            # Capa de rotación de eje libre
            for q in range(n_qubits):
                theta, phi, lam = parametros[idx], parametros[idx+1], parametros[idx+2]
                qc.u(theta, phi, lam, q)
                idx += 3
            # Capa de entrelazamiento
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
        return qc

    # Auditoría de Hardware
    qc_aud = transpile(construir_circuito(np.zeros(num_params)), basis_gates=['u', 'cx'], optimization_level=3)
    cnot_totales = qc_aud.count_ops().get('cx', 0)
    profundidad = qc_aud.depth()

    # 3. Motor de Evaluación
    def evaluar_coste(parametros):
        qc = construir_circuito(parametros)
        st = Statevector(qc).data
        probabilidades = np.abs(st)**2
        return np.dot(probabilidades, vector_energias)

    # 4. Bucle de Optimización FQS con Gate Freezing (Pankkonen et al., 2025)
    start_time = time.time()
    evals_totales = 0
    num_sweeps = 3 
    umbral_congelacion = 1e-4 # Si el cambio de energía es menor a esto, congelo la puerta
    
    parametros_actuales = np.random.uniform(-np.pi, np.pi, num_params)
    puertas_congeladas = [False] * num_gates
    
    for sweep in range(num_sweeps):
        for g_idx in range(num_gates):
            if puertas_congeladas[g_idx]:
                continue # Salto la optimización de esta puerta para ahorrar recursos
                
            p_inicio = g_idx * 3
            p_fin = p_inicio + 3
            params_puerta_original = parametros_actuales[p_inicio:p_fin].copy()
            
            energia_antes = evaluar_coste(parametros_actuales)
            evals_totales += 1
            
            # Para simular la búsqueda del eje óptimo (Fraxis/FQS), optimizo los 3 
            # ángulos de Euler de la puerta U local simultáneamente. En hardware físico,
            # esto equivaldría a las 6 mediciones proyectivas analíticas de Fraxis.
            def coste_local(params_locales):
                params_temp = parametros_actuales.copy()
                params_temp[p_inicio:p_fin] = params_locales
                return evaluar_coste(params_temp)
                
            # Optimizo la rotación libre en la esfera de Bloch
            res_local = minimize(coste_local, params_puerta_original, method='COBYLA', options={'maxiter': 15})
            evals_totales += res_local.nfev
            
            parametros_actuales[p_inicio:p_fin] = res_local.x
            energia_despues = res_local.fun
            
            # Heurística de Gate Freezing: si la mejora es despreciable, la congelo
            mejora = energia_antes - energia_despues
            if mejora > 0 and mejora < umbral_congelacion:
                puertas_congeladas[g_idx] = True

    tiempo_total_q = time.time() - start_time
    
    # 5. Extracción del Resultado Final
    qc_final = construir_circuito(parametros_actuales)
    probabilidades_finales = np.abs(Statevector(qc_final).data)**2
    
    idx_max = np.argmax(probabilidades_finales)
    best_bitstring = format(idx_max, f'0{n_qubits}b')
    precision_surrogada = resultados.get(best_bitstring, 0)

    return best_bitstring, precision_surrogada, tiempo_total_q, cnot_totales, profundidad, evals_totales

if __name__ == "__main__":
    print("Ejecutando FRAXIS/FQS (Free-Axis Selection Optimizer)...")
    print(ejecutar_fraxis())