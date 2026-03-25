"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo 35: Rotosolve (Sequential Single-Qubit Optimizer)

Implemento el optimizador analítico Rotosolve, que actualiza secuencialmente 
cada parámetro resolviendo el mínimo exacto de su función de coste sinusoidal.

Referencias implementadas y analizadas:
1. Pankkonen, Raasakka, Marchesin & Tittonen (2025): "Enhancing Hybrid Methods 
   in Parameterized Quantum Circuit Optimization".
2. Watanabe, Raymond, Ohnishi, Kaminishi & Sugawara (2023): "Optimizing 
   Parameterized Quantum Circuits With Free-Axis Single-Qubit Gates".
3. Watanabe et al. (2021): "Optimizing Parameterized Quantum Circuits with 
   Free-Axis Selection".
"""

import json
import os
import sys
import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Statevector

# Ajusto las rutas para el entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

def ejecutar_rotosolve():
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

    # 2. Arquitectura del Circuito (Ansatz de Eje Fijo)
    # Como indican Watanabe et al. (2023), Rotosolve asume ejes fijos (RY, RZ)
    ansatz = EfficientSU2(num_qubits=n_qubits, su2_gates=['ry', 'rz'], entanglement='linear', reps=2)
    num_params = ansatz.num_parameters
    
    # Auditoría de Hardware
    qc_aud = transpile(ansatz, basis_gates=['u', 'cx'], optimization_level=3)
    cnot_totales = qc_aud.count_ops().get('cx', 0)
    profundidad = qc_aud.depth()

    # 3. Motor de Evaluación
    def evaluar_coste(parametros):
        qc_bound = ansatz.assign_parameters(parametros)
        st = Statevector(qc_bound).data
        probabilidades = np.abs(st)**2
        return np.dot(probabilidades, vector_energias)

    # 4. Bucle de Optimización Rotosolve
    start_time = time.time()
    evals_totales = 0
    num_sweeps = 3 # Número de pasadas completas por todos los parámetros
    
    # Inicialización aleatoria de los parámetros
    parametros_actuales = np.random.uniform(-np.pi, np.pi, num_params)
    
    for sweep in range(num_sweeps):
        for i in range(num_params):
            # Guardo el valor original por seguridad
            val_orig = parametros_actuales[i]
            
            # Evaluación 1: theta = 0
            parametros_actuales[i] = 0.0
            e_0 = evaluar_coste(parametros_actuales)
            
            # Evaluación 2: theta = pi/2
            parametros_actuales[i] = np.pi / 2.0
            e_plus = evaluar_coste(parametros_actuales)
            
            # Evaluación 3: theta = -pi/2
            parametros_actuales[i] = -np.pi / 2.0
            e_minus = evaluar_coste(parametros_actuales)
            
            evals_totales += 3
            
            # Reconstrucción analítica de la curva f(theta) = A*sin(theta + phi) + C
            # C = (E(+) + E(-)) / 2
            # A*sin(phi) = E(0) - C
            # A*cos(phi) = (E(+) - E(-)) / 2
            
            c = (e_plus + e_minus) / 2.0
            a_sin_phi = e_0 - c
            a_cos_phi = (e_plus - e_minus) / 2.0
            
            # Calculo el desfase phi
            phi = np.arctan2(a_sin_phi, a_cos_phi)
            
            # El mínimo de A*sin(theta + phi) + C ocurre cuando (theta + phi) = -pi/2
            theta_opt = -np.pi / 2.0 - phi
            
            # Actualizo el parámetro al mínimo exacto hallado
            # Lo normalizo al rango [-pi, pi]
            theta_opt = (theta_opt + np.pi) % (2 * np.pi) - np.pi
            parametros_actuales[i] = theta_opt

    tiempo_total_q = time.time() - start_time
    
    # 5. Extracción del Resultado Final
    qc_final = ansatz.assign_parameters(parametros_actuales)
    probabilidades_finales = np.abs(Statevector(qc_final).data)**2
    
    idx_max = np.argmax(probabilidades_finales)
    best_bitstring = format(idx_max, f'0{n_qubits}b')
    precision_surrogada = resultados.get(best_bitstring, 0)

    return best_bitstring, precision_surrogada, tiempo_total_q, cnot_totales, profundidad, evals_totales

if __name__ == "__main__":
    print("Ejecutando Rotosolve (Sequential Single-Qubit Optimizer)...")
    print(ejecutar_rotosolve())