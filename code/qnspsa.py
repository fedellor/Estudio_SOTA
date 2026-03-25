"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo 28: QNSPSA (Quantum Natural SPSA)

Implemento el Descenso de Gradiente Natural Cuántico aproximado mediante SPSA,
evaluando la métrica de Fubini-Study (QFIM) de forma estocástica.

Referencias implementadas y analizadas:
1. Halla (2025): "Estimation of Quantum Fisher Information via Stein's Identity 
   in Variational Quantum Algorithms" (Uso de resampling N>=5 para estabilizar QNSPSA).
2. Britant & Pathak (2024): "Revisiting Majumdar-Ghosh spin chain model and Max-cut 
   problem using variational quantum algorithms" (Eficiencia de optimizadores zeroth-order).
3. Tecot & Hsieh (2024): "Randomized Benchmarking of Local Zeroth-Order Optimizers 
   for Variational Quantum Systems".
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

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Statevector

def ejecutar_qnspsa():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    
    # 1. Normalización del Paisaje de Energía (Hamiltoniano Diagonal)
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    vector_energias = np.zeros(2**n_qubits)
    for bitstring, acc in resultados.items():
        vector_energias[int(bitstring, 2)] = -(acc - min_acc) / (max_acc - min_acc)

    # 2. Auditoría de Hardware (Ansatz VQE)
    # Utilizamos un ansatz Hardware-Efficient de profundidad p=2
    ansatz = EfficientSU2(num_qubits=n_qubits, su2_gates=['ry', 'rz'], entanglement='linear', reps=2)
    num_params = ansatz.num_parameters
    
    qc_aud = transpile(ansatz, basis_gates=['u', 'cx'], optimization_level=3)
    cnot_totales = qc_aud.count_ops().get('cx', 0)
    profundidad = qc_aud.depth()

    # 3. Funciones de Evaluación Cuántica
    def evaluar_energia_y_estado(parametros):
        qc_bound = ansatz.assign_parameters(parametros)
        st = Statevector(qc_bound).data
        probabilidades = np.abs(st)**2
        energia = np.dot(probabilidades, vector_energias)
        return energia, st

    # 4. Bucle de Optimización QNSPSA
    start_time = time.time()
    evals_totales = 0
    
    # Hiperparámetros de SPSA estándar
    alpha_spsa, gamma_spsa = 0.602, 0.101
    a, c = 0.5, 0.1
    A = 10
    
    # Tamaño de remuestreo (N=5) sugerido por Halla (2025) para estabilizar la QFIM
    resampling_size = 5 
    max_iteraciones = 40
    
    parametros_actuales = np.random.uniform(-np.pi, np.pi, num_params)
    
    for k in range(max_iteraciones):
        ak = a / (k + 1 + A)**alpha_spsa
        ck = c / (k + 1)**gamma_spsa
        
        # Evaluación base para calcular fidelidades (métricas de Fubini-Study)
        energia_base, estado_base = evaluar_energia_y_estado(parametros_actuales)
        evals_totales += 1
        
        gradiente_acumulado = np.zeros(num_params)
        metrica_qfim_acumulada = 0.0 # Aproximación escalar de la curvatura del espacio de Hilbert
        
        # Remuestreo para estabilizar la estimación (Halla, 2025)
        for _ in range(resampling_size):
            delta = np.random.choice([-1, 1], size=num_params)
            
            # 4.1 Estimación del Gradiente de Energía
            e_plus, _ = evaluar_energia_y_estado(parametros_actuales + ck * delta)
            e_minus, _ = evaluar_energia_y_estado(parametros_actuales - ck * delta)
            gradiente_est = (e_plus - e_minus) / (2 * ck) * delta
            gradiente_acumulado += gradiente_est
            
            # 4.2 Estimación de la Métrica QFIM (Información de Fisher)
            # Evalúo un estado perturbado para medir la distancia en el espacio de Hilbert
            _, estado_perturbado = evaluar_energia_y_estado(parametros_actuales + ck * delta)
            fidelidad = np.abs(np.vdot(estado_base, estado_perturbado))**2
            
            # La aproximación de la diagonal de Fubini-Study en la dirección delta
            metrica_est = (1.0 - fidelidad) / (ck**2)
            metrica_qfim_acumulada += metrica_est
            
            evals_totales += 3 # (e_plus, e_minus, estado_perturbado)
            
        # Promedios del mini-batch
        gradiente_promedio = gradiente_acumulado / resampling_size
        metrica_promedio = metrica_qfim_acumulada / resampling_size
        
        # Regularización para evitar divisiones por cero en matrices singulares
        metrica_regularizada = max(metrica_promedio, 1e-4)
        
        # 4.3 Actualización de Gradiente Natural Cuántico (QNG)
        # El gradiente se escala inversamente a la métrica de información de Fisher
        parametros_actuales -= ak * (gradiente_promedio / metrica_regularizada)

    tiempo_total_q = time.time() - start_time
    
    # 5. Extracción del Resultado Final
    qc_final = ansatz.assign_parameters(parametros_actuales)
    probabilidades_finales = np.abs(Statevector(qc_final).data)**2
    
    idx_max = np.argmax(probabilidades_finales)
    best_bitstring = format(idx_max, f'0{n_qubits}b')
    precision_surrogada = resultados.get(best_bitstring, 0)

    # Retorno exacto en el formato exigido
    return best_bitstring, precision_surrogada, tiempo_total_q, cnot_totales, profundidad, evals_totales

if __name__ == "__main__":
    print("Ejecutando QNSPSA...")
    print(ejecutar_qnspsa())