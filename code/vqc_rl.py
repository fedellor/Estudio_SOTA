"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo 31: VQC-RL (Variational Quantum Circuit Reinforcement Learning)

Implementa un agente de Aprendizaje por Refuerzo Cuántico que utiliza 
Data Re-uploading para navegar por el espacio de hiperparámetros.

Referencias implementadas y analizadas:
1. Coelho, Sequeira & Santos (2024): "VQC-based reinforcement learning with 
   data re-uploading: performance and trainability".
2. Zhang & Tang (2026): "Quantum reinforcement learning-based active flow control".
3. Kölle et al. (2024): "A Study on Optimization Techniques for Variational 
   Quantum Circuits in Reinforcement Learning".
4. Ikhtiarudin et al. (2025): "BenchRL-QAS: Benchmarking Reinforcement 
   Learning Algorithms for Quantum Architecture Search".
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

def ejecutar_vqc_rl():
    # 1. Carga del Entorno (Environment)
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    
    # Normalización de las recompensas (Precisión)
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    vector_recompensas = np.zeros(2**n_qubits)
    for bitstring, acc in resultados.items():
        # Recompensa positiva escalada
        vector_recompensas[int(bitstring, 2)] = (acc - min_acc) / (max_acc - min_acc)

    # 2. Arquitectura de la Policy Network Cuántica (VQC)
    p_capas = 2
    num_params = p_capas * n_qubits

    def construir_policy_network(estado_actual, theta):
        """
        Construye el circuito VQC combinando Data Re-uploading y el Ansatz Parametrizado.
        """
        qc = QuantumCircuit(n_qubits)
        
        # A. Data Re-uploading (Coelho et al., 2024)
        # Codificamos el estado clásico actual en el espacio de Hilbert
        for i in range(n_qubits):
            # Si el bit es 1, aplicamos una rotación pi, si es 0, no rotamos
            qc.rx(estado_actual[i] * np.pi, i)
            
        # B. Capas Entrenables (Hardware-Efficient Ansatz - Kölle et al., 2024)
        idx = 0
        for _ in range(p_capas):
            for i in range(n_qubits):
                qc.ry(theta[idx], i)
                idx += 1
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
                
        return qc

    # Auditoría de Hardware de la Policy Network
    estado_dummy = np.zeros(n_qubits)
    theta_dummy = np.zeros(num_params)
    qc_aud = transpile(construir_policy_network(estado_dummy, theta_dummy), 
                       basis_gates=['u', 'cx'], optimization_level=3)
    cnot_totales = qc_aud.count_ops().get('cx', 0)
    profundidad = qc_aud.depth()

    # 3. Bucle de Entrenamiento del Agente RL (Policy Gradient Simulado)
    start_time = time.time()
    evals_totales = 0
    
    # Parámetros del Agente
    episodios = 4
    # Inicializamos el estado del agente en una configuración base (000...0)
    estado_agente = np.zeros(n_qubits)
    pesos_policy = np.random.uniform(-np.pi, np.pi, num_params)
    
    mejor_recompensa_historica = -1
    mejor_estado_historico = None

    

    print("Iniciando exploración del Agente VQC-RL...")
    for episodio in range(episodios):
        
        def policy_loss(theta):
            """
            Calcula la pérdida de la política: buscamos maximizar la recompensa esperada.
            Por tanto, minimizamos el negativo de la recompensa esperada.
            """
            qc = construir_policy_network(estado_agente, theta)
            probabilidades_accion = np.abs(Statevector(qc).data)**2
            recompensa_esperada = np.dot(probabilidades_accion, vector_recompensas)
            return -recompensa_esperada

        # Actualización de la política (Equivalente a la fase de optimización en PPO)
        res_opt = minimize(policy_loss, pesos_policy, method='COBYLA', options={'maxiter': 25})
        pesos_policy = res_opt.x
        evals_totales += res_opt.nfev
        
        # Interacción con el entorno (Tomar acción)
        qc_accion = construir_policy_network(estado_agente, pesos_policy)
        prob_final = np.abs(Statevector(qc_accion).data)**2
        
        # El agente elige la acción más probable (política greedy) para transicionar
        accion_idx = np.argmax(prob_final)
        accion_bitstring = format(accion_idx, f'0{n_qubits}b')
        recompensa_real = resultados.get(accion_bitstring, 0)
        
        # Transición de estado: el agente se mueve a la nueva configuración
        estado_agente = np.array([int(b) for b in accion_bitstring])
        
        if recompensa_real > mejor_recompensa_historica:
            mejor_recompensa_historica = recompensa_real
            mejor_estado_historico = accion_bitstring

    tiempo_total = time.time() - start_time

    # Retorno estricto para la integración tabular
    return mejor_estado_historico, mejor_recompensa_historica, tiempo_total, cnot_totales, profundidad, evals_totales

if __name__ == "__main__":
    print("Ejecutando Agente VQC-RL...")
    print(ejecutar_vqc_rl())