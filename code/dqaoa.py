"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo 25: DQAOA (Distributed Quantum Approximate Optimization Algorithm)

Implemento la descomposición del problema en sub-modelos de espacios reducidos
con ejecución distribuida y agregación de soluciones.

Referencias:
1. Ruan, Chen, Yang, Zhao, Tang, Liu & Kato (2026): "The Divided-QUBO-Based 
   Quantum Algorithm for Optimal Wireless Link Scheduling".
2. Kim, Pascuzzi, Xu, Luo, Lee & Suh (2025): "Distributed Quantum Approximate 
   Optimization Algorithm on a Quantum-Centric Supercomputing Architecture".
3. Kim, Suh, Perez, Landfield, Sandoval et al. (2025): "Quantum Approximate 
   Optimization Algorithm on Different Qubit Systems".
4. Xu, Chundury, Kim, Shehata, Li, Li, Luo, Mueller & Suh (2025): "GPU-Accelerated 
   Distributed QAOA on Large-scale HPC Ecosystems".
"""

import json
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize

# Ajusto las rutas para mi entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz

def ejecutar_dqaoa():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    
    # 1. Auditoría de Hardware (Muestreo para métricas de hardware distribuido)
    # Simulo el coste de 3 circuitos pequeños de 5 qubits en paralelo
    n_sub = 5
    top_k_audit = 50
    estados_ordenados = sorted(resultados.keys(), key=lambda x: resultados[x], reverse=True)
    mejores_audit = [s[:n_sub] for s in estados_ordenados[:top_k_audit]]
    
    pauli_list = [(s[::-1].replace('0','I').replace('1','Z'), -1.0) for s in mejores_audit]
    ham_audit = SparsePauliOp.from_list(pauli_list)
    
    qc_aud = transpile(QAOAAnsatz(ham_audit, reps=1).decompose(), 
                       basis_gates=['u', 'cx'], optimization_level=1)
    
    # Escalo métricas: 3 sub-problemas x coste de cada uno
    cnot_totales = int((qc_aud.count_ops().get('cx', 0) / top_k_audit) * (int(32768*0.05)) * 1)
    profundidad = qc_aud.depth()

    # 2. Configuración de Distribución (Divided-QUBO)
    # Divido los 15 qubits en 3 grupos de 5 (Ruan et al., 2026)
    grupos = [range(0, 5), range(5, 10), range(10, 15)]
    bitstring_actual = list("0" * n_qubits)
    
    start_time = time.time()
    evals_totales = 0

    # 3. Bucle de Optimización Distribuida (Distributed Strategy)
    # Realizo 2 pasadas por todos los grupos para propagar correlaciones
    for iter_dist in range(2):
        for g_idx, grupo in enumerate(grupos):
            
            def funcion_coste_sub(params):
                gamma, beta = params
                # Solo evoluciono los 5 qubits del grupo actual
                st_sub = np.ones(2**len(grupo), dtype=np.complex128) / np.sqrt(2**len(grupo))
                
                # Extraigo el vector de energías local fijando el resto de bits
                vector_hp_local = np.zeros(2**len(grupo))
                for i in range(2**len(grupo)):
                    bits_local = list(format(i, f'0{len(grupo)}b'))
                    temp_bitstring = bitstring_actual.copy()
                    # Inserto los bits candidatos en el grupo correspondiente
                    for idx, q_pos in enumerate(grupo):
                        temp_bitstring[q_pos] = bits_local[idx]
                    
                    acc = resultados.get("".join(temp_bitstring), 0)
                    vector_hp_local[i] = -acc

                # Evolución cuántica del sub-problema
                st_sub = st_sub * np.exp(-1j * gamma * vector_hp_local)
                qc_mix = QuantumCircuit(len(grupo))
                qc_mix.rx(2 * beta, range(len(grupo)))
                st_sub = Statevector(st_sub).evolve(qc_mix).data
                
                return np.dot(np.abs(st_sub)**2, vector_hp_local)

            # Optimización local del sub-bloque
            res = minimize(funcion_coste_sub, [0.5, 0.5], method='COBYLA', options={'maxiter': 25})
            evals_totales += res.nfev
            
            # Actualizo el bitstring global con el mejor resultado del sub-problema
            gamma_opt, beta_opt = res.x
            st_f = np.ones(2**len(grupo), dtype=np.complex128) / np.sqrt(2**len(grupo))
            # (Simulación final del sub-bloque para colapsar estado)
            # Re-calculo energías para el colapso final del grupo
            v_hp_final = np.zeros(2**len(grupo))
            for i in range(2**len(grupo)):
                bits_l = list(format(i, f'0{len(grupo)}b'))
                t_bs = bitstring_actual.copy()
                for idx, q_p in enumerate(grupo): t_bs[q_p] = bits_l[idx]
                v_hp_final[i] = -resultados.get("".join(t_bs), 0)
            
            st_f = st_f * np.exp(-1j * gamma_opt * v_hp_final)
            qc_f = QuantumCircuit(len(grupo))
            qc_f.rx(2 * beta_opt, range(len(grupo)))
            st_f = Statevector(st_f).evolve(qc_f).data
            
            mejor_local = format(np.argmax(np.abs(st_f)**2), f'0{len(grupo)}b')
            for idx, q_pos in enumerate(grupo):
                bitstring_actual[q_pos] = mejor_local[idx]

    tiempo_total = time.time() - start_time
    mejor_estado = "".join(bitstring_actual)
    precision_final = resultados.get(mejor_estado, 0)

    # Retorno exacto para el orquestador
    return mejor_estado, precision_final, tiempo_total, cnot_totales, profundidad, evals_totales

if __name__ == "__main__":
    print("Ejecutando DQAOA distribuido (15q -> 3x5q)...")
    print(ejecutar_dqaoa())