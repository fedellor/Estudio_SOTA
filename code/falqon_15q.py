"""
TFG: Optimizacion Cuantica de Hiperparametros (HPO)
Archivo 08: FALQON (Feedback-Based) con Escalamiento de Ganancia y con Amortiguación de Lyapunov
Basado en Legnini & Berberich (2025) y Rattighieri (2026).
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
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz

def ejecutar_falqon():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    estados_ordenados = sorted(resultados.keys(), key=lambda x: resultados[x], reverse=True)
    
    # 1. Normalización de Hp (Reescalado para maximizar el contraste del gradiente)
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    vector_hp = np.zeros(2**n_qubits)
    for b, acc in resultados.items():
        # Escalamos a [-1, 1] para que el conmutador tenga más recorrido
        vector_hp[int(b, 2)] = -2 * (acc - min_acc) / (max_acc - min_acc) + 1

    # 2. Métricas y Preparación de Hardware
    p_capas = 9 # Aumentamos capas para ver la evolucion de Lyapunov
    top_k = int(32768 * 0.05)
    mejores = estados_ordenados[:top_k]
    
    # Calculamos el coste base de una sola capa transpilada
    pauli_list = [(s[::-1].replace('0','I').replace('1','Z'), -1.0) for s in mejores]
    ham_fisico = SparsePauliOp.from_list(pauli_list)
    ansatz_una_capa = QAOAAnsatz(cost_operator=ham_fisico, reps=1).decompose(reps=3)
    qc_aud_base = transpile(ansatz_una_capa, basis_gates=['u', 'cx'], optimization_level=3)
    
    cx_capa = qc_aud_base.count_ops().get('cx', 0)
    profundidad_capa = qc_aud_base.depth()
    
    # Costes totales aproximados para el hardware real
    cnot_totales = cx_capa * p_capas
    profundidad_estimada = profundidad_capa * p_capas

    # 3. Algoritmo con Factor de Ganancia
    dt = 0.1
    ganancia = 500.0 # Amplificamos el feedback debil (At) para mover los qubits
    betas = [0.0] 
    
    # Estado inicial: Superposición uniforme
    estado = np.ones(2**n_qubits, dtype=np.complex128) / np.sqrt(2**n_qubits)
    
    start_time = time.time()
    
    # 
    for t in range(1, p_capas + 1):
        # Evolución de Coste
        estado = estado * np.exp(-1j * dt * vector_hp)
        
        # Evolución de Mezcla con beta_t
        beta_t = betas[t-1]
        qc_hd = QuantumCircuit(n_qubits)
        qc_hd.rx(2 * beta_t, range(n_qubits))
        estado = Statevector(estado).evolve(qc_hd).data
        
        # CALCULO DE At (Feedback)
        psi = estado
        psi_hd = np.zeros_like(psi)
        for q in range(n_qubits):
            dim_bloque = 2**q
            for i in range(0, 2**n_qubits, 2 * dim_bloque):
                for j in range(i, i + dim_bloque):
                    psi_hd[j], psi_hd[j + dim_bloque] = psi[j + dim_bloque], psi[j]
        
        at = 2 * np.imag(np.vdot(psi, vector_hp * psi_hd))
        
        # Aplicamos GANANCIA para que el angulo sea significativo
        betas.append(-at * ganancia)
        
    tiempo_total = time.time() - start_time
    
    idx_final = np.argmax(np.abs(estado)**2)
    best_bitstring = format(idx_final, f'0{n_qubits}b')
    precision_surrogada = resultados.get(best_bitstring, 0)
    
    # FALQON es determinista una vez fijado el Hamiltoniano, no hay optimizador clásico
    evals_totales = 0 
    
    # Retornamos en el formato estricto: 
    # (bitstring, precisión_subrogada, tiempo_cuántico, cnots, profundidad, evaluaciones)
    return best_bitstring, precision_surrogada, tiempo_total, cnot_totales, profundidad_estimada, evals_totales

if __name__ == "__main__":
    print("Este archivo debe ser llamado desde runner_experimentos.py")