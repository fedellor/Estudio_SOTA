"""
TFG: Optimizacion Cuantica de Hiperparametros (HPO)
Archivo 10: CD-QAOA (Counter-Diabatic QAOA)
Implementación basada en operadores de Pauli Y (Ruan et al., 2026 IEEE).
"""
import json
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize

# Ajusto las rutas para el entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz

def ejecutar_cd_qaoa():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    estados_ordenados = sorted(resultados.keys(), key=lambda x: resultados[x], reverse=True)
    
    # 1. Preparo las Métricas y el Hamiltoniano
    p_pasos = 2 
    top_k = int(32768 * 0.05)
    mejores = estados_ordenados[:top_k]
    
    # Construyo el Hamiltoniano físico (Pauli Z)
    pauli_list = [(s[::-1].replace('0','I').replace('1','Z'), -1.0) for s in mejores]
    ham_fisico = SparsePauliOp.from_list(pauli_list)
    
    # [NUEVO] Transpilo el circuito completo para obtener CNOTs y Depth reales
    ansatz_completo = QAOAAnsatz(cost_operator=ham_fisico, reps=p_pasos).decompose(reps=3)
    qc_aud = transpile(ansatz_completo, basis_gates=['u', 'cx'], optimization_level=3)
    cx_totales = qc_aud.count_ops().get('cx', 0)
    profundidad = qc_aud.depth()

    # 2. Normalizo el paisaje de energía (Hp)
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    vector_hp = np.zeros(2**n_qubits)
    for b, acc in resultados.items():
        vector_hp[int(b, 2)] = -(acc - min_acc) / (max_acc - min_acc)

    # 3. Defino la Función de Coste CD-QAOA
    def funcion_coste_cd(params):
        gammas = params[0:p_pasos]
        alphas = params[p_pasos:2*p_pasos]
        betas = params[2*p_pasos:3*p_pasos]
        
        st = np.ones(2**n_qubits, dtype=np.complex128) / np.sqrt(2**n_qubits)
        
        for i in range(p_pasos):
            # A. Operador de Coste
            st = st * np.exp(-1j * gammas[i] * vector_hp)
            
            # B. Operador Contra-Diabático
            qc_cd = QuantumCircuit(n_qubits)
            qc_cd.ry(2 * alphas[i], range(n_qubits))
            st = Statevector(st).evolve(qc_cd).data
            
            # C. Operador Mezclador
            qc_mix = QuantumCircuit(n_qubits)
            qc_mix.rx(2 * betas[i], range(n_qubits))
            st = Statevector(st).evolve(qc_mix).data
            
        return np.dot(np.abs(st)**2, vector_hp)

    # 4. Configuro la Optimización
    start_time = time.time()
    
    # Dejo la generación de semilla comentada o dinámica si el runner la controla
    # np.random.seed(42) 
    params_iniciales = np.random.uniform(-0.1, 0.1, 3 * p_pasos)
    
    res = minimize(funcion_coste_cd, params_iniciales, method='COBYLA', options={'maxiter': 200})
    tiempo_total_sim = time.time() - start_time
    
    # 5. Extraigo el resultado final
    gammas_opt = res.x[0:p_pasos]
    alphas_opt = res.x[p_pasos:2*p_pasos]
    betas_opt = res.x[2*p_pasos:3*p_pasos]
    
    st_f = np.ones(2**n_qubits, dtype=np.complex128) / np.sqrt(2**n_qubits)
    for i in range(p_pasos):
        st_f = st_f * np.exp(-1j * gammas_opt[i] * vector_hp)
        qc_y = QuantumCircuit(n_qubits)
        qc_y.ry(2 * alphas_opt[i], range(n_qubits))
        st_f = Statevector(st_f).evolve(qc_y).data
        qc_x = QuantumCircuit(n_qubits)
        qc_x.rx(2 * betas_opt[i], range(n_qubits))
        st_f = Statevector(st_f).evolve(qc_x).data
        
    idx = np.argmax(np.abs(st_f)**2)
    best_bitstring = format(idx, f'0{n_qubits}b')
    precision_surrogada = resultados.get(best_bitstring, 0)
    evals_totales = res.nfev

    # Retorno en el orden estricto exigido por el runner
    return best_bitstring, precision_surrogada, tiempo_total_sim, cx_totales, profundidad, evals_totales

if __name__ == "__main__":
    print("Este archivo debe ser llamado desde runner_experimentos.py")