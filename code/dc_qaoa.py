"""
TFG: Optimizacion Cuantica de Hiperparametros (HPO)
Archivo 19: DC-QAOA (Digitized-Counterdiabatic QAOA)

Implementa la conduccion contra-adiabatica digitalizada mediante un
potencial de gauge de primer orden en el mezclador para suprimir fugas 
a estados excitados de alta energia.

Referencias implementadas y analizadas:
1. Xu, Romero, Tang, Ban & Chen (2025): "Digitized counterdiabatic quantum optimization..."
2. Zhang, Li, Jiao, Zhang, Zhou & Ukil (2026): "Hybrid Digitized-counterdiabatic Quantum-classical Benders..."
3. Li, Alam & Ghosh (2023): "Large-Scale Quantum Approximate Optimization via Divide-and-Conquer"
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

def ejecutar_dc_qaoa():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    estados_ordenados = sorted(resultados.keys(), key=lambda x: resultados[x], reverse=True)
    
    # 1. Metricas Fisicas Optimizadas (Auditoria de Hardware)
    p_pasos = 2 
    top_k_total = int(32768 * 0.05)
    top_k_audit = 100 

    mejores_audit = estados_ordenados[:top_k_audit]
    pauli_list_audit = [(s[::-1].replace('0','I').replace('1','Z'), -1.0) for s in mejores_audit]
    ham_audit = SparsePauliOp.from_list(pauli_list_audit)
    
    # Transpilo solo 1 capa de la muestra para inferir el coste fisico sin congelar el PC
    ansatz_audit = QAOAAnsatz(cost_operator=ham_audit, reps=1).decompose()
    qc_aud = transpile(ansatz_audit, basis_gates=['u', 'cx'], optimization_level=1)
    
    # El termino contra-adiabatico añade rotaciones locales, manteniendo intactas las CNOTs
    cnot_por_termino = qc_aud.count_ops().get('cx', 0) / top_k_audit
    cnot_totales = int(cnot_por_termino * top_k_total * p_pasos)
    profundidad = int(qc_aud.depth() * p_pasos) + (2 * p_pasos) # +2 extra por la capa de gauge RY

    # 2. Normalizacion de Energía
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    vector_hp = np.zeros(2**n_qubits)
    for b, acc in resultados.items():
        vector_hp[int(b, 2)] = -(acc - min_acc) / (max_acc - min_acc)

    # 3. Funcion de Coste DC-QAOA
    def funcion_coste(params):
        gammas = params[:p_pasos]
        betas = params[p_pasos:2*p_pasos]
        alphas = params[2*p_pasos:] # Angulos del potencial de Gauge (RY)
        
        st = np.ones(2**n_qubits, dtype=np.complex128) / np.sqrt(2**n_qubits)
        
        for i in range(p_pasos):
            # Fase de Coste estandar
            st = st * np.exp(-1j * gammas[i] * vector_hp)
            
            # Mezclador Contra-adiabatico Digitalizado: RX + RY
            qc_mix = QuantumCircuit(n_qubits)
            for q in range(n_qubits):
                qc_mix.rx(2 * betas[i], q)
                qc_mix.ry(2 * alphas[i], q) 
            st = Statevector(st).evolve(qc_mix).data
            
        return np.dot(np.abs(st)**2, vector_hp)

    # 4. Optimizacion Clasica
    start_time = time.time()
    
    # 3 parametros por capa: gamma, beta y el termino contra-adiabatico alpha
    # El orquestador ya inyecta la semilla global antes de esta ejecucion
    params_init = np.random.uniform(-np.pi, np.pi, 3 * p_pasos)
    res = minimize(funcion_coste, params_init, method='COBYLA', options={'maxiter': 80})
    
    tiempo_total_q = time.time() - start_time
    
    # 5. Extraccion del Resultado Final
    st_f = np.ones(2**n_qubits, dtype=np.complex128) / np.sqrt(2**n_qubits)
    g_opt = res.x[:p_pasos]
    b_opt = res.x[p_pasos:2*p_pasos]
    a_opt = res.x[2*p_pasos:]
    
    for i in range(p_pasos):
        st_f = st_f * np.exp(-1j * g_opt[i] * vector_hp)
        qc_f = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            qc_f.rx(2 * b_opt[i], q)
            qc_f.ry(2 * a_opt[i], q)
        st_f = Statevector(st_f).evolve(qc_f).data
        
    idx = np.argmax(np.abs(st_f)**2)
    best_bitstring = format(idx, f'0{n_qubits}b')
    precision_surrogada = resultados.get(best_bitstring, 0)

    # Devuelvo las metricas en el formato estricto para runner_experimentos.py
    return best_bitstring, precision_surrogada, tiempo_total_q, cnot_totales, profundidad, res.nfev

if __name__ == "__main__":
    print("Ejecutando DC-QAOA optimizado...")
    print(ejecutar_dc_qaoa())