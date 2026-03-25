"""
TFG: Optimizacion Cuantica de Hiperparametros (HPO)
Archivo 09: QAOA con Inicializacion TQA (TQA Initialisation)
Basado en Lusso et al. (2026) y Brodoloni et al. (2026).
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

def ejecutar_qaoa_tqa():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    estados_ordenados = sorted(resultados.keys(), key=lambda x: resultados[x], reverse=True)
    
    # 1. Metricas Fisicas Optimizadas (Hardware Auditing)
    # En lugar de transpilar 1.600 terminos x 5 capas (que congela el PC),
    # calculo el coste de una muestra representativa y escalo linealmente.
    p_pasos = 5 
    top_k_total = int(32768 * 0.05) # Los ~1.600 terminos reales
    top_k_audit = 100               # Muestra para el transpiler

    mejores_audit = estados_ordenados[:top_k_audit]
    pauli_list_audit = [(s[::-1].replace('0','I').replace('1','Z'), -1.0) for s in mejores_audit]
    ham_audit = SparsePauliOp.from_list(pauli_list_audit)
    
    # Transpilo solo 1 capa de la muestra con optimizacion ligera
    ansatz_audit = QAOAAnsatz(cost_operator=ham_audit, reps=1).decompose()
    qc_aud = transpile(ansatz_audit, basis_gates=['u', 'cx'], optimization_level=1)
    
    # Escalo los resultados al tamaño real del problema
    cnot_por_termino = qc_aud.count_ops().get('cx', 0) / top_k_audit
    cnot_totales = int(cnot_por_termino * top_k_total * p_pasos)
    profundidad = int(qc_aud.depth() * p_pasos)

    # 2. Normalizacion de Energía (Hp) - Sigue siendo ultrarrapida con Numpy
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    vector_hp = np.zeros(2**n_qubits)
    for b, acc in resultados.items():
        vector_hp[int(b, 2)] = -(acc - min_acc) / (max_acc - min_acc)

    # 3. TQA Initialisation
    T_heuristico = p_pasos * 1.5
    params_init = []
    for i in range(1, p_pasos + 1):
        s = i / p_pasos
        params_init.append(s * (T_heuristico / p_pasos)) # Gamma
    for i in range(1, p_pasos + 1):
        s = i / p_pasos
        params_init.append((1 - s) * (T_heuristico / p_pasos)) # Beta

    # 4. Funcion de Coste QAOA (Simulacion Algebraica)
    def funcion_coste(params):
        gammas = params[:p_pasos]
        betas = params[p_pasos:]
        st = np.ones(2**n_qubits, dtype=np.complex128) / np.sqrt(2**n_qubits)
        
        for i in range(p_pasos):
            st = st * np.exp(-1j * gammas[i] * vector_hp)
            qc_mix = QuantumCircuit(n_qubits)
            qc_mix.rx(2 * betas[i], range(n_qubits))
            st = Statevector(st).evolve(qc_mix).data
        return np.dot(np.abs(st)**2, vector_hp)

    # 5. Optimizacion
    start_time = time.time()
    res = minimize(funcion_coste, params_init, method='COBYLA', options={'maxiter': 50})
    tiempo_total_q = time.time() - start_time
    
    # 6. Extraccion del Resultado Final
    st_f = np.ones(2**n_qubits, dtype=np.complex128) / np.sqrt(2**n_qubits)
    g_opt, b_opt = res.x[:p_pasos], res.x[p_pasos:]
    for i in range(p_pasos):
        st_f = st_f * np.exp(-1j * g_opt[i] * vector_hp)
        qc_f = QuantumCircuit(n_qubits)
        qc_f.rx(2 * b_opt[i], range(n_qubits))
        st_f = Statevector(st_f).evolve(qc_f).data
        
    idx = np.argmax(np.abs(st_f)**2)
    best_bitstring = format(idx, f'0{n_qubits}b')
    precision_surrogada = resultados.get(best_bitstring, 0)

    return best_bitstring, precision_surrogada, tiempo_total_q, cnot_totales, profundidad, res.nfev

if __name__ == "__main__":
    print("Ejecutando QAOA-TQA...")
    print(ejecutar_qaoa_tqa())