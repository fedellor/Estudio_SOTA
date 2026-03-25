"""
TFG: Optimizacion Cuantica de Hiperparametros (HPO)
Archivo 12: MA-QAOA (Multi-Angle QAOA)
Implementacion basada en Dash et al. (2025 arXiv).
"""
import json
import os
import sys
import time
import numpy as np

# Ajusto las rutas para mi entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def ejecutar_ma_qaoa():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    estados_ordenados = sorted(resultados.keys(), key=lambda x: resultados[x], reverse=True)
    
    # 1. Seleccion de terminos (Paper Dash et al. 2025)
    top_k = 50 
    mejores_bin = estados_ordenados[:top_k]
    
    # Mapeo de estados a indices de energia
    indices_top = [int(s, 2) for s in mejores_bin]
    num_params_totales = top_k + n_qubits # 50 gammas + 15 betas

    # Normalizacion
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    vector_hp = np.zeros(2**n_qubits)
    for b, acc in resultados.items():
        vector_hp[int(b, 2)] = -(acc - min_acc) / (max_acc - min_acc)

    # 2. Ansatz MA-QAOA de una capa (p=1)
    def funcion_coste(params):
        gammas = params[:top_k]
        betas = params[top_k:]
        
        # Estado inicial |+>
        psi = np.ones(2**n_qubits, dtype=np.complex128) / np.sqrt(2**n_qubits)
        
        # Evolucion Multi-Angle de Coste (Diagonal)
        for i, idx in enumerate(indices_top):
            psi[idx] *= np.exp(-1j * gammas[i])
            
        # Evolucion Multi-Angle Mezcladora (Rotaciones RX locales)
        qc_mix = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc_mix.rx(2 * betas[i], i)
        
        psi_final = Statevector(psi).evolve(qc_mix).data
        return np.dot(np.abs(psi_final)**2, vector_hp)

    # 3. Optimizador SPSA (Simultaneous Perturbation Stochastic Approximation)
    start_time = time.time()
    
    # El runner ya se encarga de fijar la semilla np.random antes de cada ejecucion
    theta = np.random.uniform(-0.1, 0.1, num_params_totales)
    
    # Hiperparametros SPSA estandar
    a, c, A = 0.1, 0.01, 10
    alpha, gamma_spsa = 0.602, 0.101
    
    n_iter = 100 # SPSA converge muy rapido en 100 pasos
    evals_totales = 0
    
    for k in range(n_iter):
        ak = a / (k + 1 + A)**alpha
        ck = c / (k + 1)**gamma_spsa
        
        delta = np.random.choice([-1, 1], size=num_params_totales)
        
        # Dos mediciones por iteracion
        theta_plus = theta + ck * delta
        theta_minus = theta - ck * delta
        
        f_plus = funcion_coste(theta_plus)
        f_minus = funcion_coste(theta_minus)
        evals_totales += 2
        
        # Estimacion del gradiente
        g_k = (f_plus - f_minus) / (2 * ck * delta)
        theta = theta - ak * g_k

    tiempo_total = time.time() - start_time
    
    # 4. Resultado final
    psi_init = np.ones(2**n_qubits, dtype=np.complex128) / np.sqrt(2**n_qubits)
    for i, idx in enumerate(indices_top):
        psi_init[idx] *= np.exp(-1j * theta[i])
    qc_f = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc_f.rx(2 * theta[top_k+i], i)
    st_f = Statevector(psi_init).evolve(qc_f).data
    
    idx_max = np.argmax(np.abs(st_f)**2)
    best_bitstring = format(idx_max, f'0{n_qubits}b')
    precision_surrogada = resultados.get(best_bitstring, 0)
    
    cnot_totales = 0
    profundidad_estimada = 2 # p=1 (Capa Coste local + Capa Mezclador local)

    # Retorno en el formato estricto: 
    return best_bitstring, precision_surrogada, tiempo_total, cnot_totales, profundidad_estimada, evals_totales

if __name__ == "__main__":
    print("Este archivo debe ser llamado desde runner_experimentos.py")