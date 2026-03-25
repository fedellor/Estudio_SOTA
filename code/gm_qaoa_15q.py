"""
TFG: Optimizacion Cuantica de Hiperparametros (HPO)
Archivo 16: GM-QAOA (Warm-Start / Shifting Complexity) - Datos Exactos

Referencias implementadas:
1. Bärtschi & Eidenbenz (2020 QCE): "Grover Mixers for QAOA: Shifting Complexity". 
2. Bridi & Marquezino (2024 Phys. Rev. A): "Analytical results for the QAOA with Grover mixer".
3. Kiktenko et al. (2025 arXiv): "Applying GM-QAOA to High-order Unconstrained Binary Optimization".
4. Tsvelikhovskiy et al. (2025 arXiv): "Provable avoidance of barren plateaus for the QAOA with Grover mixers".
"""
import json
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile

# Ajusto las rutas para mi entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

def ejecutar_gm_qaoa():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    N = 2**n_qubits
    p_capas = 1  
    
    # 1. Calculo de CNOTs exactas y Profundidad
    qc_mixer = QuantumCircuit(n_qubits)
    qc_mixer.h(range(n_qubits))
    qc_mixer.x(range(n_qubits))
    qc_mixer.mcp(np.pi, list(range(n_qubits - 1)), n_qubits - 1)
    qc_mixer.x(range(n_qubits))
    qc_mixer.h(range(n_qubits))
    
    qc_res = transpile(qc_mixer, basis_gates=['u', 'cx'], optimization_level=3)
    cx_mixer = qc_res.count_ops().get('cx', 0)
    profundidad_mixer = qc_res.depth()
    
    cx_fase_coste = int(n_qubits * (n_qubits - 1) / 2) # Hamiltoniano denso
    profundidad_coste = n_qubits * 2 # Estimacion para el operador de coste
    
    cx_totales = (cx_mixer + cx_fase_coste) * p_capas
    profundidad_estimada = (profundidad_mixer + profundidad_coste) * p_capas

    # 2. Shifting Complexity (Top 10%)
    estados_ordenados = sorted(resultados.keys(), key=lambda x: resultados[x], reverse=True)
    top_k = int(N * 0.10)
    indices_validos = [int(s, 2) for s in estados_ordenados[:top_k]]
    
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    costes = -(acc_vals - min_acc) / (max_acc - min_acc)

    def funcion_energia(params):
        gammas, betas = params[:p_capas], params[p_capas:]
        psi = np.zeros(N, dtype=np.complex128)
        psi[indices_validos] = 1.0 / np.sqrt(top_k)
        
        for p in range(p_capas):
            psi = psi * np.exp(-1j * gammas[p] * costes)
            promedio = np.sum(psi[indices_validos]) / top_k
            psi[indices_validos] = psi[indices_validos] - (1 - np.exp(-1j * betas[p])) * promedio
            
        return np.dot(np.abs(psi)**2, costes)

    start_time = time.time()
    
    # El orquestador inyecta la semilla antes de llamar a la funcion
    res = minimize(funcion_energia, np.random.uniform(-np.pi, np.pi, 2 * p_capas), 
                   method='COBYLA', options={'maxiter': 100})
    
    # Re-simulacion para extraer el estado final
    psi_f = np.zeros(N, dtype=np.complex128)
    psi_f[indices_validos] = 1.0 / np.sqrt(top_k)
    for p in range(p_capas):
        psi_f = psi_f * np.exp(-1j * res.x[0] * costes)
        psi_f[indices_validos] = psi_f[indices_validos] - (1 - np.exp(-1j * res.x[1])) * (np.sum(psi_f[indices_validos]) / top_k)
        
    idx_max = np.argmax(np.abs(psi_f)**2)
    best_bitstring = format(idx_max, f'0{n_qubits}b')
    precision_surrogada = resultados.get(best_bitstring, 0)
    
    tiempo_total = time.time() - start_time
    evals_totales = res.nfev

    return best_bitstring, precision_surrogada, tiempo_total, cx_totales, profundidad_estimada, evals_totales

if __name__ == "__main__":
    print("Este archivo debe ser llamado desde runner_experimentos.py")