"""
TFG: Optimizacion Cuantica de Hiperparametros (HPO)
Archivo 07: WS-QAOA (Warm-Started QAOA) - Version Final Verificada
Alineado con Yu et al. (2025) y Obst et al. (2024).
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

def ejecutar_ws_qaoa():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    # 1. Carga y Normalizacion
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    estados_ordenados = sorted(resultados.keys(), key=lambda x: resultados[x], reverse=True)
    
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    
    # Normalizo al rango [-1, 1] para estabilizar mis gradientes
    def normalizar(val): return 2 * (val - min_acc) / (max_acc - min_acc) - 1
    
    vector_energias_norm = np.zeros(2**n_qubits)
    for bitstring, acc in resultados.items():
        vector_energias_norm[int(bitstring, 2)] = -normalizar(acc)
            
    # 2. Metricas Fisicas Reales (Auditoria de Hardware)
    top_k = int(32768 * 0.05)
    mejores_estados = estados_ordenados[:top_k]
    pauli_list = [(estado[::-1].replace('0','I').replace('1','Z'), -normalizar(resultados[estado])) for estado in mejores_estados]
    ham_fisico = SparsePauliOp.from_list(pauli_list)
    
    # Construyo un circuito representativo para extraer profundidad y CNOTs
    ansatz_base = QAOAAnsatz(cost_operator=ham_fisico, reps=1)
    qc_aud = transpile(ansatz_base, basis_gates=['u', 'cx'], optimization_level=3)
    num_cx = qc_aud.count_ops().get('cx', 0)
    profundidad = qc_aud.depth()
    
    # 3. Warm-Start (Sesgo de Obst et al. 2024)
    # Calculo las frecuencias de bits en mis mejores estados clasicos
    frecuencias = np.zeros(n_qubits)
    for estado in mejores_estados:
        for i, bit in enumerate(estado[::-1]):
            if bit == '1': frecuencias[i] += 1
    
    # Determino mis angulos de rotacion iniciales
    c_i = np.clip(frecuencias / top_k, 0.15, 0.85)
    thetas = 2 * np.arcsin(np.sqrt(c_i))
    
    # 4. Motor de Evolucion Temporal (Simulacion Hibrida)
    def funcion_coste(parametros):
        gamma, beta = parametros
        # Inicializo mi estado sesgado (Warm Start)
        qc_init = QuantumCircuit(n_qubits)
        for i in range(n_qubits): qc_init.ry(thetas[i], i)
        
        st = Statevector(qc_init).data
        # Aplico mi operador de fase de coste
        st = st * np.exp(-1j * gamma * vector_energias_norm)
        
        # Aplico mi mezclador sesgado Ry(-theta) Rx(2beta) Ry(theta)
        qc_mix = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc_mix.ry(-thetas[i], i)
            qc_mix.rx(2 * beta, i)
            qc_mix.ry(thetas[i], i)
        
        st_final = Statevector(st).evolve(qc_mix)
        return np.dot(st_final.probabilities(), vector_energias_norm)

    # 5. Optimizacion en dos fases
    grid_size = 8
    gammas = np.linspace(0, np.pi, grid_size)
    betas = np.linspace(-np.pi/2, np.pi/2, grid_size)
    mejor_e_grid = float('inf')
    punto_inicio = [0.5, 0.5]
    
    # Fase 1: Mi Grid Search para evitar quedar atrapado en valles locales
    for g in gammas:
        for b in betas:
            e = funcion_coste([g, b])
            if e < mejor_e_grid:
                mejor_e_grid = e
                punto_inicio = [g, b]
                
    # Fase 2: Refinamiento mediante gradiente (L-BFGS-B)
    start_time = time.time()
    res = minimize(funcion_coste, punto_inicio, method='L-BFGS-B', bounds=[(0, np.pi), (-np.pi, np.pi)])
    tiempo_total_q = time.time() - start_time
    
    # 6. Reconstruccion del resultado final
    g_opt, b_opt = res.x
    qc_ini = QuantumCircuit(n_qubits); 
    for i in range(n_qubits): qc_ini.ry(thetas[i], i)
    st_opt = Statevector(qc_ini).data * np.exp(-1j * g_opt * vector_energias_norm)
    qc_mx = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc_mx.ry(-thetas[i], i); qc_mx.rx(2 * b_opt, i); qc_mx.ry(thetas[i], i)
    
    probs = Statevector(st_opt).evolve(qc_mx).probabilities()
    ganador_idx = np.argmax(probs)
    estado_ganador = format(ganador_idx, f'0{n_qubits}b')
    precision_final = resultados.get(estado_ganador, 0)
    evals_totales = res.nfev + (grid_size**2)
    
    # Retorno exacto para el orquestador
    return estado_ganador, precision_final, tiempo_total_q, num_cx, profundidad, evals_totales

if __name__ == "__main__":
    print("Este archivo debe ser llamado desde runner_experimentos.py")