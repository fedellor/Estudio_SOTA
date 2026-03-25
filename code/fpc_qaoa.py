"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo 38: FPC-QAOA (Fixed-Parameter-Count QAOA)

Implemento el QAOA con un número fijo de parámetros, utilizando una función 
polinómica continua para generar los ángulos de las p capas del circuito, 
mitigando el sobreajuste y los Barren Plateaus en circuitos profundos.

Referencias implementadas y analizadas:
1. Saavedra-Pino, Quispe-Mendizábal, Alvarado Barrios, Solano, Retamal 
   & Albarrán-Arriagada (2025): "Quantum Approximate Optimization Algorithm 
   with Fixed Number of Parameters".
2. Eker, Arslan, Nazlı, Demirgil & Deligöz (2026): "QANTIS: A Hardware-Validated 
   Quantum Platform for POMDP Planning and Multi-Target Data Association".
"""

import json
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz

# Ajusto las rutas para mi entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

def ejecutar_fpc_qaoa():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    estados_ordenados = sorted(resultados.keys(), key=lambda x: resultados[x], reverse=True)
    
    # 1. Configuración del FPC-QAOA (Desacoplamiento p vs k)
    p_pasos = 8 # Profundidad real del circuito cuántico
    k_params = 3 # Grado del polinomio (fijado). Total parámetros a optimizar: 2k = 6
    
    # Auditoría de Hardware (Muestreo Representativo)
    top_k_total = int(32768 * 0.05)
    top_k_audit = 100 

    mejores_audit = estados_ordenados[:top_k_audit]
    pauli_list_audit = [(s[::-1].replace('0','I').replace('1','Z'), -1.0) for s in mejores_audit]
    ham_audit = SparsePauliOp.from_list(pauli_list_audit)
    
    ansatz_audit = QAOAAnsatz(cost_operator=ham_audit, reps=1).decompose()
    qc_aud = transpile(ansatz_audit, basis_gates=['u', 'cx'], optimization_level=1)
    
    cnot_por_termino = qc_aud.count_ops().get('cx', 0) / top_k_audit
    cnot_totales = int(cnot_por_termino * top_k_total * p_pasos)
    profundidad = int(qc_aud.depth() * p_pasos)

    # 2. Normalización de Energía
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    vector_hp = np.zeros(2**n_qubits)
    for b, acc in resultados.items():
        vector_hp[int(b, 2)] = -(acc - min_acc) / (max_acc - min_acc)

    # 3. Generador de Schedule (Mapeo k -> p)
    def generar_angulos(parametros_k):
        """
        Convierte los 2k parámetros en 2p ángulos muestreando un polinomio.
        """
        coefs_gamma = parametros_k[:k_params]
        coefs_beta = parametros_k[k_params:]
        gammas, betas = [], []
        
        for i in range(1, p_pasos + 1):
            t = i / p_pasos # Tiempo normalizado en (0, 1]
            # Polinomio: c0 + c1*t + c2*t^2 + ...
            g = sum(c * (t**j) for j, c in enumerate(coefs_gamma))
            b = sum(c * (t**j) for j, c in enumerate(coefs_beta))
            gammas.append(g)
            betas.append(b)
            
        return gammas, betas

    # 4. Función de Coste (Simulación Algebraica Rápida)
    def funcion_coste_fpc(parametros_k):
        gammas, betas = generar_angulos(parametros_k)
        
        st = np.ones(2**n_qubits, dtype=np.complex128) / np.sqrt(2**n_qubits)
        
        for i in range(p_pasos):
            st = st * np.exp(-1j * gammas[i] * vector_hp)
            qc_mix = QuantumCircuit(n_qubits)
            qc_mix.rx(2 * betas[i], range(n_qubits))
            st = Statevector(st).evolve(qc_mix).data
            
        return np.dot(np.abs(st)**2, vector_hp)

    # 5. Optimización Clásica sobre el espacio reducido (2k dimensiones)
    start_time = time.time()
    
    # Inicialización aleatoria de los 2k coeficientes polinómicos
    params_init = np.random.uniform(-1.0, 1.0, 2 * k_params)
    
    res = minimize(funcion_coste_fpc, params_init, method='L-BFGS-B', options={'maxiter': 60})
    tiempo_total_q = time.time() - start_time
    
    # 6. Extracción del Resultado Final
    gammas_opt, betas_opt = generar_angulos(res.x)
    st_f = np.ones(2**n_qubits, dtype=np.complex128) / np.sqrt(2**n_qubits)
    
    for i in range(p_pasos):
        st_f = st_f * np.exp(-1j * gammas_opt[i] * vector_hp)
        qc_f = QuantumCircuit(n_qubits)
        qc_f.rx(2 * betas_opt[i], range(n_qubits))
        st_f = Statevector(st_f).evolve(qc_f).data
        
    idx_max = np.argmax(np.abs(st_f)**2)
    best_bitstring = format(idx_max, f'0{n_qubits}b')
    precision_surrogada = resultados.get(best_bitstring, 0)

    return best_bitstring, precision_surrogada, tiempo_total_q, cnot_totales, profundidad, res.nfev

if __name__ == "__main__":
    print("Ejecutando FPC-QAOA (Fixed-Parameter-Count QAOA)...")
    print(ejecutar_fpc_qaoa())