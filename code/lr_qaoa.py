"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo 39: LR-QAOA (Linear-Ramp QAOA)

Implemento el QAOA de Rampa Lineal. La profundidad del circuito (p) se 
desacopla de la optimización clásica, reduciendo el problema a encontrar 
solo dos parámetros globales (c_gamma, c_beta) que definen las pendientes.

Referencias implementadas y analizadas:
1. Montañez-Barrera & Michielsen (2025): "Toward a linear-ramp QAOA protocol: 
   evidence of a scaling advantage in solving some combinatorial optimization problems".
2. Dehn, Zaefferer, Hellstern, Jayadevan, Reiter & Wellens (2026): "Extrapolation 
   method to optimize linear-ramp QAOA parameters".
3. Coelho, Kruse & Lorenz (2026): "QAOA-Predictor: Forecasting Success Probabilities 
   and Minimal Depths for Efficient Fixed-Parameter Optimization".
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

def ejecutar_lr_qaoa():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    estados_ordenados = sorted(resultados.keys(), key=lambda x: resultados[x], reverse=True)
    
    # 1. Configuración del LR-QAOA
    # Aprovecho que solo hay 2 parámetros para usar un circuito muy profundo (p=10)
    p_pasos = 10 
    
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

    # 3. Generador de la Rampa Lineal (Linear Schedule)
    def generar_rampa(parametros):
        """
        Calcula los 2p ángulos a partir de los 2 coeficientes de pendiente.
        """
        c_gamma, c_beta = parametros
        gammas, betas = [], []
        
        for i in range(1, p_pasos + 1):
            t = i / p_pasos # Progreso normalizado de 0 a 1
            # Rampa ascendente para el Coste, descendente para el Mezclador
            gammas.append(c_gamma * t)
            betas.append(c_beta * (1.0 - t))
            
        return gammas, betas

    # 4. Función de Coste (Evolución Rápida)
    def funcion_coste_lr(parametros):
        gammas, betas = generar_rampa(parametros)
        
        st = np.ones(2**n_qubits, dtype=np.complex128) / np.sqrt(2**n_qubits)
        
        for i in range(p_pasos):
            st = st * np.exp(-1j * gammas[i] * vector_hp)
            qc_mix = QuantumCircuit(n_qubits)
            qc_mix.rx(2 * betas[i], range(n_qubits))
            st = Statevector(st).evolve(qc_mix).data
            
        return np.dot(np.abs(st)**2, vector_hp)

    # 5. Optimización Clásica de 2 dimensiones
    start_time = time.time()
    
    # Inicialización aleatoria de los topes de la rampa
    params_init = np.random.uniform(0.1, np.pi, 2)
    
    # L-BFGS-B convergerá increíblemente rápido al haber solo 2 dimensiones
    res = minimize(funcion_coste_lr, params_init, method='L-BFGS-B', options={'maxiter': 50})
    tiempo_total_q = time.time() - start_time
    
    # 6. Extracción de la Solución Final
    gammas_opt, betas_opt = generar_rampa(res.x)
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
    print("Ejecutando LR-QAOA (Linear-Ramp QAOA)...")
    print(ejecutar_lr_qaoa())