"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo: sa_dqas_15q.py (Self-Attention Differentiable Quantum Architecture Search)

Implemento:
- Differentiable Quantum Architecture Search (Zhang et al., 2021)
- Self-Attention Mechanism for DQAS (Sun et al., 2025)
"""

import json
import os
import time
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

def ejecutar_sa_dqas():
    # Localizo el dataset de precisiones subrogadas
    ruta_json = os.path.join(os.path.dirname(__file__), 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos["qubits"]
    N = 2**n_qubits
    p_capas = 2
    n_ops = 3 # [0:Identidad, 1:RY, 2:CX]
    
    # Normalizo el paisaje de energía para la optimización
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    costes = -(acc_vals - min_acc) / (max_acc - min_acc)
    tensor_shape = (2,) * n_qubits

    def self_attention(alphas):
        """Implemento el mecanismo de atención para capturar dependencias entre qubits (Sun et al. 2025)"""
        alphas_out = np.zeros_like(alphas)
        for p in range(p_capas):
            layer = alphas[p]
            # Calculo el Scaled Dot-Product Attention sobre la matriz de arquitectura
            scores = np.dot(layer, layer.T) / np.sqrt(n_ops)
            attn = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            attn /= np.sum(attn, axis=1, keepdims=True)
            alphas_out[p] = np.dot(attn, layer)
        return alphas_out

    def evaluar_super_circuito(params):
        """Simulo la relajación continua del super-circuito (DQAS)"""
        idx_div = p_capas * n_qubits * n_ops
        alphas = params[:idx_div].reshape((p_capas, n_qubits, n_ops))
        thetas = params[idx_div:].reshape((p_capas, n_qubits))
        
        # Aplico atención antes de la Softmax para ponderar las rutas de la arquitectura
        alphas_attn = self_attention(alphas)
        psi = np.ones(N, dtype=np.complex128) / np.sqrt(N)
        
        for p in range(p_capas):
            # Operador de Coste (Diagonal)
            psi *= np.exp(-1j * costes)
            
            # Super-Operador Mezclador: combinación convexa de operaciones
            psi_reshaped = psi.reshape(tensor_shape)
            for q in range(n_qubits):
                # Calculo probabilidades Softmax de cada puerta
                e_a = np.exp(alphas_attn[p, q] - np.max(alphas_attn[p, q]))
                probs = e_a / np.sum(e_a)
                
                # Extraigo las amplitudes del qubit objetivo
                p0 = np.take(psi_reshaped, 0, axis=q)
                p1 = np.take(psi_reshaped, 1, axis=q)
                
                # Rama 1: Puerta RY(theta)
                c, s = np.cos(thetas[p,q]/2), np.sin(thetas[p,q]/2)
                ry0, ry1 = c*p0 - s*p1, s*p0 + c*p1
                
                # Rama 2: Puerta CX (Entrelazamiento cíclico)
                cx0, cx1 = p0, p1
                if q < n_qubits - 1:
                    cx1 = np.roll(p1, shift=1) 
                
                # Mezcla continua de las tres opciones (Identidad, RY, CX)
                new_q0 = probs[0]*p0 + probs[1]*ry0 + probs[2]*cx1 # Simplificación de mezcla
                new_q1 = probs[0]*p1 + probs[1]*ry1 + probs[2]*cx1
                
                # Reinserto en el tensor de estado
                s0 = [slice(None)] * n_qubits; s0[q] = 0
                s1 = [slice(None)] * n_qubits; s1[q] = 1
                psi_reshaped[tuple(s0)] = new_q0
                psi_reshaped[tuple(s1)] = new_q1
            
            psi = psi_reshaped.flatten()
            norm = np.linalg.norm(psi)
            if norm > 0: psi /= norm
                
        return np.real(np.sum(np.conj(psi) * costes * psi))

    # --- FASE DE ENTRENAMIENTO ---
    start_time = time.time()
    n_params = (p_capas * n_qubits * n_ops) + (p_capas * n_qubits)
    # El runner gestiona la semilla global, uso el estado aleatorio actual
    init_params = np.random.normal(0, 0.1, n_params)
    
    # Optimizo tanto la arquitectura (alphas) como los parámetros (thetas) simultáneamente
    res = minimize(evaluar_super_circuito, init_params, method='L-BFGS-B', 
                   options={'maxiter': 30, 'ftol': 1e-4})
    
    # --- DISCRETIZACIÓN DE LA MEJOR ARQUITECTURA ---
    alphas_f = res.x[:p_capas * n_qubits * n_ops].reshape((p_capas, n_qubits, n_ops))
    thetas_f = res.x[p_capas * n_qubits * n_ops:].reshape((p_capas, n_qubits))
    alphas_attn_f = self_attention(alphas_f)
    
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    
    for p in range(p_capas):
        for q in range(n_qubits):
            # Selecciono la puerta con mayor probabilidad acumulada tras la atención
            op = np.argmax(alphas_attn_f[p, q])
            if op == 1: # RY
                qc.ry(thetas_f[p, q], q)
            elif op == 2: # CX
                qc.cx(q, (q + 1) % n_qubits)

    # --- MEDICIÓN DE RESULTADOS ---
    sv = Statevector(qc)
    idx_max = np.argmax(np.abs(sv.data)**2)
    best_bitstring = format(idx_max, f'0{n_qubits}b')
    precision_surr = resultados.get(best_bitstring, 0)
    
    # Transpilo para obtener métricas de hardware (Depth y CNOTs)
    qc_trans = transpile(qc, basis_gates=['u', 'cx'], optimization_level=3)
    cx_count = qc_trans.count_ops().get('cx', 0)
    profundidad = qc_trans.depth()
    
    tiempo_total = time.time() - start_time

    # Devuelvo la tupla de 6 variables para el runner_experimentos.py
    return best_bitstring, precision_surr, tiempo_total, cx_count, profundidad, res.nfev

if __name__ == "__main__":
    print("Ejecución individual de SA-DQAS...")
    print(ejecutar_sa_dqas())