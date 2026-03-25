"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo 29: DQAS (Differentiable Quantum Architecture Search)

Implementa la búsqueda de arquitectura cuántica diferenciable mediante 
la relajación continua del espacio de puertas y discretización final.

Referencias implementadas y analizadas:
1. Sun, Liu, Ma & Tresp (2024): "Differentiable Quantum Architecture Search 
   For Job Shop Scheduling Problem".
2. Afane, Long, Shen, Mao, Wang & Chen (2026): "Differentiable architecture 
   search for adversarially robust quantum computer vision".
3. Zhu, Pi & Peng (2023): "A Brief Survey of Quantum Architecture Search".
"""

import json
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

# Ajuste de rutas para el entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

def ejecutar_dqas():
    # Carga del dataset de hiperparámetros
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    N = 2**n_qubits
    
    # Parámetros del espacio de búsqueda (Arquitectura)
    p_capas = 2
    n_ops = 3 # Pool de operaciones: [0: Identidad, 1: RY(theta), 2: CX]
    
    # Normalización del paisaje de energía (Coste = -Precisión normalizada)
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    costes = -(acc_vals - min_acc) / (max_acc - min_acc)
    tensor_shape = (2,) * n_qubits

    def evaluar_super_circuito(params):
        """
        Simula la relajación continua del super-circuito (DQAS estándar).
        """
        idx_div = p_capas * n_qubits * n_ops
        # alphas: pesos de la arquitectura | thetas: ángulos de las rotaciones
        alphas = params[:idx_div].reshape((p_capas, n_qubits, n_ops))
        thetas = params[idx_div:].reshape((p_capas, n_qubits))
        
        psi = np.ones(N, dtype=np.complex128) / np.sqrt(N)
        
        for p in range(p_capas):
            # 1. Operador de Coste (Diagonal)
            psi *= np.exp(-1j * costes)
            
            # 2. Super-Operador Mezclador (Mezcla convexa de operaciones)
            psi_reshaped = psi.reshape(tensor_shape)
            for q in range(n_qubits):
                # Aplicación directa de Softmax sobre los pesos de arquitectura (Sin Atención)
                e_a = np.exp(alphas[p, q] - np.max(alphas[p, q]))
                probs = e_a / np.sum(e_a)
                
                # Extracción de amplitudes del qubit objetivo
                p0 = np.take(psi_reshaped, 0, axis=q)
                p1 = np.take(psi_reshaped, 1, axis=q)
                
                # Operación 1: Puerta RY(theta)
                c, s = np.cos(thetas[p,q]/2), np.sin(thetas[p,q]/2)
                ry0, ry1 = c*p0 - s*p1, s*p0 + c*p1
                
                # Operación 2: Puerta CX (Entrelazamiento con el vecino cíclico)
                cx0, cx1 = p0, p1
                if q < n_qubits - 1:
                    cx1 = np.roll(p1, shift=1) 
                
                # Construcción del estado mezclado basado en las probabilidades de la arquitectura
                new_q0 = probs[0]*p0 + probs[1]*ry0 + probs[2]*cx1
                new_q1 = probs[0]*p1 + probs[1]*ry1 + probs[2]*cx1
                
                # Re-inserción en el tensor de estado global
                s0 = [slice(None)] * n_qubits; s0[q] = 0
                s1 = [slice(None)] * n_qubits; s1[q] = 1
                psi_reshaped[tuple(s0)] = new_q0
                psi_reshaped[tuple(s1)] = new_q1
            
            psi = psi_reshaped.flatten()
            norm = np.linalg.norm(psi)
            if norm > 0: psi /= norm
                
        # Retorno la energía esperada del super-estado
        return np.real(np.sum(np.conj(psi) * costes * psi))

    # --- FASE DE ENTRENAMIENTO (BÚSQUEDA DIFERENCIABLE) ---
    start_time = time.time()
    n_params = (p_capas * n_qubits * n_ops) + (p_capas * n_qubits)
    init_params = np.random.normal(0, 0.1, n_params)
    
    # Optimización conjunta de arquitectura (alphas) y parámetros (thetas)
    res = minimize(evaluar_super_circuito, init_params, method='L-BFGS-B', 
                   options={'maxiter': 40, 'ftol': 1e-4})
    
    # --- FASE DE DISCRETIZACIÓN ---
    alphas_f = res.x[:p_capas * n_qubits * n_ops].reshape((p_capas, n_qubits, n_ops))
    thetas_f = res.x[p_capas * n_qubits * n_ops:].reshape((p_capas, n_qubits))
    
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    
    # Congelamos la arquitectura eligiendo la puerta más probable
    for p in range(p_capas):
        for q in range(n_qubits):
            op = np.argmax(alphas_f[p, q])
            if op == 1: # Seleccionada RY
                qc.ry(thetas_f[p, q], q)
            elif op == 2: # Seleccionada CX
                qc.cx(q, (q + 1) % n_qubits)

    # --- MEDICIÓN Y AUDITORÍA DE HARDWARE ---
    # Simulación final sobre el circuito discreto (físico)
    sv_final = Statevector(qc)
    probabilidades = np.abs(sv_final.data)**2
    idx_max = np.argmax(probabilidades)
    best_bitstring = format(idx_max, f'0{n_qubits}b')
    precision_surr = resultados.get(best_bitstring, 0)
    
    # Extracción de métricas de hardware mediante transpilación
    qc_trans = transpile(qc, basis_gates=['u', 'cx'], optimization_level=3)
    cx_count = qc_trans.count_ops().get('cx', 0)
    profundidad = qc_trans.depth()
    
    tiempo_total = time.time() - start_time

    return best_bitstring, precision_surr, tiempo_total, cx_count, profundidad, res.nfev

if __name__ == "__main__":
    print("Ejecutando DQAS (Differentiable Quantum Architecture Search)...")
    print(ejecutar_dqas())