"""
TFG: Optimizacion Cuantica de Hiperparametros (HPO)
Archivo 18: RQAOA (Recursive QAOA)

Implemento la versión recursiva de QAOA basándome en:
- Chen et al. (2026): Recursive QAOA for Interference-Aware Resource Allocation.
- Bae & Lee (2024): Recursive QAOA outperforms the original QAOA...
- Finžgar et al. (2024): Quantum-Informed Recursive Optimization Algorithms.
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

def ejecutar_rqaoa():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    
    # 1. Normalizo el paisaje de energía
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    vector_energias = np.zeros(2**n_qubits)
    for bitstring, acc in resultados.items():
        vector_energias[int(bitstring, 2)] = -(acc - min_acc) / (max_acc - min_acc)

    # 2. Auditoría de Hardware (Métricas del circuito base QAOA p=1)
    estados_ordenados = sorted(resultados.keys(), key=lambda x: resultados[x], reverse=True)
    top_k_audit = 100 
    mejores_audit = estados_ordenados[:top_k_audit]
    pauli_list_audit = [(s[::-1].replace('0','I').replace('1','Z'), -1.0) for s in mejores_audit]
    ham_audit = SparsePauliOp.from_list(pauli_list_audit)
    
    ansatz_audit = QAOAAnsatz(cost_operator=ham_audit, reps=1).decompose()
    qc_aud = transpile(ansatz_audit, basis_gates=['u', 'cx'], optimization_level=1)
    
    # Estimo las métricas escalando a la complejidad real
    top_k_total = int(32768 * 0.05)
    cnot_base = int((qc_aud.count_ops().get('cx', 0) / top_k_audit) * top_k_total)
    profundidad_base = qc_aud.depth()

    # 3. Configuración del RQAOA
    n_cutoff = 5 # Umbral de parada de la recursión (variables restantes)
    qubits_activos = list(range(n_qubits))
    relaciones = {} # Almaceno las variables eliminadas y sus restricciones
    
    start_time = time.time()
    evals_totales = 0
    
    # BUCLE RECURSIVO
    while len(qubits_activos) > n_cutoff:
        # A. Defino la función de coste para los qubits activos
        def funcion_coste(parametros):
            gamma, beta = parametros
            st = np.ones(2**n_qubits, dtype=np.complex128) / np.sqrt(2**n_qubits)
            
            # Aplico restricciones actuales al vector de estado (proyección)
            for q_elim, (q_ref, signo) in relaciones.items():
                mascara_valida = np.zeros(2**n_qubits, dtype=bool)
                for i in range(2**n_qubits):
                    b_str = format(i, f'0{n_qubits}b')
                    val_elim = 1 if b_str[q_elim] == '1' else -1
                    if q_ref is None:
                        # Fijación de un solo spin
                        if val_elim == signo: mascara_valida[i] = True
                    else:
                        # Correlación entre dos spins
                        val_ref = 1 if b_str[q_ref] == '1' else -1
                        if val_elim == signo * val_ref: mascara_valida[i] = True
                st[~mascara_valida] = 0
            
            norma = np.linalg.norm(st)
            if norma > 0: st /= norma
            
            # Evolución QAOA
            st = st * np.exp(-1j * gamma * vector_energias)
            qc_mix = QuantumCircuit(n_qubits)
            for q in qubits_activos: # Mezclador solo en qubits libres
                qc_mix.rx(2 * beta, q)
            st = Statevector(st).evolve(qc_mix).data
            
            return np.dot(np.abs(st)**2, vector_energias)

        # B. Optimizo el QAOA en la iteración actual
        punto_inicial = np.random.uniform(-np.pi, np.pi, 2)
        res = minimize(funcion_coste, punto_inicial, method='COBYLA', options={'maxiter': 40})
        evals_totales += res.nfev
        
        # C. Reconstruyo el estado óptimo para medir correlaciones
        gamma_opt, beta_opt = res.x
        st_opt = np.ones(2**n_qubits, dtype=np.complex128) / np.sqrt(2**n_qubits)
        for q_elim, (q_ref, signo) in relaciones.items():
            mascara_valida = np.zeros(2**n_qubits, dtype=bool)
            for i in range(2**n_qubits):
                b_str = format(i, f'0{n_qubits}b')
                val_elim = 1 if b_str[q_elim] == '1' else -1
                if q_ref is None:
                    if val_elim == signo: mascara_valida[i] = True
                else:
                    val_ref = 1 if b_str[q_ref] == '1' else -1
                    if val_elim == signo * val_ref: mascara_valida[i] = True
            st_opt[~mascara_valida] = 0
        norma = np.linalg.norm(st_opt)
        if norma > 0: st_opt /= norma
        
        st_opt = st_opt * np.exp(-1j * gamma_opt * vector_energias)
        qc_mix = QuantumCircuit(n_qubits)
        for q in qubits_activos: qc_mix.rx(2 * beta_opt, q)
        st_opt = Statevector(st_opt).evolve(qc_mix).data
        probabilidades = np.abs(st_opt)**2

        # D. Mido correlaciones <Z_i> y <Z_i Z_j>
        max_correlacion = -1
        mejor_regla = None
        
        for i in range(len(qubits_activos)):
            q_i = qubits_activos[i]
            # <Z_i>
            z_i_val = 0
            for idx, prob in enumerate(probabilidades):
                if prob > 1e-6:
                    b_str = format(idx, f'0{n_qubits}b')
                    z_i_val += prob * (1 if b_str[q_i] == '0' else -1)
                    
            if abs(z_i_val) > max_correlacion:
                max_correlacion = abs(z_i_val)
                mejor_regla = ('single', q_i, np.sign(z_i_val))
                
            # <Z_i Z_j>
            for j in range(i + 1, len(qubits_activos)):
                q_j = qubits_activos[j]
                z_ij_val = 0
                for idx, prob in enumerate(probabilidades):
                    if prob > 1e-6:
                        b_str = format(idx, f'0{n_qubits}b')
                        v_i = 1 if b_str[q_i] == '0' else -1
                        v_j = 1 if b_str[q_j] == '0' else -1
                        z_ij_val += prob * (v_i * v_j)
                        
                if abs(z_ij_val) > max_correlacion:
                    max_correlacion = abs(z_ij_val)
                    mejor_regla = ('pair', q_i, q_j, np.sign(z_ij_val))
                    
        # E. Aplico la regla de eliminación (Decimation)
        if mejor_regla[0] == 'single':
            q_elim = mejor_regla[1]
            relaciones[q_elim] = (None, mejor_regla[2])
            qubits_activos.remove(q_elim)
        else:
            q_i, q_j, signo = mejor_regla[1], mejor_regla[2], mejor_regla[3]
            relaciones[q_j] = (q_i, signo)
            qubits_activos.remove(q_j)

    # 4. Solución Exacta del Núcleo Restante
    mejor_precision = -1
    mejor_estado_final = None
    
    # Fuerzo la búsqueda bruta sobre los qubits activos
    for i in range(2**len(qubits_activos)):
        b_activos = format(i, f'0{len(qubits_activos)}b')
        estado_candidato = ['0'] * n_qubits
        
        # Asigno valores al núcleo
        for idx, q in enumerate(qubits_activos):
            estado_candidato[q] = b_activos[idx]
            
        # Reconstruyo las variables eliminadas (Back-substitution)
        # Itero hasta que todas las dependencias estén resueltas
        cambios = True
        while cambios:
            cambios = False
            for q_elim, (q_ref, signo) in relaciones.items():
                val_antiguo = estado_candidato[q_elim]
                if q_ref is None:
                    val_nuevo = '0' if signo == 1 else '1'
                else:
                    v_ref = 1 if estado_candidato[q_ref] == '0' else -1
                    v_elim = signo * v_ref
                    val_nuevo = '0' if v_elim == 1 else '1'
                    
                if val_antiguo != val_nuevo:
                    estado_candidato[q_elim] = val_nuevo
                    cambios = True
                    
        str_candidato = "".join(estado_candidato)
        acc = resultados.get(str_candidato, 0)
        if acc > mejor_precision:
            mejor_precision = acc
            mejor_estado_final = str_candidato

    tiempo_total = time.time() - start_time
    
    # El coste físico real es la suma de los circuitos QAOA ejecutados, 
    # pero como el tamaño se reduce, acoto el coste máximo al del primer circuito.
    cnot_estimadas = cnot_base * (n_qubits - n_cutoff)
    prof_estimada = profundidad_base * (n_qubits - n_cutoff)

    return mejor_estado_final, mejor_precision, tiempo_total, cnot_estimadas, prof_estimada, evals_totales

if __name__ == "__main__":
    print("Ejecutando RQAOA individual...")
    print(ejecutar_rqaoa())