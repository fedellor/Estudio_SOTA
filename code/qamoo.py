"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo 40: QAMOO (Quantum Approximate Multi-Objective Optimization) - Optimizado

Implemento un algoritmo QAOA multiobjetivo que maximiza el indicador de 
hipervolumen (HV). Utilizo contracciones tensoriales nativas de NumPy para 
evitar el cuello de botella de la exponenciación de matrices dispersas en Qiskit.

Referencias implementadas y analizadas:
1. Kotil et al. (2025): "Quantum approximate multi-objective optimization".
2. Ekstrom et al. (2025): "Variational quantum multiobjective optimization".
3. Ekstrom et al. (2026): "Improving Quantum Multi-Objective Optimization 
   with Archiving and Substitution".
"""

import json
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz

# Ajusto las rutas para mi entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

def ejecutar_qamoo():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    
    # 1. Definición Multiobjetivo (f1, f2)
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    
    espacio_moo = {}
    vector_coste_f1 = np.zeros(2**n_qubits)
    
    for bitstring, acc in resultados.items():
        # f1: Precisión (A maximizar)
        f1 = (acc - min_acc) / (max_acc - min_acc)
        # f2: Ligereza del modelo (A maximizar)
        peso_hamming = sum(int(b) for b in bitstring)
        f2 = (n_qubits - peso_hamming) / n_qubits
        
        espacio_moo[bitstring] = (f1, f2)
        # Vector para guiar la fase QAOA 
        vector_coste_f1[int(bitstring, 2)] = -f1

    # 2. Auditoría de Hardware (Cálculo de CNOTs teóricas)
    p_pasos = 2
    estados_ordenados = sorted(resultados.keys(), key=lambda x: resultados[x], reverse=True)
    mejores_audit = estados_ordenados[:50]
    pauli_list = [(s[::-1].replace('0','I').replace('1','Z'), -1.0) for s in mejores_audit]
    ham_coste = SparsePauliOp.from_list(pauli_list)
    
    ansatz_audit = QAOAAnsatz(cost_operator=ham_coste, reps=p_pasos).decompose()
    qc_aud = transpile(ansatz_audit, basis_gates=['u', 'cx'], optimization_level=1)
    cnot_totales = qc_aud.count_ops().get('cx', 0)
    profundidad = qc_aud.depth()

    # 3. Funciones de Hipervolumen y Archivo de Pareto
    archivo_pareto = {} 
    
    def actualizar_archivo(nuevas_soluciones):
        """Actualizo el archivo manteniendo solo las soluciones no dominadas."""
        candidatos = {**archivo_pareto, **nuevas_soluciones}
        no_dominados = {}
        
        for k1, v1 in candidatos.items():
            dominado = False
            for k2, v2 in candidatos.items():
                if k1 == k2: continue
                if v2[0] >= v1[0] and v2[1] >= v1[1] and (v2[0] > v1[0] or v2[1] > v1[1]):
                    dominado = True
                    break
            if not dominado:
                no_dominados[k1] = v1
        return no_dominados

    def calcular_hipervolumen_2d(frente_pareto, punto_referencia=(0.0, 0.0)):
        """Calculo el hipervolumen exacto en 2D."""
        if not frente_pareto: return 0.0
        puntos = sorted(list(frente_pareto.values()), key=lambda x: x[0])
        hv = 0.0
        altura_previa = punto_referencia[1]
        
        for p in puntos:
            ancho = p[0] - punto_referencia[0]
            alto_incremental = p[1] - altura_previa
            if ancho > 0 and alto_incremental > 0:
                hv += ancho * alto_incremental
                altura_previa = p[1]
        return hv

    # 4. Motor de Evaluación Multiobjetivo (Nativo NumPy TensorDot)
    def funcion_coste_qamoo(parametros):
        nonlocal archivo_pareto
        gammas = parametros[:p_pasos]
        betas = parametros[p_pasos:]
        
        # Evolución tensorial ultrarrápida
        tensor_st = np.ones((2,) * n_qubits, dtype=np.complex128) / np.sqrt(2**n_qubits)
        
        for i in range(p_pasos):
            st_flat = tensor_st.flatten()
            st_flat *= np.exp(-1j * gammas[i] * vector_coste_f1)
            tensor_st = st_flat.reshape((2,) * n_qubits)
            
            theta = 2 * betas[i]
            c, s = np.cos(theta/2), -1j * np.sin(theta/2)
            rx_matrix = np.array([[c, s], [s, c]], dtype=np.complex128)
            
            for q in range(n_qubits):
                tensor_st = np.tensordot(rx_matrix, tensor_st, axes=([1], [q]))
                tensor_st = np.moveaxis(tensor_st, 0, q)
                
        probabilidades = np.abs(tensor_st.flatten())**2
        
        # Simulo el muestreo del estado superpuesto
        indices_top = np.argsort(probabilidades)[-20:]
        soluciones_medidas = {}
        for idx in indices_top:
            bitstring = format(idx, f'0{n_qubits}b')
            soluciones_medidas[bitstring] = espacio_moo[bitstring]
            
        frente_local = actualizar_archivo(soluciones_medidas)
        archivo_pareto = actualizar_archivo(soluciones_medidas)
        hv = calcular_hipervolumen_2d(frente_local)
        
        return -hv

    # 5. Optimización Híbrida
    start_time = time.time()
    params_init = np.random.uniform(-np.pi, np.pi, 2 * p_pasos)
    
    res = minimize(funcion_coste_qamoo, params_init, method='COBYLA', options={'maxiter': 60})
    tiempo_total_q = time.time() - start_time
    
    # 6. Extracción del Resultado
    mejor_bitstring = max(archivo_pareto.keys(), key=lambda k: espacio_moo[k][0])
    precision_surrogada = resultados.get(mejor_bitstring, 0)

    return mejor_bitstring, precision_surrogada, tiempo_total_q, cnot_totales, profundidad, res.nfev

if __name__ == "__main__":
    print("Ejecutando QAMOO (Optimizado con TensorDot)...")
    print(ejecutar_qamoo())