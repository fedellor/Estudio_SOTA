"""
TFG: Optimizacion Cuantica de Hiperparametros (HPO)
Archivo 11: QITE-VQE (Quantum Imaginary Time Evolution VQE)
Implementacion hibrida basada en Xie et al. (2025 IEEE).
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
from qiskit.circuit.library import EfficientSU2

def ejecutar_qite_vqe():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    # Cargo el dataset de hiperparametros
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    estados_ordenados = sorted(resultados.keys(), key=lambda x: resultados[x], reverse=True)
    
    # 1. Preparo el Hamiltoniano y las métricas de hardware
    # Ansatz del VQE (Profundidad p=2 con rotaciones Y, Z de Pauli)
    ansatz = EfficientSU2(num_qubits=n_qubits, su2_gates=['ry', 'rz'], entanglement='linear', reps=2)
    num_params = ansatz.num_parameters
    
    # Transpilo para obtener métricas reales de hardware (CNOTs y Profundidad)
    qc_aud = transpile(ansatz, basis_gates=['u', 'cx'], optimization_level=3)
    cnot_totales = qc_aud.count_ops().get('cx', 0)
    profundidad = qc_aud.depth()

    # Normalizo el paisaje de energia (Hamiltoniano Hp)
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    vector_hp = np.zeros(2**n_qubits)
    for b, acc in resultados.items():
        vector_hp[int(b, 2)] = -(acc - min_acc) / (max_acc - min_acc)

    # 2. Operador de Tiempo Imaginario (Filtro Proyectivo)
    # Aplico e^{-tau * Hp} para amplificar el estado fundamental
    tau_qite = 2.5 
    filtro_qite = np.exp(-tau_qite * vector_hp)
    
    # 3. Funcion de Coste Hibrida (VQE proyectado)
    def funcion_coste_qite(parametros):
        # Genero el estado variacional
        qc = ansatz.assign_parameters(parametros)
        estado_vqe = Statevector(qc).data
        
        # Aplico la proyeccion de Tiempo Imaginario (No unitaria)
        estado_filtrado = estado_vqe * filtro_qite
        
        # Normalizo el estado tras el filtrado
        norma = np.linalg.norm(estado_filtrado)
        if norma < 1e-10: return 0.0
        estado_filtrado = estado_filtrado / norma
        
        # Calculo el valor esperado de la energia
        energia = np.dot(np.abs(estado_filtrado)**2, vector_hp)
        return energia

    # 4. Optimizacion
    start_time = time.time()
    
    # El runner gestiona la semilla global para la reproducibilidad
    params_iniciales = np.random.uniform(-np.pi, np.pi, num_params)
    
    # Utilizo COBYLA para minimizar la energía bajo el filtro QITE
    res = minimize(funcion_coste_qite, params_iniciales, method='COBYLA', options={'maxiter': 300})
    
    tiempo_total_q = time.time() - start_time
    
    # 5. Extraccion del Resultado Final
    qc_final = ansatz.assign_parameters(res.x)
    estado_final = Statevector(qc_final).data
    
    # Aplico el filtro QITE final para obtener la configuración más probable
    estado_final_filtrado = estado_final * filtro_qite
    estado_final_filtrado /= np.linalg.norm(estado_final_filtrado)
    
    idx = np.argmax(np.abs(estado_final_filtrado)**2)
    best_bitstring = format(idx, f'0{n_qubits}b')
    precision_surrogada = resultados.get(best_bitstring, 0)

    # Retorno en el formato estricto: 
    # (bitstring, precisión_subrogada, tiempo_cuántico, cnots, profundidad, evaluaciones)
    return best_bitstring, precision_surrogada, tiempo_total_q, cnot_totales, profundidad, res.nfev

if __name__ == "__main__":
    print("Este archivo debe ser llamado desde runner_experimentos.py")