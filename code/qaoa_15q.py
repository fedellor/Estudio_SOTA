"""
TFG: Optimizacion Cuantica de Hiperparametros (HPO)
Archivo 04: Simulacion QAOA a 15 Qubits (Evolucion Algebraica + Metricas Fisicas)
"""
import json
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize

# Ajusto las rutas relativas para mi entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz

def ejecutar_qaoa():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    # 1. Cargo mi mapa de precisiones subrogado
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    estados_ordenados = sorted(resultados.keys(), key=lambda x: resultados[x], reverse=True)
    
    # 2. Calculo las métricas físicas (CX y Depth)
    # Construyo el Hamiltoniano para extraer métricas de hardware real
    top_k = int(32768 * 0.05) 
    mejores_estados = estados_ordenados[:top_k]
    
    pauli_strings, coeficientes = [], []
    for estado in mejores_estados:
        estado_qiskit = estado[::-1] 
        pauli_str = estado_qiskit.replace('0', 'I').replace('1', 'Z')
        pauli_strings.append(pauli_str)
        coeficientes.append(-resultados[estado])
        
    hamiltoniano = SparsePauliOp(pauli_strings, coeffs=coeficientes)
    circuito_qaoa = QAOAAnsatz(cost_operator=hamiltoniano, reps=1)
    
    # Transpilo con nivel 3 para obtener la profundidad y CNOTs óptimas
    qc_aud = transpile(circuito_qaoa, basis_gates=['u', 'cx'], optimization_level=3)
    num_cx = qc_aud.count_ops().get('cx', 0)
    profundidad = qc_aud.depth()
    
    # 3. Construyo el Hamiltoniano Diagonal para la simulación algebraica rápida
    vector_energias = np.zeros(2**n_qubits)
    for bitstring, acc in resultados.items():
        indice = int(bitstring, 2)
        vector_energias[indice] = -acc
            
    # 4. Defino la función de coste de QAOA (p=1)
    def funcion_coste_qaoa(parametros):
        gamma, beta = parametros
        # Superposición inicial
        estado_array = np.ones(2**n_qubits, dtype=np.complex128) / np.sqrt(2**n_qubits)
        # Operador de Coste (Evolución algebraica)
        estado_array = estado_array * np.exp(-1j * gamma * vector_energias)
        # Operador Mezclador (Qiskit Mixer)
        qc_mixer = QuantumCircuit(n_qubits)
        qc_mixer.rx(2 * beta, range(n_qubits))
        estado_final = Statevector(estado_array).evolve(qc_mixer)
        probabilidades = estado_final.probabilities()
        return np.dot(probabilidades, vector_energias)

    # 5. Optimización COBYLA con Multi-Start
    n_inicios = 3
    mejor_energia = float('inf')
    mejores_parametros = None
    evals_totales = 0
    
    start_time = time.time()
    
    for intento in range(1, n_inicios + 1):
        # Utilizo el estado aleatorio global (semilla gestionada por el runner)
        punto_inicial = np.random.uniform(-np.pi, np.pi, 2)
        
        resultado = minimize(
            funcion_coste_qaoa, 
            punto_inicial, 
            method='COBYLA', 
            options={'maxiter': 300, 'tol': 1e-3}
        )
        
        evals_totales += resultado.nfev
        
        if resultado.fun < mejor_energia:
            mejor_energia = resultado.fun
            mejores_parametros = resultado.x
            
    tiempo_total = time.time() - start_time
    
    # 6. Extracción del estado ganador
    gamma_opt, beta_opt = mejores_parametros
    estado_array = np.ones(2**n_qubits, dtype=np.complex128) / np.sqrt(2**n_qubits)
    estado_array = estado_array * np.exp(-1j * gamma_opt * vector_energias)
    qc_mixer = QuantumCircuit(n_qubits)
    qc_mixer.rx(2 * beta_opt, range(n_qubits))
    estado_final = Statevector(estado_array).evolve(qc_mixer)
    
    probabilidades_finales = estado_final.probabilities()
    estado_ganador = format(np.argmax(probabilidades_finales), f'0{n_qubits}b')
    precision_surrogada = resultados.get(estado_ganador, 0)
    
    # Retorno en el formato estricto: 
    # (bitstring, precisión_subrogada, tiempo_cuántico, cnots, profundidad, evaluaciones)
    return estado_ganador, precision_surrogada, tiempo_total, num_cx, profundidad, evals_totales

if __name__ == "__main__":
    print("Este archivo debe ser llamado desde runner_experimentos.py")