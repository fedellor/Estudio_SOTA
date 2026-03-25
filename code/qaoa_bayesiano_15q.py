"""
TFG: Optimizacion Cuantica de Hiperparametros (HPO)
Archivo 05: QAOA con Optimizador Bayesiano a 15 Qubits (+ Metricas Fisicas)
"""
import json
import os
import sys
import time
import numpy as np

# Ajusto las rutas relativas para mi entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from skopt import gp_minimize
from skopt.space import Real

def ejecutar_qaoa_bayesiano():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    # 1. Cargo mi mapa de precisiones
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
        pauli_strings.append(estado_qiskit.replace('0', 'I').replace('1', 'Z'))
        coeficientes.append(-resultados[estado])
        
    hamiltoniano = SparsePauliOp(pauli_strings, coeffs=coeficientes)
    circuito_qaoa = QAOAAnsatz(cost_operator=hamiltoniano, reps=1)
    
    # Transpilo para obtener la profundidad y CNOTs óptimas
    qc_aud = transpile(circuito_qaoa, basis_gates=['u', 'cx'], optimization_level=3)
    num_cx = qc_aud.count_ops().get('cx', 0)
    profundidad = qc_aud.depth()
    
    # 3. Construyo el Hamiltoniano Diagonal para evaluar rapidamente
    vector_energias = np.zeros(2**n_qubits)
    for bitstring, acc in resultados.items():
        indice = int(bitstring, 2)
        vector_energias[indice] = -acc
            
    # 4. Defino la función de coste matemática (p=1)
    def funcion_coste_qaoa(parametros):
        gamma, beta = parametros
        estado_array = np.ones(2**n_qubits, dtype=np.complex128) / np.sqrt(2**n_qubits)
        estado_array = estado_array * np.exp(-1j * gamma * vector_energias)
        qc_mixer = QuantumCircuit(n_qubits)
        qc_mixer.rx(2 * beta, range(n_qubits))
        estado_final = Statevector(estado_array).evolve(qc_mixer)
        probabilidades = estado_final.probabilities()
        return np.dot(probabilidades, vector_energias)

    # 5. Configuro el Optimizador Bayesiano
    # Elimino random_state=42 para que el runner gestione la semilla global
    espacio_busqueda = [
        Real(-np.pi, np.pi, name='gamma'),
        Real(-np.pi, np.pi, name='beta')
    ]
    
    start_time = time.time()
    
    # Ejecuto la búsqueda con Procesos Gaussianos
    resultado_bayesiano = gp_minimize(
        func=funcion_coste_qaoa,
        dimensions=espacio_busqueda,
        n_calls=40,
        n_initial_points=10
    )
    
    tiempo_total = time.time() - start_time
    
    # 6. Extraccion del estado ganador
    gamma_opt, beta_opt = resultado_bayesiano.x
    estado_array = np.ones(2**n_qubits, dtype=np.complex128) / np.sqrt(2**n_qubits)
    estado_array = estado_array * np.exp(-1j * gamma_opt * vector_energias)
    qc_mixer = QuantumCircuit(n_qubits)
    qc_mixer.rx(2 * beta_opt, range(n_qubits))
    estado_final = Statevector(estado_array).evolve(qc_mixer)
    
    probabilidades_finales = estado_final.probabilities()
    estado_ganador = format(np.argmax(probabilidades_finales), f'0{n_qubits}b')
    precision_surrogada = resultados.get(estado_ganador, 0)
    evals_totales = len(resultado_bayesiano.func_vals)

    # Retorno exacto para el orquestador
    return estado_ganador, precision_surrogada, tiempo_total, num_cx, profundidad, evals_totales

if __name__ == "__main__":
    print("Este archivo debe ser llamado desde runner_experimentos.py")