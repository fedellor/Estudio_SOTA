"""
TFG: Optimizacion Cuantica de Hiperparametros (HPO)
Archivo 02: Simulacion VQE a 15 Qubits (Simplificado + Metricas)
"""
import json
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize

# Ajuste de rutas relativas para mi entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

from qiskit import transpile
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector

def ejecutar_vqe():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    # 1. Cargo los datos del modelo subrogado
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    
    # 2. Creo el paisaje de energia (Vector Diagonal)
    # Minimizar la energia equivale a maximizar la precision clasica
    vector_energias = np.zeros(2**n_qubits)
    
    for bitstring, acc in resultados.items():
        indice = int(bitstring, 2)
        vector_energias[indice] = -acc
            
    # 3. Defino el Circuito Cuantico (Ansatz)
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=2, entanglement='linear')
    num_parametros = ansatz.num_parameters
    
    # Extraigo metricas fisicas del circuito transpilandolo
    qc_aud = transpile(ansatz, basis_gates=['u', 'cx'], optimization_level=3)
    cnot_totales = qc_aud.count_ops().get('cx', 0)
    profundidad_estimada = qc_aud.depth()
    
    # 4. Defino la funcion de coste VQE pura
    def funcion_coste(parametros):
        qc = ansatz.assign_parameters(parametros)
        probabilidades = Statevector(qc).probabilities()
        return np.dot(probabilidades, vector_energias)
    
    # 5. Optimizacion COBYLA (Gradient-Free) con Multi-Start
    n_inicios = 3
    mejor_energia = float('inf')
    mejores_parametros = None
    evals_totales = 0
    
    start_time = time.time()
    
    for intento in range(1, n_inicios + 1):
        # El runner inyecta la semilla global, por lo que np.random 
        # explorara diferentes inicios sin fijar la seed internamente.
        punto_inicial = np.random.uniform(-np.pi, np.pi, num_parametros)
        
        resultado = minimize(
            funcion_coste, 
            punto_inicial, 
            method='COBYLA', 
            options={'maxiter': 300, 'tol': 1e-3}
        )
        
        evals_totales += resultado.nfev
        
        if resultado.fun < mejor_energia:
            mejor_energia = resultado.fun
            mejores_parametros = resultado.x
            
    tiempo_total = time.time() - start_time
    
    # 6. Extraigo los resultados colapsando la funcion de onda
    qc_final = ansatz.assign_parameters(mejores_parametros)
    probabilidades_finales = Statevector(qc_final).probabilities()
    estado_ganador = format(np.argmax(probabilidades_finales), f'0{n_qubits}b')
    precision_surrogada = resultados.get(estado_ganador, 0)
    
    # Retorno en el formato estricto del orquestador
    return estado_ganador, precision_surrogada, tiempo_total, cnot_totales, profundidad_estimada, evals_totales

if __name__ == "__main__":
    print("Este archivo debe ser llamado desde runner_experimentos.py")