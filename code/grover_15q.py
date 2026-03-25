"""
TFG: Optimizacion Cuantica de Hiperparametros (HPO)
Archivo 03: Algoritmo de Grover a 15 Qubits (Oraculo Diagonal Exacto)
"""
import json
import os
import sys
import time
import math
import numpy as np

# Me aseguro de que las rutas funcionen en el entorno del orquestador
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import Diagonal, GroverOperator
from qiskit.quantum_info import Statevector

def ejecutar_grover():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    N_total = 2**n_qubits
    
    # 1. Construcción del Oráculo (Fase de marcado)
    # Busco el máximo global (Precision >= 81.50%)
    umbral_exito = 81.50
    diagonal_elements = []
    
    for i in range(N_total):
        # Qiskit usa Little Endian, así que invierto para mi dataset JSON
        bitstring_json = format(i, f'0{n_qubits}b')[::-1]
        acc = resultados.get(bitstring_json, 0.0)
        
        # Oráculo de fase: invierto la fase (-1) de los estados que cumplen el umbral
        if acc >= umbral_exito:
            diagonal_elements.append(-1)
        else:
            diagonal_elements.append(1)
            
    num_soluciones = diagonal_elements.count(-1)
    
    # Fallback si el umbral es demasiado exigente para evitar errores
    if num_soluciones == 0:
        return "000000000000000", 0, 0, 0, 0, 0

    # 2. Parámetros óptimos
    # Aplico la fórmula matemática: $$\frac{\pi}{4} \sqrt{\frac{N}{M}}$$
    iteraciones_optimas = math.floor((math.pi / 4) * math.sqrt(N_total / num_soluciones))
    
    start_time = time.time()
    
    # 3. Construcción del Circuito y Extracción de Métricas Físicas
    oraculo = Diagonal(diagonal_elements)
    operador_grover = GroverOperator(oracle=oraculo)
    
    # Creo el circuito completo para auditar su coste en hardware
    qc_completo = QuantumCircuit(n_qubits)
    qc_completo.h(range(n_qubits))
    qc_completo.compose(operador_grover.power(iteraciones_optimas), inplace=True)
    
    # Transpilo a puertas base de IBM (cx, u) para obtener datos realistas
    qc_trans = transpile(qc_completo, basis_gates=['u', 'cx'], optimization_level=3)
    cnots = qc_trans.count_ops().get('cx', 0)
    profundidad = qc_trans.depth()
    
    # 4. Simulación (Evolución de la función de onda)
    # Ejecuto el algoritmo mediante simulación de vector de estado
    probabilidades = Statevector(qc_completo).probabilities()
    
    indice_ganador = np.argmax(probabilidades)
    estado_ganador_json = format(indice_ganador, f'0{n_qubits}b')[::-1]
    precision_surrogada = resultados.get(estado_ganador_json, 0)
    
    tiempo_total = time.time() - start_time
    
    # Grover es un algoritmo de búsqueda única, por lo que cuento como 1 evaluación cuántica total
    evals_totales = 1 

    # Retorno la tupla de 6 variables para mi orquestador
    return estado_ganador_json, precision_surrogada, tiempo_total, cnots, profundidad, evals_totales

if __name__ == "__main__":
    # Prueba de ejecución individual
    resultado = ejecutar_grover()
    print(f"Resultado Grover: {resultado}")