"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo 27: VQSD (Variational Quantum State Diagonalization)

Implemento la diagonalización variacional para extraer la configuración 
de hiperparámetros óptima como el componente principal (autovector dominante) 
del paisaje de precisión.

Referencias implementadas y analizadas:
1. LaRose, Tikku, O'Neel-Judy, Cincio & Coles (2019): "Variational quantum 
   state diagonalization".
2. Arrow (2022): "Assessing the Trainability of the Variational Quantum State 
   Diagonalization Algorithm at Scale".
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
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Statevector

def ejecutar_vqsd():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    
    # 1. Construcción del Estado Mixto (Paisaje HPO)
    # Extraigo las precisiones como espectro de autovalores
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    
    # Normalizo para que funcionen como una distribución de probabilidad (Traza = 1)
    espectro = (acc_vals - min_acc) / (max_acc - min_acc)
    suma_espectro = np.sum(espectro)
    if suma_espectro > 0:
        espectro /= suma_espectro
        
    vector_diagonal = np.zeros(2**n_qubits)
    for bitstring, acc in resultados.items():
        idx = int(bitstring, 2)
        # Asigno el autovalor correspondiente a cada estado computacional
        val_norm = (acc - min_acc) / (max_acc - min_acc)
        vector_diagonal[idx] = val_norm / suma_espectro

    # 2. Auditoría de Hardware (Ansatz VQSD)
    # Arrow (2022) evalúa la entrenabilidad usando Hardware-Efficient Ansätze.
    # Utilizo un EfficientSU2 con entrelazamiento lineal y profundidad p=2.
    reps = 2
    ansatz_vqsd = EfficientSU2(num_qubits=n_qubits, su2_gates=['ry', 'rz'], entanglement='linear', reps=reps)
    num_params = ansatz_vqsd.num_parameters
    
    # Transpilo para extraer las métricas físicas reales
    qc_aud = transpile(ansatz_vqsd, basis_gates=['u', 'cx'], optimization_level=3)
    cnot_totales = qc_aud.count_ops().get('cx', 0)
    profundidad = qc_aud.depth()

    # 3. Función de Coste VQSD (Componente Principal)
    # LaRose et al. (2019) definen el coste minimizando la no-diagonalidad.
    # Para extraer el autovector dominante (PCA cuántico), el coste equivale a 
    # maximizar el solapamiento del estado base |0...0> transformado por U^dagger 
    # con el estado de máxima probabilidad.
    
    def funcion_coste_vqsd(parametros):
        # Asigno los parámetros al ansatz para crear U
        qc_bound = ansatz_vqsd.assign_parameters(parametros)
        
        # Simulo la acción de U sobre el estado de referencia |0...0>
        # En VQSD, U^dagger diagonaliza el estado, por lo que U prepara el autovector dominante
        st = Statevector(qc_bound).data
        
        # El valor esperado de nuestro "estado mixto" (que ya es diagonal en Z)
        # equivale a la suma ponderada de las probabilidades por los autovalores
        probabilidades = np.abs(st)**2
        
        # Minimizamos el negativo del solapamiento para maximizar el componente principal
        return -np.dot(probabilidades, vector_diagonal)

    # 4. Fase de Entrenamiento (Mitigando Barren Plateaus)
    # Arrow (2022) demuestra que VQSD sufre severos Barren Plateaus a gran escala.
    # Utilizo múltiples puntos de inicio (multi-start) para aumentar la probabilidad de éxito.
    start_time = time.time()
    n_inicios = 3
    mejor_coste = float('inf')
    mejores_parametros = None
    evals_totales = 0
    
    for intento in range(n_inicios):
        # Inicio aleatorio
        params_init = np.random.uniform(-np.pi, np.pi, num_params)
        
        # Optimización con L-BFGS-B (gradientes aproximados)
        res = minimize(funcion_coste_vqsd, params_init, method='L-BFGS-B', options={'maxiter': 60})
        evals_totales += res.nfev
        
        if res.fun < mejor_coste:
            mejor_coste = res.fun
            mejores_parametros = res.x

    tiempo_total_q = time.time() - start_time
    
    # 5. Extracción del Resultado Final
    qc_final = ansatz_vqsd.assign_parameters(mejores_parametros)
    st_final = Statevector(qc_final).data
    probabilidades_finales = np.abs(st_final)**2
    
    idx_max = np.argmax(probabilidades_finales)
    best_bitstring = format(idx_max, f'0{n_qubits}b')
    precision_surrogada = resultados.get(best_bitstring, 0)

    # Retorno exacto para la integración en runner_experimentos.py
    return best_bitstring, precision_surrogada, tiempo_total_q, cnot_totales, profundidad, evals_totales

if __name__ == "__main__":
    print("Ejecutando VQSD (Variational Quantum State Diagonalization)...")
    print(ejecutar_vqsd())