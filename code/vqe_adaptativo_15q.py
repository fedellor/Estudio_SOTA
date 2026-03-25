"""
TFG: Optimizacion Cuantica de Hiperparametros (HPO)
Archivo: adapt_vqe_15q.py (ADAPT-VQE Estricto)

Fiel a la literatura: Stadelmann (2026) y Jiang (2025).
Implemento la construcción del ansatz puerta a puerta basándome en el gradiente.
"""

import json
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

# Ajusto las rutas relativas para mi entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

def ejecutar_adapt_vqe():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    
    # 1. Construyo el Hamiltoniano del problema de optimización
    vector_energias = np.zeros(2**n_qubits)
    for bitstring, acc in resultados.items():
        indice = int(bitstring, 2)
        vector_energias[indice] = -acc
            
    # 2. Defino mi Pool de Operadores (Generadores)
    # Incluyo 15 rotaciones locales (RY) y 14 entrelazadores controlados (CRY)
    pool_operadores = []
    for i in range(n_qubits):
        pool_operadores.append(('ry', i))
    for i in range(n_qubits - 1):
        pool_operadores.append(('cry', i, i+1))
    
    # 3. Funciones de simulación y evaluación
    def construir_circuito(instrucciones_ansatz, parametros):
        # Inicializo en superposición uniforme para explorar todo el espacio
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits)) 
        for instr, theta in zip(instrucciones_ansatz, parametros):
            tipo = instr[0]
            if tipo == 'ry':
                qc.ry(theta, instr[1])
            elif tipo == 'cry':
                qc.cry(theta, instr[1], instr[2])
        return qc

    def evaluar_energia(instrucciones_ansatz, parametros):
        qc = construir_circuito(instrucciones_ansatz, parametros)
        probabilidades = Statevector(qc).probabilities()
        return np.dot(probabilidades, vector_energias)

    # 4. Bucle Principal ADAPT-VQE con SOAP y Anti-Trough
    max_iteraciones_adapt = 10
    umbral_gradiente = 0.05 
    paciencia_maxima = 1 
    paciencia = paciencia_maxima
    
    ansatz_actual = []
    parametros_actuales = np.array([])
    evals_totales = 0
    
    start_time = time.time()
    
    for iteracion in range(1, max_iteraciones_adapt + 1):
        # FASE A: Evaluación de Gradientes (Selección del mejor operador)
        gradientes = []
        eps = 0.01 
        
        for operador in pool_operadores:
            # Calculo gradientes por diferencias finitas
            e_mas = evaluar_energia(ansatz_actual + [operador], np.append(parametros_actuales, eps))
            e_menos = evaluar_energia(ansatz_actual + [operador], np.append(parametros_actuales, -eps))
            grad = (e_mas - e_menos) / (2 * eps)
            gradientes.append(abs(grad))
            evals_totales += 2
            
        indice_mejor_op = np.argmax(gradientes)
        max_grad = gradientes[indice_mejor_op]
        mejor_op = pool_operadores[indice_mejor_op]
        
        # FASE B: Estrategia Anti-Trough para escapar de mesetas áridas
        if max_grad < umbral_gradiente:
            if paciencia > 0:
                paciencia -= 1
            else:
                break # Convergencia alcanzada
        else:
            paciencia = paciencia_maxima 
            
        ansatz_actual.append(mejor_op)
        
        # FASE C: Optimizador SOAP (Fase Local + Global)
        # 1. Optimización Local (Solo el nuevo parámetro)
        def coste_local(theta_nuevo):
            return evaluar_energia(ansatz_actual, np.append(parametros_actuales, theta_nuevo[0]))
        res_local = minimize(coste_local, [0.0], method='BFGS')
        evals_totales += res_local.nfev
        
        # 2. Optimización Global (Ajuste de todo el circuito)
        def coste_global(todos_params):
            return evaluar_energia(ansatz_actual, todos_params)
        punto_ini = np.append(parametros_actuales, res_local.x[0])
        res_global = minimize(coste_global, punto_ini, method='BFGS')
        evals_totales += res_global.nfev
        
        parametros_actuales = res_global.x

    tiempo_total = time.time() - start_time
    
    # 5. Extracción de Resultados y Métricas de Hardware
    qc_final = construir_circuito(ansatz_actual, parametros_actuales)
    # Transpilo para obtener métricas reales de puertas y profundidad
    qc_aud = transpile(qc_final, basis_gates=['u', 'cx'], optimization_level=3)
    cnot_totales = qc_aud.count_ops().get('cx', 0)
    profundidad = qc_aud.depth()
    
    prob_final = Statevector(qc_final).probabilities()
    idx_max = np.argmax(prob_final)
    best_bitstring = format(idx_max, f'0{n_qubits}b')
    precision_surrogada = resultados.get(best_bitstring, 0)

    # Devuelvo las 6 variables para el orquestador
    return best_bitstring, precision_surrogada, tiempo_total, cnot_totales, profundidad, evals_totales

if __name__ == "__main__":
    print("Ejecutando ADAPT-VQE individual...")
    print(ejecutar_adapt_vqe())