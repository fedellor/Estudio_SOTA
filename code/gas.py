"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo 34: GAS (Grover Adaptive Search)

Implemento la búsqueda adaptativa de Grover con umbral dinámico y
reducción heurística del número de consultas (rotaciones).

Referencias implementadas y analizadas:
1. Giuffrida, Volpe, Cirillo, Zamboni & Turvani (2022): "Engineering Grover 
   Adaptive Search: Exploring the Degrees of Freedom for Efficient QUBO Solving".
2. Ominato, Ohyama & Yamaguchi (2024): "Grover Adaptive Search With Fewer Queries".
3. Norimoto, Mikuriya & Ishikawa (2024): "Quantum Speedup for Multiuser Detection 
   With Optimized Parameters in Grover Adaptive Search".
"""

import json
import os
import sys
import time
import math
import numpy as np

# Ajusto las rutas para mi entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import GroverOperator, Diagonal

def ejecutar_gas():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    N_total = 2**n_qubits

    # 1. Auditoría de Hardware (Coste de 1 iteración de Grover)
    # Construyo un oráculo denso de prueba para extraer las métricas físicas
    oraculo_dummy = Diagonal([1] * (N_total - 1) + [-1])
    operador_grover = GroverOperator(oracle=oraculo_dummy)
    qc_aud = transpile(operador_grover.decompose(), basis_gates=['u', 'cx'], optimization_level=1)
    
    cnot_por_iteracion = qc_aud.count_ops().get('cx', 0)
    depth_por_iteracion = qc_aud.depth()

    # 2. Inicialización del GAS (Warm-Start Clásico)
    # Norimoto (2024) demuestra que optimizar el umbral inicial reduce las consultas
    muestras_iniciales = np.random.choice(list(resultados.keys()), 15, replace=False)
    mejor_estado_actual = max(muestras_iniciales, key=lambda x: resultados[x])
    umbral_actual = resultados[mejor_estado_actual]

    start_time = time.time()
    consultas_totales = 15 # Contabilizo las muestras iniciales
    cnot_totales = 0
    profundidad_total = 0

    # Parámetros de control de GAS (Giuffrida 2022)
    max_intentos_sin_mejora = 10 
    intentos_fallidos = 0
    lambda_rotaciones = 1.0 # Multiplicador de rotaciones (Ominato 2024)

    # 3. Bucle Adaptativo de Grover
    while intentos_fallidos < max_intentos_sin_mejora:
        # Construyo el oráculo de fase para el umbral actual (marco estados > umbral)
        diagonal_elements = []
        soluciones_validas = 0
        for i in range(N_total):
            # Qiskit ordena en Little-Endian, invierto el string
            bitstring = format(i, f'0{n_qubits}b')[::-1]
            acc = resultados.get(bitstring, 0.0)
            if acc > umbral_actual:
                diagonal_elements.append(-1)
                soluciones_validas += 1
            else:
                diagonal_elements.append(1)

        # Si no quedan soluciones por encima del umbral, he encontrado el óptimo
        if soluciones_validas == 0:
            break 

        # Estrategia "Fewer Queries": elijo un número de rotaciones aleatorio 
        # acotado por lambda para no sobrerrotar el estado (heurística Dürr-Høyer)
        limite_superior = math.ceil(lambda_rotaciones)
        iteraciones_grover = np.random.randint(0, limite_superior + 1)
        if iteraciones_grover == 0: 
            iteraciones_grover = 1

        # Registro la auditoría física de este paso
        cnot_totales += cnot_por_iteracion * iteraciones_grover
        profundidad_total += depth_por_iteracion * iteraciones_grover
        consultas_totales += iteraciones_grover

        # 4. Evolución del Estado (Simulación Algebraica Rápida)
        # Aplico el operador directamente sobre el vector para máxima velocidad
        estado_superposicion = np.ones(N_total) / np.sqrt(N_total)

        for _ in range(iteraciones_grover):
            # A. Aplicación del Oráculo (Inversión de fase)
            estado_superposicion = estado_superposicion * np.array(diagonal_elements)
            # B. Aplicación del Operador de Difusión (Inversión sobre la media)
            media = np.mean(estado_superposicion)
            estado_superposicion = 2 * media - estado_superposicion

        # 5. Medición (Colapso de la función de onda)
        probabilidades = np.abs(estado_superposicion)**2
        medicion_idx = np.random.choice(N_total, p=probabilidades)
        estado_medido = format(medicion_idx, f'0{n_qubits}b')[::-1]
        precision_medida = resultados.get(estado_medido, 0.0)

        # 6. Actualización del Umbral
        if precision_medida > umbral_actual:
            umbral_actual = precision_medida
            mejor_estado_actual = estado_medido
            intentos_fallidos = 0
            lambda_rotaciones = 1.0 # Reseteo la ventana de búsqueda
        else:
            intentos_fallidos += 1
            lambda_rotaciones *= 1.2 # Expando el límite de rotaciones (Adaptive schedule)

    tiempo_total = time.time() - start_time

    return mejor_estado_actual, umbral_actual, tiempo_total, cnot_totales, profundidad_total, consultas_totales

if __name__ == "__main__":
    print("Ejecutando Grover Adaptive Search (GAS)...")
    print(ejecutar_gas())