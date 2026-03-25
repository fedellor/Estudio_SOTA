"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo 45: QACO (Quantum Ant Colony Optimization)

Implemento un algoritmo híbrido cuántico de colonia de hormigas.
Represento los rastros de feromonas mediante las amplitudes de probabilidad
de un registro de qubits y utilizo rotaciones de Pauli para la actualización.

Referencias implementadas y analizadas:
1. "Implementable hybrid quantum ant colony optimization algorithm" (Representación 
   de feromonas como qubits y actualización con puertas cuánticas).
2. "A Novel Quantum Algorithm for Ant Colony" (Construcción de soluciones por colapso).
3. "RefSCAT-2.0..." (Uso de QACO para verificación formal y optimización a gran escala).
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

def ejecutar_qaco():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    
    start_time = time.time()
    evaluaciones_totales = 0
    
    # 1. Parámetros de la Colonia QACO
    n_hormigas = 25
    max_iteraciones = 40
    # Ángulo base para las rotaciones de Pauli (tasa de aprendizaje)
    theta_paso = 0.02 * math.pi 
    
    # 2. Inicialización de las Feromonas Cuánticas
    # Inicializo las "feromonas" como ángulos theta para cada dimensión (qubit).
    # Un ángulo de pi/4 significa superposición uniforme (50% prob de '0', 50% prob de '1').
    feromonas_theta = np.ones(n_qubits) * (math.pi / 4.0)
    
    mejor_estado_global = None
    mejor_fitness_global = -1.0

    print("Inicio la ejecución de QACO (Quantum Ant Colony Optimization)...")

    # 3. Bucle Evolutivo de la Colonia
    for iteracion in range(max_iteraciones):
        soluciones_iteracion = []
        fitness_iteracion = []
        
        # A. Construcción de Caminos (Colapso Cuántico)
        for h in range(n_hormigas):
            camino_hormiga = ""
            for i in range(n_qubits):
                # Calculo la probabilidad de colapsar al estado |1> usando la amplitud sin(theta)^2
                prob_1 = math.sin(feromonas_theta[i]) ** 2
                
                if np.random.rand() < prob_1:
                    camino_hormiga += "1"
                else:
                    camino_hormiga += "0"
            
            # Evalúo el camino construido
            fit = resultados.get(camino_hormiga, 0.0)
            evaluaciones_totales += 1
            
            soluciones_iteracion.append(camino_hormiga)
            fitness_iteracion.append(fit)
            
            # Guardo el mejor global
            if fit > mejor_fitness_global:
                mejor_fitness_global = fit
                mejor_estado_global = camino_hormiga

        # Encuentro a la mejor hormiga de la iteración actual
        idx_mejor_local = np.argmax(fitness_iteracion)
        mejor_estado_local = soluciones_iteracion[idx_mejor_local]

        # B. Actualización de Feromonas Cuánticas (Rotaciones de Pauli)
        # Comparo la mejor solución local con la matriz de feromonas actual
        for i in range(n_qubits):
            bit_optimo = int(mejor_estado_local[i])
            theta_actual = feromonas_theta[i]
            
            # Determino la dirección de la rotación para acercar la probabilidad al bit óptimo
            # Utilizo el equivalente a una rotación basada en Pauli-Y
            delta_theta = 0.0
            if bit_optimo == 1 and theta_actual < (math.pi / 2.0):
                delta_theta = theta_paso
            elif bit_optimo == 0 and theta_actual > 0.0:
                delta_theta = -theta_paso
                
            # Aplico la rotación de Pauli simulada y mantengo los límites [0, pi/2]
            feromonas_theta[i] += delta_theta
            feromonas_theta[i] = max(0.0, min(math.pi / 2.0, feromonas_theta[i]))

    tiempo_total = time.time() - start_time

    # Devuelvo los resultados. Las métricas de CNOTs y Profundidad son 0 al ser simulación Quantum-Inspired.
    return mejor_estado_global, mejor_fitness_global, tiempo_total, 0, 0, evaluaciones_totales

if __name__ == "__main__":
    print("Ejecutando QACO...")
    print(ejecutar_qaco())