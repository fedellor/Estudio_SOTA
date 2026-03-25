"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo 43: QGA (Quantum Genetic Algorithm)

Implemento un Algoritmo Genético Cuántico donde los cromosomas se representan 
mediante amplitudes de probabilidad de qubits y evolucionan utilizando 
puertas de rotación cuántica.

Referencias implementadas y analizadas:
1. "Analog Integrated Circuit Optimization With an Enhanced Adaptive Quantum 
   Genetic Algorithm" (Estrategia adaptativa de rotación y actualización).
2. "A Quantum Genetic Algorithm Framework" (Framework general, representación 
   de cromosomas y colapso probabilístico).
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

def ejecutar_qga():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    
    start_time = time.time()
    evaluaciones_totales = 0
    
    # 1. Parámetros del QGA
    tam_poblacion = 20
    max_generaciones = 50
    theta_paso = 0.05 * math.pi # Ángulo base de rotación
    
    # 2. Inicialización de la Población Cuántica
    # Inicializo cada gen (qubit) en superposición uniforme (1/sqrt(2)) 
    # Matriz de dimensiones (tam_poblacion, n_qubits, 2) -> [alpha, beta]
    q_poblacion = np.ones((tam_poblacion, n_qubits, 2)) / math.sqrt(2.0)
    
    mejor_estado_global = None
    mejor_fitness_global = -1.0
    
    # Función de Observación (Colapso)
    def medir_cromosoma(q_cromosoma):
        bitstring = ""
        for i in range(n_qubits):
            # La probabilidad de medir '1' es |beta|^2 
            prob_1 = q_cromosoma[i][1] ** 2
            if np.random.rand() < prob_1:
                bitstring += "1"
            else:
                bitstring += "0"
        return bitstring

    print("Iniciando QGA (Quantum Genetic Algorithm)...")

    # 3. Bucle Evolutivo
    for gen in range(max_generaciones):
        poblacion_clasica = []
        fitness_poblacion = []
        
        # A. Observación y Evaluación
        for i in range(tam_poblacion):
            bitstring = medir_cromosoma(q_poblacion[i])
            fit = resultados.get(bitstring, 0.0)
            evaluaciones_totales += 1
            
            poblacion_clasica.append(bitstring)
            fitness_poblacion.append(fit)
            
            # Actualizo el mejor global
            if fit > mejor_fitness_global:
                mejor_fitness_global = fit
                mejor_estado_global = bitstring

        # B. Actualización mediante Puertas de Rotación Cuántica 
        for i in range(tam_poblacion):
            fit_actual = fitness_poblacion[i]
            x_actual = poblacion_clasica[i]
            
            for j in range(n_qubits):
                bit_actual = int(x_actual[j])
                bit_mejor = int(mejor_estado_global[j])
                
                # Estrategia adaptativa (Lookup Table simplificada) 
                # Si el bit actual no coincide con el mejor, roto hacia el mejor
                delta_theta = 0.0
                
                if bit_actual != bit_mejor:
                    # Direccionalidad de la rotación basada en los signos de alpha y beta
                    alpha = q_poblacion[i][j][0]
                    beta = q_poblacion[i][j][1]
                    
                    # Dirijo la probabilidad hacia '1' o '0' según el mejor bit
                    direccion = 1.0 if bit_mejor == 1 else -1.0
                    if alpha * beta < 0:
                        direccion *= -1.0
                    elif alpha * beta == 0 and alpha < 0: # Evito estancamientos
                        direccion *= -1.0
                        
                    delta_theta = direccion * theta_paso
                
                # Aplico la Matriz de Rotación Cuántica 
                if delta_theta != 0.0:
                    alpha_old = q_poblacion[i][j][0]
                    beta_old = q_poblacion[i][j][1]
                    
                    q_poblacion[i][j][0] = math.cos(delta_theta) * alpha_old - math.sin(delta_theta) * beta_old
                    q_poblacion[i][j][1] = math.sin(delta_theta) * alpha_old + math.cos(delta_theta) * beta_old

    tiempo_total = time.time() - start_time

    # Como es un algoritmo bioinspirado clásico, CNOTs y Depth son 0
    return mejor_estado_global, mejor_fitness_global, tiempo_total, 0, 0, evaluaciones_totales

if __name__ == "__main__":
    print("Ejecutando QGA...")
    print(ejecutar_qga())