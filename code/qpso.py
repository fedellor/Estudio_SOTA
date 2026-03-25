"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo 42: QPSO (Quantum-behaved Particle Swarm Optimization)

Implemento un algoritmo de inteligencia de enjambre inspirado en la mecánica
cuántica. Las partículas se comportan según la ecuación de Schrödinger en un 
pozo de potencial Delta, eliminando la necesidad de vectores de velocidad.

Referencias implementadas y analizadas:
1. "Hybrid intelligent model for strength prediction and parameter sensitivity 
   analysis of cemented paste backfill"  (Teoría QPSO, mbest, coeficiente alpha).
2. "A swarm intelligence optimization algorithm on riemannian manifolds"  
   (Adaptación de algoritmos de enjambre a espacios topológicos complejos).
"""

import json
import os
import sys
import time
import numpy as np

# Ajusto las rutas para mi entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

def ejecutar_qpso():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    
    start_time = time.time()
    evaluaciones_totales = 0
    
    # 1. Parámetros del Enjambre QPSO
    n_particulas = 30
    max_iter = 50
    # Coeficiente de Contracción-Expansión (alpha) decreciente para favorecer 
    # la exploración inicial y la explotación final.
    alpha_max = 1.0
    alpha_min = 0.5
    
    # 2. Inicialización
    # Inicializo las posiciones en un espacio continuo [0, 1]
    posiciones = np.random.rand(n_particulas, n_qubits)
    pbest_posiciones = np.copy(posiciones)
    pbest_fitness = np.zeros(n_particulas)
    
    gbest_posicion = np.zeros(n_qubits)
    gbest_fitness = -1.0
    
    # Función para discretizar la posición continua a un bitstring y evaluarlo
    def evaluar_particula(pos_continua):
        # Topología proyectada: mapeo el continuo al espacio binario (Riemanniano/Discreto) 
        bitstring = "".join(['1' if x >= 0.5 else '0' for x in pos_continua])
        return resultados.get(bitstring, 0.0), bitstring

    # Evaluación inicial
    for i in range(n_particulas):
        fit, bs = evaluar_particula(posiciones[i])
        evaluaciones_totales += 1
        pbest_fitness[i] = fit
        
        if fit > gbest_fitness:
            gbest_fitness = fit
            gbest_posicion = np.copy(posiciones[i])
            mejor_bitstring_global = bs

    print("Iniciando QPSO (Quantum-behaved Particle Swarm Optimization)...")

    # 3. Bucle Evolutivo QPSO
    for t in range(max_iter):
        # Actualizo el coeficiente alpha linealmente 
        alpha = alpha_max - (alpha_max - alpha_min) * (t / max_iter)
        
        # Calculo el "Mean Best" (mbest): el centro de las mejores posiciones personales 
        mbest = np.mean(pbest_posiciones, axis=0)
        
        for i in range(n_particulas):
            # A. Genero los atractores cuánticos locales para cada dimensión
            phi = np.random.rand(n_qubits)
            # p_i es el atractor estocástico entre el pbest personal y el gbest global 
            p_atractor = phi * pbest_posiciones[i] + (1.0 - phi) * gbest_posicion
            
            # B. Actualización de la posición (Colapso de la función de onda)
            u = np.random.rand(n_qubits)
            # Evito log(0)
            u = np.clip(u, 1e-10, 1.0) 
            
            # L_i es la longitud del pozo de potencial
            L = alpha * np.abs(mbest - posiciones[i])
            
            # La partícula "salta" a izquierda o derecha del atractor 
            signo = np.where(np.random.rand(n_qubits) > 0.5, 1.0, -1.0)
            pos_nueva = p_atractor + signo * L * np.log(1.0 / u)
            
            # Restrinjo la nueva posición al espacio de búsqueda válido [0, 1]
            posiciones[i] = np.clip(pos_nueva, 0.0, 1.0)
            
            # C. Evaluación y actualización de memorias
            fit, bs = evaluar_particula(posiciones[i])
            evaluaciones_totales += 1
            
            if fit > pbest_fitness[i]:
                pbest_fitness[i] = fit
                pbest_posiciones[i] = np.copy(posiciones[i])
                
                if fit > gbest_fitness:
                    gbest_fitness = fit
                    gbest_posicion = np.copy(posiciones[i])
                    mejor_bitstring_global = bs

    tiempo_total = time.time() - start_time

    # Como es un algoritmo de simulación clásica inspirado en la cuántica,
    # el hardware real no ejecuta CNOTs lógicas.
    return mejor_bitstring_global, gbest_fitness, tiempo_total, 0, 0, evaluaciones_totales

if __name__ == "__main__":
    print("Ejecutando QPSO...")
    print(ejecutar_qpso())