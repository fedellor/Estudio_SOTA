"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo 25: EQNAS (Evolutionary Quantum Neural Architecture Search)

Este script implementa una búsqueda de arquitectura cuántica basada en algoritmos 
evolutivos de inspiración cuántica (QEA). A diferencia de los métodos de gradiente, 
utiliza una población de individuos para explorar el espacio de circuitos.

Referencias Técnicas:
1. Li et al. (2023): Representación de arquitecturas mediante Q-bits.
2. Li et al. (2025): AQEA-QAS (Algoritmo Evolutivo Adaptativo con Catástrofe).
3. Shi et al. (2026): Optimización mediante Caching para reducir el coste computacional.
"""

import json
import os
import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

def ejecutar_eqnas():
    # Cargo la ruta del dataset generado en los primeros archivos del proyecto
    ruta_json = os.path.join(os.path.dirname(__file__), 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = 15
    p_capas = 2
    n_ops = 3 # Operaciones: 0=Identidad (nada), 1=RY, 2=CX

    # --- CONFIGURACIÓN DEL MOTOR EVOLUTIVO ---
    pop_size = 20     
    max_generaciones = 15
    cache_arquitecturas = {} 
    conteo_estancamiento = 0
    limite_catastrofe = 3  
    
    q_pop = np.full((pop_size, p_capas, n_qubits, n_ops), 1.0 / np.sqrt(n_ops))

    def medir_arquitectura(q_individuo):
        arch = np.zeros((p_capas, n_qubits), dtype=int)
        for p in range(p_capas):
            for q in range(n_qubits):
                probs = np.abs(q_individuo[p, q])**2
                probs /= np.sum(probs) 
                arch[p, q] = np.random.choice(n_ops, p=probs)
        return arch

    def calcular_fitness(arquitectura):
        arch_id = str(arquitectura.flatten())
        if arch_id in cache_arquitecturas:
            return cache_arquitecturas[arch_id]
        
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits)) 
        for p in range(p_capas):
            for q in range(n_qubits):
                op = arquitectura[p, q]
                if op == 1: qc.ry(np.pi/4, q)
                elif op == 2: qc.cx(q, (q+1)%n_qubits)
        
        sv = Statevector(qc)
        idx_max = np.argmax(np.abs(sv.data)**2)
        bitstring = format(idx_max, f'0{n_qubits}b')
        precision = resultados.get(bitstring, 0)
        
        cache_arquitecturas[arch_id] = precision
        return precision

    # --- BUCLE PRINCIPAL DE EVOLUCIÓN ---
    start_time = time.time()
    mejor_global_arch = None
    mejor_global_fit = -1
    evals_reales = 0

    for gen in range(max_generaciones):
        mejor_gen_fit = -1
        
        for i in range(pop_size):
            arch_discreta = medir_arquitectura(q_pop[i])
            fit = calcular_fitness(arch_discreta)
            evals_reales += 1
            
            if fit > mejor_gen_fit:
                mejor_gen_fit = fit
            
            if fit > mejor_global_fit:
                mejor_global_fit = fit
                mejor_global_arch = arch_discreta.copy()
                conteo_estancamiento = 0 

        if mejor_gen_fit <= mejor_global_fit:
            conteo_estancamiento += 1

        if conteo_estancamiento >= limite_catastrofe:
            # Reseteo la mitad de la población para inyectar entropía
            for i in range(pop_size // 2):
                q_pop[i] = np.full((p_capas, n_qubits, n_ops), 1.0 / np.sqrt(n_ops))
            conteo_estancamiento = 0

        for i in range(pop_size):
            for p in range(p_capas):
                for q in range(n_qubits):
                    mejor_op = mejor_global_arch[p, q]
                    delta_theta = 0.08 * np.pi * (1.0 - gen/max_generaciones)
                    q_pop[i, p, q, mejor_op] += np.sin(delta_theta)
                    q_pop[i, p, q, :] /= np.linalg.norm(q_pop[i, p, q, :])

    # --- EXTRACCIÓN Y RESULTADOS ---
    qc_final = QuantumCircuit(n_qubits)
    qc_final.h(range(n_qubits))
    for p in range(p_capas):
        for q in range(n_qubits):
            op = mejor_global_arch[p, q]
            if op == 1: qc_final.ry(np.pi/4, q)
            elif op == 2: qc_final.cx(q, (q+1)%n_qubits)
    
    # Transpilo para obtener métricas reales de hardware
    qc_aud = transpile(qc_final, basis_gates=['u', 'cx'], optimization_level=3)
    cx_finales = qc_aud.count_ops().get('cx', 0)
    profundidad = qc_aud.depth() # [NUEVO] Extraigo la profundidad real del circuito
    
    tiempo_total = time.time() - start_time
    
    sv = Statevector(qc_final)
    idx_max = np.argmax(np.abs(sv.data)**2)
    best_bitstring = format(idx_max, f'0{n_qubits}b')

    # Devuelvo los 6 datos al Orquestador
    return best_bitstring, mejor_global_fit, tiempo_total, cx_finales, profundidad, evals_reales

if __name__ == "__main__":
    print("Este archivo debe ser llamado desde runner_experimentos.py")