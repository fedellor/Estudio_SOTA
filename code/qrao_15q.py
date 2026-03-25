"""
TFG: Optimizacion Cuantica de Hiperparametros (HPO)
Archivo: qaoa_qrao_15q.py (Non-Variational QAOA-QRAO)

Implemento una aproximación no-variacional para optimizar 15 variables 
usando solo 5 qubits mediante codificación QRAC (3,1).

Referencias: He et al. (2025), Sharma & Lau (2025), Ngo & Nguyen (2024), Teramoto et al. (2023).
"""

import json
import os
import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Pauli

def ejecutar_qaoa_qrao():
    # Cargo la ruta del dataset de hiperparámetros
    ruta_json = os.path.join(os.path.dirname(__file__), 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_variables = 15
    n_qubits = n_variables // 3  # (3,1)-QRAC Encoding -> 5 qubits
    p_capas = 2
    
    # --- AUDITORÍA DE HARDWARE ---
    # Construyo un circuito representativo para medir el coste físico real
    qc_auditoria = QuantumCircuit(n_qubits)
    for p in range(p_capas):
        for q in range(n_qubits - 1):
            qc_auditoria.rzx(1.0, q, q + 1)
            qc_auditoria.rzz(1.0, q, q + 1)
        qc_auditoria.rx(1.0, range(n_qubits))
        
    qc_trans = transpile(qc_auditoria, basis_gates=['u', 'cx'], optimization_level=3)
    cx_finales = qc_trans.count_ops().get('cx', 0)
    profundidad = qc_trans.depth()
    
    # 1. Decodificador Magic State Rounding (Sharma & Lau, 2025)
    def decodificar_qrac(sv):
        bits_recuperados = []
        for q in range(n_qubits):
            # Extraigo la información de 3 bits clásicos de 1 solo qubit usando ejes X, Y, Z
            for eje in ['X', 'Y', 'Z']:
                pauli_str = ['I'] * n_qubits
                pauli_str[n_qubits - 1 - q] = eje
                exp_val = sv.expectation_value(Pauli("".join(pauli_str))).real
                # Si el valor esperado es negativo, asumo bit 1
                bits_recuperados.append('1' if exp_val < 0 else '0')
        return "".join(bits_recuperados)

    # 2. Alternating Operator Ansatz (He et al., 2025)
    def simular_qaoa_qrao(gamma, beta):
        # Inicio mi estado en superposición en el eje X
        sv = Statevector.from_label('+' * n_qubits)
        
        for p in range(p_capas):
            # Operador de Coste Relajado (Hamiltoniano QRAO)
            qc_coste = QuantumCircuit(n_qubits)
            for q in range(n_qubits - 1):
                qc_coste.rzx(gamma, q, q + 1)
                qc_coste.rzz(gamma, q, q + 1)
            sv = sv.evolve(qc_coste)
            
            # Operador Mezclador (Rotaciones Rx globales)
            qc_mixer = QuantumCircuit(n_qubits)
            qc_mixer.rx(2 * beta, range(n_qubits))
            sv = sv.evolve(qc_mixer)
            
        return sv

    # 3. EXPLORACIÓN NO-VARIACIONAL (Grid-Search)
    start_time = time.time()
    mejor_estado = None
    mejor_precision_surr = -1
    
    evals_reales = 0
    # Escaneo mi cuadrícula de parámetros para saltarme el optimizador clásico
    pasos = 20
    for gamma in np.linspace(-np.pi, np.pi, pasos):
        for beta in np.linspace(-np.pi, np.pi, pasos):
            evals_reales += 1
            sv_actual = simular_qaoa_qrao(gamma, beta)
            bitstring = decodificar_qrac(sv_actual)
            precision = resultados.get(bitstring, 0)
            
            if precision > mejor_precision_surr:
                mejor_precision_surr = precision
                mejor_estado = bitstring

    tiempo_total = time.time() - start_time

    # Devuelvo los 6 valores clave al orquestador runner_experimentos.py
    return mejor_estado, mejor_precision_surr, tiempo_total, cx_finales, profundidad, evals_reales

if __name__ == "__main__":
    print("Iniciando ejecución individual de QAOA-QRAO...")
    resultado = ejecutar_qaoa_qrao()
    print(f"Mejor bitstring hallado: {resultado[0]} con {resultado[1]}% de precisión.")