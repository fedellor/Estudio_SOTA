"""
TFG: Optimizacion Cuantica de Hiperparametros (HPO)
Archivo 18: CVaR-VQE (Conditional Value-at-Risk)

Referencias:
1. Wang et al. (2025): "Achieving High-Quality Portfolio Optimization with VQE".
2. Uttarkar et al. (2026): "CVaR-optimized VQE".
"""
import json
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

# Ajusto las rutas para el entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

def ejecutar_cvar_vqe():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    # Cargo mi dataset de hiperparametros
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    N = 2**n_qubits
    
    # Fijo mi umbral de CVaR en el 10% de los mejores estados
    alpha = 0.10
    
    # Extraigo y normalizo mis costes (Invierto la precision para minimizar)
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    costes = -(acc_vals - min_acc) / (max_acc - min_acc)

    # Ordeno previamente los indices de menor a mayor coste para acelerar mi calculo
    indices_ordenados = np.argsort(costes)

    # Diseno mi ansatz eficiente en hardware utilizando rotaciones Pauli Y
    def construir_ansatz(params):
        qc = QuantumCircuit(n_qubits)
        
        # Primera capa de rotaciones Pauli Y (Exploracion local)
        for q in range(n_qubits):
            qc.ry(params[q], q)
        
        # Capa de entrelazamiento lineal (14 CNOTs para propagar correlaciones)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
            
        # Segunda capa de rotaciones Pauli Y (Ajuste fino)
        for q in range(n_qubits):
            qc.ry(params[n_qubits + q], q)
            
        return qc

    def funcion_cvar(params):
        # Simulo mi circuito y extraigo las probabilidades directamente de las rotaciones Pauli
        qc = construir_ansatz(params)
        psi = Statevector(qc).data
        probabilidades = np.abs(psi)**2
        
        # Calculo mi CVaR: acumulo probabilidad solo de la cola de la distribucion
        prob_acumulada = 0.0
        cvar_coste = 0.0
        
        for idx in indices_ordenados:
            p_estado = probabilidades[idx]
            if prob_acumulada + p_estado < alpha:
                cvar_coste += p_estado * costes[idx]
                prob_acumulada += p_estado
            else:
                # Tomo solo la fraccion necesaria para alcanzar exactamente mi alpha
                peso_restante = alpha - prob_acumulada
                cvar_coste += peso_restante * costes[idx]
                prob_acumulada += peso_restante
                break
                
        # Retorno mi valor condicionado y normalizado
        return cvar_coste / alpha

    # Inicio mi optimizacion con COBYLA (30 parametros clasicos a optimizar)
    start_time = time.time()
    
    # El orquestador ya fija la semilla antes de llamar a la función,
    # por lo que np.random usará un estado controlado para cada run.
    parametros_iniciales = np.random.uniform(-np.pi, np.pi, n_qubits * 2)
    
    # Doy algo de margen al optimizador ya que este ansatz requiere mas ajuste parametrico
    res = minimize(funcion_cvar, parametros_iniciales, method='COBYLA', options={'maxiter': 200})
    
    # Reconstruyo mi estado final optimizado
    qc_final = construir_ansatz(res.x)
    psi_final = Statevector(qc_final).data
    
    idx_max = np.argmax(np.abs(psi_final)**2)
    best_bitstring = format(idx_max, f'0{n_qubits}b')
    precision_surrogada = resultados.get(best_bitstring, 0)
    tiempo_total = time.time() - start_time
    
    # Transpilo para obtener las métricas físicas reales
    qc_aud = transpile(qc_final, basis_gates=['u', 'cx'], optimization_level=3)
    cx_finales = qc_aud.count_ops().get('cx', 0)
    profundidad = qc_aud.depth()

    # Retorno en el formato estricto: 
    # (bitstring, precisión_subrogada, tiempo_cuántico, cnots, profundidad, evaluaciones)
    return best_bitstring, precision_surrogada, tiempo_total, cx_finales, profundidad, res.nfev

if __name__ == "__main__":
    print("Este archivo debe ser llamado desde runner_experimentos.py")