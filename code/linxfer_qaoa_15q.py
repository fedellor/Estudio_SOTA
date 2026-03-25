"""
TFG: Optimizacion Cuantica de Hiperparametros (HPO)
Archivo 23: LINXFER-QAOA (Linear Parameter Transfer)

Referencias implementadas:
1. Sakai et al. (2025): "Transferring linearly fixed QAOA angles..."
2. Matsuyama & Yamashiro (2025): "Sampling-based Quantum Optimization..."
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

def ejecutar_linxfer_qaoa():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    # Cargo mi dataset de hiperparametros
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    N = 2**n_qubits
    
    # Aumento mi profundidad a 4 para evaluar la resiliencia del modelo
    p_capas = 4 
    
    # Normalizo mis costes
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    costes = -(acc_vals - min_acc) / (max_acc - min_acc)
    tensor_shape = (2,) * n_qubits

    def evaluar_qaoa_lineal(params):
        # Desempaqueto mis 4 parametros maestros (pendientes e intersecciones)
        c_g, d_g, c_b, d_b = params
        
        # Construyo mis angulos siguiendo la funcion lineal para todas las capas
        gammas = [c_g * i + d_g for i in range(1, p_capas + 1)]
        betas = [c_b * i + d_b for i in range(1, p_capas + 1)]
        
        # Inicializo mi superposicion uniforme
        psi = np.ones(N, dtype=np.complex128) / np.sqrt(N)
        for p in range(p_capas):
            psi *= np.exp(-1j * gammas[p] * costes)
            psi_reshaped = psi.reshape(tensor_shape)
            cos_b, isin_b = np.cos(betas[p]), -1j * np.sin(betas[p])
            for q in range(n_qubits):
                s0, s1 = [slice(None)] * n_qubits, [slice(None)] * n_qubits
                s0[q], s1[q] = 0, 1
                p0, p1 = psi_reshaped[tuple(s0)], psi_reshaped[tuple(s1)]
                psi_reshaped[tuple(s0)] = cos_b * p0 + isin_b * p1
                psi_reshaped[tuple(s1)] = isin_b * p0 + cos_b * p1
            psi = psi_reshaped.flatten()
            
        probabilidades = np.abs(psi)**2
        return np.sum(probabilidades * costes)

    start_time = time.time()
    
    # Establezco mi punto de partida basandome en la teoria de Quantum Annealing
    # (El optimizador inyectara su propia semilla np.random antes de esta ejecucion si hace falta ruido)
    params_iniciales = np.array([0.1, 0.0, -0.1, 1.0]) 
    
    # Utilizo L-BFGS-B para un ajuste fino rapido sobre los 4 parametros
    res = minimize(evaluar_qaoa_lineal, params_iniciales, method='L-BFGS-B', options={'maxiter': 100})
    
    c_g_opt, d_g_opt, c_b_opt, d_b_opt = res.x
    gammas_opt = [c_g_opt * i + d_g_opt for i in range(1, p_capas + 1)]
    betas_opt = [c_b_opt * i + d_b_opt for i in range(1, p_capas + 1)]
    
    # Reconstruyo mi estado final optimo
    psi_final = np.ones(N, dtype=np.complex128) / np.sqrt(N)
    for p in range(p_capas):
        psi_final *= np.exp(-1j * gammas_opt[p] * costes)
        psi_reshaped = psi_final.reshape(tensor_shape)
        cos_b, isin_b = np.cos(betas_opt[p]), -1j * np.sin(betas_opt[p])
        for q in range(n_qubits):
            s0, s1 = [slice(None)] * n_qubits, [slice(None)] * n_qubits
            s0[q], s1[q] = 0, 1
            p0, p1 = psi_reshaped[tuple(s0)], psi_reshaped[tuple(s1)]
            psi_reshaped[tuple(s0)] = cos_b * p0 + isin_b * p1
            psi_reshaped[tuple(s1)] = isin_b * p0 + cos_b * p1
        psi_final = psi_reshaped.flatten()

    idx_max = np.argmax(np.abs(psi_final)**2)
    estado_bin = format(idx_max, f'0{n_qubits}b')
    precision_surrogada = resultados.get(estado_bin, 0)
    
    tiempo_total = time.time() - start_time
    
    # Metricas de hardware estandar para QAOA p=4 sobre grafo completo
    cnot_estimadas = p_capas * 22698
    profundidad_estimada = p_capas * (n_qubits * 2) 
    evals_totales = res.nfev

    # Retorno exacto en el formato exigido por runner_experimentos.py
    return estado_bin, precision_surrogada, tiempo_total, cnot_estimadas, profundidad_estimada, evals_totales

if __name__ == "__main__":
    print("Este archivo debe ser llamado desde runner_experimentos.py")