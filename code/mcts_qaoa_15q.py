"""
TFG: Optimizacion Cuantica de Hiperparametros (HPO)
Archivo 22: RL-QAOA (Monte Carlo Tree Search Hybrid Optimization)

Referencias:
- Yao et al. (2022): "Monte Carlo Tree Search based Hybrid Optimization of Variational Quantum Circuits"
- Patel et al. (2024): "Reinforcement learning assisted recursive QAOA"
"""
import json
import os
import sys
import time
import math
import numpy as np
from scipy.optimize import minimize

# Ajusto las rutas para mi entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

def ejecutar_mcts_qaoa():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    N = 2**n_qubits
    p_capas = 2  # QAOA de profundidad 2
    
    # Normalización para la simulación
    acc_vals = np.array(list(resultados.values()))
    max_acc, min_acc = np.max(acc_vals), np.min(acc_vals)
    costes = -(acc_vals - min_acc) / (max_acc - min_acc)
    tensor_shape = (2,) * n_qubits

    # 1. SIMULADOR VECTORIZADO ULTRARRÁPIDO
    def evaluar_qaoa(gammas, betas):
        psi = np.ones(N, dtype=np.complex128) / np.sqrt(N)
        for p in range(len(gammas)):
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

    # 2. DEFINICIÓN DEL MDP PARA EL AGENTE RL (MCTS)
    # Espacio de acciones discreto: diferentes inicializaciones heurísticas de (gamma, beta)
    espacio_acciones = [
        (0.1, 0.1), (0.5, -0.5), (-0.1, 0.8), 
        (1.0, 1.0), (-1.0, -1.0), (0.0, 0.5)
    ]

    class MCTSNode:
        def __init__(self, estado, parent=None):
            self.estado = estado  # Lista de acciones seleccionadas hasta ahora
            self.parent = parent
            self.children = []
            self.visits = 0
            self.value = 0.0
            self.acciones_no_probadas = espacio_acciones.copy()

        def ucb1(self, exploracion=1.41):
            if self.visits == 0:
                return float('inf')
            return (self.value / self.visits) + exploracion * math.sqrt(math.log(self.parent.visits) / self.visits)

    start_time = time.time()
    
    # 3. FASE 1: MONTE CARLO TREE SEARCH (Planificación)
    # El runner ya se encargó de inyectar la semilla estocástica
    raiz = MCTSNode([])
    iteraciones_mcts = 100
    evals_mcts = 0

    for _ in range(iteraciones_mcts):
        nodo = raiz
        
        # Selección
        while not nodo.acciones_no_probadas and len(nodo.estado) < p_capas:
            nodo = max(nodo.children, key=lambda c: c.ucb1())
            
        # Expansión
        if nodo.acciones_no_probadas and len(nodo.estado) < p_capas:
            accion = nodo.acciones_no_probadas.pop(np.random.randint(len(nodo.acciones_no_probadas)))
            nuevo_estado = nodo.estado + [accion]
            nuevo_nodo = MCTSNode(nuevo_estado, parent=nodo)
            nodo.children.append(nuevo_nodo)
            nodo = nuevo_nodo
            
        # Rollout (Simulación hasta el final)
        estado_rollout = nodo.estado.copy()
        while len(estado_rollout) < p_capas:
            estado_rollout.append(espacio_acciones[np.random.randint(len(espacio_acciones))])
            
        # Evaluación del entorno
        gammas_r = [a[0] for a in estado_rollout]
        betas_r = [a[1] for a in estado_rollout]
        coste = evaluar_qaoa(gammas_r, betas_r)
        evals_mcts += 1
        recompensa = -coste # Maximizar el inverso del coste
        
        # Retropropagación
        while nodo is not None:
            nodo.visits += 1
            nodo.value += recompensa
            nodo = nodo.parent

    # Selección de la mejor política discreta encontrada por el agente
    nodo_actual = raiz
    mejor_secuencia = []
    while nodo_actual.children:
        nodo_actual = max(nodo_actual.children, key=lambda c: c.visits) # Robust child
        mejor_secuencia.append(nodo_actual.estado[-1])

    # El MCTS puede haber explorado menos profundidad si p_capas es alto, rellenamos para evitar index errors
    while len(mejor_secuencia) < p_capas:
        mejor_secuencia.append((0.0, 0.0))

    gammas_optimos_rl = [a[0] for a in mejor_secuencia]
    betas_optimos_rl = [a[1] for a in mejor_secuencia]

    # 4. FASE 2: OPTIMIZACIÓN HÍBRIDA (Ajuste fino continuo)
    params_iniciales = gammas_optimos_rl + betas_optimos_rl
    
    def funcion_objetivo(params):
        g, b = params[:p_capas], params[p_capas:]
        return evaluar_qaoa(g, b)

    res = minimize(funcion_objetivo, params_iniciales, method='L-BFGS-B', options={'maxiter': 50})
    gammas_finales, betas_finales = res.x[:p_capas], res.x[p_capas:]
    
    # 5. EXTRACCIÓN DEL RESULTADO
    psi_final = np.ones(N, dtype=np.complex128) / np.sqrt(N)
    for p in range(p_capas):
        psi_final *= np.exp(-1j * gammas_finales[p] * costes)
        psi_reshaped = psi_final.reshape(tensor_shape)
        cos_b, isin_b = np.cos(betas_finales[p]), -1j * np.sin(betas_finales[p])
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
    evals_totales = evals_mcts + res.nfev
    cnot_teoricas = p_capas * 22698 # Estimación de grafo completo estándar
    profundidad_estimada = p_capas * (n_qubits * 2)

    # Retorno en el formato estricto: 
    return estado_bin, precision_surrogada, tiempo_total, cnot_teoricas, profundidad_estimada, evals_totales

if __name__ == "__main__":
    print("Este archivo debe ser llamado desde runner_experimentos.py")