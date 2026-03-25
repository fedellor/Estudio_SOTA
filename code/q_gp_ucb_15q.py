"""
TFG: Optimizacion Cuantica de Hiperparametros (HPO)
Archivo 15: Q-GP-UCB (Quantum-Gaussian Process UCB) con QAWA Encoding
Referencias: 
- Dai et al. (2023 NeurIPS): Creadores del framework Q-GP-UCB.
- Siam et al. (2025 arXiv): Analisis de dimensionalidad sobre Q-GP-UCB.
- Guo et al. (2025 arXiv): Mapeo arccos y profundidad O(n) para correlaciones.
"""
import json
import os
import sys
import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel

# Ajusto las rutas para mi entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

def ejecutar_q_gp_ucb_qawa():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    cadenas_todas = list(resultados.keys())
    
    # 1. Quantum Kernel con Mapeo SOTA (Guo et al. 2025)
    class QAWAQuantumKernel(Kernel):
        def __init__(self):
            self.cache = {}
            
        def _get_state(self, x_array):
            x_bin = "".join(str(int(b)) for b in x_array)
            if x_bin not in self.cache:
                qc = QuantumCircuit(n_qubits)
                
                # Mapeo de Amplitud arccos (Guo et al. 2025)
                for i, b_str in enumerate(x_bin):
                    b_val = float(b_str)
                    angulo = np.arccos(1 - b_val) 
                    qc.ry(angulo, i)
                    
                # Entrelazamiento Lineal O(n)
                for i in range(n_qubits - 1):
                    qc.cx(i, i+1)
                
                self.cache[x_bin] = Statevector(qc).data
            return self.cache[x_bin]

        def __call__(self, X, Y=None, eval_gradient=False):
            if Y is None: Y = X
            K = np.zeros((len(X), len(Y)))
            for i, x in enumerate(X):
                st_x = self._get_state(x)
                for j, y in enumerate(Y):
                    st_y = self._get_state(y)
                    # El kernel es la fidelidad (overlap) en el espacio de Hilbert
                    K[i, j] = np.abs(np.dot(st_x.conj(), st_y))**2
            if eval_gradient:
                return K, np.zeros((len(X), len(Y), 0))
            return K
        
        def diag(self, X):
            return np.ones(len(X))
            
        def is_stationary(self):
            return False

    # 2. Métricas Físicas Exactas
    # Genero un circuito de ejemplo para medir profundidad y CX reales
    qc_test = QuantumCircuit(n_qubits)
    for i in range(n_qubits): qc_test.ry(np.pi/4, i)
    for i in range(n_qubits - 1): qc_test.cx(i, i+1)
    
    qc_aud = transpile(qc_test, basis_gates=['u', 'cx'], optimization_level=3)
    cnot_totales = qc_aud.count_ops().get('cx', 0)
    profundidad = qc_aud.depth()

    # 3. Configuración del Proceso Gaussiano
    n_inicial = 50 
    # La semilla la gestiona el runner antes de llamar a la función
    x_train_bin = list(np.random.choice(cadenas_todas, n_inicial, replace=False))
    X_train = np.array([[int(b) for b in x] for x in x_train_bin])
    y_train = np.array([resultados[x] for x in x_train_bin])
    
    gpr = GaussianProcessRegressor(kernel=QAWAQuantumKernel(), alpha=0.1, normalize_y=True)
    start_time = time.time()

    # 4. BUCLE Q-GP-UCB (Optimización de Hiperparámetros)
    n_ciclos_bo = 4 
    muestras_exploracion = 500
    batch_size = 3
    
    for ciclo in range(1, n_ciclos_bo + 1):
        gpr.fit(X_train, y_train)
        
        candidatos_bin = [c for c in cadenas_todas if c not in x_train_bin]
        x_candidatos_bin = np.random.choice(candidatos_bin, muestras_exploracion, replace=False)
        X_cand = np.array([[int(b) for b in x] for x in x_candidatos_bin])
        
        mean, std = gpr.predict(X_cand, return_std=True)
        
        # Criterio UCB (Upper Confidence Bound)
        kappa = 1.96 
        ucb_scores = mean + kappa * std
        top_indices = np.argsort(ucb_scores)[-batch_size:]
        
        for idx in reversed(top_indices): 
            mejor_candidato_bin = x_candidatos_bin[idx]
            precision_real_ciclo = resultados[mejor_candidato_bin]
            
            x_train_bin.append(mejor_candidato_bin)
            X_train = np.vstack([X_train, X_cand[idx]])
            y_train = np.append(y_train, precision_real_ciclo)

    tiempo_total = time.time() - start_time
    
    mejor_historico_idx = np.argmax(y_train)
    best_bitstring = x_train_bin[mejor_historico_idx]
    precision_surrogada = y_train[mejor_historico_idx]
    evals_totales = len(X_train)

    return best_bitstring, precision_surrogada, tiempo_total, cnot_totales, profundidad, evals_totales

if __name__ == "__main__":
    print("Este archivo debe ser llamado desde runner_experimentos.py")