"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo 00: Classical Baseline - Bayesian Optimization (BO)

Implemento un optimizador bayesiano clásico basado en Procesos Gaussianos (GP) 
y la función de adquisición Expected Improvement (EI) para establecer un 
benchmark estricto de 250 evaluaciones.
"""

import json
import os
import sys
import time
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# Ajusto las rutas para mi entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

def ejecutar_bo_baseline():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    cadenas_todas = list(resultados.keys())
    
    # 1. Configuración del Presupuesto (Baseline)
    presupuesto_total = 250
    puntos_iniciales = 20 # Muestreo aleatorio inicial para calentar el GP
    
    start_time = time.time()
    
    # Función para convertir bitstrings a arrays de características
    def bitstring_a_array(bs):
        return np.array([int(b) for b in bs])
        
    # Inicializo el conjunto de datos observados
    x_evaluados = np.random.choice(cadenas_todas, puntos_iniciales, replace=False).tolist()
    y_evaluados = [resultados[x] for x in x_evaluados]
    
    # Kernel Matern: Ideal para funciones rugosas pero continuas (típico en HPO)
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, normalize_y=True)
    
    print(f"Iniciando Optimización Bayesiana Clásica (Presupuesto: {presupuesto_total} evals)...")
    
    # 2. Bucle de Optimización Bayesiana
    for i in range(puntos_iniciales, presupuesto_total):
        # Entreno el modelo subrogado con los datos actuales
        X_train = np.array([bitstring_a_array(x) for x in x_evaluados])
        Y_train = np.array(y_evaluados)
        gp.fit(X_train, Y_train)
        
        # Para acelerar la búsqueda de la función de adquisición, 
        # muestreo un pool de candidatos no evaluados
        candidatos_restantes = list(set(cadenas_todas) - set(x_evaluados))
        pool_size = min(2000, len(candidatos_restantes))
        pool_candidatos = np.random.choice(candidatos_restantes, pool_size, replace=False)
        X_pool = np.array([bitstring_a_array(c) for c in pool_candidatos])
        
        # Predicción del Proceso Gaussiano (Media y Desviación Estándar)
        mu, sigma = gp.predict(X_pool, return_std=True)
        
        # 3. Función de Adquisición: Expected Improvement (EI)
        mu_sample_opt = np.max(Y_train)
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - 0.01 # Factor de exploración (xi)
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
            
        # Selecciono el candidato que maximiza el Expected Improvement
        mejor_idx_pool = np.argmax(ei)
        siguiente_candidato = pool_candidatos[mejor_idx_pool]
        
        # Evalúo la precisión real del entorno subrogado
        siguiente_acc = resultados[siguiente_candidato]
        
        # Actualizo mis registros
        x_evaluados.append(siguiente_candidato)
        y_evaluados.append(siguiente_acc)
        
        if (i + 1) % 50 == 0:
            print(f" -> Evaluación {i + 1}/{presupuesto_total} | Mejor precisión actual: {np.max(y_evaluados):.2f}%")

    tiempo_total = time.time() - start_time
    
    # 4. Extracción de Resultados
    idx_mejor = np.argmax(y_evaluados)
    mejor_estado = x_evaluados[idx_mejor]
    mejor_precision = y_evaluados[idx_mejor]

    # Retorno en el formato estricto: 
    # (bitstring, precisión_subrogada, tiempo, cnots, profundidad, evaluaciones)
    # CNOTs y Profundidad son 0 porque es un algoritmo 100% clásico.
    return mejor_estado, mejor_precision, tiempo_total, 0, 0, presupuesto_total

if __name__ == "__main__":
    print("Ejecutando Baseline Clásico (Bayesian Optimization)...")
    print(ejecutar_bo_baseline())