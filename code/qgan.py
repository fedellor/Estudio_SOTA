"""
TFG: Optimización Cuántica de Hiperparámetros (HPO)
Archivo 46: QGAN (Hybrid Quantum Generative Adversarial Network)

Implemento una arquitectura QGAN híbrida. Un circuito cuántico parametrizado 
(Generador) aprende a imitar la distribución de probabilidad de las mejores 
configuraciones de hiperparámetros, intentando engañar a un clasificador 
clásico (Discriminador).

Referencias implementadas y analizadas:
1. "Designing Effective Quantum Generators: A Comparative Study of Variational 
   Ansätze in Hybrid QGANs" (Arquitectura híbrida, diseño del ansatz y entrenamiento).
2. "Entanglement-assisted Hamiltonian dynamics learning" (El rol del entrelazamiento 
   en la capacidad expresiva y aprendizaje de distribuciones complejas).
"""

import json
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Statevector

# Ajusto las rutas para mi entorno local
ruta_script = os.path.dirname(os.path.abspath(__file__))
if ruta_script not in sys.path:
    sys.path.append(ruta_script)

def ejecutar_qgan():
    ruta_json = os.path.join(ruta_script, 'datos_hpo_15q.json')
    
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    resultados = datos["resultados_precision"]
    n_qubits = datos.get("qubits", 15)
    
    start_time = time.time()
    evaluaciones_totales = 0
    
    # 1. Definición de los "Datos Reales" (True Distribution)
    # El QGAN debe aprender a generar configuraciones parecidas al Top 5% de tu dataset
    estados_ordenados = sorted(resultados.keys(), key=lambda x: resultados[x], reverse=True)
    top_k = int(32768 * 0.05)
    datos_reales_bs = estados_ordenados[:top_k]
    
    def bitstring_a_array(bs):
        return np.array([int(b) for b in bs])
        
    X_real = np.array([bitstring_a_array(bs) for bs in datos_reales_bs])
    y_real = np.ones(top_k) # Etiqueta 1: "Real"

    # 2. Configuración del Generador Cuántico (Ansatz Entrelazado)
    # Utilizo EfficientSU2 para maximizar la expresividad con entrelazamiento lineal
    #
    p_capas = 1
    ansatz_gen = EfficientSU2(num_qubits=n_qubits, su2_gates=['ry', 'rz'], entanglement='linear', reps=p_capas)
    num_params = ansatz_gen.num_parameters
    
    # Auditoría de hardware
    qc_aud = transpile(ansatz_gen, basis_gates=['u', 'cx'], optimization_level=1)
    cnot_totales = qc_aud.count_ops().get('cx', 0)
    profundidad = qc_aud.depth()

    # 3. Configuración del Entrenamiento Antagónico
    epocas = 15
    parametros_gen = np.random.uniform(-np.pi, np.pi, num_params)
    discriminador = LogisticRegression(solver='liblinear') # Clasificador clásico rápido
    
    print("Iniciando entrenamiento híbrido QGAN (Generador Cuántico vs. Discriminador Clásico)...")

    for epoca in range(epocas):
        # A. FASE DEL DISCRIMINADOR
        # 1. Genero muestras "Fake" desde el Generador Cuántico
        qc_bound = ansatz_gen.assign_parameters(parametros_gen)
        probabilidades = np.abs(Statevector(qc_bound).data)**2
        
        # Muestreo el circuito para obtener N configuraciones generadas
        indices_fake = np.random.choice(2**n_qubits, size=top_k, p=probabilidades)
        datos_fake_bs = [format(idx, f'0{n_qubits}b') for idx in indices_fake]
        
        X_fake = np.array([bitstring_a_array(bs) for bs in datos_fake_bs])
        y_fake = np.zeros(top_k) # Etiqueta 0: "Fake"
        
        # 2. Entreno el Discriminador Clásico para que distinga Real vs Fake
        X_train = np.vstack((X_real, X_fake))
        y_train = np.hstack((y_real, y_fake))
        discriminador.fit(X_train, y_train)
        
        # B. FASE DEL GENERADOR CUÁNTICO
        # Defino la función de pérdida del Generador: maximizar la probabilidad 
        # de que el Discriminador clasifique sus muestras como "Reales" (1).
        def loss_generador(params):
            qc_temp = ansatz_gen.assign_parameters(params)
            probs_temp = np.abs(Statevector(qc_temp).data)**2
            
            # Tomo las muestras más probables que el generador produciría ahora
            top_indices_gen = np.argsort(probs_temp)[-50:]
            X_gen_eval = np.array([bitstring_a_array(format(idx, f'0{n_qubits}b')) for idx in top_indices_gen])
            
            # El discriminador devuelve la probabilidad de la clase 1 (Real)
            probs_discriminador = discriminador.predict_proba(X_gen_eval)[:, 1]
            
            # El objetivo es engañar al discriminador, así que minimizamos el log-loss negativo
            loss = -np.mean(np.log(probs_discriminador + 1e-10))
            return loss

        # Actualizo el Generador Cuántico usando COBYLA
        res_gen = minimize(loss_generador, parametros_gen, method='COBYLA', options={'maxiter': 20})
        parametros_gen = res_gen.x
        evaluaciones_totales += res_gen.nfev
        
    tiempo_total_q = time.time() - start_time
    
    # 4. Inferencia Final (Generación de la mejor configuración)
    qc_final = ansatz_gen.assign_parameters(parametros_gen)
    probabilidades_finales = np.abs(Statevector(qc_final).data)**2
    
    # El Generador ya está entrenado, le pido la configuración más probable
    idx_max = np.argmax(probabilidades_finales)
    best_bitstring = format(idx_max, f'0{n_qubits}b')
    precision_surrogada = resultados.get(best_bitstring, 0)

    return best_bitstring, precision_surrogada, tiempo_total_q, cnot_totales, profundidad, evaluaciones_totales

if __name__ == "__main__":
    print("Ejecutando QGAN (Quantum Generative Adversarial Network)...")
    print(ejecutar_qgan())