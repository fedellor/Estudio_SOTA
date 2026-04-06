"""
TFG: Optimizacion Cuantica de Hiperparametros (HPO)
Experimento Baseline Clásico: Búsqueda Exhaustiva del Surrogate (Max)
"""
import json
import time
import random
import os
import sys
import numpy as np
import torch

# Ajusto las rutas para importar tus funciones reales
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from benchmark_subrogado import evaluar_transformer_real, bitstring_a_hiperparametros

def borrar_modelo_cache():
    """Fuerzo el entrenamiento desde cero eliminando los pesos guardados del Transformer."""
    rutas_posibles = [
        os.path.join(os.getcwd(), "data", "models", "env_permit_fold0_event_predictor.pth"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "models", "env_permit_fold0_event_predictor.pth")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "models", "env_permit_fold0_event_predictor.pth"))
    ]
    for ruta_modelo in rutas_posibles:
        ruta_modelo = ruta_modelo.replace("//", os.sep) 
        if os.path.exists(ruta_modelo):
            try:
                os.remove(ruta_modelo)
            except Exception as e:
                pass

def ejecutar_max_exhaustivo(n_runs=5):
    print("\n" + "="*65)
    print(" EXPERIMENTO BASELINE: Búsqueda Exhaustiva Clásica (max)")
    print("="*65)

    # 1. BÚSQUEDA EXHAUSTIVA CLÁSICA (Haciendo "trampa" leyendo todo de golpe)
    start_search = time.time()
    
    with open('datos_hpo_15q.json', 'r') as f:
        datos = json.load(f)
    resultados = datos["resultados_precision"]
    
    # Python encuentra el máximo en milisegundos
    best_bitstring = max(resultados, key=resultados.get)
    surrogate_acc = resultados[best_bitstring]
    
    tiempo_busqueda = time.time() - start_search

    print(f"Búsqueda exhaustiva completada en {tiempo_busqueda:.6f} segundos.")
    print(f"Mejor Bitstring predicho por RF : {best_bitstring}")
    print(f"Precisión Predicha (Surrogate)  : {surrogate_acc:.2f}%\n")

    # 2. VALIDACIÓN REAL (Entrenando el Transformer 5 veces)
    print(">> Fase 2: Entrenando Transformer Real...")
    arr_real = []
    arr_tiempo = []

    for i in range(n_runs):
        print(f"--- [ Run {i+1}/{n_runs} ] ---")
        # Control estricto de semillas para que sea comparable con tu runner
        seed = 42 + i
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        borrar_modelo_cache()
        
        params = bitstring_a_hiperparametros(best_bitstring)
        
        start_real = time.time()
        real_acc = evaluar_transformer_real(params)
        t_real = time.time() - start_real
        
        arr_real.append(real_acc)
        arr_tiempo.append(t_real)
        
        print(f"    Resultado Real: {real_acc:.2f}% | Tiempo: {t_real:.2f}s")

    # 3. RESULTADOS ESTADÍSTICOS
    gap_medio = np.mean(arr_real) - surrogate_acc
    
    print("\n" + "="*65)
    print(" RESUMEN ESTADÍSTICO BASELINE MAX() ")
    print("="*65)
    print(f"Precisión Surrogate (La Promesa) : {surrogate_acc:.2f}%")
    print(f"Precisión REAL (La Verdad)       : {np.mean(arr_real):.2f}% ± {np.std(arr_real):.2f}%")
    print(f"Gap Medio                        : {gap_medio:.2f}%")
    print(f"Tiempo de Búsqueda (Python)      : {tiempo_busqueda:.6f} s")
    print(f"T. Medio Entrenamiento PyTorch   : {np.mean(arr_tiempo):.2f} s")
    print("="*65 + "\n")

if __name__ == "__main__":
    ejecutar_max_exhaustivo()