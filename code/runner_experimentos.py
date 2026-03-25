from math import e
import sys
import os
import time
import random
import numpy as np
import torch

# Me aseguro de que Python encuentre los módulos en la carpeta actual
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importo la evaluación real y el algoritmo
from benchmark_subrogado import evaluar_transformer_real, bitstring_a_hiperparametros
from qk_lstm import ejecutar_qk_lstm_efectivo

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
                print(f"     [!] No pude borrar el modelo en {ruta_modelo}: {e}")

def ejecutar_pipeline_generico(nombre_algoritmo, funcion_algoritmo, n_runs=5):
    print("\n" + "="*65)
    print(f" EXPERIMENTO SOTA: {nombre_algoritmo} (N={n_runs} RUNS)")
    print("="*65)

    resultados_totales = []

    for i in range(n_runs):
        print(f"\n--- [ Run {i+1}/{n_runs} ] ---")
        
        # Fijo la semilla iterativa para asegurar caminos evolutivos distintos
        seed = 42 + i
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Búsqueda Cuántica
        print("  >> Fase 1: Búsqueda Cuántica (Surrogate)...")
        best_bitstring, surrogate_acc, t_cuantico, cnots, depth, evals_q = funcion_algoritmo()
        print(f"     Solución: {best_bitstring} | Precisión: {surrogate_acc:.2f}% | CNOTs: {cnots} | Depth: {depth}")
        
        # Validación Real Clásica
        print("  >> Fase 2: Entrenando Transformer Real...")
        params = bitstring_a_hiperparametros(best_bitstring)
        
        borrar_modelo_cache() # Elimino pesos antiguos
        
        start_real = time.time()
        real_acc = evaluar_transformer_real(params)
        t_real = time.time() - start_real
        
        if real_acc is None or np.isnan(real_acc) or real_acc < 0:
            print("     [!] Error en validación real. Run descartado.")
            continue
            
        gap = real_acc - surrogate_acc
        coste_total_real = t_cuantico + t_real # [NUEVO] Coste GPU/CPU total
        
        print(f"     Resultado -> Real: {real_acc:.2f}% | Gap: {gap:.2f}%")
        print(f"     T. Cuántico: {t_cuantico:.2f}s | T. Real: {t_real:.2f}s | Coste Total: {coste_total_real:.2f}s")
        
        resultados_totales.append({
            "run": i + 1,
            "bitstring": best_bitstring,
            "surrogate_acc": surrogate_acc,
            "real_acc": real_acc,
            "gap": gap,
            "cnots": cnots,
            "depth": depth,
            "t_cuantico": t_cuantico,
            "t_real": t_real,
            "coste_total": coste_total_real,
            "evals_q": evals_q
        })

    # Extraigo arrays limpios para la estadística del TFG
    if not resultados_totales:
        print(f"\n[ERROR] No obtuve runs válidos para {nombre_algoritmo}.")
        return None

    arr_surr  = [r["surrogate_acc"] for r in resultados_totales]
    arr_real  = [r["real_acc"] for r in resultados_totales]
    arr_gap   = [r["gap"] for r in resultados_totales]
    arr_cnots = [r["cnots"] for r in resultados_totales]
    arr_depth = [r["depth"] for r in resultados_totales]
    arr_t_q   = [r["t_cuantico"] for r in resultados_totales]
    arr_t_r   = [r["t_real"] for r in resultados_totales]
    arr_coste = [r["coste_total"] for r in resultados_totales]
    arr_evals = [r["evals_q"] for r in resultados_totales]

    print("\n" + "="*65)
    print(f" RESUMEN ESTADÍSTICO PARA LA TABLA: {nombre_algoritmo} ")
    print("="*65)
    print(f"Precisión Surrogate : {np.mean(arr_surr):.2f}% ± {np.std(arr_surr):.2f}%")
    print(f"Precisión REAL      : {np.mean(arr_real):.2f}% ± {np.std(arr_real):.2f}%")
    print(f"Gap Medio           : {np.mean(arr_gap):.2f}%")
    print(f"CNOTs Medias        : {np.mean(arr_cnots):.1f} puertas")
    print(f"Depth Medio         : {np.mean(arr_depth):.1f} capas")
    print(f"Evals Cuánticas     : {np.mean(arr_evals):.1f}")
    print(f"T. Búsqueda Cuánt.  : {np.mean(arr_t_q):.2f} s")
    print(f"T. Entrenam. PyTorch: {np.mean(arr_t_r):.2f} s")
    print(f"Coste Total REAL    : {np.mean(arr_coste):.2f} s")
    print("="*65 + "\n")
    
    return resultados_totales

if __name__ == "__main__":
    ejecutar_pipeline_generico("QK-LSTM", ejecutar_qk_lstm_efectivo, n_runs=5)
    #pass