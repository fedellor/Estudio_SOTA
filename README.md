# Optimización Cuántica de Hiperparámetros (HPO) - Estado del Arte (SOTA) ⚛️🧠

Este repositorio contiene el código fuente, los datos empíricos y la bibliografía de mi Trabajo de Fin de Grado (TFG) en Inteligencia Artificial. El objetivo principal de este proyecto es realizar una revisión exhaustiva y experimental del estado del arte (SOTA) en la **Optimización Cuántica de Hiperparámetros (HPO)** para modelos de Machine Learning. 

Para este estudio, **he implementado y evaluado 50 algoritmos** diferentes (tanto puramente cuánticos como *quantum-inspired*), contrastando su rendimiento empírico frente a *baselines* clásicos de alta solvencia como la Optimización Bayesiana y los modelos Random Forest.

## 🗂 Estructura del Repositorio

He diseñado una arquitectura de software modular para garantizar la reproducibilidad y el aislamiento de cada experimento.

```text
📦 Raíz del Proyecto
 ┣ 📂 code/                    # Núcleo de la investigación algorítmica
 ┃ ┣ 📜 runner_experimentos.py # Orquestador central y benchmarking unificado
 ┃ ┣ 📜 datos_hpo_15q.json     # Dataset de evaluación subyacente (15 qubits)
 ┃ ┣ 📜 resultados_reales.json # Resultados reales para el cálculo del gap de precisión
 ┃ ┣ 📂 data/                  # Particiones del dataset (K-Folds) y modelos pre-entrenados
 ┃ ┣ 📂 data_processors/       # Scripts de extracción de atributos y logs
 ┃ ┗ 📜 [50+ scripts .py]      # Implementaciones modulares de cada algoritmo SOTA
 ┣ 📂 resultados/              # Datos empíricos extraídos de las simulaciones
 ┃ ┗ 📊 Algoritmos.xlsx        # Tabla completa con las métricas de los 50 algoritmos
 ┣ 📂 lecturas/                # Documentación de respaldo
 ┃ ┗ 📜 bibliografia.bib       # Archivo BibTeX con las 130 referencias del SOTA (2024-2026)
 ┗ 📜 README.md
```

## 📊 Resultados Destacados

El estudio evalúa el rendimiento computacional, la precisión de los hiperparámetros y la viabilidad física (era NISQ) de cada estrategia. A continuación, se muestra un extracto comparativo de los mejores enfoques cuánticos frente a los *baselines* clásicos. **Para consultar la tabla con los 50 algoritmos evaluados en detalle, revisa el archivo `resultados/Algoritmos.xlsx`.**

| Algoritmo / Estrategia | Precisión Real | Gap de Precisión | Coste Total (Tiempo) | CNOTs (Ruido Físico) | Depth (Capas) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Baseline Clásico)** | 81.73% | 0.00% | ~ 13.5 h | N/A | N/A |
| **Optimización Bayesiana (Clásica)** | 81.83% | 0.81% | 235.37 s | N/A | N/A |
| **QACO (Quantum Ant Colony)** | **81.89%** | 0.92% | 256.48 s | 0 (Simulado) | 0.0 |
| **QPSO (Quantum Swarm)** | **81.86%** | 0.62% | 187.80 s | 0 (Simulado) | 0.0 |
| **QGA (Quantum Genetic)** | 81.60% | 0.43% | **166.40 s** | 0 (Simulado) | 0.0 |

> *Nota: Se observa cómo las metaheurísticas poblacionales cuánticas (QACO, QPSO) logran superar el techo de precisión de los métodos clásicos reduciendo el tiempo de exploración de horas a apenas unos minutos, ejecutándose en hardware clásico sin sufrir la penalización de puertas CNOT impuesta por las arquitecturas cuánticas actuales.*

## 🧬 Taxonomía Algorítmica Implementada

He clasificado y programado las estrategias SOTA en **5 grandes familias** de la computación cuántica actual:

1. **Algoritmos Variacionales (VQA):** * Variantes de VQE (Pauli Z, Adaptativo, CVaR).
   * Variantes de QAOA (Mixers alternativos, Warm-Start, multiobjetivo, reducción de parámetros).
   * Optimizadores sin gradientes (FALQON, FRAXIS, Rotosolve).
2. **Búsqueda Cuántica:** * Búsqueda tipo Grover y Grover Adaptativo (GAS).
   * Caminatas Cuánticas Continuas (NV-QWOA).
   * Recocido Cuántico (Quantum Annealing).
3. **Búsqueda de Arquitecturas Cuánticas (QAS):**
   * Enfoques diferenciables y evolutivos (DQAS, SA-DQAS, EQNAS, QuantumNAS).
4. **Aprendizaje Cuántico (QML):**
   * Optimización Bayesiana Cuántica (Q-GP-UCB).
   * Aprendizaje por Refuerzo (VQC-RL).
   * Modelos Generativos (QGAN).
   * Meta-Aprendizaje Secuencial Cuántico (QK-LSTM).
5. **Metaheurísticas Poblacionales Cuánticas (*Quantum-Inspired*):**
   * Inteligencia de Enjambre (QPSO, QACO).
   * Algoritmos Evolutivos Cuánticos (QGA).

## ⚙️ Metodología y Ejecución

La metodología de evaluación sigue condiciones de contorno estrictamente idénticas para todos los algoritmos:
* **Espacio de Búsqueda:** 2^15 combinaciones (dataset de 15 qubits).
* **Métricas Extraídas:** Precisión subrogada, precisión real, gap de precisión, tiempo de búsqueda, tiempo de entrenamiento en PyTorch, número de evaluaciones del circuito, profundidad (*Depth*) y recuento de puertas CNOTs.

Para lanzar las simulaciones y reproducir la tabla comparativa del estudio, simplemente hay que posicionarse en el núcleo algorítmico y ejecutar el orquestador principal:

```bash
cd code
python runner_experimentos.py
```

*Nota: Todas las simulaciones cuánticas se ejecutan en simuladores de estado ideal (sin ruido) utilizando la librería Qiskit de IBM.*

## 📚 Bibliografía

Debido a la extensa magnitud del soporte documental necesario para esta investigación, la bibliografía no se detalla estáticamente en el código. He compilado los **130 estudios científicos base**, publicados en su mayoría entre 2024 y marzo de 2026, en el archivo estructurado `lecturas/bibliografia.bib`. Este archivo permite su importación directa a gestores como Zotero o Mendeley para auditar la trazabilidad del estudio SOTA.
