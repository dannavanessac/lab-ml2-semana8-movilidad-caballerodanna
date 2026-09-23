# Laboratorio de Machine Learning — Semana 8: GMM, KDE y DBSCAN aplicados a Movilidad Urbana

[![Python](https://img.shields.io/badge/Python-3.14-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-Académico-green)]()

## Descripción del Proyecto

Este repositorio contiene el desarrollo completo del **Laboratorio Formativo — Comité de Movilidad** de la asignatura de Machine Learning II (Semana 8 — Parcial 2). El objetivo central es proponer **dos zonas candidatas para estudio operativo** y **una ubicación para revisión** a partir de un conjunto de recogidas vehiculares históricas del año 2014 en la zona metropolitana de Nueva York.

El análisis triangula tres fuentes de evidencia estadística complementarias:

| Método | Objetivo matemático | Pregunta que responde |
|---|---|---|
| **GMM** (Gaussian Mixture Models) | Densidad paramétrica con variables latentes | ¿Cuántos núcleos de concentración y con qué orientación? |
| **KDE** (Kernel Density Estimation) | Superficie continua no paramétrica | ¿Qué fracción del espacio urbano concentra la masa de recogidas? |
| **DBSCAN** | Componentes conexas por densidad local | ¿Qué zonas forman clústeres contiguos y cuáles son periféricas? |

---

## ⚠️ Advertencia Metodológica Fundamental

> **Un centroide estadístico NO es una estación operativa.** Los resultados de este análisis identifican *zonas candidatas para estudio* operativo, no ubicaciones precisas de infraestructura. Se requiere información adicional (demanda O-D, red vial, costos, regulación) que **no está contenida en este conjunto de datos** para tomar decisiones de inversión.

---

## Estructura del Repositorio

```text
LABORATORIO SEMANA 8 - PARCIAL 2/
├── README.md                              # Este archivo
├── INFORME_COMITE_MOVILIDAD.md            # Informe ejecutivo completo y defensa oral
├── requirements.txt                       # Dependencias verificadas (Python 3.14)
├── semana-08.html                         # Enunciado oficial y rúbrica de la actividad
├── .gitignore                             # Exclusiones para el repositorio Git
│
├── datasets/
│   └── semana-08/
│       ├── README.md                      # Procedencia, protocolo y reglas de los datos
│       ├── recogidas_desarrollo.csv       # 6.000 filas ene–sep 2014 (semilla 42)
│       ├── recogidas_test_temporal.csv    # 7.311 filas oct–dic 2014 (reservado)
│       ├── recogidas_jerarquico.csv       # 500 filas submuestra para refuerzo jerárquico
│       ├── auditoria.csv                  # Conteos de exclusión secuencial
│       ├── diccionario.csv                # Variables, tipos y unidades
│       └── manifest.json                  # SHA-256, semilla, dimensiones y metadatos
│
├── notebooks/
│   └── semana-08/
│       ├── movilidad.py                   # Módulo auxiliar: carga, figuras y métricas
│       ├── sesion_1_gmm.ipynb             # ✅ GMM: paso EM, BIC, transferencia temporal
│       ├── sesion_2_kde.ipynb             # ✅ KDE: kernel, ancho de banda, masa espacial
│       ├── sesion_3_dbscan.ipynb          # ✅ DBSCAN: vecindades, sensibilidad, comité
│       ├── refuerzo_kmeans.ipynb          # ✅ K-means: inercia, Lloyd, centros vs estaciones
│       └── refuerzo_jerarquico.ipynb      # ✅ Jerárquico: Ward, dendrograma, comparación
│
├── assets/
│   └── semana-08/                         # 17 figuras generadas (.png, 160 dpi)
│       ├── 01_recogidas.png               # Dispersión espacial del conjunto de desarrollo
│       ├── 02_em.png                      # Convergencia de log-verosimilitud en EM
│       ├── 03_geometria.png               # Comparación K-means vs GMM en experimento controlado
│       ├── 04_bic.png                     # ★ Selección de K por BIC (full vs. diag)
│       ├── 05_gmm_espacial.png            # ★ Componentes, elipses y entropía de pertenencia
│       ├── 06_bandwidth_1d.png            # Efecto del ancho de banda h en 1D
│       ├── 07_kde_cv.png                  # Validación temporal de KDE por folds expansivos
│       ├── 08_kde_espacial.png            # ★ Superficie continua de densidad (3 anchos de banda)
│       ├── 09_kernels.png                 # Gaussian vs. Epanechnikov
│       ├── 10_dbscan_manual.png           # Vecindades manuales en ejemplo de 5 puntos 1D
│       ├── 11_dbscan_geometrias.png       # Formas no convexas y densidades disímiles
│       ├── 12_vecinos.png                 # Curva de distancia al k-ésimo vecino
│       ├── 13_dbscan_espacial.png         # ★ Conectividad y ruido a 3 valores de epsilon
│       ├── 14_kmeans.png                  # Inercia y Silhouette vs. K en K-means
│       ├── 15_kmeans_espacial.png         # Partición espacial con 4 centros
│       ├── 16_dendrogramas.png            # Dendrogramas (simple, complete, average, Ward)
│       └── 17_ward_espacial.png           # Partición espacial Ward a 2, 4 y 8 grupos
│
└── resultados/
    └── semana-08/                         # 8 tablas CSV de métricas y selección
        ├── gmm_seleccion.csv              # BIC, AIC, convergencia por (K, tipo covarianza)
        ├── gmm_test_temporal.csv          # Log-densidad mensual en conjunto de test
        ├── kde_folds.csv                  # Scores por fold y ancho de banda
        ├── kde_seleccion.csv              # Media y desviación del CV por h
        ├── dbscan_sensibilidad.csv        # ★ Grupos, ruido y Silhouette por (eps, m)
        ├── densidad_test_comparacion.csv  # Comparación GMM vs. KDE en test temporal
        ├── kmeans_referencia.csv          # Inercia y Silhouette K-means para K=1..8
        └── jerarquico_cortes.csv          # Silhouette Ward para K=2, 4 y 8
```

> **★ Figuras y Tablas Clave** — Estas son las evidencias centrales para la presentación al Comité de Movilidad.

---

## Hallazgos Principales

### 1. Selección de Modelos

| Modelo | Configuración Óptima | Criterio de Selección | Resultado en Test Temporal |
|---|---|---|---|
| **GMM** | $K=8$, covarianza `full` | BIC mínimo: $45\,282.13$ | Log-densidad media: $-3.724$ |
| **KDE** | $h = 0.4\text{ km}$, kernel gaussiano | CV temporal 3-folds: $-3.897 \pm 0.061$ | Log-densidad media: $-3.786$ |
| **DBSCAN** | $\varepsilon = 0.4\text{ km}$, $m = 15$ | Argumento espacial + sensibilidad | 3 grupos, $5.1\%$ ruido, Silhouette: $0.679$ |

### 2. Zonas Candidatas Identificadas

| Zona | Coordenadas Relativas | Evidencia de Modelos | Prioridad |
|---|---|---|---|
| **Corredor Midtown–Downtown Manhattan** | $x \in [-1.5, 1.5]\text{ km}$, $y \in [0, 4]\text{ km}$ | GMM + KDE (pico máximo) + DBSCAN (núcleo $>5400$ pts) | **Alta (A)** |
| **Polo LaGuardia** | $x \in [7.5, 10]\text{ km}$, $y \in [3, 5.5]\text{ km}$ | GMM (componente aislado) + KDE (cresta satélite) + DBSCAN (clúster 113 pts) | **Media (B)** |
| **Borde Río Hudson / GWB** (revisión) | $x \in [-3, -1.5]\text{ km}$, $y \in [5, 7.5]\text{ km}$ | Conflicto KDE-GMM vs. DBSCAN sobre barrera fluvial | **Vigilancia** |

---

## Instalación y Reproducción

### Requisitos
- Python 3.14
- Todas las dependencias especificadas en `requirements.txt`

### Instrucciones

```powershell
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/lab-ml2-semana8.git
cd "lab-ml2-semana8"

# 2. Instalar dependencias
python -m pip install --user -r requirements.txt

# 3. Ejecutar los notebooks en Jupyter
jupyter notebook notebooks/semana-08/

# Orden recomendado de ejecución:
#   1. sesion_1_gmm.ipynb       (GMM y Algoritmo EM)
#   2. sesion_2_kde.ipynb       (KDE y superficie de densidad)
#   3. sesion_3_dbscan.ipynb    (DBSCAN y comité de movilidad)
#   4. refuerzo_kmeans.ipynb    (K-means como referencia)
#   5. refuerzo_jerarquico.ipynb (Clustering jerárquico Ward)
```

> **Nota importante:** Los notebooks deben ejecutarse desde dentro de la carpeta `notebooks/semana-08/` o desde la raíz del repositorio. El módulo `movilidad.py` localiza automáticamente la carpeta `datasets/semana-08/` recorriendo los directorios padre.

### Reproducción de Figuras y Tablas

Al ejecutar cada notebook de inicio a fin, se regeneran automáticamente:
- Las **17 figuras** en `assets/semana-08/` (formato PNG, 160 dpi).
- Las **8 tablas** de resultados en `resultados/semana-08/` (formato CSV).

---

## Contrato Experimental

| Decisión | Acuerdo |
|---|---|
| Unidad de análisis | Un evento de recogida vehicular registrado (2014) |
| Variables espaciales | $x_{\text{km}}$ (este relativo) e $y_{\text{km}}$ (norte relativo) en kilómetros UTM |
| Periodo de desarrollo | Enero–septiembre 2014 ($n = 6\,000$) |
| Test temporal reservado | Octubre–diciembre 2014 ($n = 7\,311$) — **Abierto una sola vez al final** |
| Limpieza vs. recorte espacial | Limpieza = errores físicos de GPS. Recorte = decisión metodológica de área de estudio |
| Qué observamos | Recogidas registradas históricamente — **NO** demanda total de la ciudad |
| Información faltante | Matriz O-D, tiempos de viaje, congestión, costos operativos y flota |

---

## Advertencias y Limitaciones del Estudio

1. **Los datos no atribuyen origen al proveedor de transporte.** No se afirma que los datos provengan de ninguna empresa en particular.
2. **La muestra no captura la demanda insatisfecha.** Solo se observan viajes registrados, no la demanda latente o rechazada.
3. **Los centroides matemáticos NO son estaciones operativas.** Pueden caer en cuerpos de agua, parques o zonas privadas.
4. **KDE y GMM proyectan densidad sobre barreras físicas** (ríos, costas). Se requieren máscaras hidrológicas para uso operativo.
5. **La dependencia espacial rompe el supuesto i.i.d.** del BIC: los viajes en zonas contiguas están correlacionados espacialmente.
6. **DBSCAN es transductivo.** No produce una función de densidad ni puede evaluar nuevas observaciones fuera de muestra.

---

## Referencias

- Microcurrículo y guía docente ML2, sesiones 19–24.
- [scikit-learn: Gaussian Mixture Models](https://scikit-learn.org/stable/modules/mixture.html)
- [scikit-learn: Kernel Density Estimation](https://scikit-learn.org/stable/modules/density.html)
- [scikit-learn: Clustering (DBSCAN, K-means)](https://scikit-learn.org/stable/modules/clustering.html)
- [SciPy: Hierarchical Clustering (linkage, dendrogram)](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
- Dataset suministrado por el docente. Protocolo y huellas criptográficas en `datasets/semana-08/manifest.json`.
