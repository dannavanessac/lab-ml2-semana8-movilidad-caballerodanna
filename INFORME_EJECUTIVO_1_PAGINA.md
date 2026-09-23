# Informe Ejecutivo · Comité de Movilidad
**Análisis Espacial de Recogidas Históricas (NYC 2014) Mediante Aprendizaje No Supervisado**  
*Autora: Danna Vanessa Caballero Urrego · Asignatura: Machine Learning II*

---

### 1. Planteamiento y Contrato Experimental
El objetivo fue identificar zonas candidatas para priorizar futuros estudios operativos de transporte a partir de una muestra de $6\,000$ recogidas vehiculares (enero–septiembre de 2014), evaluando su transferencia temporal en $7\,311$ registros reservados (octubre–diciembre de 2014).  
Para preservar la geometría euclidiana sin distorsionar distancias reales, las coordenadas se proyectaron a UTM 18N expresadas en kilómetros relativos ($x_{\text{km}}, y_{\text{km}}$) respecto a un origen común, evitando escalados independientes. La unidad de análisis es el **evento de recogida registrado**; no representa la demanda agregada de la ciudad, los viajes insatisfechos ni rutas viales.

---

### 2. Evidencia Triangulada por Método
Ningún algoritmo describe la realidad por sí solo; combinamos tres perspectivas complementarias:
1. **GMM (Mezclas Gaussianas):** Con covarianza completa (`full`) y selección por BIC ($45\,282$), el modelo óptimo eligió el límite explorado ($K=8$). Esto no prueba que existan ocho barrios reales, sino que la distribución urbana exige múltiples elipses orientadas en dirección suroeste–noreste para aproximar la asimetría del corredor central. Log-densidad en test: $-3.724$.
2. **KDE (Densidad Continua):** Seleccionado mediante validación temporal expansiva (3 folds) con un ancho de banda óptimo $h = 0.4\text{ km}$ (400 m, escala caminable de 4 cuadras). La superficie continua reveló que el **$88.6\%$ de la probabilidad espacial** se concentra en el rectángulo central $[-3, 3] \times [-4, 4]\text{ km}$. Log-densidad en test: $-3.786$.
3. **DBSCAN (Conectividad por Grafos):** Con $\varepsilon = 0.4\text{ km}$ y $m = 15$, recuperó la forma no convexa de Manhattan como un único megagrupo conexo de $5\,457$ puntos y separó a los aeropuertos como componentes satélites, identificando un $5.1\%$ de observaciones periféricas como ruido (densidad local insuficiente, no datos erróneos).

---

### 3. Zonas Propuestas y Ubicación Crítica
* **Zona Candidata 1 (Estudio Operativo Prioritario): Corredor Midtown–Downtown Manhattan**  
  *Coordenadas:* $x \in [-1.5, 1.5]\text{ km}, y \in [0.0, 4.0]\text{ km}$.  
  *Sustento:* Concentra el pico absoluto de densidad en KDE ($\widehat{f} > 1.4\text{ km}^{-2}$), el mayor peso acumulado en GMM y el núcleo hiperconectado en DBSCAN. Es el área natural para aforos viales e infraestructura de parada.
* **Zona Candidata 2 (Estudio Operativo Secundario): Polo Intermodal LaGuardia**  
  *Coordenadas:* $x \in [7.5, 10.0]\text{ km}, y \in [3.0, 5.5]\text{ km}$.  
  *Sustento:* Aparece como una cresta de densidad compacta y autónoma en KDE, una componente normal sin solapamiento en GMM y un clúster disyunto de $113$ puntos en DBSCAN. Ideal para evaluar servicios dedicados punto a punto.
* **Ubicación para Revisión: Riberas del Río Hudson / Cruce GWB**  
  *Coordenadas:* $x \in [-3.0, -1.5]\text{ km}, y \in [5.0, 7.5]\text{ km}$.  
  *Sustento:* Conflicto metodológico: KDE y GMM suavizan la densidad derramando masa sobre el río (barrera física), mientras DBSCAN rompe la conectividad al no encontrar núcleos en el agua. Requiere incorporar máscaras geográficas que anulen la probabilidad sobre cuerpos hídricos.

---

### 4. Dictamen Estructurado del Comité
> **Observo** una fuerte concentración espacial estructurada en un corredor continuo hiperdenso en Manhattan ($[-1.5, 1.5] \times [0, 4]\text{ km}$) y dos núcleos satélites en los aeropuertos LGA y JFK, con un $5.1\%$ de recogidas periféricas catalogadas como ruido en DBSCAN ($\varepsilon=0.4\text{ km}, m=15$).  
> **Interpreto** que la demanda responde a una geometría no convexa continua articulada con polos intermodales, que no puede modelarse fielmente mediante particiones rígidas esféricas (K-means) ni con elipses que ignoran barreras geográficas.  
> **Recomendaría estudiar** operativamente el corredor de Midtown Manhattan ($[-1.5, 1.5] \times [0, 4]\text{ km}$) y la terminal de LaGuardia ($[7.5, 10] \times [3, 5.5]\text{ km}$) para aforos en campo y diseño de estaciones.  
> **Vigilaría** las riberas del Río Hudson ($x \in [-3, -1.5], y \in [5, 7.5]\text{ km}$), donde el suavizado de los modelos continuos asigna probabilidad artificial sobre el agua.  
> **No puedo concluir** sobre la ubicación exacta de estaciones, tamaño de flota ni demanda insatisfecha sin integrar matrices de origen-destino, tiempos de viaje en congestión y costos operativos de servicio.
