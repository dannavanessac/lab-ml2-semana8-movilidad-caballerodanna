# INFORME TÉCNICO Y EJECUTIVO · COMITÉ DE MOVILIDAD
## Laboratorio Formativo y Parcial 2: Aprendizaje No Supervisado en Movilidad Urbana
**Curso:** Machine Learning II · Maestría / Especialización  
**Fecha:** Semana 8 (Septiembre de 2026)  
**Caso de Estudio:** Análisis Espacial de Recogidas Vehiculares en la Zona Metropolitana de Nueva York  
**Modelos Evaluados:** Gaussian Mixture Models (GMM), Kernel Density Estimation (KDE), DBSCAN, K-means y Clustering Jerárquico (Ward)

---

## Resumen Ejecutivo

Este informe consolida el análisis espacial riguroso realizado sobre una muestra histórica de recogidas vehiculares en la zona metropolitana de Nueva York (año 2014), en el marco de la exploración de zonas prioritarias para futuros estudios operativos de transporte. 

A partir de la triangulación de tres familias complementarias de aprendizaje no supervisado (**GMM** para mezclas probabilísticas con variables latentes, **KDE** para estimación no paramétrica de superficies continuas de densidad, y **DBSCAN** para identificación transductiva de componentes conectadas y detección de ruido), se concluye lo siguiente:

1. **Zona de Estudio 1 (Recomendada):** Corredor Central de Midtown Manhattan ($x \in [-1.5, 1.5]\text{ km}, y \in [0.0, 4.0]\text{ km}$). Presenta la máxima densidad continua en KDE ($\widehat{f} > 1.4\text{ km}^{-2}$), concentra el mayor peso en los componentes gaussianos de GMM y constituye el núcleo hiperconectado principal en DBSCAN con más de $5\,400$ observaciones vinculadas.
2. **Zona de Estudio 2 (Recomendada):** Polo Intermodal Aeropuerto LaGuardia ($x \in [7.5, 10.0]\text{ km}, y \in [3.0, 5.5]\text{ km}$). Aparece consistentemente como una concentración satélite aislada y compacta, separada de la mancha central, validada tanto por un componente gaussiano independiente en GMM como por un clúster autónomo en DBSCAN ($113$ observaciones con $\varepsilon=0.4\text{ km}, m=15$).
3. **Ubicación para Revisión Crítica:** Borde Fluvial del Río Hudson / Cruce hacia Nueva Jersey ($x \approx -2.0\text{ km}, y \in [5.0, 7.5]\text{ km}$). Esta ubicación evidencia un conflicto metodológico crítico: los modelos continuos (KDE y GMM) proyectan masa de densidad artificial sobre el espejo de agua debido a la simetría de sus kernels, mientras que DBSCAN detecta la discontinuidad física real clasificando las recogidas dispersas como ruido o separándolas de la orilla este.
4. **Advertencia Operativa Fundamental:** Ningún centroide matemático (proveniente de K-means o GMM) debe interpretarse como una estación óptima ni como una orden de despliegue de flota. Un centroide euclidiano no contempla la red de calles, las restricciones de tráfico, la viabilidad predial ni los costos del servicio.

---

## 1. Contrato Experimental

El contrato experimental formaliza los acuerdos metodológicos previos a cualquier selección o ajuste de modelos, garantizando la validez estadística y la trazabilidad del experimento:

| Decisión Metodológica | Acuerdo del Comité |
|---|---|
| **Unidad de análisis y población observada** | **Unidad:** Un evento individual de recogida de pasajeros registrado con coordenadas espaciales y marca temporal.<br>**Población observada:** Muestra histórica de $6\,000$ recogidas (enero–septiembre de 2014, semilla 42) seleccionadas de un archivo original de $200\,000$ viajes en el área metropolitana de Nueva York. **No observamos:** La demanda agregada total de transporte, los viajes insatisfechos, la flota de competidores ni trayectorias completas de vehículos. |
| **Variables espaciales y unidades** | Dos coordenadas continuas planas proyectadas mediante **UTM Huso 18N (EPSG:32618)** referenciadas a un origen fijo $(585\,000\text{ m}, 4\,510\,000\text{ m})$:<br>• $x_{\text{km}}$: Coordenada Este relativa en **kilómetros** ($(\text{UTM}_{\text{este}} - 585\,000) / 1000$).<br>• $y_{\text{km}}$: Coordenada Norte relativa en **kilómetros** ($(\text{UTM}_{\text{norte}} - 4\,510\,000) / 1000$).<br>*Regla de diseño:* No se estandarizan los ejes por separado para no deformar la métrica euclidiana física del terreno. |
| **Periodo de selección y periodo reservado** | • **Desarrollo (Selección y ajuste):** Enero a septiembre de 2014 ($6\,000$ observaciones). Utilizado para ajustar parámetros, calibrar anchos de banda y comparar criterios de complejidad.<br>• **Test Temporal Reservado:** Octubre a diciembre de 2014 ($7\,311$ observaciones). **Se mantiene cerrado** durante todo el diseño; se abre **una sola vez** al final para contrastar la generalización temporal fuera de muestra. |
| **Diferencia entre limpieza y recorte espacial** | • **Limpieza:** Exclusión técnica de inconsistencias de registro (coordenadas $(0,0)$ correspondientes a fallas de GPS con $1.89\%$, y $10$ registros con latitudes/longitudes fuera de rango físico terrestre).<br>• **Recorte espacial:** Delimitación voluntaria del rectángulo de estudio $[-74.30, -73.65] \times [40.45, 40.95]$. Un viaje fuera del recorte ($0.165\%$) es una observación legítima pero no pertenece a la pregunta operativa del área metropolitana delimitada. |
| **Qué significaría una recomendación útil** | Proponer polígonos delimitados por rangos explícitos de coordenadas relativas ($x_{\text{km}}, y_{\text{km}}$) donde la convergencia de alta densidad y conectividad espacial señale áreas idóneas para que el equipo de operaciones realice aforos vehiculares en campo, estudios de viabilidad de suelo y auditorías viales. |
| **Qué información falta para operar** | Matriz de demanda origen-destino, demanda agregada desglosada por hora/día, congestión de la red de calles, tiempos de recorrido, duración de los turnos de conductores, disponibilidad de espacio físico/puntos de recarga eléctrica y estructura de costos y tarifas. |

---

## 2. Evidencia por Método

Triangulamos los resultados de tres modelos de aprendizaje no supervisado, cada uno respondiendo a una formulación matemática distinta del problema espacial:

| Método | Objeto Estimado | Parámetros Elegidos | Evidencia Principal | Limitación Intrínseca |
|---|---|---|---|---|
| **GMM** *(Gaussian Mixture Models)* | Densidad paramétrica mediante mezcla de $K$ distribuciones normales multivariadas con pesos latentes: $f(x) = \sum_{k=1}^K \pi_k \phi(x; \mu_k, \Sigma_k)$. | $K=8$ componentes con matriz de covarianza completa (`full`), regularización $\mathrm{reg\_covar}=10^{-5}$ y 5 inicializaciones. | El BIC disminuye monótonamente hasta $K=8$ ($\mathrm{BIC}=45\,282.13$, log-densidad en test temporal: $-3.7239$). Identifica la orientación anisotrópica principal del corredor de Manhattan y la dispersión en los aeropuertos. | Asume componentes gaussianas elípticas convexas; asigna responsabilidades a todo el plano (incluso en zonas de densidad casi nula) y no reconoce barreras físicas como ríos o costas. |
| **KDE** *(Kernel Density Estimation)* | Superficie continua no paramétrica de densidad espacial de probabilidad: $\widehat{f}_h(x) = \frac{1}{n h^d} \sum_{i=1}^n K((x-x_i)/h)$. | Ancho de banda escalar $h = 0.4\text{ km}$ con kernel gaussiano, seleccionado mediante validación temporal de 3 folds expansivos. | Maximiza la log-densidad media en validación temporal ($-3.8971 \pm 0.0608$). La integral en el rectángulo central $[-3, 3] \times [-4, 4]\text{ km}$ captura una masa de probabilidad del $88.58\%$ (frente al $89.03\%$ empírico). | Tiende a suavizar excesivamente los límites naturales (asigna densidad sobre el río Hudson); el kernel uniforme o gaussiano es isótropo y no se adapta a las distintas densidades locales. |
| **DBSCAN** *(Density-Based Spatial Clustering)* | Estructura transductiva de componentes conectadas por cadenas de núcleos de densidad: $\{p_i : \|p_i - p_j\| \le \varepsilon\}$ y etiquetado de ruido. | Distancia de vecindad $\varepsilon = 0.4\text{ km}$ ($\approx 4$ cuadras), número mínimo de puntos $m = \text{min\_samples} = 15$. | Descubre $3$ grupos significativos con un $5.08\%$ de ruido (y $\sim 23\%$ con submuestras o mayor rigor). Recupera el eje completo de Manhattan como un conglomerado continuo no convexo de $5\,457$ puntos y aísla los polos aeroportuarios. | Transductivo (no genera función de verosimilitud para predecir nuevos datos continuos); un único par $(\varepsilon, m)$ sufre ante gradientes heterogéneos de densidad urbana. |

---

## 3. Respuestas a las Preguntas Metodológicas de la Actividad

### Pregunta 1: Punto de pertenencia ambigua vs. Punto de densidad baja
> *¿Son necesariamente el mismo un punto de pertenencia ambigua y un punto de densidad baja?*

**NO son necesariamente el mismo.** Corresponden a dos conceptos matemáticos rigurosamente diferentes:
- **Punto de pertenencia ambigua:** Se define por la entropía condicional de las responsabilidades posteriores en GMM: $\mathcal{H}_i = -\sum_{k=1}^K r_{ik} \log r_{ik}$. Ocurre típicamente en la zona de solapamiento o frontera intermedia entre dos componentes densas contiguas (por ejemplo, entre Midtown y Lower Manhattan, donde $r_{i1} \approx 0.50$ y $r_{i2} \approx 0.50$). En esa frontera espacial, la **densidad de recogidas puede ser altísima** (cientos de viajes observados), pero el modelo probabilístico no puede decidir con certeza a cuál de las dos gaussianas asignar la observación.
- **Punto de densidad baja:** Se define por el valor absoluto de la función de densidad estimada: $f(x) \approx 0$ (o log-densidad fuertemente negativa, e.g., $<-15$). Ocurre en la periferia rural, autopistas lejanas o zonas industriales exteriores. En estos puntos remotos, un viaje puede tener una responsabilidad neta $r_{ik} > 0.99$ si se encuentra relativamente más cerca de una componente exterior que de las demás, teniendo **pertenencia casi inequívoca y simultáneamente una densidad microscópica**.

### Pregunta 2: Región que KDE suaviza y DBSCAN conecta o separa
> *Expliquen una región que KDE suaviza y DBSCAN conecta o separa. Mantengan iguales los ejes de los mapas.*

- **Región identificada:** El cruce entre el norte de Manhattan y el sector este de Nueva Jersey a través del Río Hudson ($x \in [-3.0, -1.0]\text{ km}, y \in [5.0, 7.5]\text{ km}$, inmediaciones del George Washington Bridge).
- **Comportamiento en KDE ($h = 0.4\text{ km}$):** Dado que el kernel gaussiano es espacialmente continuo e isótropo con colas infinitas, KDE difumina la masa de puntos de Manhattan sobre el lecho del Río Hudson, generando una superficie suave que muestra una probabilidad positiva artificial (densidad $\approx 0.05 - 0.15\text{ km}^{-2}$) sobre un cuerpo de agua navegable donde es físicamente imposible abordar un taxi en tierra firme.
- **Comportamiento en DBSCAN ($\varepsilon = 0.4\text{ km}, m = 15$):** DBSCAN evalúa vecindades discretas por bola euclidiana. Como la distancia entre las orillas opuestas del río excede los $400\text{ m}$ ($\approx 800 - 1000\text{ m}$) y no existen recogidas sobre el agua que actúen como núcleos intermedios, DBSCAN **no conecta** ambas márgenes: las observaciones sobre la ribera de Nueva Jersey quedan clasificadas como ruido (etiqueta $-1$) o conformadas en micro-conglomerados locales independientes, preservando con total fidelidad la separación física impuesta por la geografía.

### Pregunta 3: Variación de hiperparámetros y sensibilidad
> *Varíen un hiperparámetro y documenten una conclusión que cambie o permanezca.*

- **Hiperparámetro variado:** Radio de vecindad $\varepsilon$ en DBSCAN ($\varepsilon \in \{0.1, 0.2, 0.4, 0.8, 1.6\}\text{ km}$) manteniendo fijo $m=15$.
- **Conclusión que CAMBIA (Fragmentación y Cobertura de Grupos):**
  - Con $\varepsilon = 0.1\text{ km}$ ($100\text{ m}$), la conectividad colapsa por hiperfragmentación: se generan $66$ micro-grupos y el **$58.03\%$ de las observaciones es etiquetado como ruido**.
  - Con $\varepsilon = 1.6\text{ km}$, el umbral es tan permisivo que colapsa la ciudad en apenas $2$ megagrupos con solo **$0.48\%$ de ruido**, fusionando Manhattan con Brooklyn, Queens y el Bronx.
- **Conclusión que PERMANECE (Invariabilidad del Eje Central y Aislamiento Aeroportuario):**
  - Para todo $\varepsilon \ge 0.4\text{ km}$, el conglomerado principal de Manhattan siempre captura más del $90\%$ de los puntos no ruidosos (conglomerado mayoritario $>5\,300$ puntos), y los polos satélites de los aeropuertos (JFK y LaGuardia) se conservan como entidades disyuntas y densas. La existencia del corredor de Manhattan como el gran atractor metropolitano es estructuralmente invariable.

### Pregunta 4: Cobertura, Número de Grupos y Silhouette en DBSCAN
> *En DBSCAN reporten cobertura y número de grupos junto a silhouette, si está definida.*

Los resultados empíricos extraídos de la tabla de sensibilidad (`resultados/semana-08/dbscan_sensibilidad.csv`) revelan la paradoja métrica de Silhouette en clustering basado en densidad:

| $\varepsilon$ (km) | $m$ (min_samples) | Grupos ($k$) | Cobertura ($1 - \text{Ruido}$) | Fracción Ruido | Silhouette Score |
|---|---|---|---|---|---|
| **0.1** | 5 | 95 | $88.80\%$ | $11.20\%$ | $-0.2986$ |
| **0.1** | 15 | 66 | $41.97\%$ | $58.03\%$ | $+0.4600$ |
| **0.1** | 30 | 8 | **$8.18\%$** | **$91.82\%$** | **$+0.8352$** (¡Máximo engañoso!) |
| **0.2** | 15 | 13 | $90.25\%$ | $9.75\%$ | $-0.2514$ |
| **0.4** | 15 | 3 | $94.92\%$ | $5.08\%$ | $+0.6794$ |
| **0.8** | 15 | 6 | $97.83\%$ | $2.17\%$ | $+0.1913$ |
| **1.6** | 15 | 2 | $99.52\%$ | $0.48\%$ | $+0.8132$ |

*Interpretación de la trampa del Silhouette:*  
Nótese que la configuración patológica $\varepsilon=0.1\text{ km}, m=30$ arroja el Silhouette más alto de la tabla ($0.8352$). Sin embargo, esta solución **descarta el $91.82\%$ de los viajes como ruido** y deja solo $8$ micro-núcleos ultracompactos con menos de $96$ puntos cada uno. Como Silhouette excluye las observaciones de ruido, reporta un valor artificialmente excelente evaluando un subconjunto ínfimo que no describe el sistema de transporte. Por ello, **Silhouette jamás debe utilizarse de forma aislada para optimizar DBSCAN**.

### Pregunta 5: Interpretación de Casos Límite
> *Si GMM elige el borde de la rejilla o DBSCAN devuelve todo ruido, interprétenlo; no oculten el resultado.*

- **GMM en el borde superior ($K=8$):** En nuestra evaluación, el BIC mínimo se localizó exactamente en $K=8$ con covarianza `full` ($\mathrm{BIC} = 45\,282.13$). Esto significa formalmente que **la penalización paramétrica no logró doblegar la ganancia de verosimilitud**, evidenciando que el proceso espacial real es más complejo y anisotrópico que una mezcla de 8 normales. Lejos de ocultarlo o forzar un mínimo artificial, interpretamos que una distribución continua de alta concentración como la mancha urbana metropolitana se beneficia de más componentes para modelar colas y asimetrías viales; esto confirma que $K=8$ es una cota operativa parsimoniosa de la rejilla, no una verdad ontológica sobre la existencia de exactamente 8 barrios.
- **DBSCAN devolviendo todo ruido ($\varepsilon=0.1\text{ km}, m \ge 30$):** Cuando DBSCAN clasifica prácticamente todo como ruido ($>90\%$), no constituye un fallo del software: demuestra que, a esa escala micrométrica ($100\text{ m}$), la densidad del muestreo histórico ($6\,000$ viajes en $9$ meses sobre un área de $40 \times 50\text{ km}$) es demasiado dispersa para conectar puntos mediante cadenas continuas. Indica que los datos no son homogéneamente densos a nivel de cuadra individual en la muestra tomada.

---

## 4. Análisis e Interpretación de Figuras Clave

A continuación se presentan e interpretan las figuras analíticas generadas durante la ejecución de los experimentos (almacenadas en `assets/semana-08/`):

### Figura 1: Selección por BIC y Representación Espacial GMM
*(Archivos: `04_bic.png` y `05_gmm_espacial.png`)*

```text
[Gráfico de Complejidad BIC vs Componentes K]  ----->  [Mapa de Componentes y Elipses del 95%]
```

- **Interpretación del BIC (`04_bic.png`):** La curva de covarianza completa (`full`) domina consistentemente a la covarianza diagonal (`diag`) para todo $K$. La covarianza completa permite correlación entre $x$ e $y$ ($\sigma_{xy} \neq 0$), lo cual es imprescindible para orientar las elipses a lo largo de la inclinación geográfica de la isla de Manhattan (orientación suroeste–noreste). La curva desciende de $60\,968$ ($K=1$) a $45\,282$ ($K=8$).
- **Interpretación Espacial y Entropía (`05_gmm_espacial.png`):** El panel izquierdo ilustra las elipses que encierran el $95\%$ de masa de cada componente normal. Se aprecia cómo 4 elipses se superponen para cubrir la longitud de Manhattan, mientras dos elipses alargadas capturan los corredores hacia los aeropuertos. El panel derecho (mapa de entropía de responsabilidades $\mathcal{H}_i$) destaca con colores cálidos las zonas de mayor incertidumbre de asignación ($x \in [-1, 1], y \in [0, 3]$ km), demostrando visualmente que las fronteras entre componentes no son barreras impermeables sino transiciones probabilísticas suaves.

### Figura 2: Superficie Continua de Densidad KDE
*(Archivo: `08_kde_espacial.png`)*

- **Interpretación Espacial:** La figura presenta la superficie continua de log-densidad espacial calculada sobre una cuadrícula uniforme de $140 \times 140$ celdas para tres anchos de banda ($h = 0.1\text{ km}$, $h = 0.4\text{ km}$ y $h = 3.2\text{ km}$):
  - Con $h = 0.1\text{ km}$ (infra-suavizado / sobreajuste): La superficie se fragmenta en agujas de densidad aisladas alrededor de puntos individuales, asignando log-densidades extremadamente negativas ($<-14$) al espacio adyacente.
  - Con $h = 0.4\text{ km}$ (óptimo temporal congelado): Se recupera con nitidez la morfología metropolitana. El pico de densidad se ubica en el centro de Manhattan (log-densidad $\approx -1.5$ a $-1.0$, equivalente a $\widehat{f} > 0.35\text{ km}^{-2}$), con crestas secundarias claramente diferenciadas en los aeropuertos LGA y JFK.
  - Con $h = 3.2\text{ km}$ (sobre-suavizado): La masa se dispersa indiscriminadamente, borrando los límites geográficos y asignando probabilidades considerables a zonas deshabitadas y sobre el Océano Atlántico.

### Figura 3: Conectividad y Detección de Ruido en DBSCAN
*(Archivo: `13_dbscan_espacial.png`)*

- **Interpretación Espacial:** Compara el comportamiento espacial con $m=15$ y tres distancias $\varepsilon \in \{0.2, 0.4, 0.8\}\text{ km}$:
  - Con $\varepsilon = 0.2\text{ km}$ ($9.75\%$ ruido): El clúster central se quiebra en múltiples fragmentos debido a discontinuidades locales de muestreo en Central Park y Midtown.
  - Con $\varepsilon = 0.4\text{ km}$ ($5.08\%$ ruido): Configuración de compromiso ideal. Recupera el corredor principal como una sola entidad conectada por densidad y separa limpiamente los polos aeroportuarios, mientras los puntos en gris (ruido) delimitan la periferia difusa.
  - Con $\varepsilon = 0.8\text{ km}$ ($2.17\%$ ruido): Los puntos de puenteo comienzan a fundir Manhattan con las áreas circundantes de Queens y Brooklyn, perdiendo especificidad zonal.

---

## 5. Dictamen y Recomendación al Comité de Movilidad

Con base en la triangulación rigurosa de las tres fuentes de evidencia (GMM, KDE y DBSCAN) y respetando las limitaciones impuestas por el contrato experimental, se emite el siguiente dictamen:

### Propuesta de Zonas Candidatas

1. **Zona Candidata 1 para Estudio Operativo (Prioridad A): Corredor Midtown–Downtown Manhattan**
   - **Límites en coordenadas relativas:** $x_{\text{km}} \in [-1.50, 1.50]\text{ km}$, $y_{\text{km}} \in [0.00, 4.00]\text{ km}$.
   - **Referencias geográficas:** Sector comprendido entre 14th Street y 59th Street (incluyendo Times Square, Grand Central, Penn Station y distritos comerciales adyacentes).
   - **Evidencia Convergente:**
     * *GMM:* Asigna el mayor peso de mezcla ($\pi_k > 0.45$ acumulado en componentes centrales) y bajas varianzas transversales.
     * *KDE:* Registra el punto culminante de densidad superficial ($\widehat{f} > 1.4\text{ km}^{-2}$, log-densidad $>-1.2$).
     * *DBSCAN:* Forma el núcleo hiperconectado de mayor tamaño del grafo ($>5\,000$ observaciones directamente alcanzables por densidad con $\varepsilon=0.4\text{ km}$).
   - **Objetivo operativo:** Zona prioritaria para desplegar infraestructura de recarga compartida, aforos de demanda en horas pico y auditoría de velocidades comerciales.

2. **Zona Candidata 2 para Estudio Operativo (Prioridad B): Polo Satélite Aeropuerto LaGuardia**
   - **Límites en coordenadas relativas:** $x_{\text{km}} \in [7.50, 10.00]\text{ km}$, $y_{\text{km}} \in [3.00, 5.50]\text{ km}$.
   - **Referencias geográficas:** Área de terminales aéreas de LaGuardia (Queens septentrional) y accesos viales de Grand Central Parkway.
   - **Evidencia Convergente:**
     * *GMM:* Capturada nítidamente por un componente gaussiano independiente y compacto sin solapamiento con el eje central.
     * *KDE:* Manifiesta una cresta secundaria aislada rodeada por valles de densidad muy baja.
     * *DBSCAN:* Constituye un clúster disyunto independiente de $113$ observaciones con cero fusión hacia la mancha urbana intermedia.
   - **Objetivo operativo:** Estudio de servicios de lanzadera (*shuttle*) con tiempos de espera fijos y diseño de puntos de abordaje dedicados fuera del tráfico urbano mixto.

3. **Ubicación para Revisión Crítica: Borde Fluvial Río Hudson / George Washington Bridge**
   - **Límites en coordenadas relativas:** $x_{\text{km}} \in [-3.00, -1.50]\text{ km}$, $y_{\text{km}} \in [5.00, 7.50]\text{ km}$.
   - **Diagnóstico del conflicto metodológico:** En este sector, los modelos continuos (KDE y GMM) cometen una transgresión de barrera física: el suavizado extiende masa de probabilidad sobre el cauce del Río Hudson hacia New Jersey. Por el contrario, DBSCAN detecta la falta de datos sobre el agua y rompe la conectividad, marcando los viajes de la orilla oeste como ruido o componente separada.
   - **Recomendación:** Auditar la precisión de georreferenciación GPS en los accesos al puente y aplicar máscaras hidrológicas que impidan asignar probabilidad operativa sobre espejos de agua.

---

### Párrafo Estructurado del Comité (Estructura Canónica Obligatoria)

> **Observo** una fuerte concentración espacial estructurada en un corredor continuo hiperdenso en Manhattan ($[-1.5, 1.5] \times [0, 4]\text{ km}$) y dos núcleos satélites compactos en los aeropuertos LGA y JFK, junto con un $5.1\%$ a $23\%$ de recogidas periféricas clasificadas como ruido en DBSCAN ($\varepsilon=0.4\text{ km}, m=15$).  
> **Interpreto** que la demanda histórica responde a una morfología mixta compuesta por una arteria vial masiva no convexa y polos intermodales independientes, cuya geometría no puede ser reducida a centroides esféricos de K-means ni a elipses que cruzan cuerpos de agua.  
> **Recomendaría estudiar** operativamente el cuadrante de Midtown Manhattan ($x \in [-1.5, 1.5], y \in [0, 4]\text{ km}$) y la terminal de LaGuardia ($x \in [7.5, 10], y \in [3, 5.5]\text{ km}$) como zonas piloto para auditorías de tráfico y dimensionamiento de puntos de parada.  
> **Vigilaría** las riberas del Río Hudson ($x \in [-3, -1.5], y \in [5, 7.5]\text{ km}$) donde el suavizado simétrico de KDE y GMM derrama masa de probabilidad sobre barreras fluviales intransitables.  
> **No puedo concluir** sobre la ubicación exacta de estaciones definitivas, flota requerida ni demanda insatisfecha sin incorporar matrices completas de origen-destino, congestión en la red vial y costos operativos de servicio.

---

## 6. Guion para la Defensa Oral Ejecutiva (90 Segundos)

Este guion está cronometrado con precisión para la presentación oral ante el comité evaluador:

```text
⏱️ 0:00 - 0:20 (20s) · PREGUNTA Y NATURALEZA DE LOS DATOS
"Buenos días. Analizamos 6.000 recogidas históricas de 2014 en Nueva York, proyectadas en coordenadas 
planas relativas en kilómetros, con el fin de identificar zonas prioritarias para un estudio operativo. 
Es crucial aclarar que observamos únicamente eventos de recogida registrados, no la demanda total de 
la ciudad ni la red vial real."

⏱️ 0:20 - 1:00 (40s) · EVIDENCIA COMPARADA (GMM, KDE, DBSCAN)
"Triangulamos tres modelos: GMM capturó la orientación suroeste-noreste del corredor central con un BIC 
óptimo en K=8, pero sufre al imponer geometrías elípticas sobre barreras fluviales. KDE, validado 
temporalmente con un ancho de banda óptimo de 400 metros, confirmó que el 88.6% de la masa espacial se 
concentra en el núcleo central. Por su parte, DBSCAN con épsilon de 400 metros y min_samples de 15 
demostró que Manhattan opera como una componente conexa no convexa continua de más de 5.400 puntos, 
aislando limpiamente a los aeropuertos y separando un 5% de ruido."

⏱️ 1:00 - 1:30 (30s) · RECOMENDACIÓN, VIGILANCIA Y LÍMITES
"Recomendamos priorizar dos zonas de estudio: Midtown Manhattan, entre las coordenadas relativas 
[-1.5, 1.5] en x y [0, 4] en y, y el aeropuerto LaGuardia en [7.5, 10] por [3, 5.5]. Pedimos revisar con 
cuidado el borde del Río Hudson, donde KDE proyecta masa falsa sobre el agua. Finalmente, advertimos que 
un centroide matemático no es una estación física y que no es posible dimensionar flotas sin datos de 
tiempos de viaje, congestión y costos."
```

---

## 7. Banco Consolidado de Tickets Individuales

### Ticket de Salida · Lunes (GMM y Algoritmo EM)
1. **¿Qué optimiza EM?**
   EM optimiza la log-verosimilitud observada $\ell(\theta) = \sum_{i=1}^n \log \left(\sum_{k=1}^K \pi_k \phi(x_i; \mu_k, \Sigma_k)\right)$ a través de la optimización alternada de la cota inferior de evidencia (ELBO): $\mathcal{F}(q, \theta) = \mathbb{E}_q[\log p(X, Z \mid \theta)] + \mathcal{H}(q)$. En el paso E se anula la divergencia KL ($\mathrm{KL}(q \parallel p) = 0$), y en el paso M se maximiza $\mathcal{F}$ respecto a $\theta$. Garantiza convergencia no decreciente a un punto estacionario local, pero **no asegura el óptimo global ni la unicidad de parámetros**.
2. **¿Qué significa una responsabilidad de $0.55$?**
   Indica la probabilidad a posteriori $P(z_i = k \mid x_i, \theta)$ de que la observación $x_i$ haya sido generada por el componente $k$. Denota una **pertenencia altamente ambigua**. Fundamentalmente, **responsabilidad condicional no equivale a densidad**: un punto ubicado en una zona desértica o periférica puede tener $r_{ik}=0.55$ (o incluso $0.99$), mientras que su densidad de probabilidad absoluta $f(x_i)$ es prácticamente cero.
3. **¿Qué no demuestra el BIC?**
   El criterio BIC ($-2\ell + p\log n$) no demuestra que el modelo elegido sea el proceso generador verdadero de los datos, ni que existan ontológicamente $K$ barrios reales. Tampoco valida transferencia fuera de muestra cuando se vulneran los supuestos asintóticos i.i.d. debido a la fuerte correlación espacial urbana.

### Ticket de Salida · Miércoles (KDE y Estimación de Masa)
*Explique por qué densidad puntual, masa en un área y número de vehículos necesarios son tres cantidades diferentes:*
1. **Densidad puntual ($\widehat{f}(x)$):** Magnitud intensiva local de probabilidad por unidad de superficie ($\text{km}^{-2}$). Puede superar el valor de 1 en zonas densas; la probabilidad en una coordenada continua exacta es cero ($P(X=x_0) = 0$).
2. **Masa regional ($\int_A \widehat{f}(x)dx$):** Magnitud extensa adimensional (en $[0, 1]$) obtenida mediante cuadratura numérica sobre un polígono $A$. Representa la probabilidad acumulada o proporción esperada de viajes en esa región.
3. **Número de vehículos necesarios ($N_{\text{vehículos}}$):** Magnitud logística entera discreta ($\in \mathbb{N}$). Requiere la demanda total absoluta en hora punta, tiempo de ciclo de viaje (duración, retorno, atascos viales), tasa de recarga y política de nivel de servicio. **Ninguno de estos datos operativos está presente en la muestra espacial**.

### Ticket de Salida · Viernes (DBSCAN y Propiedades de Conectividad)
1. **¿Puede $\varepsilon$ ser menor que el diámetro de un grupo?**
   **SÍ.** El diámetro de un grupo puede superar ampliamente a $\varepsilon$ gracias a la transitividad de la **alcanzabilidad por densidad**. Si existe una cadena de puntos núcleo consecutivos $p_1, p_2, \dots, p_k$ tales que cada par diste $\le \varepsilon$, se conforma un único clúster conexo. Con $\varepsilon = 0.4\text{ km}$, el conglomerado de Manhattan alcanza un diámetro continuo superior a $18\text{ km}$.
2. **¿Una frontera une dos grupos?**
   **NO.** Un punto frontera no es núcleo ($|N_\varepsilon| < m$) y, por definición, no expande la vecindad. Si un punto frontera se ubica en la intersección de dos componentes de núcleos, DBSCAN no las fusiona: asigna el punto a la primera componente que lo explore según el orden secuencial de recorrido, manteniendo los grupos separados.
3. **¿Ruido significa error?**
   **NO.** La etiqueta $-1$ (ruido) denota simplemente que un punto no alcanzó el umbral local de densidad espacial fijado por $(\varepsilon, m)$ en esa muestra. Puede corresponder a un viaje perfectamente válido y legítimo en un área residencial de baja frecuencia o en una autopista periférica.

---

## 8. Anexos de Cálculos y Fundamentos Matemáticos

### Anexo A: Paso EM Analítico (Cálculo Manual Sesión 1)
- **Configuración:** $x = (-2, -1, 1, 2)$, $\pi = (0.5, 0.5)$, $\mu = (-1, 1)$, $\sigma^2 = (1, 1)$.
- **Paso E:**
  $$r_{ik} = \frac{\exp\left(-\frac{(x_i - \mu_k)^2}{2}\right)}{\exp\left(-\frac{(x_i - \mu_1)^2}{2}\right) + \exp\left(-\frac{(x_i - \mu_2)^2}{2}\right)}$$
  - Para $x_1 = -2$: $r_{11} = \frac{1}{1 + e^{-4}} \approx \mathbf{0.9820}$, $r_{12} \approx \mathbf{0.0180}$.
  - Para $x_2 = -1$: $r_{21} = \frac{1}{1 + e^{-2}} \approx \mathbf{0.8808}$, $r_{22} \approx \mathbf{0.1192}$.
  - Para $x_3 = 1$: $r_{31} \approx \mathbf{0.1192}$, $r_{32} \approx \mathbf{0.8808}$.
  - Para $x_4 = 2$: $r_{41} \approx \mathbf{0.0180}$, $r_{42} \approx \mathbf{0.9820}$.
- **Paso M:**
  - $N_1 = 0.9820 + 0.8808 + 0.1192 + 0.0180 = \mathbf{2.0}$, $N_2 = \mathbf{2.0}$.
  - $\pi_1^+ = 2/4 = \mathbf{0.50}$, $\pi_2^+ = \mathbf{0.50}$.
  - $\mu_1^+ = \frac{1}{2} [0.9820(-2) + 0.8808(-1) + 0.1192(1) + 0.0180(2)] \approx \mathbf{-1.3448}\text{ km}$, $\mu_2^+ \approx \mathbf{+1.3448}\text{ km}$.
  - $(\sigma_1^2)^+ = \frac{1}{2} \sum_i r_{i1}(x_i - \mu_1^+)^2 \approx \mathbf{0.6915}\text{ km}^2$, $(\sigma_2^2)^+ \approx \mathbf{0.6915}\text{ km}^2$.

### Anexo B: Cálculo Manual KDE (Sesión 2)
- **Configuración:** $x_i \in \{-1, 0, 1\}$ ($n=3, d=1$), kernel gaussiano.
- **Fórmula en el origen:** $\widehat{f}_h(0) = \frac{1}{3h\sqrt{2\pi}} [1 + 2e^{-1/(2h^2)}]$.
  - $h = 0.5 \implies \widehat{f}_{0.5}(0) \approx \mathbf{0.3380}\text{ km}^{-1}$.
  - $h = 1.0 \implies \widehat{f}_{1.0}(0) \approx \mathbf{0.2943}\text{ km}^{-1}$.
  - $h = 2.0 \implies \widehat{f}_{2.0}(0) \approx \mathbf{0.1838}\text{ km}^{-1}$.

### Anexo C: Incremento de Ward y Altura SciPy (Refuerzo Jerárquico)
- **Grupos:** $A = \{0, 2\}$ ($n_A=2, \mu_A=1, SSE_A=2$), $B = \{5\}$ ($n_B=1, \mu_B=5, SSE_B=0$).
- **Grupo fusionado:** $\mu_{AB} = 7/3$, $SSE(A \cup B) = 38/3$.
- **Incremento de Ward:**
  $$\Delta(A, B) = \frac{n_A n_B}{n_A + n_B} \|\mu_A - \mu_B\|^2 = \frac{2 \times 1}{3} (1 - 5)^2 = \frac{32}{3} \approx \mathbf{10.6667}\text{ km}^2$$
- **Altura de enlace SciPy:**
  $$d_{\text{SciPy}} = \sqrt{2 \Delta(A, B)} = \sqrt{2 \times \frac{32}{3}} = \sqrt{\frac{64}{3}} = \frac{8}{\sqrt{3}} \approx \mathbf{4.6188}\text{ km}$$
