# LiteratureRetrieval

> **Guía rápida de revisión (Etapa 2 — "Passages"):** La primera sección mapea directamente cada punto de **11. Required deliverables** a los archivos exactos dentro del repositorio. Debajo se conserva íntegra la documentación técnica original del proyecto.

---

## 1. Guía de evaluación y entregables (Etapa 2: Pasajes)

Tabla de verificación directa frente a los entregables requeridos:

| Requisito de evaluación | Dónde verlo en el repositorio | Detalles clave verificables |
| :--- | :--- | :--- |
| **1. Código fuente con Pipelines, entorno, semillas e instrucciones reproducibles** | [requirements.txt](requirements.txt)<br>[Passages/split.py](Passages/split.py)<br>[Passages/normalize.py](Passages/normalize.py)<br>[Passages/explore_vocab.py](Passages/explore_vocab.py)<br>[Passages/representation.py](Passages/representation.py)<br>[Passages/results.py](Passages/results.py)<br>[Passages/kmeans.py](Passages/kmeans.py)<br>[Passages/agglomerative.py](Passages/agglomerative.py)<br>[Passages/lda.py](Passages/lda.py)<br>[Passages/nmf.py](Passages/nmf.py)<br>[Passages/interfaz.py](Passages/interfaz.py) | • **Pipelines:** encapsulados con `scikit-learn` (`armar_vectorizador` en `representation.py`, ajustados solo sobre `train`).<br>• **Semilla fija:** `SEED = 3` compartida desde `representation.py` y `split.py`.<br>• **Entorno:** `pandas`, `spacy` (`es_core_news_lg`), `scikit-learn==1.7.2`, `numpy` (ver instrucciones de reproducción abajo). |
| **2. `documents.tsv`** | [Passages/TSV/documents.tsv](Passages/TSV/documents.tsv) | • 500 fragmentos (`train`: 357, `dev`: 72, `test`: 71) partidos por capítulo (`chapter_group`) sin fuga.<br>• Columnas iniciales: `fragment_id`, `book_id`, `title`, `author`, `text`, `split` + metadatos de Etapa 1.<br>• `text` conserva el pasaje literario original sin etiquetas objetivo. |
| **3. `clustering_results.tsv` + tamaños, top 10 palabras y pasajes representativos (K-Means y Agglomerative)** | Consolidados:<br>[clustering_results.tsv](Passages/TSV/Results/clustering_results.tsv)<br>[cluster_summary.tsv](Passages/TSV/Results/cluster_summary.tsv)<br><br>Por algoritmo:<br>[TSV/Results/kmeans/](Passages/TSV/Results/kmeans/) (`K38`)<br>[TSV/Results/agglomerative/](Passages/TSV/Results/agglomerative/) (`A23`) | • `clustering_results.tsv` incluye `experiment_id`, `algorithm`, `fragment_id`, `split`, `cluster_id`, `assignment_note`.<br>• En **Agglomerative (`A23`)**, `assignment_note` distingue explícitamente `fit_predict on train only` (357 filas de `train`) de `approximate assignment (nearest centroid)` (72 de `dev` y 71 de `test`).<br>• `cluster_summary.tsv` reporta `size`, `top_words` (10 palabras originales) y `representative_passages` (`fragment_id`).<br>• Gráfica 2D de Agglomerative: [figura1_agglomerative_A23.png](Passages/TSV/Results/agglomerative/figura1_agglomerative_A23.png). |
| **4. `topic_results.tsv` + palabras por tema, pasajes representativos, perplejidad (LDA) y error de reconstrucción (NMF)** | Consolidados:<br>[topic_results.tsv](Passages/TSV/Results/topic_results.tsv)<br>[topic_words.tsv](Passages/TSV/Results/topic_words.tsv)<br><br>Por algoritmo:<br>[TSV/Results/lda/](Passages/TSV/Results/lda/) (`L43`)<br>[TSV/Results/nmf/](Passages/TSV/Results/nmf/) (`N19`) | • `topic_results.tsv` contiene `experiment_id`, `algorithm`, `fragment_id`, `split`, `topic_id`, `topic_weight`.<br>• `topic_words.tsv` incluye las 10 palabras principales y pasajes representativos por tema.<br>• `metrics.tsv` registra `perplexity` para LDA y `reconstruction_error` para NMF. |
| **5. `experiments.tsv` (registro completo de los 4 métodos)** | Individuales (completos):<br>• [kmeans/experiments.tsv](Passages/TSV/Results/kmeans/experiments.tsv) (`K01`–`K61`, sel. `K38`)<br>• [agglomerative/experiments.tsv](Passages/TSV/Results/agglomerative/experiments.tsv) (`A01`–`A31`, sel. `A23`)<br>• [lda/experiments.tsv](Passages/TSV/Results/lda/experiments.tsv) (`L01`–`L43`, sel. `L43`)<br>• [nmf/experiments.tsv](Passages/TSV/Results/nmf/experiments.tsv) (`N01`–`N25`, sel. `N19`)<br><br>Consolidado:<br>• [Results/experiments.tsv](Passages/TSV/Results/experiments.tsv) y [Results/metrics.tsv](Passages/TSV/Results/metrics.tsv) | • Columnas: `experiment_id`, `task`, `method`, `recipe`, `normalization`, `representation`, `ngram_range`, `vocabulary`, `dim_reduction`, `hyperparameters`, `dev_metric`, `dev_value`, `selected`, `notes`.<br>• Cada archivo registra todas las recetas (`N0`–`N4`, `R1`–`R3`, n-gramas, vocabulario, SVD), búsqueda de hiperparámetros, evidencia en `dev` (o `train` en Agglomerative) y decisión `selected` (`yes`/`no`). |
| **6. Reporte PDF** | Entrega externa / carpeta de reporte | • Reporte `ProjectReport_Stage2-Passages.pdf` con tablas, figuras, análisis de errores/limitaciones e integrantes del equipo. |

### Reproducción rápida (Etapa 2)

```bash
# 1. Instalar dependencias (Python >= 3.10 recomendado)
pip install -r requirements.txt
python -m spacy download es_core_news_lg

# 2. (Opcional: los TSV ya están generados) Regenerar partición, normalización y análisis exploratorio
python Passages/split.py
python Passages/normalize.py
python Passages/explore_vocab.py
python Passages/representation.py

# 3. Ejecutar los 4 algoritmos (cada uno guarda en Passages/TSV/Results/<algoritmo>/)
python Passages/kmeans.py
python Passages/agglomerative.py
python Passages/lda.py
python Passages/nmf.py

# 4. Consolidar los 4 algoritmos en Passages/TSV/Results/*.tsv
python Passages/results.py

# 5. Consultar clusters, pesos de temas y métricas en la interfaz CLI
python Passages/interfaz.py pasaje 432        # cluster y temas de un pasaje
python Passages/interfaz.py resumen           # clusters y temas con palabras y pasajes
python Passages/interfaz.py resumen temas     # solo temas (o: resumen clustering)
python Passages/interfaz.py metricas          # receta, hiperparámetros y métricas de K38, A23, L43 y N19
python Passages/interfaz.py                   # menú interactivo
```

---

## 2. Documentación completa del proyecto (Etapa 1 y Etapa 2)

Sistema integral de recuperacion de informacion (Information Retrieval - IR) sobre textos literarios, basado en el Modelo de Espacio Vectorial (Vector Space Model - VSM) y similitud coseno para la practica de Procesamiento de Lenguaje Natural (PLN).

El sistema permite indexar, normalizar, representar vectorialmente, consultar y evaluar sistematicamente un corpus literario de 500 fragmentos (~500 palabras por fragmento) distribuidos equitativamente entre 5 obras fundamentales de la literatura universal:

1. **La insoportable levedad del ser** (Milan Kundera) - Fragmentos 1 al 100
2. **La Odisea** (Homero) - Fragmentos 101 al 200
3. **El mundo de Sofia** (Jostein Gaarder) - Fragmentos 201 al 300
4. **Mi Lucha** (Adolf Hitler) - Fragmentos 301 al 400
5. **1984** (George Orwell) - Fragmentos 401 al 500

---

## Estructura del proyecto

```text
LiteratureRetrieval/
│
├── corpus/                                 # Archivos fuente originales (.csv y .tsv) por obra
│   ├── 1984/                               # 1984 (George Orwell)
│   ├── ElMundoDeSofia/                     # El mundo de Sofia (Jostein Gaarder)
│   ├── Levedad/                            # La insoportable levedad del ser (Milan Kundera)
│   ├── MiLucha/                            # Mi Lucha (Adolf Hitler)
│   └── Odisea/                             # La Odisea (Homero)
│
├── src/                                    # Codigo fuente modular
│   ├── __init__.py
│   ├── corpus.py                           # Fase 1: Ingestion, unificacion y consolidacion del corpus
│   ├── nlp.py                              # Fases 2 y 3: Normalizacion linguistica con spaCy
│   ├── vsm.py                              # Fase 4: Vectorizacion y modelos VSM (6 configuraciones)
│   ├── retrieval.py                        # Fase 5: Motor de busqueda por similitud coseno
│   └── benchmark.py                        # Fase 6: Evaluacion del benchmark y metricas de IR
│
├── main.py                                 # Interfaz de linea de comandos (CLI) principal
├── requirements.txt                        # Dependencias del proyecto
│
├── literature_raw_corpus.tsv               # Corpus crudo consolidado (Fase 1, 500 filas + cabecera)
├── literature_normalized_corpus.tsv        # Corpus lematizado y filtrado (Fase 2)
│
├── models/                                 # Modelos entrenados y serializados (.pkl)
│   ├── binary_unigram.pkl
│   ├── binary_bigram.pkl
│   ├── frequency_unigram.pkl
│   ├── frequency_bigram.pkl
│   ├── tfidf_unigram.pkl
│   ├── tfidf_bigram.pkl
│   └── csv/                                # Exportaciones en CSV de las matrices de termino-documento
│
├── queries.tsv                             # Conjunto de 15 consultas de prueba (Fases 5 y 6)
├── qrels.tsv                               # Juicios de relevancia ground-truth (escala 0, 1, 2)
├── rankings.tsv                            # Ranking oficial Top-10 generado por la mejor configuracion
├── combined_view.tsv                       # Ranking oficial combinado con los juicios de relevancia
│
├── results/                                # Salidas de recuperacion y evaluacion
│   ├── <timestamp>_<slug>_<repr>_<feat>.txt # Resultados detallados de consultas individuales
│   └── benchmark/                          # Reportes detallados del benchmark
│       ├── metrics_by_config.tsv           # Promedios comparativos de las 6 configuraciones (Tabla 9)
│       ├── metrics_by_query.tsv            # Desglose de metricas por consulta y configuracion (Tabla 10)
│       ├── metrics_by_level.tsv            # Metricas desglosadas por nivel de dificultad (Tabla 11)
│       ├── rankings_<config>.tsv           # Rankings Top-10 de cada configuracion
│       └── combined_view_<config>.tsv      # Vista combinada ranking + relevancia por configuracion
│
├── tools/                                  # Scripts auxiliares y herramientas de desarrollo
│   ├── ver_pkl.py                          # Inspeccion y exportacion de modelos .pkl a CSV
│   ├── epubtxt.py                          # Extraccion de texto plano desde archivos EPUB
│   ├── filter.py                           # Segmentacion y filtrado de fragmentos
│   └── keywords.py                         # Generacion de palabras clave por obra
│
├── Passages/                               # Etapa 2: Clustering y modelado de temas sobre pasajes
│   ├── split.py                            # Particion reproducible train / dev / test por capitulo
│   ├── normalize.py                        # 5 niveles de normalizacion (N0 - N4) con spaCy
│   ├── explore_vocab.py                    # Analisis exploratorio de vocabulario y varianza SVD
│   ├── representation.py                   # 13 recetas y constructor comun de vectorizadores
│   ├── results.py                          # Funciones compartidas de guardado y consolidacion TSV
│   ├── kmeans.py                           # Experimentos y evaluacion de K-Means
│   ├── agglomerative.py                    # Experimentos y evaluacion de Agglomerative Clustering
│   ├── lda.py                              # Experimentos y evaluacion de LDA
│   ├── nmf.py                              # Experimentos y evaluacion de NMF
│   ├── interfaz.py                         # Interfaz CLI para consultar clusters, temas y metricas
│   └── TSV/                                # Particiones, corpus normalizados y resultados por metodo
│
└── docs/                                   # Documentacion tecnica detallada
    ├── README.md                           # Indice de la documentacion
    ├── ARCHITECTURE.md                     # Arquitectura de software y flujo de datos
    ├── CORPUS_SPEC.md                      # Especificacion de columnas y obras
    ├── NLP_PIPELINE.md                     # Reglas del pipeline de normalizacion
    ├── SPACY_CONSIDERATIONS.md             # Consideraciones y sesgos del modelo linguistico
    └── AGENT_NOTES.md                      # Notas tecnicas de contexto para desarrolladores
```

---

## Requisitos e instalacion

Se requiere Python 3.9 o superior.

### 1. Clonar el repositorio y preparar el entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

En sistemas Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Descargar el modelo en espanol de spaCy

Se recomienda el modelo grande (`lg`) para maxima precision lematizadora y sintactica:
```bash
python -m spacy download es_core_news_lg
```

*Nota*: El pipeline cuenta con tolerancia automatica a fallos; si `es_core_news_lg` no esta presente, intentara cargar `es_core_news_md` o `es_core_news_sm`.

---

## Modo de uso: Paso a paso por fases

El sistema puede gestionarse de forma centralizada a traves del CLI principal `main.py` o mediante llamadas modulares dedicadas.

### Fase 1: Consolidar el corpus crudo

Examina recursivamente todos los archivos de corpus en `corpus/`, maneja codificaciones con BOM (`utf-8-sig`), estandariza separadores heterogeneos (comas y tabuladores), descarta columnas no estandar y preserva los identificadores originales del 1 al 500 y las columnas oficiales (`fragment_id`, `book_id`, `title`, `author`, `chapter/position`, `text`, `language`, `source`, `url`, `keywords`).

```bash
python main.py --phase1
```

Genera el archivo: `literature_raw_corpus.tsv` (500 registros + cabecera).

---

### Fase 2: Normalizar el corpus linguisticamente

Toma `literature_raw_corpus.tsv` y procesa el campo `text` mediante spaCy en lotes (`nlp.pipe(batch_size=100)`). Aplica limpieza de caracteres, filtrado estricto por categorias sintacticas (POS), eliminacion de puntuacion y simbolos, separacion de pronombres encliticos y lematizacion en minusculas. Conserva intactos los campos `title` y `keywords`.

```bash
python main.py --phase2
```

Genera el archivo: `literature_normalized_corpus.tsv`.

---

### Ejecutar Fases 1 y 2 consecutivamente

Para regenerar ambos datasets de forma continua:

```bash
python main.py --all
```

*(Si no se pasan argumentos a `main.py`, se ejecuta `--all` por defecto).*

---

### Fase 3: Probar normalizacion de una consulta libre

Permite verificar como el pipeline linguistico normaliza, filtra y lematiza una consulta de texto libre en lenguaje natural:

```bash
python main.py --query "descripcion de una tormenta violenta en alta mar"
```

Salida en consola:
- Consulta original
- Tokens conservados y lematizados
- Texto final normalizado listo para vectorizacion

---

### Fase 4: Construccion de representaciones VSM (Modelos vectoriales)

Entrena las 6 configuraciones oficiales de representacion en espacio vectorial sobre el corpus normalizado utilizando `scikit-learn`:

- **Representaciones**: Frecuencia (`CountVectorizer`), Binaria (`CountVectorizer(binary=True)`), TF-IDF (`TfidfVectorizer`).
- **Caracteristicas**: Unigramas (`ngram_range=(1, 1)`), Bigramas (`ngram_range=(2, 2)`).

Ejecucion directa:
```bash
python -m src.vsm
```

O con rutas personalizadas:
```bash
python -m src.vsm --input literature_normalized_corpus.tsv --out models
```

Cada configuracion guarda su vectorizador entrenado, matriz dispersa y lista de identificadores en `models/<representacion>_<n-grama>.pkl`:
- `models/frequency_unigram.pkl`
- `models/frequency_bigram.pkl`
- `models/binary_unigram.pkl`
- `models/binary_bigram.pkl`
- `models/tfidf_unigram.pkl`
- `models/tfidf_bigram.pkl`

#### Inspeccion y exportacion de modelos a CSV

Para inspeccionar las dimensiones, vocabulario, dispersion de ceros y exportar matrices a formato CSV:

```bash
# Exporta un archivo CSV con las ternas (fragment_id, termino, valor) omitiendo ceros:
python tools/ver_pkl.py models

# Exporta la matriz completa con ceros (terminos x documentos) en models/csv_completo/:
python tools/ver_pkl.py models --completo
```

---

### Fase 5: Motor de recuperacion de informacion (Similitud coseno)

Recibe una consulta libre en lenguaje natural, la normaliza a traves de `src.nlp.normalize_query()`, la proyecta en el espacio vectorial del modelo seleccionado y calcula la similitud coseno frente a los 500 fragmentos del corpus literario.

Muestra el ranking Top-10 formateado en consola y genera automaticamente un archivo de texto exhaustivo con los fragmentos completos, puntuaciones de similitud y metadatos en la carpeta `results/`.

#### Ejecucion con configuracion recomendada (TF-IDF y unigramas):

```bash
python -m src.retrieval -q "¿que relacion tiene la vida y la muerte?"
```

#### Seleccionar representacion y n-gramas:

```bash
python -m src.retrieval -q "evadirse sin ser visto de una guarida" -r tfidf -f unigram
python -m src.retrieval -q "el odio y la propaganda en el pueblo" -r frequency -f unigram
python -m src.retrieval -q "libertad y control del pensamiento" -r binary -f bigram
```

#### Parametros disponibles:
- `-q`, `--query`: Consulta en lenguaje natural (obligatorio).
- `-r`, `--representation`: Representacion vectorial (`tfidf`, `frequency`, `binary`). Por defecto: `tfidf`.
- `-f`, `--feature`: Tipo de n-grama (`unigram`, `bigram`). Por defecto: `unigram`.

#### Archivo generado:
Cada busqueda guarda un archivo unico con timestamp en `results/`:
`results/<timestamp>_<slug_consulta>_<representacion>_<n-grama>.txt`

---

### Fase 6: Evaluacion del Benchmark y Generacion de Metricas

Evalua el rendimiento de las 6 configuraciones VSM sobre el conjunto de 15 consultas de prueba (`queries.tsv`) contra los juicios de relevancia anotados en `qrels.tsv` (150 juicios con relevancia graduada: 0 = No relevante, 1 = Parcialmente relevante, 2 = Altamente relevante).

#### Ejecucion rapida desde CLI principal:

```bash
python main.py --benchmark
```

#### Ejecucion modular avanzada:

```bash
python -m src.benchmark --queries queries.tsv --qrels qrels.tsv --output rankings.tsv --results-dir results/benchmark
```

#### Metricas calculadas (sobre los primeros 10 resultados):

- **P@10 (Precision at 10)**: Proporcion de documentos recuperados en el Top-10 que son relevantes.
- **R@10 (Recall at 10)**: Proporcion de documentos relevantes recuperados respecto al total de relevantes existentes para la consulta.
- **F1@10**: Media armonica balanceada entre P@10 y R@10.
- **MRR (Mean Reciprocal Rank)**: Posicion inversa ($1 / \text{rank}$) del primer documento relevante encontrado.
- **nDCG@10 (Normalized Discounted Cumulative Gain at 10)**: Calidad del ranking considerando la relevancia graduada (ganancia $2^{\text{rel}} - 1$) y penalizando posiciones tardias mediante descuento logaritmico $\log_2(\text{rank} + 1)$.

#### Salidas generadas por el benchmark:

1. **`rankings.tsv`**: Ranking Top-10 oficial generado por la mejor configuracion (`query_id`, `document_id`, `rank`, `score`).
2. **`combined_view.tsv`**: Ranking Top-10 oficial con la columna de relevancia de cada documento segun `qrels.tsv` (`relevance`, `relevant: Yes/No`).
3. **`results/benchmark/metrics_by_config.tsv`**: Tabla comparativa de desempeno promedio entre las 6 configuraciones (Tabla 9 de la especificacion).
4. **`results/benchmark/metrics_by_query.tsv`**: Desglose detallado de todas las metricas para cada consulta individual y configuracion (Tabla 10).
5. **`results/benchmark/metrics_by_level.tsv`**: Promedios agrupados segun la dificultad de la consulta (Fácil, Media, Difícil) para cada configuracion (Tabla 11).
6. **`results/benchmark/rankings_<config>.tsv`**: Rankings individuales Top-10 para cada una de las 6 configuraciones.
7. **`results/benchmark/combined_view_<config>.tsv`**: Vistas combinadas individuales para cada una de las 6 configuraciones.

---

## Resultados del Benchmark

Resumen de resultados obtenidos sobre las 15 consultas del benchmark (`queries.tsv` y `qrels.tsv`):

### Comparacion de configuraciones (Promedios globales)

| Representacion | N-gramas | P@10 | R@10 | F1@10 | MRR | nDCG@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TF-IDF** | **Unigramas** | **0.33** | **0.33** | **0.33** | **0.58** | **0.399** |
| Binaria | Unigramas | 0.25 | 0.25 | 0.25 | 0.49 | 0.316 |
| Frecuencia | Unigramas | 0.22 | 0.22 | 0.22 | 0.47 | 0.277 |
| TF-IDF | Bigramas | 0.07 | 0.07 | 0.07 | 0.19 | 0.067 |
| Frecuencia | Bigramas | 0.07 | 0.07 | 0.07 | 0.20 | 0.068 |
| Binaria | Bigramas | 0.07 | 0.07 | 0.07 | 0.15 | 0.063 |

> **Conclusion**: La configuracion **TF-IDF con unigramas** es la mas efectiva globalmente, ponderando negativamente terminos omnipresentes y priorizando terminos discriminativos. Las representaciones de bigramas sufren un problema severo de dispersion léxica (escasez de coocurrencias exactas entre la consulta y los fragmentos), generando vectores de consulta vacios en 8 de las 15 consultas evaluadas.

### Desempeno por nivel de dificultad (Mejor configuracion: TF-IDF Unigramas)

| Nivel de Dificultad | Consultas | P@10 | R@10 | F1@10 | MRR | nDCG@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fácil (Nivel 1)** | Q01 - Q05 | 0.38 | 0.38 | 0.38 | 0.75 | 0.455 |
| **Media (Nivel 2)** | Q06 - Q10 | 0.54 | 0.54 | 0.54 | 0.87 | 0.650 |
| **Difícil (Nivel 3)** | Q11 - Q15 | 0.08 | 0.08 | 0.08 | 0.12 | 0.093 |
| **Promedio Global** | Q01 - Q15 | **0.33** | **0.33** | **0.33** | **0.58** | **0.399** |

- En el **nivel medio**, las consultas poseen terminos concretos y tematicos claros (ej. nombres propios o conceptos especificos como *Karenin*, *Odiseo*, *pensadores*), alcanzando un MRR de 0.87 y nDCG@10 de 0.650.
- En el **nivel dificil**, las consultas utilizan lenguaje metaforico, abstracto o descriptivo sin coincidencia lexica directa con los terminos del pasaje, lo que evidencia la limitacion de los modelos de bolsa de palabras frente al vacio semantico (*vocabulary mismatch*).

---

## Reglas de normalizacion linguistica

El modulo `src/nlp.py` implementa las siguientes reglas sistematicas:

1. **Limpieza de caracteres**: Saltos de linea (`\n`, `\r`), tabulaciones (`\t`) y espacios contiguos se reemplazan por un unico espacio en blanco.
2. **Filtrado por categorias sintacticas (POS)**:
   - **Conservados**: Sustantivos (`NOUN`, `PROPN`), verbos y auxiliares (`VERB`, `AUX`), adjetivos (`ADJ`) y adverbios (`ADV`).
   - **Descartados**: Articulos (`DET`), preposiciones (`ADP`), conjunciones (`CCONJ`, `SCONJ`), pronombres (`PRON`), signos de puntuacion (`PUNCT`) y espacios (`SPACE`).
3. **Lematizacion y resolucion de cliticos**: Cada token aceptado se convierte a su lema en minusculas (`token.lemma_.lower()`). Si spaCy separa pronombres encliticos (por ejemplo, `meterse` -> `meter él`), se toma unicamente la raiz base (`meter`).
4. **Campos protegidos**: La columna `title` y la columna `keywords` se mantienen intactas sin lematizar ni podar para preservar la integridad de los metadatos bibliograficos.

---

## Etapa 2: Pasajes (Clustering y Modelado de Temas)

La carpeta `Passages/` extiende el proyecto con particion reproducible (`train` 357 / `dev` 72 / `test` 71, semilla `SEED = 3`), cinco normalizaciones (`N0`–`N4`), trece recetas de representacion vectorial y una interfaz comun de resultados en `Passages/TSV/Results/`.

### Comandos principales (Etapa 2)

```bash
# Verificar recetas y dimensiones por split
python Passages/representation.py

# Ejecutar experimentos de Agglomerative Clustering (A01-A31)
python Passages/agglomerative.py

# Consolidar los resultados de todos los algoritmos en Passages/TSV/Results/
python Passages/results.py
```

| Tarea | Script | Carpeta de salida | Estado |
| :--- | :--- | :--- | :--- |
| Base y vectorizacion | `Passages/representation.py`, `Passages/results.py` | `Passages/TSV/` | Listo |
| Agglomerative Clustering | `Passages/agglomerative.py` | `Passages/TSV/Results/agglomerative/` | Listo (`A23`) |
| K-Means | `Passages/kmeans.py` | `Passages/TSV/Results/kmeans/` | Listo (`K38`) |
| LDA | `Passages/lda.py` | `Passages/TSV/Results/lda/` | Listo (`L43`) |
| NMF | `Passages/nmf.py` | `Passages/TSV/Results/nmf/` | Listo (`N19`) |
| Interfaz CLI | `Passages/interfaz.py` | Consola (`pasaje`, `resumen`, `metricas`) | Listo |

---

## Referencias tecnicas

Para una revision exhaustiva de los disenos, contratos de datos y notas de desarrollo, consultar los documentos en `docs/`:

- [ARCHITECTURE.md](docs/ARCHITECTURE.md): Diagrama de arquitectura y flujo de datos modular.
- [CORPUS_SPEC.md](docs/CORPUS_SPEC.md): Esquema de datos y especificaciones tecnicas de las obras.
- [NLP_PIPELINE.md](docs/NLP_PIPELINE.md): Especificacion del pipeline linguistico.
- [SPACY_CONSIDERATIONS.md](docs/SPACY_CONSIDERATIONS.md): Analisis de sesgos y limitaciones lematizadoras.
- [AGENT_NOTES.md](docs/AGENT_NOTES.md): Bitacora de contexto tecnico y decisiones de ingenieria.
