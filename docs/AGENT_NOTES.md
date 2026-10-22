# Notas para Agentes de IA (Guia de Contexto Rapido)

Lee este documento antes de realizar cambios o implementar modificaciones adicionales en el repositorio. Contiene el estado del proyecto, decisiones tecnicas de ingenieria, contratos de datos y convenciones obligatorias.

---

## 1. Estado Actual del Repositorio (Fases 1 a 6 Completadas)

- **Fase 1 (Consolidacion del Corpus Crudo)**:
  - Modulo: `src/corpus.py` (`build_raw_corpus`).
  - Archivo de salida: `literature_raw_corpus.tsv` (500 documentos + cabecera).
  - Consolida 5 obras (100 fragmentos por obra, ~500 palabras por fragmento).
  - Esquema oficial de 10 columnas: `fragment_id` (1 a 500), `book_id`, `title`, `author`, `chapter/position`, `text`, `language`, `source`, `url`, `keywords`.

- **Fase 2 (Normalizacion Linguistica con spaCy)**:
  - Modulo: `src/nlp.py` (`normalize_corpus`).
  - Archivo de salida: `literature_normalized_corpus.tsv`.
  - Pipeline spaCy con filtrado estricto por POS tags, lematizacion en minusculas y desacoplamiento de pronombres encliticos. Conserva `title` y `keywords` sin alterar.

- **Fase 3 (Normalizacion de Consultas Libres)**:
  - Modulo: `src/nlp.py` (`normalize_query`, `tokenize_and_normalize`).
  - Asegura identidad de preprocesamiento entre documentos y consultas.
  - Probable desde CLI: `python main.py --query "<texto>"`.

- **Fase 4 (Representaciones VSM)**:
  - Modulo: `src/vsm.py` (`build_all`, `build_vectorizer`, `load_model`).
  - Genera 6 modelos serializados en `models/`:
    - `frequency_unigram.pkl`, `frequency_bigram.pkl`
    - `binary_unigram.pkl`, `binary_bigram.pkl`
    - `tfidf_unigram.pkl`, `tfidf_bigram.pkl`
  - Herramienta de analisis: `tools/ver_pkl.py` exporta ternas a CSV y matrices completas término-documento a `models/csv/`.

- **Fase 5 (Motor de Recuperacion de Informacion)**:
  - Modulo: `src/retrieval.py` (`retrieve`, `print_results`, `save_results_txt`).
  - Calcula similitud coseno (`sklearn.metrics.pairwise.cosine_similarity`) sobre la matriz del modelo seleccionado.
  - Despliega ranking Top-10 en consola y genera automaticamente un archivo de texto exhaustivo con fragmentos completos y metadatos en `results/<timestamp>_<slug>_<repr>_<feat>.txt`.

- **Fase 6 (Benchmark y Evaluacion de Metricas IR)**:
  - Modulos: `src/benchmark.py` y `main.py --benchmark`.
  - Entradas: `queries.tsv` (15 consultas etiquetadas por dificultad 1, 2, 3) y `qrels.tsv` (150 juicios con relevancias 1 y 2).
  - Metricas evaluadas sobre Top-10: P@10, R@10, F1@10, MRR, nDCG@10.
  - Salidas oficiales en raiz: `rankings.tsv` y `combined_view.tsv` (generados con la mejor configuracion `tfidf_unigram`).
  - Salidas analiticas en `results/benchmark/`:
    - `metrics_by_config.tsv` (Tabla 9 comparativa global de las 6 configuraciones).
    - `metrics_by_query.tsv` (Tabla 10 con desglose por consulta individual).
    - `metrics_by_level.tsv` (Tabla 11 agrupada por nivel de dificultad).
    - `rankings_<cfg>.tsv` y `combined_view_<cfg>.tsv` para cada configuracion.

---

## 2. Reglas y Convenciones del Usuario

- **Sin emojis**: El usuario solicito explicitamente no utilizar emojis en la documentacion, codigo ni en las respuestas de chat.
- **Sin Makefiles**: No crear Makefiles en la raiz ni sugerir `make`. La interfaz unificada del proyecto es el CLI en Python `main.py` y ejecuciones modulares `python -m src.<modulo>`.
- **Formato TSV y UTF-8**: Todos los datasets generados (`literature_raw_corpus.tsv`, `literature_normalized_corpus.tsv`, `queries.tsv`, `qrels.tsv`, `rankings.tsv`, `combined_view.tsv`) deben ser archivos delimitados por tabuladores (`\t`) en codificacion UTF-8 estandar.
- **Columna `title` protegida**: La especificacion exige no lematizar ni podar el titulo; debe conservarse identico al original en todas las fases.
- **Columna `keywords` preservada**: Las etiquetas de palabras clave deben conservarse intactas en los archivos TSV.

---

## 3. Discrepancias Tecnicas Resueltas en el Corpus

- `corpus/1984/corpus_1984.csv` contiene BOM (`\ufeff`) en la primera columna (`fragment_id`). Usar siempre `utf-8-sig` para abrirlo.
- `corpus/MiLucha/corpus_301_400.tsv` utiliza tabuladores (`\t`), mientras que los otros 4 libros usan comas (`,`).
- Los IDs originales de cada obra ya estaban asignados por rangos de 100 en 100:
  - Levedad (Kundera): 1 a 100
  - La Odisea (Homero): 101 a 200
  - El mundo de Sofia (Gaarder): 201 a 300
  - Mi Lucha (Hitler): 301 a 400
  - 1984 (Orwell): 401 a 500
- El consolidado simplemente conserva su `fragment_id` original y los ordena de menor a mayor (1 a 500).
- La columna no estandar `context` (presente solo en 1984) se descarta.

---

## 4. Decisiones de Diseno e Implementacion en PLN e IR

### Preprocesamiento y spaCy (`src/nlp.py`)
- **Categorias POS conservadas**: `NOUN`, `PROPN`, `VERB`, `AUX`, `ADJ`, `ADV`.
- **Categorias descartadas**: Articulos (`DET`), preposiciones (`ADP`), conjunciones (`CCONJ`, `SCONJ`), pronombres (`PRON`), puntuacion (`PUNCT`), espacios (`SPACE`).
- **Cliticos pronominales**: Al lematizar formas con pronombres encliticos (ej. `meterse` -> `meter él`), se aplica `token.lemma_.lower().split()[0]` para conservar unicamente la base verbal (`meter`).
- **Carga tolerante de modelos**: Busca en orden `es_core_news_lg` -> `es_core_news_md` -> `es_core_news_sm`. Lanza excepcion clara si ningun modelo esta presente.

### Representaciones VSM (`src/vsm.py`)
- Token pattern por defecto de scikit-learn (`r"(?u)\b\w\w+\b"`): descarta simbolos sueltos y caracteres aislados de una letra.
- Frecuencia y Binaria se modelan mediante `CountVectorizer` (con `binary=False` y `binary=True` respectivamente).
- TF-IDF se modela mediante `TfidfVectorizer` (sublineal estandar con suavizado laplaciano de scikit-learn).
- Unigramas usan `ngram_range=(1, 1)`; Bigramas usan `ngram_range=(2, 2)`.

### Evaluacion del Benchmark (`src/benchmark.py`)
- **Penalizacion estricta en P@10**: Si el ranking recupera menos de 10 documentos o no recupera ninguno, el denominador siempre es `TOP_K = 10`.
- **Empates en score 0**: Si no hay coincidencias de terminos entre la consulta y el corpus, la similitud coseno da 0. Los 10 primeros se ordenan de forma estable por orden en el corpus; son contabilizados y reportados en `metrics_by_query.tsv` como `zero_score_hits` y `empty_query_vector`.
- **Criterio de mejor configuracion**: Aquella con mayor `ndcg10`; en caso de empate, la de mayor `f1_10`.
- **Hallazgo experimental clave**: `tfidf_unigram` es la configuracion optima global (nDCG@10 = 0.3994, MRR = 0.5796). Los modelos basados en bigramas sufren un colapso severo de exhaustividad (8 de las 15 consultas del benchmark producen vectores nulos por falta de pares de palabras contiguas identicas).

---

## 5. Comandos CLI Principales de Referencia Rapida

```bash
# Ingestion y normalizacion
python main.py --phase1          # Genera literature_raw_corpus.tsv
python main.py --phase2          # Genera literature_normalized_corpus.tsv
python main.py --all             # Fases 1 y 2 consecutivas
python main.py --query "<texto>" # Prueba de normalizacion de consulta

# Entrenamiento de modelos vectoriales (Fase 4)
python -m src.vsm                # Entrena las 6 configuraciones en models/*.pkl
python tools/ver_pkl.py models   # Inspecciona matrices y exporta a CSV

# Busqueda y recuperacion (Fase 5)
python -m src.retrieval -q "<consulta>" -r tfidf -f unigram

# Evaluacion del Benchmark (Fase 6)
python main.py --benchmark       # Ejecuta evaluacion completa y genera reportes
```
