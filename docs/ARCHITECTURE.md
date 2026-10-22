# Arquitectura y Diseno Modular del Sistema

Este documento describe la arquitectura modular del software y el flujo de trabajo entre componentes.

---

## 1. Diagrama de Flujo de Datos

```
corpus/ (archivos .csv / .tsv individuales)
     │
     ▼  [src/corpus.py: build_raw_corpus()]
literature_raw_corpus.tsv (500 documentos, 10 columnas)
     │
     ▼  [src/nlp.py: normalize_corpus()]
literature_normalized_corpus.tsv (texto lematizado y filtrado por POS)
     │
     ▼  [src/vsm.py: build_all()]
models/*.pkl (6 configuraciones VSM: Frecuencia, Binario, TF-IDF x Unigramas, Bigramas)
     │
     ├───────────────────────────────────────────────┐
     ▼  [src/retrieval.py: retrieve()]               ▼  [src/benchmark.py: run_benchmark()]
results/*.txt (consultas individuales Top-10)       rankings.tsv, combined_view.tsv
                                                    results/benchmark/ (tablas 9, 10 y 11)
```

---

## 2. Responsabilidades por Modulo

### `src/corpus.py` (Fase 1)
- **Proposito**: Ingestion, unificacion y limpieza estructural del corpus literario.
- **Funciones clave**:
  - `build_raw_corpus(corpus_dir, output_path)`: Escanea recursivamente archivos de corpus, elimina columnas ajenas, respeta identificadores de 1 a 500 y escribe `literature_raw_corpus.tsv`.
- **Dependencias**: Solo libreria estandar de Python (`csv`, `glob`, `pathlib`, `re`).

### `src/nlp.py` (Fases 2 y 3)
- **Proposito**: Procesamiento de lenguaje natural mediante spaCy.
- **Funciones clave**:
  - `clean_text(text: str) -> str`: Normaliza espacios y caracteres de control.
  - `tokenize_and_normalize(text: str) -> List[str]`: Filtra stopwords por POS y extrae lemas en minusculas.
  - `normalize_query(text: str) -> str`: Normaliza una consulta individual para busqueda.
  - `normalize_corpus(raw_path, output_path)`: Normaliza el corpus en bloque utilizando `nlp.pipe()` para maxima velocidad y genera `literature_normalized_corpus.tsv`.
- **Dependencias**: `spacy`, `pathlib`, `csv`, `re`.

### `src/vsm.py` (Fase 4)
- **Proposito**: Construccion y serializacion de representaciones en el Modelo de Espacio Vectorial (VSM).
- **Funciones clave**:
  - `build_vectorizer(representation, feature)`: Configura vectorizadores de scikit-learn (`CountVectorizer` o `TfidfVectorizer` para unigramas o bigramas).
  - `build_all(normalized_path, out_dir)`: Entrena y almacena las 6 combinaciones en archivos `.pkl` dentro de `models/`.
  - `load_model(representation, feature, out_dir)`: Carga vectorizador, matriz y mapeo de IDs.
- **Dependencias**: `scikit-learn`, `joblib`, `csv`, `pathlib`.

### `src/retrieval.py` (Fase 5)
- **Proposito**: Motor de busqueda sobre el corpus mediante similitud coseno.
- **Funciones clave**:
  - `retrieve(query, representation, feature)`: Normaliza la consulta, la proyecta en el espacio vectorial correspondiente y obtiene los 10 fragmentos mas similares.
  - `print_results(...)`: Despliega el ranking formateado en consola.
  - `save_results_txt(...)`: Exporta el resultado completo con fragmentos y metadatos a `results/`.
- **Dependencias**: `scikit-learn`, `numpy`, `csv`, `pathlib`.

### `src/benchmark.py` (Fase 6)
- **Proposito**: Evaluacion sistematica de las 6 configuraciones sobre el conjunto de consultas y juicios ground-truth.
- **Funciones clave**:
  - `evaluate_ranking(top_docs, qrel)`: Calcula P@10, R@10, F1@10, RR y nDCG@10.
  - `run_benchmark(...)`: Ejecuta la comparativa global, detecta la mejor configuracion (`tfidf_unigram`), genera `rankings.tsv`, `combined_view.tsv` y almacena las tablas de metricas en `results/benchmark/`.
- **Dependencias**: `scikit-learn`, `numpy`, `csv`, `pathlib`, `math`.

### `main.py`
- **Proposito**: Punto de entrada de linea de comandos (CLI) para ejecutar tareas individuales o el flujo completo sin requerir comandos de compilacion adicionales como Make.
- **Banderas soportadas**:
  - `--phase1`: Ejecuta unicamente la consolidacion del corpus crudo.
  - `--phase2`: Ejecuta unicamente la normalizacion con spaCy.
  - `--all`: Ejecuta ambas fases de forma encadenada.
  - `--query "<texto>"`: Ejecuta una prueba puntual de normalizacion sobre una consulta dada.
  - `--benchmark`: Ejecuta la evaluacion completa del benchmark y actualiza rankings y metricas.
