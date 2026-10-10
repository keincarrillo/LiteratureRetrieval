# Arquitectura y Diseno Modular del Sistema

Este documento describe la arquitectura modular del software y el flujo de trabajo entre componentes.

---

## 1. Diagrama de Flujo de Datos

```
corpus/ (archivos .csv / .tsv individuales)
     │
     ▼  [src/corpus.py: build_raw_corpus()]
raw_corpus.tsv (500 documentos, 10 columnas)
     │
     ▼  [src/nlp.py: normalize_corpus()]
normalized_corpus.tsv (texto lematizado y filtrado por POS)
     │
     ▼  [Modulo VSM / scikit-learn (Fase posterior)]
Modelos Vectoriales (Frecuencia, Binario, TF-IDF / Unigramas, Bigramas)
     │
     ▼  [Modulo de Recuperacion y Evaluacion]
Rankings Top-10 (similitud coseno) y Metricas (P@10, R@10, F1@10, MRR, nDCG@10)
```

---

## 2. Responsabilidades por Modulo

### `src/corpus.py` (Fase 1)
- **Proposito**: Ingestion, unificacion y limpieza estructural del corpus literario.
- **Funciones clave**:
  - `build_raw_corpus(corpus_dir, output_path)`: Escanea recursivamente archivos de corpus, elimina columnas ajenas, respeta identificadores de 1 a 500 y escribe `raw_corpus.tsv`.
- **Dependencias**: Solo libreria estandar de Python (`csv`, `glob`, `pathlib`, `re`).

### `src/nlp.py` (Fases 2 y 3)
- **Proposito**: Procesamiento de lenguaje natural mediante spaCy.
- **Funciones clave**:
  - `clean_text(text: str) -> str`: Normaliza espacios y caracteres de control.
  - `tokenize_and_normalize(text: str) -> List[str]`: Filtra stopwords por POS y extrae lemas en minusculas.
  - `normalize_query(text: str) -> str`: Normaliza una consulta individual para busqueda.
  - `normalize_corpus(raw_path, output_path)`: Normaliza el corpus en bloque utilizando `nlp.pipe()` para maxima velocidad y genera `normalized_corpus.tsv`.
- **Dependencias**: `spacy`, `pathlib`, `csv`, `re`.

### `main.py`
- **Proposito**: Punto de entrada de linea de comandos (CLI) para ejecutar tareas individuales o el flujo completo sin requerir comandos de compilacion adicionales como Make.
- **Banderas soportadas**:
  - `--phase1`: Ejecuta unicamente la consolidacion del corpus crudo.
  - `--phase2`: Ejecuta unicamente la normalizacion con spaCy.
  - `--all`: Ejecuta ambas fases de forma encadenada.
  - `--query "<texto>"`: Ejecuta una prueba puntual de normalizacion sobre una consulta dada.
