# Notas para Agentes de IA (Guia de Contexto Rapido)

Lee este documento antes de realizar cambios o implementar modulos adicionales. Contiene el estado del proyecto, decisiones tecnicas tomadas y especificaciones para las siguientes fases.

---

## 1. Estado Actual del Repositorio

- **Fase 1 (Completada)**:
  - Archivo de salida: `literature_raw_corpus.tsv` (500 filas + 1 cabecera).
  - Modulo: `src/corpus.py`.
  - Generado a partir de 5 obras en `corpus/` (100 filas por obra).
  - Incluye 10 columnas oficiales: `fragment_id` (1 a 500), `book_id`, `title`, `author`, `chapter/position`, `text`, `language`, `source`, `url`, `keywords`.
- **Fase 2 (Implementada en codigo)**:
  - Modulo: `src/nlp.py`.
  - Pipeline de spaCy con filtrado estricto por POS y lematizacion (`es_core_news_lg`).
  - Archivo de salida esperado: `literature_normalized_corpus.tsv`.
- **Fase 3 (Preparada)**:
  - Funcion expuesta: `normalize_query(text: str) -> str` en `src/nlp.py` para asegurar igualdad de preprocesamiento entre documentos y consultas.

---

## 2. Reglas y Convenciones del Usuario

- **Sin emojis**: El usuario solicito explicitamente no utilizar emojis en la documentacion ni en las respuestas.
- **Sin Makefiles**: No agregar Makefiles en la raiz ni sugerir `make`. La interfaz oficial es el CLI de Python `main.py`.
- **Formato TSV y UTF-8**: Todos los datasets generados (`literature_raw_corpus.tsv`, `literature_normalized_corpus.tsv`, `queries.tsv`, `qrels.tsv`, `rankings.tsv`) deben ser archivos delimitados por tabuladores (`\t`) en codificacion UTF-8 estandar.
- **Columna `title` protegida**: La especificacion exige no lematizar ni podar el titulo; debe conservarse identico al original.
- **Columna `keywords` preservada**: Las etiquetas de palabras clave deben conservarse intactas en los archivos TSV.

---

## 3. Discrepancias Tecnicas Resueltas en el Corpus

- `corpus/1984/corpus_1984.csv` contiene BOM (`\ufeff`) en la primera columna (`fragment_id`). Usar siempre `utf-8-sig` para abrirlo.
- `corpus/MiLucha/corpus_301_400.tsv` utiliza tabuladores (`\t`), mientras que los otros 4 libros usan comas (`,`).
- Los IDs originales de cada obra ya estaban asignados por rangos de 100 en 100:
  - Levedad: 1 a 100
  - La Odisea: 101 a 200
  - El mundo de Sofia: 201 a 300
  - Mi Lucha: 301 a 400
  - 1984: 401 a 500
- El consolidado simplemente conserva su `fragment_id` original y los ordena de menor a mayor (1 a 500).
- La columna no estandar `context` (presente solo en 1984) se descarta.

---

## 4. Especificacion de las Siguientes Fases (del PDF 04_Literature_Retrieval_EN.pdf)

Cuando el usuario pida implementar las siguientes fases, consultar estas directrices:

### Modelo de Espacio Vectorial (VSM) y Representaciones (Seccion 7)
- Crear modulo (ej. `src/vsm.py`).
- Implementar con `scikit-learn` tres tipos de representacion:
  1. Frecuencia (CountVectorizer)
  2. Binaria (CountVectorizer con `binary=True`)
  3. TF-IDF (TfidfVectorizer)
- Experimentar con caracteristicas de unigramas (`ngram_range=(1, 1)`) y bigramas (`ngram_range=(2, 2)` o `(1, 2)`).
- Permitir vectorizacion sobre `text` y opcionalmente sobre `title`.
- Guardar matrices y vectorizadores en archivos `.pkl` si es necesario.

### Motor de Recuperacion (Seccion 8)
- Entrada: Consulta libre en lenguaje natural (ej. "description of a storm").
- Preprocesamiento: Aplicar obligatoriamente `normalize_query()` de `src/nlp.py`.
- Representar consulta en el espacio vectorial seleccionado.
- Calcular similitud coseno (`sklearn.metrics.pairwise.cosine_similarity`).
- Ordenar descendentemente y retornar el Top-10 con su score.

### Benchmark y Evaluacion (Secciones 9 y 10)
- **Archivos requeridos**:
  - `queries.tsv`: 15 consultas (5 faciles, 5 medias, 5 dificiles) con formato:
    `query_id \t difficulty_level \t query`
  - `qrels.tsv`: 10 a 15 documentos relevantes por consulta, con escala de 3 puntos:
    - 0: No relevante (se excluye de qrels)
    - 1: Parcialmente relevante
    - 2: Altamente relevante
    Formato: `query_id \t document_id \t relevance`
  - `rankings.tsv`: Top-10 predicho para cada consulta con formato:
    `query_id \t document_id \t rank \t score`
  - Columna `title` protegida: no lematizar ni podar.
  - Columna `keywords` protegida: mantener intacta.
  - Limitaciones y sesgos de spaCy (ej. `olía` -> `oliar`, cliticos como `hallar él`): documentados en `docs/SPACY_CONSIDERATIONS.md` para ser citados en el analisis de errores de la Seccion 12.
- **Metricas de evaluacion obligatorias (Top-10)**:
  - Precision@10 (P@10)
  - Recall@10 (R@10)
  - F1@10
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (nDCG@10) (ver formulas en la pagina 8 del PDF).
