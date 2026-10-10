# Pipeline de Normalizacion Linguistica (spaCy)

Este documento detalla las reglas de transformacion de texto aplicadas tanto a los documentos literarios (Fase 2) como a las consultas de busqueda de los usuarios (Fase 3).

---

## 1. Reglas de Normalizacion de Texto

Toda cadena de texto analizada por el sistema pasa por tres etapas secuenciales:

### Paso 1: Limpieza de caracteres y espacios
- Reemplazo de caracteres de control de nueva linea (`\n`, `\r`) y tabulaciones (`\t`) por un unico espacio en blanco (`' '`).
- Colapso de espacios multiples contiguos a un solo espacio.
- Eliminacion de espacios al inicio y al final (`strip()`).

### Paso 2: Filtrado por categorias sintacticas (Universal POS Tags)
Se utiliza el modelo en espanol de spaCy (`es_core_news_md` con fallback a `es_core_news_sm`).

- **Categorias descartadas (stopwords sintacticas y ruido)**:
  - Articulos y determinantes: `DET`
  - Preposiciones: `ADP`
  - Conjunciones coordinantes y subordinantes: `CCONJ`, `SCONJ`
  - Pronombres: `PRON`
  - Signos de puntuacion: `PUNCT`
  - Espacios: `SPACE`

- **Categorias conservadas (terminos de contenido)**:
  - Sustantivos: `NOUN`
  - Nombres propios: `PROPN`
  - Verbos: `VERB`
  - Verbos auxiliares: `AUX`
  - Adjetivos: `ADJ`
  - Adverbios: `ADV`

### Paso 3: Lematizacion y minusculas
- Para cada token aceptado, se extrae su lema base (`token.lemma_`).
- El lema se convierte a minusculas estrictas (`token.lemma_.strip().lower()`).
- Los lemas resultantes se unen mediante un unico espacio en blanco.

---

## 2. Politica de Columnas en el Dataset Normalizado

- **Columna `text`**: Es la unica columna sobre la cual se aplica el filtrado POS y la lematizacion.
- **Columna `title`**: Se mantiene estrictamente con su texto original sin podar ni lematizar (requisito explicito de la practica).
- **Columna `keywords`**: Se mantiene con su texto original.
- **Resto de columnas**: Conservan sus metadatos sin modificacion.

---

## 3. Tokenizacion y Normalizacion de Consultas (Fase 3)

La funcion `tokenize_and_normalize(text: str) -> List[str]` en `src/nlp.py` expone explicitamente la lista de tokens resultantes:
- Descompone la cadena en tokens mediante el tokenizador de spaCy.
- Aplica el filtrado sintactico descartando categorias irrelevantes.
- Retorna la lista de lemas validos (ejemplo: `['descripcion', 'tormenta', 'violento', 'alto', 'mar']`).

La funcion `normalize_query(text: str) -> str` une dichos tokens con espacios (`' '.join(...)`) para asegurar igualdad de formato con el corpus y facilitar la alimentacion a los vectorizadores (CountVectorizer, TfidfVectorizer) de scikit-learn.
