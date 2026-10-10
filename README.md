# LiteratureRetrieval

Sistema de recuperacion de informacion literaria basado en el Modelo de Espacio Vectorial (VSM) y similitud coseno para la practica de Procesamiento de Lenguaje Natural.

El objetivo del proyecto es permitir la busqueda de pasajes literarios relevantes a partir de consultas en lenguaje natural sobre un corpus de 500 fragmentos distribuidos en 5 obras literarias:
- 1984 (George Orwell)
- El mundo de Sofia (Jostein Gaarder)
- La insoportable levedad del ser (Milan Kundera)
- Mi Lucha (Adolf Hitler)
- La Odisea (Homero)

---

## Estructura del proyecto

- `corpus/`: Directorios con los archivos fuente (.csv y .tsv) de cada una de las obras literarias.
- `src/`: Codigo fuente modular.
  - `src/corpus.py`: Modulo de la Fase 1. Consolida archivos con delimitadores mixtos, estandariza columnas y reasigna identificadores secuenciales (1 a 500).
  - `src/nlp.py`: Modulo de las Fases 2 y 3. Limpieza de texto, tokenizacion, filtrado gramatical por POS, lematizacion con spaCy y normalizacion de consultas.
- `main.py`: Interfaz de linea de comandos (CLI) para ejecutar las distintas etapas del sistema.
- `requirements.txt`: Dependencias del proyecto (pandas, spacy, scikit-learn, numpy).
- `literature_raw_corpus.tsv`: Corpus crudo unificado generado en la Fase 1.
- `literature_normalized_corpus.tsv`: Corpus procesado linguisticamente generado en la Fase 2.

---

## Requisitos e instalacion

Se requiere Python 3.9 o superior.

1. Crear y activar el entorno virtual:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Descargar el modelo en espanol de spaCy:
```bash
python -m spacy download es_core_news_md
```

---

## Modo de uso

El flujo se controla a traves de `main.py`:

### 1. Consolidar el corpus crudo (Fase 1)
Busca recursivamente todos los fragmentos en `corpus/`, unifica los separadores, asigna IDs continuos del 1 al 500, conserva las columnas requeridas (incluyendo keywords) y genera `literature_raw_corpus.tsv`:
```bash
python main.py --phase1
```

### 2. Normalizar el corpus con spaCy (Fase 2)
Toma `literature_raw_corpus.tsv`, procesa el campo de texto con el pipeline de spaCy (`nlp.pipe()`), elimina stopwords segun categorias sintacticas (articulos, preposiciones, conjunciones, pronombres), descarta signos de puntuacion, conserva categorias de contenido (sustantivos, verbos, adjetivos, adverbios), lematiza en minusculas y exporta `literature_normalized_corpus.tsv`:
```bash
python main.py --phase2
```

### 3. Ejecutar ambas fases de forma consecutiva
```bash
python main.py --all
```

### 4. Probar la normalizacion de una consulta libre (Fase 3)
Aplica exactamente el mismo pipeline linguisticamente a una consulta de texto libre:
```bash
python main.py --query "descripcion de una tormenta violenta en alta mar"
```

---

## Reglas de normalizacion linguistica

1. **Limpieza de caracteres**: Saltos de linea (`\n`, `\r`), tabulaciones (`\t`) y espacios multiples contiguos se reemplazan por un solo espacio.
2. **Filtrado por categorias sintacticas (POS)**:
   - Descartados: Articulos (DET), preposiciones (ADP), conjunciones (CCONJ, SCONJ), pronombres (PRON), signos de puntuacion (PUNCT) y espacios (SPACE).
   - Conservados: Sustantivos (NOUN, PROPN), verbos y auxiliares (VERB, AUX), adjetivos (ADJ) y adverbios (ADV).
3. **Lematizacion**: Cada token conservado se transforma a su lema en minusculas (`token.lemma_.lower()`).
4. **Campos protegidos**: La columna `title` y la columna `keywords` se mantienen con su texto original sin podar ni lematizar.
