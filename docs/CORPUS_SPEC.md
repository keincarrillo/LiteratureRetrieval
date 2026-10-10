# Especificacion del Corpus de Datos

Este documento define la estructura, el esquema de datos y los detalles tecnicos del corpus para evitar tener que inspeccionar los archivos fuente manualmente.

---

## 1. Obras Literarias y Distribucion

El corpus contiene un total de 500 fragmentos (100 fragmentos por obra) distribuidos en:

| Directorio | Obra | Autor | Rango fragment_id | Formato Origen | Separador |
|---|---|---|---|---|---|
| `corpus/Levedad/` | La insoportable levedad del ser | Milan Kundera | 1 - 100 | `corpus_kundera.csv` | Coma (`,`) |
| `corpus/Odisea/` | La Odisea | Homero | 101 - 200 | `corpus_odisea.csv` | Coma (`,`) |
| `corpus/ElMundoDeSofia/` | El mundo de Sofia | Jostein Gaarder | 201 - 300 | `corpus_gaarder.csv` | Coma (`,`) |
| `corpus/MiLucha/` | Mi Lucha | Adolf Hitler | 301 - 400 | `corpus_301_400.tsv` | Tabulador (`\t`) |
| `corpus/1984/` | 1984 | George Orwell | 401 - 500 | `corpus_1984.csv` | Coma (`,`), con BOM UTF-8 |

---

## 2. Esquema Oficial de Columnas (TSV)

Todo archivo de corpus unificado (`literature_raw_corpus.tsv` y `literature_normalized_corpus.tsv`) respeta estrictamente 10 columnas en este orden:

1. `fragment_id`: Identificador numerico entero unico y secuencial global (de 1 a 500).
2. `book_id`: Identificador textual de la obra (ej. `orwell_1984`, `kundera_levedad`).
3. `title`: Titulo de la obra o capitulo.
4. `author`: Autor de la obra.
5. `chapter/position`: Ubicacion del fragmento (capitulo, libro o rango de paginas).
6. `text`: Texto del fragmento literario (crudo o normalizado segun la fase).
7. `language`: Codigo de idioma ISO (`es`).
8. `source`: Fuente documental de origen (ej. `Internet Archive`, `Lectulandia`).
9. `url`: Enlace original a la fuente digital.
10. `keywords`: Etiquetas tematicas y motivos literarios extraidos en la curacion.

---

## 3. Reglas Criticas de Ingestion (Fase 1)

Al procesar o consolidar los datos de entrada:

- **Codificacion**: Siempre usar `encoding='utf-8-sig'` al leer para eliminar de forma transparente el BOM (`\ufeff`) presente en `corpus_1984.csv`.
- **Deteccion de delimitador**: Determinar por extension (`\t` para `.tsv`, `,` para `.csv`).
- **Columnas no estandar descartadas**: Columnas como `context` (presente solo en 1984) se descartan totalmente. La columna `keywords` se conserva.
- **Reasignacion de IDs**: Los identificadores locales de cada archivo (que inician en 1) se reemplazan obligatoriamente por un ID global secuencial del 1 al 500.
- **Formato de salida**: Archivo separado por tabulaciones (`\t`), codificacion UTF-8 estandar, sin indice numerico de pandas (`index=False`).
