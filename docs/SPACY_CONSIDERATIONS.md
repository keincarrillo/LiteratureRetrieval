# Consideraciones y Limitaciones de spaCy en Espanol

Este documento recopila discrepancias linguisticas, comportamientos del modelo `es_core_news_md` y fenomenos observados durante la normalizacion del corpus. Sirve como evidencia y justificacion tecnica para el analisis de errores (Seccion 12 de la practica).

---

## 1. Sobre-regularizacion en Verbos Irregulares

### Caso detectado: Verbo "oler" (`olía` -> `oliar` vs `oler`)
- **Texto original**: *"El vestibulo olia a legumbres cocidas..."* (1984, Fragmento 401).
- **Comportamiento en `es_core_news_md`**:
  - Predice morfologia: `Mood=Ind|Number=Sing|Person=3|Tense=Pres` (Presente).
  - Regla heuristica: Como cree que termina en `-a` de presente, asume un verbo en `-iar` y produce el lema erroneo `oliar`.
- **Comportamiento en el modelo grande `es_core_news_lg`**:
  - Predice morfologia: `Mood=Ind|Number=Sing|Person=3|Tense=Imp` (Imperfecto).
  - Asigna el lema correcto: **`oler`**.
- **Conclusion**: El modelo `es_core_news_lg` cuenta con un modelo morfologico mas preciso que resuelve esta anomalia. Se configuro `src/nlp.py` para priorizar `es_core_news_lg` automaticamente cuando este disponible.

---

## 2. Enclisis Pronominal en Verbos

### Caso detectado: Verbos con pronombres pegados (`hallarse` -> `hallar él`)
- **Texto original**: *"para hallarse en un interior..."* (1984, Fragmento 1).
- **Resultado en spaCy**: El verbo genera el lema compuesto `hallar él`.
- **Causa**: El tokenizador / lematizador de spaCy descompone formas encliticas en su verbo base y el pronombre subyacente. Aunque la regla de POS descarta pronombres (`PRON`), el lematizador del verbo conserva la descomposicion como una cadena interna.
- **Impacto en recuperacion**: Introduce el termino `él` junto al verbo en el texto normalizado.

---

## 3. Ambigüedad de Participios (ADJ vs VERB)

- Formas verbales en participio como `cocidas`, `pegado` o `cerrados` pueden recibir alternativamente la etiqueta `ADJ` o `VERB` segun la estructura sintactica inmediata.
- En nuestro pipeline, ambas categorias (`ADJ` y `VERB`) se conservan intencionalmente, por lo que el termino no se pierde, aunque su lema puede variar entre la forma masculina singular del adjetivo (`cocido`) o el infinitivo del verbo (`cocer`) segun la prediccion del modelo.

---

## 4. Registro Linguistico y Arcaismos

- El modelo `es_core_news_md` fue entrenado con texto contemporaneo (noticias y web).
- En obras con traducciones clasicas, poeticas o arcaicas (particularmente *La Odisea*), construcciones como *"varon de multiforme ingenio"* o vocabulario helenico pueden recibir clasificaciones morfosintacticas imprecisas frente a textos narrativos modernos como *1984* o *El mundo de Sofia*.

---

## 5. Aplicacion en el Analisis de Errores (Seccion 12)

La especificacion de la practica solicita identificar causas de bajo desempeno en el benchmark de consultas. Los fenomenos documentados aqui corresponden directamente a las causas senaladas en la guia:
- Normalizacion inadecuada (inadequate normalization).
- Falta de solapamiento lexico por desalineacion de lemas (lack of lexical overlap).
- Vocabulario de dominio especifico o de epoca (domain-specific vocabulary).
