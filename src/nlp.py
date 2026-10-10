"""Módulo de normalización lingüística con spaCy (Fases 2 y 3).

Aplica limpieza de caracteres, tokenización, filtrado por categorías POS
y lematización en español según las especificaciones de la práctica.
"""

import csv
import re
from pathlib import Path
from typing import List, Optional, Set, Union

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import spacy
    from spacy.language import Language
except ImportError:
    spacy = None
    Language = None

# Categorías sintácticas (Universal POS Tags)
# Descartadas explícitamente:
# DET (artículos/determinantes), ADP (preposiciones), CCONJ (conjunciones coordinantes),
# SCONJ (conjunciones subordinantes), PRON (pronombres), PUNCT (puntuación), SPACE (espacios).
DISCARDED_POS: Set[str] = {"DET", "ADP", "CCONJ", "SCONJ", "PRON", "PUNCT", "SPACE"}

# Categorías a conservar explícitamente:
# NOUN (sustantivos), PROPN (nombres propios), VERB (verbos), AUX (verbos auxiliares),
# ADJ (adjetivos), ADV (adverbios).
KEPT_POS: Set[str] = {"NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV"}

_CACHED_NLP: Optional[Language] = None


def get_spacy_pipeline(
    model_name: str = "es_core_news_md",
    fallback_model: str = "es_core_news_sm",
    disable_ner: bool = True,
) -> "Language":
    """Carga o reutiliza el pipeline de spaCy en español.

    Intenta cargar el modelo preferido (es_core_news_md) y recurre a fallback si no está presente.
    """
    global _CACHED_NLP
    if _CACHED_NLP is not None:
        return _CACHED_NLP

    if spacy is None:
        raise ImportError(
            "spaCy no está instalado en el entorno. "
            "Por favor, instala los requisitos ejecutando: pip install -r requirements.txt"
        )

    # Lista de componentes opcionales a desactivar para optimizar rendimiento
    disable = ["ner"] if disable_ner else []

    try:
        _CACHED_NLP = spacy.load(model_name, disable=disable)
        print(f"[NLP] Modelo spaCy cargado exitosamente: {model_name}")
    except OSError:
        try:
            print(f"[NLP] Modelo '{model_name}' no encontrado. Intentando fallback: '{fallback_model}'...")
            _CACHED_NLP = spacy.load(fallback_model, disable=disable)
            print(f"[NLP] Modelo spaCy cargado exitosamente: {fallback_model}")
        except OSError:
            raise OSError(
                f"No se encontró ni '{model_name}' ni '{fallback_model}'. "
                f"Descárgalo ejecutando: python -m spacy download {model_name} "
                f"o python -m spacy download {fallback_model}"
            )

    return _CACHED_NLP


def clean_whitespace(text: str) -> str:
    """Reemplaza saltos de línea (\\n, \\r), tabulaciones (\\t) y múltiples

    espacios contiguos por un único espacio en blanco.
    """
    if not text:
        return ""
    # Reemplazar saltos de línea y tabuladores por espacios
    text = re.sub(r"[\r\n\t]+", " ", text)
    # Colapsar múltiples espacios continuos
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_and_normalize(text: str, nlp: Optional["Language"] = None) -> List[str]:
    """Tokeniza el texto usando el pipeline de spaCy, filtra stopwords

    segun categorias sintacticas POS y retorna la lista de tokens lematizados.
    """
    if nlp is None:
        nlp = get_spacy_pipeline()
    cleaned = clean_whitespace(text)
    doc = nlp(cleaned)
    return [
        token.lemma_.strip().lower()
        for token in doc
        if token.pos_ in KEPT_POS
        and token.pos_ not in DISCARDED_POS
        and not token.is_punct
        and not token.is_space
        and token.lemma_.strip()
    ]


def normalize_doc(doc) -> str:
    """Extrae los lemas en minusculas de los tokens que pertenecen a las

    categorias POS conservadas (NOUN, PROPN, VERB, AUX, ADJ, ADV), descartando
    articulos, preposiciones, conjunciones, pronombres, puntuacion y espacios.
    """
    lemmas: List[str] = []
    for token in doc:
        pos = token.pos_
        if pos in KEPT_POS and pos not in DISCARDED_POS and not token.is_punct and not token.is_space:
            lemma = token.lemma_.strip().lower()
            if lemma:
                lemmas.append(lemma)
    return " ".join(lemmas)


def normalize_query(text: str, nlp: Optional["Language"] = None) -> str:
    """Normaliza una consulta de texto libre aplicando exactamente el mismo pipeline:

    limpieza de caracteres, tokenizacion, filtrado por POS y lematizacion.
    Reutilizable para el motor de recuperacion (Fase 3).
    """
    tokens = tokenize_and_normalize(text, nlp=nlp)
    return " ".join(tokens)


def normalize_corpus(
    raw_corpus_path: Union[str, Path] = "raw_corpus.tsv",
    output_path: Union[str, Path] = "normalized_corpus.tsv",
    model_name: str = "es_core_news_md",
    batch_size: int = 100,
) -> Union["pd.DataFrame", List[dict]]:
    """Normaliza el corpus crudo utilizando nlp.pipe() para procesamiento en lote:

    - Mantiene intacta la columna 'title' (sin lematizar ni podar).
    - Limpia caracteres y aplica tokenización, filtrado POS y lematización sobre 'text'.
    - Conserva las 9 columnas oficiales y exporta el archivo TSV final.
    """
    raw_path = Path(raw_corpus_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"El archivo de corpus crudo {raw_corpus_path} no existe. Ejecuta primero la Fase 1.")

    with open(raw_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        records: List[dict] = list(reader)

    if not records:
        raise ValueError("El archivo de corpus crudo está vacío.")

    nlp = get_spacy_pipeline(model_name=model_name)

    # Limpieza preliminar de texto antes de entrar al pipeline de spaCy
    cleaned_texts = [clean_whitespace(rec.get("text", "")) for rec in records]

    print(f"[Fase 2] Procesando {len(records)} fragmentos con nlp.pipe() (batch_size={batch_size})...")
    docs = nlp.pipe(cleaned_texts, batch_size=batch_size)

    for rec, doc in zip(records, docs):
        # La columna 'title' se mantiene exactamente con su texto original
        rec["text"] = normalize_doc(doc)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)

    print(f"[Fase 2] Corpus normalizado exportado exitosamente en: {output_path}")
    print(f"[Fase 2] Total de registros normalizados: {len(records)}")

    if pd is not None:
        return pd.DataFrame(records)
    return records


if __name__ == "__main__":
    normalize_corpus()
